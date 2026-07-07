"""Experiment 6 — can teacher-guided training make the FULL-data model M100 better?

A weaker teacher can't help M100, so we test four teacher SOURCES that are >= or
complementary to a single full-data model. The fit() teacher path is reused as-is;
only the teacher prediction VECTOR (encoded, over the train rows) changes.

  A_self    : teacher = this seed's own unguided M100 (in-fold)        [self-distillation]
  A_labels  : teacher = the labels (pe_teacher=0 -> pure feedback attenuation) [control]
  B_oof     : teacher = K-fold OUT-OF-FOLD M100 predictions            [no memorization]
  C_ens     : teacher = mean of E full-data M100 preds (ensemble)      [genuinely better]
  D_bigTM   : teacher = a higher-capacity TM (clause_mult x clauses)   [capacity]
  D_nn      : teacher = the PyTorch MLP's predictions (encoded)        [cross-model]

Protocol: train pool b = full train split; evaluate every model on the held-out TEST
split. Guided vs unguided are PAIRED by seed (cancels TM run-to-run noise). Report the
paired improvement = rmse(unguided) - rmse(guided)  (>0 == the teacher helped).

Outputs:
  outputs/distillation/exp6/<dataset>__makebetter.csv     (per-seed rows)
  outputs/distillation/exp6/makebetter_summary.csv / .md  (per dataset x method)
  outputs/distillation/exp6/makebetter__<dataset>.png

Usage:
  python experiment6_makebetter.py --datasets ccpp california energy --seeds 8 --epochs 100
"""
import argparse
import csv
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor

import config
from common import datasets, metrics
from common.data import get_split_data
from models.nn_model import NNModel
from experiment5_distillation import teacher_encoded_predictions

METHODS = ["A_self", "A_labels", "B_oof", "C_ens", "D_bigTM", "D_nn"]


def encode_y(y, g_min, g_max, T):
    return np.clip(((np.asarray(y, float) - g_min) / (g_max - g_min) * T).round().astype(np.int64), 0, T)


def train_tm(Xb, y, epochs, g_min, g_max, seed, num_clauses=config.TM_NUM_CLAUSES,
             teacher_vec=None, f=0.0, f_opposite=0.0, p=0.0):
    tm = TMRegressor(num_clauses, config.TM_T, config.TM_S, platform="CPU",
                     weighted_clauses=config.TM_WEIGHTED_CLAUSES, seed=seed)
    trng = np.random.RandomState(seed + 777)
    for _ in range(epochs):
        tm.fit(Xb, y, teacher_pred_encoded=teacher_vec, f=f, f_opposite=f_opposite,
               p_teacher=p, teacher_rng=trng, global_y_min=g_min, global_y_max=g_max)
    return tm


# --- teacher-vector factories (all return encoded preds over the train rows) ---
def teacher_oof(Xb, y, epochs, g_min, g_max, seed, K=5):
    oof = np.zeros(len(Xb), dtype=np.int64)
    for tr, te in KFold(n_splits=K, shuffle=True, random_state=seed).split(Xb):
        m = train_tm(Xb[tr], y[tr], epochs, g_min, g_max, seed)
        oof[te] = teacher_encoded_predictions(m, Xb[te])
    return oof


def teacher_ensemble(Xb, y, Xb_te, epochs, g_min, g_max, seeds):
    """Return (encoded train-pred vector, mean real test prediction) over E models."""
    tr_preds, te_preds = [], []
    for s in seeds:
        m = train_tm(Xb, y, epochs, g_min, g_max, s)
        tr_preds.append(teacher_encoded_predictions(m, Xb))
        te_preds.append(m.predict(Xb_te))
    return np.mean(tr_preds, axis=0).round().astype(np.int64), np.mean(te_preds, axis=0)


def teacher_bigtm(Xb, y, epochs, g_min, g_max, seed, mult):
    m = train_tm(Xb, y, epochs, g_min, g_max, seed, num_clauses=config.TM_NUM_CLAUSES * mult)
    return teacher_encoded_predictions(m, Xb), m


def teacher_nn(Xtr_raw, y, g_min, g_max, T, seed):
    nn = NNModel(seed=seed); nn.fit(Xtr_raw, y)
    return encode_y(nn.predict(Xtr_raw), g_min, g_max, T), nn


def run_dataset(key, args):
    Xtr_raw, Xte_raw, ytr, yte = get_split_data(key)
    binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
    Xb = binr.fit_transform(Xtr_raw).astype(np.uint32)
    Xb_te = binr.transform(Xte_raw).astype(np.uint32)
    g_min, g_max, T = float(ytr.min()), float(ytr.max()), config.TM_T
    print(f"\n=== {key}: {len(ytr)} train / {len(yte)} test ===", flush=True)

    # shared (knob/seed-independent) teacher vectors, built once
    tv_labels = encode_y(ytr, g_min, g_max, T)
    tv_oof = teacher_oof(Xb, ytr, args.epochs, g_min, g_max, 999, K=args.K)
    tv_ens, ens_te = teacher_ensemble(Xb, ytr, Xb_te, args.epochs, g_min, g_max, list(range(100, 100 + args.E)))
    tv_big, big_m = teacher_bigtm(Xb, ytr, args.epochs, g_min, g_max, 999, args.clause_mult)
    tv_nn, nn_m = teacher_nn(Xtr_raw, ytr, g_min, g_max, T, 999)
    # reference test RMSEs of the strong teachers (context / upper bounds)
    ref = dict(bigTM=metrics.rmse(yte, big_m.predict(Xb_te)),
               nn=metrics.rmse(yte, nn_m.predict(Xte_raw)),
               ens=metrics.rmse(yte, ens_te))
    print(f"  ref test RMSE  bigTM={ref['bigTM']:.4f}  nn={ref['nn']:.4f}  ens={ref['ens']:.4f}", flush=True)

    shared = {"A_labels": tv_labels, "B_oof": tv_oof, "C_ens": tv_ens, "D_bigTM": tv_big, "D_nn": tv_nn}
    rows = []
    for seed in [config.RANDOM_SEED + i for i in range(args.seeds)]:
        base = train_tm(Xb, ytr, args.epochs, g_min, g_max, seed)
        rb = metrics.rmse(yte, base.predict(Xb_te))
        tv_self = teacher_encoded_predictions(base, Xb)
        for name in METHODS:
            tv = tv_self if name == "A_self" else shared[name]
            g = train_tm(Xb, ytr, args.epochs, g_min, g_max, seed,
                         teacher_vec=tv, f=args.f, f_opposite=args.f_opposite, p=args.p_teacher)
            rg = metrics.rmse(yte, g.predict(Xb_te))
            rows.append(dict(dataset=key, seed=seed, method=name,
                             rmse_base=round(rb, 5), rmse_guided=round(rg, 5),
                             improvement=round(rb - rg, 5)))
        print(f"  seed {seed}: base={rb:.4f} | "
              + " ".join(f"{m}={[r['improvement'] for r in rows if r['seed']==seed and r['method']==m][0]:+.3f}"
                         for m in METHODS), flush=True)
    return rows, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["energy", "ccpp", "california"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--f", type=float, default=0.5)
    ap.add_argument("--f-opposite", type=float, default=0.0)
    ap.add_argument("--p-teacher", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=5, help="OOF folds")
    ap.add_argument("--E", type=int, default=5, help="ensemble size")
    ap.add_argument("--clause-mult", type=int, default=4, help="bigTM clause multiplier")
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation", "exp6"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for key in args.datasets:
        rows, ref = run_dataset(key, args)
        with open(os.path.join(args.out, f"{key}__makebetter.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        for m in METHODS:
            imp = np.array([r["improvement"] for r in rows if r["method"] == m])
            t = float(np.mean(imp) / (np.std(imp) / np.sqrt(len(imp)))) if np.std(imp) > 0 else 0.0
            summary.append(dict(dataset=key, method=m, n=len(imp),
                                mean_improvement=round(float(np.mean(imp)), 5),
                                std=round(float(np.std(imp)), 5),
                                winrate=round(float(np.mean(imp > 0)), 3),
                                t_stat=round(t, 2),
                                works=("YES" if np.mean(imp) > 0 and t > 2 else
                                       "hurts" if np.mean(imp) < 0 and t < -2 else "~ns")))
        _plot(key, rows, ref, args.out)

    with open(os.path.join(args.out, "makebetter_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    _write_md(summary, args)
    print("\n===== SUMMARY (paired improvement = base - guided, >0 helps) =====")
    print(f"  {'dataset':12s} {'method':9s} {'mean':>8s} {'std':>7s} {'win':>5s} {'t':>6s}  works")
    for s in summary:
        print(f"  {s['dataset']:12s} {s['method']:9s} {s['mean_improvement']:+8.4f} {s['std']:7.4f} "
              f"{s['winrate']:5.2f} {s['t_stat']:+6.2f}  {s['works']}")


def _plot(key, rows, ref, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [np.mean([r["improvement"] for r in rows if r["method"] == m]) for m in METHODS]
    stds = [np.std([r["improvement"] for r in rows if r["method"] == m]) for m in METHODS]
    colors = ["#0984e3" if x > 0 else "#d63031" for x in means]
    ax.bar(METHODS, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("paired test-RMSE improvement (base - guided)")
    ax.set_title(f"{key}: making M100 better via teacher source  "
                 f"(>0 helps; refs bigTM={ref['bigTM']:.3f} nn={ref['nn']:.3f} ens={ref['ens']:.3f})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"makebetter__{key}.png"), dpi=120); plt.close(fig)


def _write_md(summary, args):
    with open(os.path.join(args.out, "makebetter_summary.md"), "w") as fh:
        fh.write(f"# Experiment 6 — making M100 better (f={args.f}, p={args.p_teacher}, "
                 f"f_opposite={args.f_opposite}, {args.seeds} seeds, {args.epochs}ep)\n\n")
        fh.write("Paired improvement = rmse(unguided M100) - rmse(guided M100) on held-out TEST; "
                 ">0 = teacher helps. `works` = mean>0 & t>2 (paired).\n\n")
        fh.write("| dataset | method | mean impr | std | winrate | t | verdict |\n|---|---|---|---|---|---|---|\n")
        for s in summary:
            fh.write(f"| {s['dataset']} | {s['method']} | {s['mean_improvement']:+.4f} | {s['std']:.4f} "
                     f"| {s['winrate']} | {s['t_stat']:+.2f} | {s['works']} |\n")
        fh.write("\n_NOTE: single config (f, p) only — winners deserve an f/p sweep. "
                 "A_labels is the pure-attenuation control; A_self - A_labels = dark-knowledge gain._\n")


if __name__ == "__main__":
    main()
