"""Experiment 11 — teacher-bounded magnitude, generalized.

E10's transferable finding: sourcing the update MAGNITUDE from the frozen teacher's (bounded)
error is what both stabilized teacher_sign AND softened bad-teacher harm. E11 applies that lever
two more ways and asks whether it helps the modes that don't yet use it:

  force_ii_tmag : E9's best single win (force_ii, always Type-II on opposite-side) but with the
                  update magnitude taken from the teacher's error. Does it KEEP force_ii's
                  airquality win while CUTTING the energy harm (the way tmag did for teacher_sign)?
  sameside_tmag : opposite-side untouched (= uniform), but on SAME-side teacher samples the update
                  magnitude is taken from the teacher's error. Isolates whether teacher-bounded
                  magnitude helps on its own, independent of any opposite-side override.

Arms (masked-student task, teacher 20% -> student 30% mask, structured, f=0.75, p=0.5, 100ep,
held-out TEST, paired by seed):
  unguided | uniform (= E8/E9 baseline) | force_ii (E9 ref) | force_ii_tmag (NEW)
           | teacher_sign_tmag (E10 winner, ref) | sameside_tmag (NEW)

Out: outputs/distillation/exp11/{<ds>__tmag.csv, tmag_summary.{csv,md}, tmag__<ds>.png}
"""
import argparse, csv, logging, os, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor_v4 import TMRegressor   # V4: + force_ii_tmag / same_side_tmag

import config
from common import datasets, metrics
from common.data import get_split_data
from experiment5_distillation import nested_structured, teacher_encoded_predictions

# method -> (opposite_mode, same_side_tmag)
CFG = {
    "uniform":           (None,                 False),
    "force_ii":          ("force_ii",           False),
    "force_ii_tmag":     ("force_ii_tmag",      False),
    "teacher_sign_tmag": ("teacher_sign_tmag",  False),
    "sameside_tmag":     (None,                 True),
}
METHODS = list(CFG.keys())


def train_student(Xb, y, epochs, g_min, g_max, seed, teacher_vec=None, f=0.0, p=0.0,
                  opposite_mode=None, same_side_tmag=False):
    tm = TMRegressor(config.TM_NUM_CLAUSES, config.TM_T, config.TM_S, platform="CPU",
                     weighted_clauses=config.TM_WEIGHTED_CLAUSES, seed=seed)
    trng = np.random.RandomState(seed + 777)
    for ep in range(epochs):
        tm.fit(Xb, y, teacher_pred_encoded=teacher_vec, f=f, f_opposite=0.0, p_teacher=p,
               opposite_mode=opposite_mode, same_side_tmag=same_side_tmag,
               teacher_rng=trng, global_y_min=g_min, global_y_max=g_max)
    return tm


def run_dataset(key, args):
    Xtr, Xte, ytr, yte = get_split_data(key)
    if getattr(args, "subsample_train", 0) and len(ytr) > args.subsample_train:
        ridx = np.random.default_rng(12345).choice(len(ytr), args.subsample_train, replace=False)
        Xtr, ytr = Xtr[ridx], ytr[ridx]
        print(f"  [subsampled train -> {args.subsample_train}]", flush=True)
    binr = StandardBinarizer(max_bits_per_feature=config.TM_MAX_BITS_PER_FEATURE)
    Xb, Xb_te = binr.fit_transform(Xtr).astype(np.uint32), binr.transform(Xte).astype(np.uint32)
    Xs = StandardScaler().fit_transform(Xtr)
    g_min, g_max, n = float(ytr.min()), float(ytr.max()), len(ytr)
    print(f"\n=== {key}: {n} train / {len(yte)} test ===", flush=True)

    rows = []
    for seed in [config.RANDOM_SEED + i for i in range(args.seeds)]:
        rem_t, rem_s = nested_structured(Xs, [0.2, 0.3], seed, hole_frac=0.05)
        kept_t, kept_s = np.setdiff1d(np.arange(n), rem_t), np.setdiff1d(np.arange(n), rem_s)
        Xb_s, y_s = Xb[kept_s], ytr[kept_s]
        M00 = train_student(Xb, ytr, args.epochs, g_min, g_max, seed)        # FULL data, no hole (ceiling)
        M20 = train_student(Xb[kept_t], ytr[kept_t], args.epochs, g_min, g_max, seed)
        M30u = train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed)
        tvec = teacher_encoded_predictions(M20, Xb_s)
        guided = {m: train_student(Xb_s, y_s, args.epochs, g_min, g_max, seed, tvec, args.f,
                                   args.p_teacher, opposite_mode=CFG[m][0], same_side_tmag=CFG[m][1])
                  for m in METHODS}
        ru = metrics.rmse(yte, M30u.predict(Xb_te))
        rec = dict(dataset=key, seed=seed, rmse_unguided=round(ru, 5),
                   rmse_M00=round(metrics.rmse(yte, M00.predict(Xb_te)), 5),
                   rmse_M20=round(metrics.rmse(yte, M20.predict(Xb_te)), 5))
        for m in METHODS:
            rg = metrics.rmse(yte, guided[m].predict(Xb_te))
            rec[f"rmse_{m}"] = round(rg, 5)
            rec[f"impr_{m}"] = round(ru - rg, 5)              # >0 = helps vs unguided
        rows.append(rec)
        print(f"  seed {seed}: full={rec['rmse_M00']:.4f} M20={rec['rmse_M20']:.4f} ung={ru:.4f} | "
              + " ".join(f"{m}:{rec['impr_'+m]:+.3f}" for m in METHODS), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ccpp", "airquality", "california", "energy"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--f", type=float, default=0.75)
    ap.add_argument("--p-teacher", type=float, default=0.5)
    ap.add_argument("--subsample-train", type=int, default=0,
                    help="cap training rows (0=all); use for Tier A on the large datasets")
    ap.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "distillation", "exp11"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for key in args.datasets:
        rows = run_dataset(key, args)
        with open(os.path.join(args.out, f"{key}__tmag.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        for m in METHODS:
            impr = np.array([r[f"impr_{m}"] for r in rows])
            t = float(np.mean(impr) / (np.std(impr) / np.sqrt(len(impr)))) if np.std(impr) > 0 else 0.0
            gain = impr - np.array([r["impr_uniform"] for r in rows])     # mode - uniform (paired)
            tg = float(np.mean(gain) / (np.std(gain) / np.sqrt(len(gain)))) if np.std(gain) > 0 else 0.0
            summary.append(dict(dataset=key, method=m, n=len(impr),
                                mean_impr=round(float(np.mean(impr)), 5), impr_t=round(t, 2),
                                impr_std=round(float(np.std(impr)), 5),
                                mean_gain_vs_uniform=round(float(np.mean(gain)), 5),
                                gain_t=("-" if m == "uniform" else round(tg, 2))))
        _plot(key, rows, args.out)

    with open(os.path.join(args.out, "tmag_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print("\n===== SUMMARY (impr = unguided - guided RMSE; gain = mode - uniform, paired) =====")
    print(f"  {'dataset':11s} {'method':18s} {'mean_impr':>9s} {'t':>6s} {'std':>8s} {'gain_vs_uni':>11s} {'gain_t':>7s}")
    for s in summary:
        print(f"  {s['dataset']:11s} {s['method']:18s} {s['mean_impr']:+9.4f} {s['impr_t']:+6.2f} "
              f"{s['impr_std']:8.4f} {s['mean_gain_vs_uniform']:+11.4f} {str(s['gain_t']):>7s}")


def _plot(key, rows, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    xs = ["unguided"] + METHODS
    means = [0.0] + [np.mean([r[f"impr_{m}"] for r in rows]) for m in METHODS]
    stds = [0.0] + [np.std([r[f"impr_{m}"] for r in rows]) for m in METHODS]
    ax.bar(xs, means, yerr=stds, capsize=4,
           color=["#b2bec3"] + ["#0984e3" if x >= 0 else "#d63031" for x in means[1:]])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("RMSE improvement vs unguided")
    ax.set_title(f"{key}: teacher-bounded magnitude (E11)"); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"tmag__{key}.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
