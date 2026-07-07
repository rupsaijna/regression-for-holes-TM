"""Phase-7 follow-up: a FAIR effort-to-recover metric.

The masking experiment (experiment4_masking.py) measured "epochs to reach the
full-data target", where the target was each model's RMSE at its DEFAULT epoch
budget (TM=30, NN=100). For TM that bar is loose -- TM keeps improving well past
30 epochs -- so a masked TM "reaches" it in a couple of epochs trivially, which
makes the TM effort metric uninformative.

This recomputes effort against a FAIRER bar: each model's own *best* RMSE within
the EXTENDED budget on full data (the min over its extended-budget convergence
curve). Baseline and masked models are trained with fit_with_history at the
extended budget; we report, per (dataset, model, mode, p), the epochs to reach
the extended target alongside the old default-target effort for comparison.

Clean datasets only (ccpp, california) -- on heavy-tailed nyse effort is
uninformative regardless (the tail, not geometry, sets the error; see
masking_summary.md), so it is excluded here.

Outputs:
  outputs/masking/effort/<ds>__effort.csv
  outputs/masking/effort/<ds>__effort_vs_p.png   (extended-target effort, TM & NN)
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

import config
from common import datasets, masking
from common.data import get_split_data, standardize
from models.tm_model import TMModel
from models.nn_model import NNModel

MODELS = ["TM", "NN"]
DEFAULT_EPOCHS = {"TM": config.TM_EPOCHS, "NN": config.NN_EPOCHS}        # 30 / 100
EXTENDED_EPOCHS = {"TM": 100, "NN": 300}                                 # matches experiment4


def _first_at_or_below(hist, target):
    for i, v in enumerate(hist):
        if v <= target:
            return i + 1
    return None  # censored: never reached within the extended budget


def _make_model(name, seed):
    return TMModel() if name == "TM" else NNModel(seed=seed)


def run_dataset(key, args):
    Xtr, Xte, ytr, yte = get_split_data(key)
    _, Xtr_s, _ = standardize(Xtr, Xte)
    n = len(ytr)
    print(f"\n=== {key}: {n} train / {len(yte)} test ===", flush=True)

    rows = []
    for model_name in MODELS:
        ext = EXTENDED_EPOCHS[model_name]
        deph = DEFAULT_EPOCHS[model_name]
        # per-repeat full-data baseline at the EXTENDED budget; the first `deph`
        # entries of this curve ARE the default-budget curve (same training run).
        tgt_ext, tgt_def = {}, {}
        for r in range(args.repeats):
            m = _make_model(model_name, seed=config.RANDOM_SEED + r)
            hist = m.fit_with_history(Xtr, ytr, Xte, yte, max_epochs=ext)
            tgt_ext[r] = float(np.min(hist))                 # fair bar: best on full data
            tgt_def[r] = float(hist[deph - 1])               # old bar: default-epoch RMSE
            rows.append(dict(dataset=key, model=model_name, mode="none", p=0.0, repeat=r,
                             target_ext=round(tgt_ext[r], 6), target_def=round(tgt_def[r], 6),
                             e2t_ext="", e2t_def="", rmse_final=round(float(hist[-1]), 6),
                             censored_ext=0, censored_def=0))
            print(f"  baseline {model_name} r{r}: target_ext={tgt_ext[r]:.5g} "
                  f"target_def={tgt_def[r]:.5g}", flush=True)

        for frac in args.fractions:
            if frac <= 0:
                continue
            for mode in args.modes:
                for r in range(args.repeats):
                    if mode == "uniform":
                        kept, _ = masking.mask_uniform(n, frac, seed=r)
                    else:
                        kept, _ = masking.mask_structured(Xtr_s, frac, seed=r,
                                                          hole_frac=args.hole_frac)
                    m = _make_model(model_name, seed=config.RANDOM_SEED + r)
                    hist = m.fit_with_history(Xtr[kept], ytr[kept], Xte, yte, max_epochs=ext)
                    e_ext = _first_at_or_below(hist, tgt_ext[r])
                    e_def = _first_at_or_below(hist, tgt_def[r])
                    rows.append(dict(
                        dataset=key, model=model_name, mode=mode, p=frac, repeat=r,
                        target_ext=round(tgt_ext[r], 6), target_def=round(tgt_def[r], 6),
                        e2t_ext=(e_ext if e_ext else ""), e2t_def=(e_def if e_def else ""),
                        rmse_final=round(float(hist[-1]), 6),
                        censored_ext=int(e_ext is None), censored_def=int(e_def is None)))
                print(f"  p={frac:.2f} {mode:10s} {model_name}: done "
                      f"(ext e2t mean over reps)", flush=True)
    return rows


def _agg(rows, model, mode, field):
    ps = sorted({r["p"] for r in rows if r["mode"] == mode})
    out_ps, means, stds = [], [], []
    for pv in ps:
        vals = [r[field] for r in rows if r["model"] == model and r["mode"] == mode
                and r["p"] == pv and isinstance(r[field], (int, float))]
        if vals:
            out_ps.append(pv); means.append(float(np.mean(vals))); stds.append(float(np.std(vals)))
    return out_ps, means, stds


def _censor_rate(rows, model, mode, field):
    ps = sorted({r["p"] for r in rows if r["mode"] == mode})
    out = []
    for pv in ps:
        sel = [r for r in rows if r["model"] == model and r["mode"] == mode and r["p"] == pv]
        if sel:
            out.append((pv, sum(r[field] for r in sel), len(sel)))
    return out


def plot_dataset(key, rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model in zip(axes, MODELS):
        for mode in ("uniform", "structured"):
            ps, means, stds = _agg(rows, model, mode, "e2t_ext")
            if ps:
                ax.errorbar(ps, means, yerr=stds, marker="o", capsize=3, label=mode)
        # annotate censoring (reached-never) counts on the extended target
        notes = []
        for mode in ("uniform", "structured"):
            cr = _censor_rate(rows, model, mode, "censored_ext")
            cens = [f"p{p}:{c}/{t}" for p, c, t in cr if c]
            if cens:
                notes.append(f"{mode} censored " + ", ".join(cens))
        ax.set_title(f"{key} — {model}: effort vs p (FAIR extended-budget target)"
                     + ("\n" + " | ".join(notes) if notes else ""))
        ax.set_xlabel("masking fraction p")
        ax.set_ylabel("epochs to reach own extended-budget full-data RMSE")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{key}__effort_vs_p.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"plot -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ccpp", "california"])
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["uniform", "structured"])
    ap.add_argument("--hole-frac", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = os.path.join(config.OUTPUTS_DIR, "masking", "effort")
    os.makedirs(out_dir, exist_ok=True)
    cols = ["dataset", "model", "mode", "p", "repeat", "target_ext", "target_def",
            "e2t_ext", "e2t_def", "rmse_final", "censored_ext", "censored_def"]
    for key in args.datasets:
        rows = run_dataset(key, args)
        csv_path = os.path.join(out_dir, f"{key}__effort.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"CSV -> {csv_path}", flush=True)
        plot_dataset(key, rows, out_dir)


if __name__ == "__main__":
    main()
