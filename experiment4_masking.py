"""Phase-7 masking experiment: remove training points stepwise and measure the
effect on final performance and on training effort. See PLAN_masking.md.

At each removal fraction `p`, the training set is masked two ways:
  * uniform    -- drop p.n points at random (control: sparser, still even)
  * structured -- carve holes (treatment: genuinely uneven)
Fresh TM + NN are trained on each (repeated R times). We record, per setting:
  * RMSE on the fixed original test set, and on its region-split
    (in-hole = local training support removed, vs surviving)
  * epochs / minibatch-iterations / wall-time to reach the full-data target RMSE
    (censored if never reached within the extended epoch budget)

Outputs:
  outputs/masking/<dataset>__curve.csv
  outputs/masking/<dataset>__rmse_vs_p.png, __gap_vs_p.png, __effort_vs_p.png

Usage:
  python experiment4_masking.py --dataset energy            # quick prototype
  python experiment4_masking.py --dataset ccpp --repeats 3
  python experiment4_masking.py --dataset nyse --fractions 0 0.2 0.4 0.6 --repeats 2
"""
import argparse
import csv
import logging
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from common import datasets, masking, metrics
from common.data import get_split_data, standardize
from models.tm_model import TMModel
from models.nn_model import NNModel

MODELS = ["TM", "NN"]


def _epochs_to_target(hist, target):
    for i, v in enumerate(hist):
        if v <= target:
            return i + 1
    return None  # censored: never reached


def _rmse_sub(y, pred, mask):
    return metrics.rmse(y[mask], pred[mask]) if mask.any() else float("nan")


def _train(model_name, Xtr_m, ytr_m, Xte, yte, max_epochs, seed):
    if model_name == "TM":
        m = TMModel()
    else:
        m = NNModel(seed=seed)
    t0 = time.perf_counter()
    hist = m.fit_with_history(Xtr_m, ytr_m, Xte, yte, max_epochs=max_epochs)
    secs = time.perf_counter() - t0
    preds = m.predict(Xte)
    return hist, preds, secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="energy", choices=list(datasets.DATASETS))
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--modes", nargs="+", default=["uniform", "structured"])
    p.add_argument("--hole-frac", type=float, default=0.05, help="structured hole size (fraction of n per hole)")
    p.add_argument("--k", type=int, default=5, help="neighbours for the in-hole test-point split")
    p.add_argument("--max-epochs-nn", type=int, default=300, help="extended NN budget for masked runs")
    p.add_argument("--max-epochs-tm", type=int, default=100, help="extended TM budget for masked runs")
    args = p.parse_args()

    key = args.dataset
    Xtr, Xte, ytr, yte = get_split_data(key)
    _, Xtr_s, Xte_s = standardize(Xtr, Xte)
    n_train = len(ytr)
    max_ep = {"TM": args.max_epochs_tm, "NN": args.max_epochs_nn}
    print(f"Dataset '{key}': {n_train} train / {len(yte)} test, fractions={args.fractions}, "
          f"modes={args.modes}, repeats={args.repeats}", flush=True)

    # --- baseline (p=0): the full-data target RMSE per model (standard epoch budget) ---
    target = {}
    rows = []
    for model_name in MODELS:
        rmses = []
        for r in range(args.repeats):
            hist, preds, secs = _train(model_name, Xtr, ytr, Xte, yte, max_epochs=None,
                                       seed=config.RANDOM_SEED + r)
            rmse0 = metrics.rmse(yte, preds)
            rmses.append(rmse0)
            rows.append(dict(dataset=key, p=0.0, mode="none", repeat=r, model=model_name,
                             n_kept=n_train, n_removed=0, n_inhole=0,
                             rmse_orig=round(rmse0, 5), rmse_inhole="", rmse_surviving="",
                             target_rmse="", epochs_to_target="", iters_to_target="",
                             secs_to_target=round(secs, 2), secs_total=round(secs, 2),
                             epochs_run=len(hist)))
        target[model_name] = float(np.mean(rmses))
        print(f"  baseline {model_name}: target RMSE = {target[model_name]:.5f}", flush=True)

    # --- masked runs ---
    for frac in args.fractions:
        if frac <= 0:
            continue
        for mode in args.modes:
            for r in range(args.repeats):
                if mode == "uniform":
                    kept, removed = masking.mask_uniform(n_train, frac, seed=r)
                else:
                    kept, removed = masking.mask_structured(Xtr_s, frac, seed=r, hole_frac=args.hole_frac)
                in_hole = masking.region_split(Xtr_s, removed, Xte_s, k=args.k)
                Xtr_m, ytr_m = Xtr[kept], ytr[kept]
                for model_name in MODELS:
                    hist, preds, secs = _train(model_name, Xtr_m, ytr_m, Xte, yte,
                                               max_epochs=max_ep[model_name],
                                               seed=config.RANDOM_SEED + r)
                    e2t = _epochs_to_target(hist, target[model_name])
                    epochs_run = len(hist)
                    if model_name == "NN":
                        per_epoch_iters = math.ceil(len(kept) / config.NN_BATCH_SIZE)
                    else:
                        per_epoch_iters = 1
                    iters = e2t * per_epoch_iters if e2t else ""
                    secs_to = round(secs * e2t / epochs_run, 2) if e2t else ""
                    rows.append(dict(
                        dataset=key, p=frac, mode=mode, repeat=r, model=model_name,
                        n_kept=len(kept), n_removed=len(removed), n_inhole=int(in_hole.sum()),
                        rmse_orig=round(metrics.rmse(yte, preds), 5),
                        rmse_inhole=round(_rmse_sub(yte, preds, in_hole), 5),
                        rmse_surviving=round(_rmse_sub(yte, preds, ~in_hole), 5),
                        target_rmse=round(target[model_name], 5),
                        epochs_to_target=(e2t if e2t else ""),
                        iters_to_target=iters, secs_to_target=secs_to,
                        secs_total=round(secs, 2), epochs_run=epochs_run))
                print(f"  p={frac:.2f} {mode:10s} r={r}: "
                      f"TM/NN done (n_kept={len(kept)}, in_hole={int(in_hole.sum())})", flush=True)

    # --- write CSV ---
    out_dir = os.path.join(config.OUTPUTS_DIR, "masking")
    os.makedirs(out_dir, exist_ok=True)
    cols = ["dataset", "p", "mode", "repeat", "model", "n_kept", "n_removed", "n_inhole",
            "rmse_orig", "rmse_inhole", "rmse_surviving", "target_rmse",
            "epochs_to_target", "iters_to_target", "secs_to_target", "secs_total", "epochs_run"]
    csv_path = os.path.join(out_dir, f"{key}__curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nCSV -> {csv_path}", flush=True)

    _plots(key, rows, out_dir, args)


def _agg(rows, model, mode, field):
    """mean, std of `field` over repeats, per p, for (model, mode). Returns (ps, means, stds)."""
    ps = sorted({r["p"] for r in rows if r["mode"] in (mode, "none")})
    means, stds = [], []
    out_ps = []
    for pv in ps:
        vals = [r[field] for r in rows
                if r["model"] == model and r["p"] == pv
                and (r["mode"] == mode or (pv == 0.0 and r["mode"] == "none"))
                and isinstance(r[field], (int, float))]
        if vals:
            out_ps.append(pv); means.append(float(np.mean(vals))); stds.append(float(np.std(vals)))
    return out_ps, means, stds


def _plots(key, rows, out_dir, args):
    # 1) RMSE vs p (orig test), per model x mode
    plt.figure(figsize=(7, 5))
    for model in MODELS:
        for mode in args.modes:
            ps, means, stds = _agg(rows, model, mode, "rmse_orig")
            if ps:
                plt.errorbar(ps, means, yerr=stds, marker="o", capsize=3,
                             label=f"{model} {mode}")
    plt.xlabel("masking fraction p"); plt.ylabel("RMSE on original test set")
    plt.title(f"{key}: final RMSE vs masking fraction"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{key}__rmse_vs_p.png"), dpi=110); plt.close()

    # 2) structured - uniform gap (the pure unevenness penalty), per model
    if "uniform" in args.modes and "structured" in args.modes:
        plt.figure(figsize=(7, 5))
        for model in MODELS:
            pu, mu, _ = _agg(rows, model, "uniform", "rmse_orig")
            psr, msr, _ = _agg(rows, model, "structured", "rmse_orig")
            common = sorted(set(pu) & set(psr))
            if common:
                du = dict(zip(pu, mu)); ds = dict(zip(psr, msr))
                plt.plot(common, [ds[c] - du[c] for c in common], marker="s", label=f"{model}")
        plt.axhline(0, color="k", lw=0.8)
        plt.xlabel("masking fraction p"); plt.ylabel("RMSE(structured) - RMSE(uniform)")
        plt.title(f"{key}: unevenness penalty (data quantity held constant)")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{key}__gap_vs_p.png"), dpi=110); plt.close()

    # 3) effort (epochs to target) vs p
    plt.figure(figsize=(7, 5))
    for model in MODELS:
        for mode in args.modes:
            ps, means, stds = _agg(rows, model, mode, "epochs_to_target")
            if ps:
                plt.errorbar(ps, means, yerr=stds, marker="o", capsize=3, label=f"{model} {mode}")
    plt.xlabel("masking fraction p"); plt.ylabel("epochs to reach full-data target (censored omitted)")
    plt.title(f"{key}: convergence effort vs masking fraction"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{key}__effort_vs_p.png"), dpi=110); plt.close()
    print(f"plots -> {out_dir}/{key}__(rmse|gap|effort)_vs_p.png", flush=True)


if __name__ == "__main__":
    main()
