"""Spacing-vs-error correlation across ALL registered datasets.

For every dataset it trains `--runs` full-strength TM + NN models on that
dataset's fixed split, averages each test point's absolute error over the runs
(to denoise), then correlates every spacing definition (standardized feature
space) with each model's averaged |error|.

Outputs:
  * a consolidated table on stdout
  * outputs/spacing_error_correlation.csv   (machine readable)
  * outputs/spacing_error_correlation.md    (table for reading / catch-up)

Usage:
  python correlation_summary.py                 # 1 run per dataset (fast-ish)
  python correlation_summary.py --runs 3        # average over 3 runs (steadier)
  python correlation_summary.py --datasets ccpp california energy
"""
import argparse
import csv
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # silence tmu/pycuda CPU-fallback noise
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import pearsonr, spearmanr

import config
from common import datasets, metrics
from common.data import get_split_data, standardize
from models.tm_model import TMModel
from models.nn_model import NNModel

SPACINGS = ["nn_to_train_k1", "knn_to_train_k5", "local_density_k5", "nn_to_test_k1"]


def corr(a, b, n):
    if n < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1, help="models per dataset; |error| averaged over them")
    p.add_argument("--datasets", nargs="+", default=list(datasets.DATASETS))
    args = p.parse_args()

    rows = []
    for key in args.datasets:
        Xtr, Xte, ytr, yte = get_split_data(key)
        _, Xtr_s, Xte_s = standardize(Xtr, Xte)
        Xall_s = np.vstack([Xtr_s, Xte_s])
        sp = metrics.all_spacings(Xte_s, Xtr_s, Xall_s)
        n = len(yte)

        tm_err = np.zeros(n)
        nn_err = np.zeros(n)
        tm_rmse, nn_rmse = [], []
        for r in range(args.runs):
            tm = TMModel().fit(Xtr, ytr)
            nn = NNModel(seed=config.RANDOM_SEED + r).fit(Xtr, ytr)
            tp, npd = tm.predict(Xte), nn.predict(Xte)
            tm_err += np.abs(tp - yte)
            nn_err += np.abs(npd - yte)
            tm_rmse.append(metrics.rmse(yte, tp))
            nn_rmse.append(metrics.rmse(yte, npd))
        tm_err /= args.runs
        nn_err /= args.runs

        for s in SPACINGS:
            tmp, tms = corr(sp[s], tm_err, n)
            nnp, nns = corr(sp[s], nn_err, n)
            rows.append(dict(dataset=key, n_test=n, spacing=s,
                             TM_pearson=round(tmp, 3), TM_spearman=round(tms, 3),
                             NN_pearson=round(nnp, 3), NN_spearman=round(nns, 3)))
        print(f"[done] {key:11s} n_test={n:5d}  RMSE TM={np.mean(tm_rmse):.3f} NN={np.mean(nn_rmse):.3f}",
              flush=True)

    cols = ["dataset", "n_test", "spacing", "TM_pearson", "TM_spearman", "NN_pearson", "NN_spearman"]
    csv_path = os.path.join(config.OUTPUTS_DIR, "spacing_error_correlation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    md_path = os.path.join(config.OUTPUTS_DIR, "spacing_error_correlation.md")
    with open(md_path, "w") as f:
        f.write(f"# Spacing vs |error| correlation (standardized space, runs={args.runs})\n\n")
        f.write("Pearson / Spearman of each spacing definition against each model's "
                "mean absolute error. |corr| under ~0.1, or any dataset with small "
                "n_test, is essentially noise.\n\n")
        f.write("| dataset | n_test | spacing | TM Pearson | TM Spearman | NN Pearson | NN Spearman |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['n_test']} | {r['spacing']} | "
                    f"{r['TM_pearson']} | {r['TM_spearman']} | {r['NN_pearson']} | {r['NN_spearman']} |\n")

    print(f"\n=== Pearson/Spearman: spacing vs |error| (standardized space, runs={args.runs}) ===")
    print(f"{'dataset':11s} {'n':>5s} {'spacing':17s} {'TM_pear':>8s} {'TM_spr':>7s} {'NN_pear':>8s} {'NN_spr':>7s}")
    for r in rows:
        print(f"{r['dataset']:11s} {r['n_test']:5d} {r['spacing']:17s} "
              f"{r['TM_pearson']:8} {r['TM_spearman']:7} {r['NN_pearson']:8} {r['NN_spearman']:7}")
    print(f"\nCSV -> {csv_path}\nMD  -> {md_path}")


if __name__ == "__main__":
    main()
