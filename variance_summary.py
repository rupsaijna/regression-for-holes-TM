"""Spacing-VARIANCE vs performance, across all registered datasets.

Tests a different hypothesis from correlation_summary.py. That script asked
"does a point being *far* from its neighbours predict larger error?". This one
asks "does a point's neighbourhood being *uneven* (some neighbours close, some
far) predict larger error?", at two scales:

  (B) per-point  -- correlate each test point's local spacing DISPERSION
                    (CV / d_k-over-d_1 / std of its 5 nearest-train distances)
                    with that point's |error|, within each dataset.
  (A) per-dataset -- correlate a dataset-level spacing-heterogeneity scalar with
                    that dataset's normalised performance, ACROSS datasets
                    (weak: ~6 reliable datasets, heavily confounded).

Outputs:
  * outputs/spacing_variance_correlation.csv / .md   (B table + A section)

Usage:
  python variance_summary.py                 # 1 model per dataset (quick)
  python variance_summary.py --runs 10       # denoise |error| over 10 models
  python variance_summary.py --datasets ccpp nyse airquality
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

# per-point local-dispersion metrics (see common/metrics.variance_spacings)
VAR_SPACINGS = ["knn_cv_k5", "knn_ratio_k5", "knn_std_k5"]
RELIABLE_N = 100  # datasets with n_test below this are flagged as unreliable for (A)


def corr(a, b):
    """Pearson, Spearman over the finite-in-both entries (NaN-safe)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1, help="models per dataset; |error| averaged over them")
    p.add_argument("--datasets", nargs="+",
                   default=[k for k in datasets.DATASETS if k != "stock"])
    args = p.parse_args()

    rows = []          # (B) per-point correlation rows
    per_ds = []        # (A) per-dataset heterogeneity + performance
    for key in args.datasets:
        Xtr, Xte, ytr, yte = get_split_data(key)
        _, Xtr_s, Xte_s = standardize(Xtr, Xte)
        vsp = metrics.variance_spacings(Xte_s, Xtr_s, k=5)
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

        for s in VAR_SPACINGS:
            tmp, tms = corr(vsp[s], tm_err)
            nnp, nns = corr(vsp[s], nn_err)
            rows.append(dict(dataset=key, n_test=n, metric=s,
                             TM_pearson=round(tmp, 3), TM_spearman=round(tms, 3),
                             NN_pearson=round(nnp, 3), NN_spearman=round(nns, 3)))

        # (A) dataset-level scalars
        std_y = float(np.std(yte)) or float("nan")
        nrmse_tm = float(np.mean(tm_rmse)) / std_y
        nrmse_nn = float(np.mean(nn_rmse)) / std_y
        # spacing-heterogeneity, two notions:
        sp5 = metrics.spacing_nn_to_train(Xte_s, Xtr_s, k=5)          # per-point mean spacing
        het_global = float(np.std(sp5) / np.mean(sp5))               # spacing varies BETWEEN points
        het_local = float(np.nanmedian(vsp["knn_cv_k5"]))            # typical WITHIN-neighbourhood unevenness
        per_ds.append(dict(dataset=key, n_test=n,
                           het_global_cv=round(het_global, 3), het_local_cv=round(het_local, 3),
                           norm_rmse_tm=round(nrmse_tm, 3), norm_rmse_nn=round(nrmse_nn, 3)))
        print(f"[done] {key:11s} n={n:5d}  het_global={het_global:.3f} het_local={het_local:.3f}  "
              f"nRMSE TM={nrmse_tm:.3f} NN={nrmse_nn:.3f}", flush=True)

    # ---- write (B) table -------------------------------------------------
    bcols = ["dataset", "n_test", "metric", "TM_pearson", "TM_spearman", "NN_pearson", "NN_spearman"]
    csv_path = os.path.join(config.OUTPUTS_DIR, "spacing_variance_correlation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols); w.writeheader(); w.writerows(rows)

    # ---- compute (A) across-dataset correlations -------------------------
    def across(field_x, field_y, subset):
        xs = np.array([d[field_x] for d in subset], dtype=np.float64)
        ys = np.array([d[field_y] for d in subset], dtype=np.float64)
        return corr(xs, ys)

    reliable = [d for d in per_ds if d["n_test"] >= RELIABLE_N]
    a_lines = []
    for label, subset in [("all datasets", per_ds), (f"reliable (n>={RELIABLE_N})", reliable)]:
        for hx in ("het_global_cv", "het_local_cv"):
            for hy in ("norm_rmse_tm", "norm_rmse_nn"):
                pr, sp = across(hx, hy, subset)
                a_lines.append((label, len(subset), hx, hy, round(pr, 3), round(sp, 3)))

    # ---- write markdown --------------------------------------------------
    md_path = os.path.join(config.OUTPUTS_DIR, "spacing_variance_correlation.md")
    with open(md_path, "w") as f:
        f.write(f"# Spacing-VARIANCE vs performance (standardized space, runs={args.runs})\n\n")
        f.write("Does *uneven* spacing (some neighbours close, some far) predict error, "
                "rather than just *large* spacing? Compare with `spacing_error_correlation.md` "
                "(which uses mean spacing). |corr| < ~0.1 or small n_test = noise.\n\n")
        f.write("## (B) Per-point: local spacing dispersion vs |error|\n\n")
        f.write("`knn_cv_k5` = std/mean of the 5 nearest-train distances; "
                "`knn_ratio_k5` = d5/d1; `knn_std_k5` = raw std (absolute).\n\n")
        f.write("| dataset | n_test | metric | TM Pearson | TM Spearman | NN Pearson | NN Spearman |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['n_test']} | {r['metric']} | "
                    f"{r['TM_pearson']} | {r['TM_spearman']} | {r['NN_pearson']} | {r['NN_spearman']} |\n")
        f.write("\n## (A) Per-dataset: spacing heterogeneity vs normalised RMSE\n\n")
        f.write("`het_global_cv` = CV of per-point mean-spacing across the test set "
                "(some regions dense, some sparse); `het_local_cv` = median within-neighbourhood CV. "
                "`norm_rmse` = RMSE / std(y). **Weak evidence: few datasets, heavy confounding.**\n\n")
        f.write("| dataset | n_test | het_global_cv | het_local_cv | norm_rmse_tm | norm_rmse_nn |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d in per_ds:
            f.write(f"| {d['dataset']} | {d['n_test']} | {d['het_global_cv']} | {d['het_local_cv']} | "
                    f"{d['norm_rmse_tm']} | {d['norm_rmse_nn']} |\n")
        f.write("\n**Across-dataset correlation (heterogeneity vs norm_rmse):**\n\n")
        f.write("| subset | n_ds | heterogeneity | performance | Pearson | Spearman |\n")
        f.write("|---|---|---|---|---|---|\n")
        for label, nds, hx, hy, pr, sp in a_lines:
            f.write(f"| {label} | {nds} | {hx} | {hy} | {pr} | {sp} |\n")

    print(f"\nCSV -> {csv_path}\nMD  -> {md_path}")
    print("\n=== (A) across-dataset heterogeneity vs norm_rmse ===")
    for label, nds, hx, hy, pr, sp in a_lines:
        print(f"  {label:22s} n={nds:2d}  {hx:14s} vs {hy:13s}  Pearson={pr:6}  Spearman={sp:6}")


if __name__ == "__main__":
    main()
