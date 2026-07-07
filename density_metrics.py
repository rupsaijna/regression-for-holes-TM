"""Per-dataset DENSITY / spacing battery, for correlating against teacher
effectiveness (experiment 5). Computes a range of feature-space density/unevenness
descriptors on the standardized training features, plus pulls the Phase 5/6
"does spacing predict error" scores from the existing correlation CSVs.

One row per dataset -> outputs/distillation/density_metrics.csv.
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
from common import datasets, metrics
from common.data import get_split_data, standardize

SUBSAMPLE = 5000          # cap points for the O(n log n) neighbour computations
SEED = 12345


def norm_entropy(x, bins=20):
    h, _ = np.histogram(x, bins=bins)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))


def phase56_scores(key):
    """max |spacing<->error| correlation (NN & TM) for this dataset, from Phase 5/6."""
    out = {"spc_err_strength": np.nan, "spc_err_nn_k1": np.nan, "var_err_strength": np.nan}
    p5 = os.path.join(config.OUTPUTS_DIR, "spacing_error_correlation.csv")
    if os.path.exists(p5):
        rows = [r for r in csv.DictReader(open(p5)) if r["dataset"] == key]
        if rows:
            mags = []
            for r in rows:
                for c in ("TM_pearson", "TM_spearman", "NN_pearson", "NN_spearman"):
                    try: mags.append(abs(float(r[c])))
                    except (TypeError, ValueError): pass
                if r["spacing"] == "nn_to_train_k1":
                    try: out["spc_err_nn_k1"] = abs(float(r["NN_spearman"]))
                    except (TypeError, ValueError): pass
            if mags: out["spc_err_strength"] = max(mags)
    p6 = os.path.join(config.OUTPUTS_DIR, "spacing_variance_correlation.csv")
    if os.path.exists(p6):
        rows = [r for r in csv.DictReader(open(p6)) if r["dataset"] == key]
        mags = []
        for r in rows:
            for c in ("TM_pearson", "TM_spearman", "NN_pearson", "NN_spearman"):
                try: mags.append(abs(float(r[c])))
                except (TypeError, ValueError): pass
        if mags: out["var_err_strength"] = max(mags)
    return out


def dataset_density(key):
    Xtr, Xte, ytr, yte = get_split_data(key)
    _, Xs, _ = standardize(Xtr, Xte)
    if len(Xs) > SUBSAMPLE:
        idx = np.random.default_rng(SEED).choice(len(Xs), SUBSAMPLE, replace=False)
        Xs = Xs[idx]
    knn5 = metrics.spacing_local_density(Xs, Xs, k=5)          # per-point mean dist to 5 NN
    nn1 = metrics.spacing_nn_to_self(Xs, k=1)                  # per-point nearest-neighbour dist
    knn_cv = metrics.spacing_knn_cv(Xs, Xs, k=5)
    knn_ratio = metrics.spacing_knn_ratio(Xs, Xs, k=5)
    row = dict(
        dataset=key, n=len(Xtr), d=Xs.shape[1],
        spread=round(metrics.spread(Xs), 5),
        nn1_mean=round(float(np.mean(nn1)), 5),
        knn5_mean=round(float(np.mean(knn5)), 5),          # local spacing (inverse density)
        knn5_median=round(float(np.median(knn5)), 5),
        het_global_cv=round(float(np.std(knn5) / np.mean(knn5)), 5),   # global unevenness
        knn_cv_mean=round(float(np.nanmean(knn_cv)), 5),   # local dispersion
        knn_ratio_mean=round(float(np.nanmean(knn_ratio)), 5),
        norm_entropy=round(norm_entropy(knn5), 5),
        min_pairwise=round(metrics.min_pairwise_distance(Xs), 5),
    )
    row.update({k: (round(v, 5) if v == v else "") for k, v in phase56_scores(key).items()})
    return row


def main():
    keys = [k for k in datasets.DATASETS if k != "stock"]
    out_dir = os.path.join(config.OUTPUTS_DIR, "distillation")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k in keys:
        try:
            r = dataset_density(k)
            rows.append(r)
            print(f"{k:12s} n={r['n']:6d} d={r['d']:2d} knn5={r['knn5_mean']:.3f} "
                  f"het_cv={r['het_global_cv']:.3f} norm_ent={r['norm_entropy']:.3f} "
                  f"spc_err={r['spc_err_strength']}", flush=True)
        except Exception as e:
            print(f"{k}: SKIP ({e})", flush=True)
    path = os.path.join(out_dir, "density_metrics.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
