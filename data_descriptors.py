"""Richer model-free DATA DESCRIPTORS per dataset, to hunt for one that correlates
with teacher effectiveness (experiment 5) better than density / target-shape did.

Descriptors (on standardized features, train split, subsampled):
  linear_r2     : 5-fold R^2 of linear regression        (linearity / easy learnability)
  knn_r2        : 5-fold R^2 of kNN(5) regressor          (local smoothness / learnability)
  local_y_cv    : mean std(y over 5 feature-NN) / std(y)  (local target roughness / noise)
  snr           : var(kNN-smoothed y) / var(residual)     (signal-to-noise)
  intrinsic_dim : two-NN intrinsic dimension (Facco 2017) (effective manifold dim)
  pca_pr        : PCA participation ratio                 (effective #features)
  feat_corr     : mean |off-diagonal feature correlation| (redundancy)
  target_mi_max : max mutual_info(feature; y)             (strongest single-feature signal)
  uniq_y_frac   : #unique(y)/n                            (target granularity)

Writes outputs/distillation/data_descriptors.csv and prints the strongest correlations
with the effectiveness metrics.
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression

import config
from common import datasets
from common.data import get_split_data, standardize

SUB, SEED = 4000, 12345


def intrinsic_dim_twonn(X):
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    mu = d[:, 2] / np.where(d[:, 1] > 0, d[:, 1], np.nan)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.sum(np.log(mu))) if len(mu) else np.nan


def descriptors(key):
    Xtr, Xte, ytr, yte = get_split_data(key)
    _, Xs, _ = standardize(Xtr, Xte)
    y = np.asarray(ytr, float)
    if len(Xs) > SUB:
        idx = np.random.default_rng(SEED).choice(len(Xs), SUB, replace=False)
        Xs, y = Xs[idx], y[idx]
    cv = min(5, len(Xs) // 2)
    lin = float(np.mean(cross_val_score(LinearRegression(), Xs, y, cv=cv, scoring="r2")))
    knn = float(np.mean(cross_val_score(KNeighborsRegressor(n_neighbors=min(5, len(Xs) - 1)),
                                        Xs, y, cv=cv, scoring="r2")))
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    _, nidx = nn.kneighbors(Xs)
    yn = y[nidx[:, 1:]]
    local_y_cv = float(yn.std(axis=1).mean() / (y.std() or 1))
    y_sm = yn.mean(axis=1); resid = y - y_sm
    snr = float(y_sm.var() / (resid.var() or 1e-12))
    lam = np.linalg.eigvalsh(np.cov(Xs.T)) if Xs.shape[1] > 1 else np.array([1.0])
    lam = np.clip(lam, 0, None)
    pca_pr = float((lam.sum() ** 2) / (np.sum(lam ** 2) or 1))
    C = np.abs(np.corrcoef(Xs.T)) if Xs.shape[1] > 1 else np.array([[1.0]])
    feat_corr = float((C.sum() - np.trace(C)) / (C.size - len(C))) if Xs.shape[1] > 1 else 0.0
    mi = mutual_info_regression(Xs, y, random_state=SEED)
    return dict(dataset=key, n=len(Xtr), d=Xs.shape[1],
                linear_r2=round(lin, 4), knn_r2=round(knn, 4),
                local_y_cv=round(local_y_cv, 4), snr=round(snr, 4),
                intrinsic_dim=round(intrinsic_dim_twonn(Xs), 3),
                pca_pr=round(pca_pr, 3), feat_corr=round(feat_corr, 4),
                target_mi_max=round(float(mi.max()), 4),
                uniq_y_frac=round(len(np.unique(ytr)) / len(ytr), 4))


def main():
    ddir = os.path.join(config.OUTPUTS_DIR, "distillation")
    rows = []
    for k in datasets.DATASETS:
        if k == "stock":
            continue
        try:
            r = descriptors(k); rows.append(r)
            print(f"{k:12s} linR2={r['linear_r2']:+.2f} knnR2={r['knn_r2']:+.2f} "
                  f"local_y_cv={r['local_y_cv']:.2f} snr={r['snr']:.2f} id={r['intrinsic_dim']} "
                  f"mi={r['target_mi_max']:.2f}", flush=True)
        except Exception as e:
            print(f"{k}: SKIP ({e})", flush=True)
    path = os.path.join(ddir, "data_descriptors.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {path}")

    # correlate with effectiveness
    eff = {r["dataset"]: r for r in csv.DictReader(open(os.path.join(ddir, "effectiveness_metrics.csv")))}
    desc = {r["dataset"]: r for r in rows}
    EFFK = ["eff_raw", "eff_norm", "eff_winrate", "eff_vs_teacher", "eff_beat_m100"]
    DK = ["linear_r2", "knn_r2", "local_y_cv", "snr", "intrinsic_dim", "pca_pr",
          "feat_corr", "target_mi_max", "uniq_y_frac", "n", "d"]
    keys = [k for k in desc if k in eff]
    pairs = []
    for subset, ks in [("all", keys), ("n>=100", [k for k in keys if int(desc[k]["n"]) >= 100])]:
        for ek in EFFK:
            for dk in DK:
                x = np.array([float(eff[k][ek]) for k in ks])
                yv = np.array([float(desc[k][dk]) for k in ks])
                if len(ks) < 5 or np.std(x) == 0 or np.std(yv) == 0:
                    continue
                rho, p = stats.spearmanr(x, yv)
                pairs.append((subset, ek, dk, round(float(rho), 3), round(float(p), 3), len(ks)))
    with open(os.path.join(ddir, "data_descriptor_correlations.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subset", "eff", "descriptor", "spearman", "p", "n"]); w.writerows(pairs)
    for subset in ("all", "n>=100"):
        sub = sorted([p for p in pairs if p[0] == subset], key=lambda r: -abs(r[3]))
        print(f"\nStrongest |Spearman| ({subset}):")
        for s in sub[:8]:
            print(f"  {s[1]:14s} vs {s[2]:14s} rho={s[3]:+.2f} (p={s[4]}, n={s[5]})")


if __name__ == "__main__":
    main()
