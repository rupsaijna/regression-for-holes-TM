# Spacing-VARIANCE vs performance (standardized space, runs=10)

Does *uneven* spacing (some neighbours close, some far) predict error, rather than just *large* spacing? Compare with `spacing_error_correlation.md` (which uses mean spacing). |corr| < ~0.1 or small n_test = noise.

## (B) Per-point: local spacing dispersion vs |error|

`knn_cv_k5` = std/mean of the 5 nearest-train distances; `knn_ratio_k5` = d5/d1; `knn_std_k5` = raw std (absolute).

| dataset | n_test | metric | TM Pearson | TM Spearman | NN Pearson | NN Spearman |
|---|---|---|---|---|---|---|
| california | 4128 | knn_cv_k5 | -0.029 | -0.034 | -0.038 | -0.028 |
| california | 4128 | knn_ratio_k5 | -0.024 | -0.034 | -0.032 | -0.028 |
| california | 4128 | knn_std_k5 | -0.005 | 0.009 | 0.037 | 0.009 |
| ccpp | 1914 | knn_cv_k5 | 0.041 | 0.046 | -0.031 | -0.019 |
| ccpp | 1914 | knn_ratio_k5 | 0.016 | 0.039 | -0.033 | -0.019 |
| ccpp | 1914 | knn_std_k5 | 0.12 | 0.053 | -0.042 | -0.035 |
| energy | 154 | knn_cv_k5 | 0.197 | 0.195 | -0.12 | -0.068 |
| energy | 154 | knn_ratio_k5 | 0.265 | 0.216 | -0.039 | 0.003 |
| energy | 154 | knn_std_k5 | 0.26 | 0.225 | -0.067 | -0.016 |
| autompg | 79 | knn_cv_k5 | 0.305 | 0.233 | 0.141 | 0.121 |
| autompg | 79 | knn_ratio_k5 | 0.283 | 0.241 | 0.112 | 0.12 |
| autompg | 79 | knn_std_k5 | 0.412 | 0.329 | 0.18 | 0.211 |
| realestate | 83 | knn_cv_k5 | 0.013 | 0.049 | -0.062 | -0.038 |
| realestate | 83 | knn_ratio_k5 | 0.073 | 0.085 | 0.018 | -0.049 |
| realestate | 83 | knn_std_k5 | 0.101 | 0.137 | -0.083 | -0.087 |
| mortality | 12 | knn_cv_k5 | 0.434 | 0.329 | 0.212 | -0.028 |
| mortality | 12 | knn_ratio_k5 | 0.476 | 0.231 | 0.22 | -0.021 |
| mortality | 12 | knn_std_k5 | 0.505 | 0.308 | 0.197 | -0.35 |
| bloodfat | 5 | knn_cv_k5 | -0.352 | -0.6 | 0.267 | 0.2 |
| bloodfat | 5 | knn_ratio_k5 | -0.175 | -0.6 | -0.107 | 0.2 |
| bloodfat | 5 | knn_std_k5 | -0.138 | -0.4 | 0.521 | 0.2 |
| metro | 9639 | knn_cv_k5 | 0.05 | 0.043 | 0.06 | 0.054 |
| metro | 9639 | knn_ratio_k5 | 0.095 | 0.007 | 0.009 | -0.004 |
| metro | 9639 | knn_std_k5 | 0.021 | -0.024 | -0.0 | -0.051 |
| beijing | 8352 | knn_cv_k5 | -0.048 | -0.063 | -0.09 | -0.119 |
| beijing | 8352 | knn_ratio_k5 | -0.042 | -0.051 | -0.073 | -0.103 |
| beijing | 8352 | knn_std_k5 | -0.08 | -0.11 | -0.095 | -0.148 |
| airquality | 1389 | knn_cv_k5 | -0.021 | -0.01 | -0.066 | -0.058 |
| airquality | 1389 | knn_ratio_k5 | 0.006 | -0.005 | -0.038 | -0.055 |
| airquality | 1389 | knn_std_k5 | 0.156 | 0.129 | 0.216 | 0.228 |
| nyse | 10610 | knn_cv_k5 | 0.0 | 0.029 | -0.031 | -0.129 |
| nyse | 10610 | knn_ratio_k5 | -0.002 | 0.029 | -0.029 | -0.126 |
| nyse | 10610 | knn_std_k5 | 0.591 | 0.441 | 0.512 | 0.336 |
| nyse_log | 10610 | knn_cv_k5 | 0.004 | -0.007 | 0.011 | 0.023 |
| nyse_log | 10610 | knn_ratio_k5 | -0.0 | -0.008 | 0.007 | 0.022 |
| nyse_log | 10610 | knn_std_k5 | 0.236 | 0.046 | 0.033 | 0.016 |

## (A) Per-dataset: spacing heterogeneity vs normalised RMSE

`het_global_cv` = CV of per-point mean-spacing across the test set (some regions dense, some sparse); `het_local_cv` = median within-neighbourhood CV. `norm_rmse` = RMSE / std(y). **Weak evidence: few datasets, heavy confounding.**

| dataset | n_test | het_global_cv | het_local_cv | norm_rmse_tm | norm_rmse_nn |
|---|---|---|---|---|---|
| california | 4128 | 1.917 | 0.115 | 0.537 | 0.454 |
| ccpp | 1914 | 0.421 | 0.227 | 0.294 | 0.223 |
| energy | 154 | 0.123 | 0.165 | 0.119 | 0.146 |
| autompg | 79 | 0.369 | 0.202 | 0.372 | 0.349 |
| realestate | 83 | 0.438 | 0.244 | 0.531 | 0.455 |
| mortality | 12 | 0.353 | 0.071 | 0.87 | 0.495 |
| bloodfat | 5 | 0.21 | 0.303 | 1.183 | 0.898 |
| metro | 9639 | 18.649 | 0.646 | 0.996 | 0.967 |
| beijing | 8352 | 0.526 | 0.28 | 0.805 | 0.52 |
| airquality | 1389 | 0.325 | 0.142 | 0.216 | 0.016 |
| nyse | 10610 | 1.461 | 0.13 | 0.813 | 0.398 |
| nyse_log | 10610 | 1.461 | 0.13 | 0.567 | 0.342 |

**Across-dataset correlation (heterogeneity vs norm_rmse):**

| subset | n_ds | heterogeneity | performance | Pearson | Spearman |
|---|---|---|---|---|---|
| all datasets | 12 | het_global_cv | norm_rmse_tm | 0.381 | 0.357 |
| all datasets | 12 | het_global_cv | norm_rmse_nn | 0.606 | 0.357 |
| all datasets | 12 | het_local_cv | norm_rmse_tm | 0.432 | 0.196 |
| all datasets | 12 | het_local_cv | norm_rmse_nn | 0.7 | 0.448 |
| reliable (n>=100) | 8 | het_global_cv | norm_rmse_tm | 0.627 | 0.814 |
| reliable (n>=100) | 8 | het_global_cv | norm_rmse_nn | 0.85 | 0.826 |
| reliable (n>=100) | 8 | het_local_cv | norm_rmse_tm | 0.587 | 0.216 |
| reliable (n>=100) | 8 | het_local_cv | norm_rmse_nn | 0.815 | 0.275 |

## Reading (denoised, runs=10)

The hypothesis ("uneven spacing — some neighbours close, some far — hurts performance")
**splits by scale**, and the runs=10 averaging confirms the earlier runs=1 read:

**(B) Per-point: NO.** On every reliable dataset the *scale-free* dispersion metrics
(`knn_cv_k5`, `knn_ratio_k5`) are ≈ 0 — local unevenness does **not** identify which test
points a model gets wrong. The one dispersion metric that lights up — `knn_std_k5` on `nyse`
(TM 0.59, NN 0.51) — is the **raw** (un-normalised) std, i.e. spacing *magnitude* leaking
back in. Decisive check: on **`nyse_log`** (identical geometry, log target) `knn_std_k5`
collapses to TM 0.24 / **Spearman 0.05**, NN ≈ 0 — so even that was the heavy-tail target,
not unevenness. Once magnitude is divided out, neighbourhood evenness carries no error info.

**(A) Per-dataset: suggestively YES, by rank.** Across the 8 reliable datasets, dataset-level
heterogeneity `het_global_cv` tracks normalised RMSE with **Spearman ≈ 0.81–0.83** for both
models (Pearson is inflated by the `metro` outlier — trust the rank version). `het_local_cv`
is a weaker, inconsistent predictor (Spearman 0.22–0.28).

**Caveat — confounding.** Only 8 datasets, and heterogeneity is confounded with general
"messiness": heavy-tailed / outlier-rich datasets (metro spikes, california caps, nyse
volume) have **both** uneven spacing **and** high normalised error. Telling detail: `nyse`
and `nyse_log` share *identical* spacing (same `het_global_cv` = 1.461) but `nyse_log` has
lower norm_rmse (0.567 vs 0.813 TM) — so spacing heterogeneity alone does not fix performance;
the target's tail does. (A) is a between-dataset association, not a within-dataset mechanism —
consistent with (B) finding no per-point effect.
