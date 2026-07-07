# E12 Tier A - spacing/density vs distillation benefit (10 datasets)

Datasets: airquality, autompg, beijing, california, ccpp, energy, metro, nyse, nyse_log, realestate. Modes: uniform, force_ii, force_ii_tmag, teacher_sign_tmag, sameside_tmag.

## Strongest |Spearman| pairings for eff_norm (fractional benefit)

| mode | predictor | Spearman | p | Pearson |
|---|---|---|---|---|
| force_ii_tmag | spc_err_strength | -0.76 | 0.011 | -0.66 |
| force_ii_tmag | spread | -0.62 | 0.056 | -0.60 |
| force_ii_tmag | nn1_mean | -0.57 | 0.084 | -0.56 |
| force_ii_tmag | min_pairwise | -0.57 | 0.088 | -0.37 |
| force_ii_tmag | knn5_median | -0.52 | 0.121 | -0.50 |
| force_ii_tmag | spc_err_nn_k1 | -0.52 | 0.121 | -0.55 |
| force_ii_tmag | n | +0.51 | 0.132 | +0.40 |
| force_ii_tmag | knn5_mean | -0.50 | 0.143 | -0.50 |
| force_ii_tmag | het_global_cv | +0.50 | 0.143 | +0.10 |
| force_ii_tmag | var_err_strength | -0.49 | 0.15 | -0.27 |
| uniform | het_global_cv | -0.48 | 0.16 | -0.22 |
| sameside_tmag | knn_cv_mean | +0.48 | 0.16 | +0.02 |
| force_ii | knn_cv_mean | +0.43 | 0.213 | +0.07 |
| sameside_tmag | het_global_cv | -0.42 | 0.228 | -0.14 |
| uniform | teacher_edge | +0.42 | 0.229 | -0.10 |

## Best DENSITY predictor vs teacher_edge (eff_norm), per mode

| mode | best density predictor | rho | teacher_edge rho |
|---|---|---|---|
| uniform | het_global_cv | -0.48 | +0.42 |
| force_ii | knn_cv_mean | +0.43 | +0.06 |
| force_ii_tmag | spc_err_strength | -0.76 | -0.19 |
| teacher_sign_tmag | knn5_mean | -0.39 | +0.03 |
| sameside_tmag | knn_cv_mean | +0.48 | +0.07 |

_NOTE: small n datasets; correlations illustrative. less_is_more_margin (rmse_M00 - rmse_unguided > 0) is in modes_effectiveness.csv per dataset._
