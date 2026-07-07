# Teacher edge vs distillation benefit (exp11)

`teacher_edge = rmse_unguided - rmse_M20` (>0 = teacher is the more accurate model). `benefit = impr` (>0 = mode beats no-teacher). One point per dataset.

| dataset | teacher_edge | uniform | force_ii | force_ii_tmag | teacher_sign_tmag | sameside_tmag |
|---|---|---|---|---|---|---|
| airquality | +0.078 | +0.078 | +0.175 | -0.140 | +0.107 | +0.072 |
| california | +0.007 | -0.013 | -0.015 | -0.017 | -0.019 | -0.029 |
| ccpp | +0.323 | +0.360 | +0.251 | +0.431 | +0.367 | +0.392 |
| energy | +0.210 | -0.062 | -0.237 | -0.116 | -0.151 | -0.013 |

## Correlation across datasets

| method | Pearson r | Spearman rho | Pearson (clipped) | Spearman (clipped) |
|---|---|---|---|---|
| uniform | +0.673 | +0.400 | +0.673 | +0.400 |
| force_ii | +0.197 | +0.400 | +0.197 | +0.400 |
| force_ii_tmag | +0.711 | +0.400 | +0.711 | +0.400 |
| teacher_sign_tmag | +0.518 | +0.400 | +0.518 | +0.400 |
| sameside_tmag | +0.778 | +0.800 | +0.778 | +0.800 |

n = 4 datasets (correlation is illustrative, not significant at this n). A strongly positive r would mean 'more accurate teacher -> more benefit'; near-zero / negative refutes teacher accuracy as a predictor.
