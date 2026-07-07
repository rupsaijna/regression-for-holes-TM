# Teacher edge vs distillation benefit (exp10)

`teacher_edge = rmse_unguided - rmse_M20` (>0 = teacher is the more accurate model). `benefit = impr` (>0 = mode beats no-teacher). One point per dataset.

| dataset | teacher_edge | uniform | teacher_sign | teacher_sign_ii | teacher_sign_tmag |
|---|---|---|---|---|---|
| airquality | +0.078 | +0.078 | -17.386 | -0.001 | +0.107 |
| california | +0.007 | -0.013 | -0.024 | -0.018 | -0.019 |
| ccpp | +0.323 | +0.360 | +0.427 | +0.340 | +0.367 |
| energy | +0.210 | -0.062 | -0.377 | -0.366 | -0.151 |

## Correlation across datasets

| method | Pearson r | Spearman rho | Pearson (clipped) | Spearman (clipped) |
|---|---|---|---|---|
| uniform | +0.673 | +0.400 | +0.673 | +0.400 |
| teacher_sign | +0.377 | +0.400 | +0.507 | +0.400 |
| teacher_sign_ii | +0.330 | +0.400 | +0.330 | +0.400 |
| teacher_sign_tmag | +0.518 | +0.400 | +0.518 | +0.400 |

n = 4 datasets (correlation is illustrative, not significant at this n). A strongly positive r would mean 'more accurate teacher -> more benefit'; near-zero / negative refutes teacher accuracy as a predictor.
