# Experiment 7 — per-sample: which test points does the teacher rescue?

Config: teacher 20% -> student 30% mask, structured, f=0.75, p=0.5, 100ep, 5 seeds. teacher_help = (|err_unguided| - |err_guided|)/std(y) per test point; >0 = teacher helped.

## In-hole vs surviving

| dataset | in-hole | surviving | in-hole frac |
|---|---|---|---|
| ccpp | +0.0307 | +0.0244 | 0.3043 |
| california | -0.0129 | -0.0157 | 0.3097 |
| energy | -0.0050 | -0.0044 | 0.2597 |
| airquality | +0.0039 | +0.0179 | 0.3178 |

## Spearman(descriptor, teacher_help) per dataset

| descriptor | ccpp | california | energy | airquality |
|---|---|---|---|---|
| in_hole | +0.03 | +0.01 | -0.01 | -0.03 |
| local_density | +0.03 | +0.05 | -0.01 | -0.00 |
| dist_nearest | +0.01 | +0.04 | -0.08 | +0.00 |
| local_y_rough | -0.03 | +0.02 | +0.04 | +0.03 |
| support_loss | +0.03 | +0.02 | -0.00 | -0.04 |
| teacher_extra | +0.00 | -0.04 | -0.10 | -0.02 |
| ts_disagree | +0.09 | +0.00 | +0.22 | +0.07 |
| err_unguided | +0.48 | +0.13 | +0.39 | +0.45 |
