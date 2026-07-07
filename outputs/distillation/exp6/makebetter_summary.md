# Experiment 6 — making M100 better (f=0.5, p=0.5, f_opposite=0.0, 8 seeds, 100ep)

Paired improvement = rmse(unguided M100) - rmse(guided M100) on held-out TEST; >0 = teacher helps. `works` = mean>0 & t>2 (paired).

| dataset | method | mean impr | std | winrate | t | verdict |
|---|---|---|---|---|---|---|
| energy | A_self | +0.0297 | 0.0801 | 0.625 | +1.05 | ~ns |
| energy | A_labels | +0.0345 | 0.1210 | 0.375 | +0.81 | ~ns |
| energy | B_oof | +0.0146 | 0.1032 | 0.5 | +0.40 | ~ns |
| energy | C_ens | +0.0335 | 0.1196 | 0.625 | +0.79 | ~ns |
| energy | D_bigTM | +0.0746 | 0.1327 | 0.75 | +1.59 | ~ns |
| energy | D_nn | +0.0190 | 0.1376 | 0.375 | +0.39 | ~ns |
| ccpp | A_self | +0.0230 | 0.3421 | 0.5 | +0.19 | ~ns |
| ccpp | A_labels | +0.1582 | 0.3255 | 0.625 | +1.37 | ~ns |
| ccpp | B_oof | +0.1633 | 0.3131 | 0.625 | +1.47 | ~ns |
| ccpp | C_ens | +0.1226 | 0.2249 | 0.875 | +1.54 | ~ns |
| ccpp | D_bigTM | +0.0282 | 0.3101 | 0.5 | +0.26 | ~ns |
| ccpp | D_nn | -0.0370 | 0.3025 | 0.375 | -0.35 | ~ns |
| california | A_self | -0.0019 | 0.0062 | 0.25 | -0.86 | ~ns |
| california | A_labels | -0.0035 | 0.0056 | 0.25 | -1.75 | ~ns |
| california | B_oof | +0.0006 | 0.0072 | 0.5 | +0.25 | ~ns |
| california | C_ens | -0.0006 | 0.0087 | 0.375 | -0.21 | ~ns |
| california | D_bigTM | -0.0156 | 0.0137 | 0.125 | -3.21 | hurts |
| california | D_nn | -0.0061 | 0.0085 | 0.25 | -2.04 | hurts |

_NOTE: single config (f, p) only — winners deserve an f/p sweep. A_labels is the pure-attenuation control; A_self - A_labels = dark-knowledge gain._
