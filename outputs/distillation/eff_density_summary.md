# Teacher effectiveness vs data density (12 datasets, focused config)

Config: epochs=100, student_frac=0.3, f=0.75, f_opposite=0, p_teacher=0.5, structured masking. eff_raw>0 = teacher lowers RMSE.

## Strongest |Spearman| pairings (all datasets)

| effectiveness | density | Spearman | p | Pearson |
|---|---|---|---|---|
| eff_vs_teacher | spc_err_nn_k1 | -0.81 | 0.001 | -0.31 |
| eff_winrate | var_err_strength | -0.64 | 0.024 | -0.67 |
| eff_vs_teacher | d | -0.59 | 0.045 | -0.42 |
| eff_norm | knn5_mean | -0.58 | 0.047 | -0.35 |
| eff_norm | var_err_strength | -0.58 | 0.048 | -0.38 |
| eff_vs_teacher | spc_err_strength | -0.57 | 0.055 | -0.43 |
| eff_raw | var_err_strength | -0.50 | 0.095 | -0.49 |
| eff_raw_median | var_err_strength | -0.50 | 0.095 | -0.49 |
| eff_vs_teacher | var_err_strength | -0.49 | 0.106 | -0.36 |
| eff_norm | knn5_median | -0.46 | 0.137 | -0.35 |
| eff_vs_teacher | spread | -0.46 | 0.137 | -0.43 |
| eff_beat_m100 | var_err_strength | -0.45 | 0.145 | -0.49 |

_NOTE: single config / one student mask / structured only; tiny-n datasets (bloodfat, mortality) included — see n>=100 subset in eff_density_correlations.csv. Future work: more configs, student masks, uniform mask, more seeds._
