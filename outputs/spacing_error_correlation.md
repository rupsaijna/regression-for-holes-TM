# Spacing vs |error| correlation (standardized space, runs=10)

Pearson / Spearman of each spacing definition against each model's mean absolute error
(averaged over 10 runs to denoise). **|corr| under ~0.1, or any dataset with small
n_test, is essentially noise.** Datasets are grouped by reliability of `n_test`.

| dataset | n_test | spacing | TM Pearson | TM Spearman | NN Pearson | NN Spearman |
|---|---|---|---|---|---|---|
| california | 4128 | nn_to_train_k1 | 0.053 | 0.067 | 0.096 | 0.047 |
| california | 4128 | knn_to_train_k5 | 0.021 | 0.068 | 0.065 | 0.041 |
| california | 4128 | local_density_k5 | 0.019 | 0.068 | 0.065 | 0.045 |
| california | 4128 | nn_to_test_k1 | 0.01 | 0.058 | 0.056 | 0.055 |
| ccpp | 1914 | nn_to_train_k1 | -0.0 | -0.032 | 0.001 | -0.003 |
| ccpp | 1914 | knn_to_train_k5 | 0.084 | 0.014 | -0.029 | -0.033 |
| ccpp | 1914 | local_density_k5 | 0.052 | -0.006 | -0.024 | -0.021 |
| ccpp | 1914 | nn_to_test_k1 | 0.034 | -0.015 | -0.031 | 0.009 |
| energy | 154 | nn_to_train_k1 | 0.075 | 0.067 | 0.077 | 0.022 |
| energy | 154 | knn_to_train_k5 | 0.237 | 0.211 | 0.042 | 0.154 |
| energy | 154 | local_density_k5 | 0.281 | 0.291 | 0.093 | 0.185 |
| energy | 154 | nn_to_test_k1 | 0.149 | 0.235 | 0.05 | 0.039 |
| autompg | 79 | nn_to_train_k1 | -0.031 | 0.012 | 0.148 | 0.081 |
| autompg | 79 | knn_to_train_k5 | 0.141 | 0.121 | 0.198 | 0.138 |
| autompg | 79 | local_density_k5 | 0.186 | 0.122 | 0.201 | 0.159 |
| autompg | 79 | nn_to_test_k1 | 0.424 | 0.209 | 0.143 | 0.188 |
| realestate | 83 | nn_to_train_k1 | 0.02 | -0.048 | -0.018 | -0.049 |
| realestate | 83 | knn_to_train_k5 | 0.067 | 0.033 | -0.07 | -0.17 |
| realestate | 83 | local_density_k5 | 0.036 | 0.024 | -0.066 | -0.144 |
| realestate | 83 | nn_to_test_k1 | -0.081 | -0.131 | -0.045 | -0.135 |
| mortality | 12 | nn_to_train_k1 | -0.071 | 0.126 | -0.166 | 0.077 |
| mortality | 12 | knn_to_train_k5 | 0.03 | 0.21 | -0.14 | -0.119 |
| mortality | 12 | local_density_k5 | 0.03 | 0.259 | -0.137 | -0.077 |
| mortality | 12 | nn_to_test_k1 | 0.154 | 0.102 | -0.016 | -0.159 |
| bloodfat | 5 | nn_to_train_k1 | 0.534 | 0.6 | 0.171 | -0.1 |
| bloodfat | 5 | knn_to_train_k5 | 0.316 | 0.5 | 0.348 | 0.3 |
| bloodfat | 5 | local_density_k5 | 0.399 | 0.6 | 0.389 | 0.1 |
| bloodfat | 5 | nn_to_test_k1 | 0.352 | 0.211 | 0.402 | 0.527 |
| metro | 9639 | nn_to_train_k1 | -0.003 | -0.026 | -0.014 | -0.053 |
| metro | 9639 | knn_to_train_k5 | -0.002 | -0.033 | -0.007 | -0.059 |
| metro | 9639 | local_density_k5 | -0.004 | -0.034 | -0.009 | -0.064 |
| metro | 9639 | nn_to_test_k1 | 0.001 | -0.023 | -0.004 | -0.048 |
| beijing | 8352 | nn_to_train_k1 | -0.038 | -0.105 | -0.012 | -0.095 |
| beijing | 8352 | knn_to_train_k5 | -0.06 | -0.111 | -0.043 | -0.114 |
| beijing | 8352 | local_density_k5 | -0.057 | -0.125 | -0.039 | -0.129 |
| beijing | 8352 | nn_to_test_k1 | -0.051 | -0.06 | -0.062 | -0.086 |
| airquality | 1389 | nn_to_train_k1 | 0.395 | 0.209 | 0.614 | 0.465 |
| airquality | 1389 | knn_to_train_k5 | 0.449 | 0.283 | 0.689 | 0.608 |
| airquality | 1389 | local_density_k5 | 0.465 | 0.29 | 0.69 | 0.588 |
| airquality | 1389 | nn_to_test_k1 | 0.4 | 0.217 | 0.544 | 0.393 |
| nyse | 10610 | nn_to_train_k1 | 0.672 | 0.489 | 0.57 | 0.534 |
| nyse | 10610 | knn_to_train_k5 | 0.686 | 0.535 | 0.579 | 0.538 |
| nyse | 10610 | local_density_k5 | 0.686 | 0.532 | 0.583 | 0.545 |
| nyse | 10610 | nn_to_test_k1 | 0.652 | 0.508 | 0.568 | 0.512 |
| nyse_log | 10610 | nn_to_train_k1 | 0.255 | 0.038 | 0.032 | -0.003 |
| nyse_log | 10610 | knn_to_train_k5 | 0.261 | 0.036 | 0.033 | -0.001 |
| nyse_log | 10610 | local_density_k5 | 0.262 | 0.039 | 0.034 | 0.0 |
| nyse_log | 10610 | nn_to_test_k1 | 0.257 | 0.048 | 0.038 | 0.009 |

## Cross-dataset reading

The 12 datasets split cleanly into **two regimes**:

**1. Spacing does NOT predict error** (all |corr| ≲ 0.1) — the "well-behaved" sets:
`california`, `ccpp`, `metro`, `beijing`. Notably **`metro` is the most unevenly-spaced
dataset in the whole registry** (norm-entropy ≈ 0.00) yet its spacing–error correlation
is ~0 for both models. So **uneven spacing by itself does not make spacing predictive of
error** — the earlier "TM tracks spacing" hint does not survive on the large clean sets.

**2. Spacing strongly predicts error** (|corr| ≈ 0.4–0.7) — the **heavy-tailed-target**
sets with large, trustworthy `n_test`:
- **`nyse`** (raw next-day volume, n=10610): **TM Pearson 0.65–0.69**, NN 0.57–0.58.
- **`airquality`** (benzene, n=1389): **NN Pearson 0.54–0.69**, TM 0.40–0.47.

**The heavy-tail test (`nyse` vs `nyse_log`).** `nyse_log` is the *identical* dataset
(same features, same rows) with the target log-transformed. Its correlation **collapses**
to TM Pearson 0.26 / **Spearman 0.04** and NN ≈ 0. The Pearson 0.26 with Spearman ≈ 0 means
even the residual is driven by a handful of extreme points, not a monotonic relationship.
**Conclusion: the strong raw-`nyse` correlation was largely a heavy-tail artefact** — far
points scored high error mainly because extreme-volume outliers are *also* geometric
outliers. Once the target is well-behaved, spacing carries almost no error information,
even though the feature-space geometry is unchanged.

So the discriminator is not how uneven the spacing is, nor spacing magnitude per se, but a
**hard/heavy-tailed target whose extremes coincide with sparse regions**. Which model is
more spacing-sensitive is dataset-dependent (TM > NN on nyse, NN > TM on airquality) — there
is **no universal "TM tracks spacing more than NN" rule**.

**Caveats.** `energy` (n=154) and below are small; `mortality` (12) and `bloodfat` (5) are
statistically meaningless and shown only for completeness.
