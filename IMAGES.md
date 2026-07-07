# Image Guide

What every figure produced by this project contains and what it is meant to
reveal. Paths are relative to the project root, so this file renders the images
inline in any Markdown viewer (VS Code preview, GitHub, etc.).

Figures fall into three families:

1. **Baseline prediction plots** — `experiment1_baseline.py` (`outputs/`)
2. **Per-spacing diagnostic panels** — `experiment2_analytics.py` (`outputs/analytics/panel_*.png`)
3. **RMSE-vs-spacing bucket curves** — `experiment2_analytics.py` (`outputs/analytics/buckets_*.png`)

### Shared conventions
- **RegressionTM** is always drawn in **blue (squares)**, **NeuralNet** in **orange (triangles)**, and the **true value** in **black (circles)**.
- The figures embedded below are for the default **California Housing** target (*median house value*, \$100k). The same figure families are produced for every other dataset.
- "Test sample index" is the position **within the selected slice** of the fixed test set, not the global row id.

### Per-dataset outputs
Running with `--dataset <key>` writes to per-dataset paths so datasets never clobber
each other: baseline plot `outputs/baseline_<key>_*.png`, and analytics figures under
`outputs/analytics/<key>/{panel,buckets}_<space>_<spacing>.png` (note: for non-California
datasets the space tag is `standardized` rather than `standardized8d`). The cross-dataset
tables live in `outputs/spacing_distribution.{csv,md}` and
`outputs/spacing_error_correlation.{csv,md}`.

**Where the spacing→error effect is actually visible:** on most datasets the
`buckets_*` curves are flat (spacing does not predict error). The two exceptions worth
opening are **`outputs/analytics/nyse/buckets_*`** (next-day volume; TM Pearson ≈ 0.65–0.69)
and **`outputs/analytics/airquality/buckets_*`** (benzene; NN Pearson ≈ 0.54–0.69) — there
the error rises monotonically with spacing. By contrast `outputs/analytics/metro/buckets_*`
is flat despite metro being the most unevenly-spaced dataset. See
`outputs/spacing_error_correlation.md` for the full reading.

### Spacing definitions and feature spaces (families 2 & 3)
Each analytics figure is produced for every combination of a *spacing definition* and a *feature space*. The filename encodes both: `…_<space>_<spacing>.png`.

| spacing key | meaning (per test point) |
|---|---|
| `nn_to_train_k1` | distance to the **nearest training point** |
| `knn_to_train_k5` | mean distance to the **5 nearest training points** |
| `local_density_k5` | mean distance to the **5 nearest points in the whole dataset** |
| `nn_to_test_k1` | distance to the **nearest other test point** |

| space tag | meaning |
|---|---|
| `standardized8d` | Euclidean distance in the full **z-scored 8-D** feature space (scaler fit on train) |
| `pca2d` | Euclidean distance after **PCA → 2 components** (fit on the standardized train set) |

---

## 1. Baseline prediction plots

**Contains:** for a chosen slice of the test set, the true house value and each
model's prediction at every sample, drawn as three overlaid lines.
**Captures:** at-a-glance, per-sample agreement — where each model tracks the
truth, where it under/over-shoots, and how TM vs NN differ on the same points.
The `--slice` flag selects which test samples appear; the figure saves to PNG.

`outputs/baseline_predictions_0_49.png` — current **full-strength** models (5-run canonical), test samples 0–49:

![Baseline predictions, slice 0:49](./outputs/baseline_predictions_0_49.png)

`outputs/baseline_predictions_0_29.png` — same format, slice 0–29 (from the initial quick/smoke model, so predictions are coarser):

![Baseline predictions, slice 0:29](./outputs/baseline_predictions_0_29.png)

`outputs/loadtest_100_160.png` — same format, slice 100–160, produced via `--load-model` to confirm a reloaded model reproduces predictions:

![Baseline predictions, slice 100:160](./outputs/loadtest_100_160.png)

---

## 2. Per-spacing diagnostic panels

Each panel is **three stacked sub-plots** for one spacing definition over the
selected test slice:

- **Top — predictions:** true value vs TM vs NN across the slice (same as family 1), giving the local context.
- **Middle — spacing bars:** the spacing value of each sample in the slice, so you can line up "tall bar" (isolated point) against the top panel.
- **Bottom — spacing vs |error|:** a scatter of spacing against absolute error, one series per model. **This is the core test of the hypothesis** "wider spacing → larger error": an upward-sloping cloud supports it; a flat cloud refutes it.

### Standardized-8D space

`panel_standardized8d_nn_to_train_k1.png` — spacing = distance to nearest training point:
![panel standardized8d nn_to_train_k1](./outputs/analytics/panel_standardized8d_nn_to_train_k1.png)

`panel_standardized8d_knn_to_train_k5.png` — spacing = mean distance to 5 nearest training points:
![panel standardized8d knn_to_train_k5](./outputs/analytics/panel_standardized8d_knn_to_train_k5.png)

`panel_standardized8d_local_density_k5.png` — spacing = mean distance to 5 nearest points overall (local crowding):
![panel standardized8d local_density_k5](./outputs/analytics/panel_standardized8d_local_density_k5.png)

`panel_standardized8d_nn_to_test_k1.png` — spacing = distance to nearest other test point:
![panel standardized8d nn_to_test_k1](./outputs/analytics/panel_standardized8d_nn_to_test_k1.png)

### PCA-2D space

`panel_pca2d_nn_to_train_k1.png` — nearest training point, in PCA-2D:
![panel pca2d nn_to_train_k1](./outputs/analytics/panel_pca2d_nn_to_train_k1.png)

`panel_pca2d_knn_to_train_k5.png` — 5 nearest training points, in PCA-2D:
![panel pca2d knn_to_train_k5](./outputs/analytics/panel_pca2d_knn_to_train_k5.png)

`panel_pca2d_local_density_k5.png` — local density, in PCA-2D:
![panel pca2d local_density_k5](./outputs/analytics/panel_pca2d_local_density_k5.png)

`panel_pca2d_nn_to_test_k1.png` — nearest other test point, in PCA-2D:
![panel pca2d nn_to_test_k1](./outputs/analytics/panel_pca2d_nn_to_test_k1.png)

---

## 3. RMSE-vs-spacing bucket curves

**Contains:** the **whole test set** is sorted by a spacing definition and split
into equal-count quantile buckets; each curve plots a model's RMSE within each
bucket against the bucket's median spacing (x = increasing spacing →).
**Captures:** the hypothesis aggregated over all 4128 test points rather than one
slice — a rising curve means that model regresses *worse* where data is sparser.
Comparing the blue (TM) and orange (NN) curves shows which model is more
sensitive to spacing.

### Standardized-8D space

`buckets_standardized8d_nn_to_train_k1.png`:
![buckets standardized8d nn_to_train_k1](./outputs/analytics/buckets_standardized8d_nn_to_train_k1.png)

`buckets_standardized8d_knn_to_train_k5.png`:
![buckets standardized8d knn_to_train_k5](./outputs/analytics/buckets_standardized8d_knn_to_train_k5.png)

`buckets_standardized8d_local_density_k5.png`:
![buckets standardized8d local_density_k5](./outputs/analytics/buckets_standardized8d_local_density_k5.png)

`buckets_standardized8d_nn_to_test_k1.png`:
![buckets standardized8d nn_to_test_k1](./outputs/analytics/buckets_standardized8d_nn_to_test_k1.png)

### PCA-2D space

`buckets_pca2d_nn_to_train_k1.png`:
![buckets pca2d nn_to_train_k1](./outputs/analytics/buckets_pca2d_nn_to_train_k1.png)

`buckets_pca2d_knn_to_train_k5.png`:
![buckets pca2d knn_to_train_k5](./outputs/analytics/buckets_pca2d_knn_to_train_k5.png)

`buckets_pca2d_local_density_k5.png`:
![buckets pca2d local_density_k5](./outputs/analytics/buckets_pca2d_local_density_k5.png)

`buckets_pca2d_nn_to_test_k1.png`:
![buckets pca2d nn_to_test_k1](./outputs/analytics/buckets_pca2d_nn_to_test_k1.png)

---

> **Note:** the analytics figures are regenerated every time `experiment2_analytics.py`
> runs, against whatever models are currently saved in `saved_models/`. The
> baseline `--slice`/`--out` flags control which prediction PNGs (family 1) exist.
> Filenames embed the slice and the (space, spacing) pair, so re-runs with new
> settings add new files rather than silently overwriting unrelated ones.
