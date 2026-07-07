# Regression for Holes — Teacher-Guided Distillation for the Regression Tsetlin Machine

Code and paper for a study of **teacher-guided knowledge distillation in the Regression
Tsetlin Machine (RegressionTM)** under data scarcity. A frozen *teacher* TM (trained on
less-masked data) steers a data-starved *student* TM by adjusting its per-sample error
inside the clause-level Type I/Type II feedback loop. Using a masked-distillation
protocol (nested structured k-NN "holes"), we ask **when, where, and why** such guidance
helps. Headline findings: every mechanism is a *sharpener* (scales the teacher's signal,
never rescues a bad one); teacher accuracy does **not** predict benefit; a 70%-data
guided student can match or beat a full-data model where extra data hurts; and **data
spacing/density**, not teacher accuracy, governs benefit — distillation acts as a
predominantly *global* reshaping of the learned function.

## Repository layout
| path | contents |
|------|----------|
| `paper/` | **IEEE conference paper** (LaTeX, `IEEEtran`) — the writeup of this work. See `paper/README.md`; Overleaf-syncable (set main document to `paper/main.tex`). |
| `experiment{1..11}_*.py` | experiment drivers (E1 baseline → E11 teacher-bounded magnitude) |
| `spacing_buckets.py`, `correlate_*.py`, `density_metrics.py`, `data_descriptors.py` | E12 spacing/density analysis (Windows-runnable, no `tmu`) |
| `explanations_of_runs.py` | narrative documentation of the whole arc + worked examples |
| `tmu/` | vendored Tsetlin Machine library; our custom `tmu/tmu/models/regression/vanilla_regressor{,_v2,_v3,_v4}.py` hold the teacher-guided fit variants |
| `common/`, `models/`, `config.py` | dataset registry, split cache, spacing metrics, model wrappers, hyperparameters |
| `outputs/distillation/exp{8..12}*/` | per-experiment verdicts, result CSVs, and figures |
| `abstract.md`, `HISTORY.md`, `exp12_spacing_scope.md`, `DATASETS.md`, `IMAGES.md` | notes and catalogues |

**Not tracked** (see `.gitignore`): `datasets/`, `saved_models/`, `splits/`, and build
artifacts — regenerate datasets/splits from the loaders in `common/` (keys listed below),
then retrain. `tmu/` source is tracked but its build products are not; build it in the
WSL venv (below) before running.

---

## Regression Experiment — RegressionTM vs Neural Network (original baseline)

California Housing regression: a Tsetlin Machine (`tmu` RegressionTM) and a
PyTorch MLP on a **single fixed train/test split**, plus a data-analytics program
that studies how each model's error relates to the *spacing* between data points.

## Environment
`tmu` builds C extensions and does **not** work on the Windows Python 3.14 here.
Everything runs in **WSL Ubuntu** via a venv at `~/regexp-venv`:

```bash
wsl ~/regexp-venv/bin/python experiment1_baseline.py --quick --runs 2
```

(Run the commands from this folder; on Windows prefix with `wsl`.)

## Files
| path | purpose |
|------|---------|
| `config.py` | all paths, the fixed-split seed, and every hyperparameter |
| `common/datasets.py` | registry of all selectable regression datasets (+ loaders/cache) |
| `common/data.py` | per-dataset persisted fixed split (`splits/<key>.npz`) |
| `common/metrics.py` | RMSE, pairwise-distance stats, all spacing definitions |
| `common/viz.py` | PNG plots (predictions, per-spacing panels, RMSE-vs-spacing) |
| `models/tm_model.py` | RegressionTM wrapper (owns its binarizer; save/load) |
| `models/nn_model.py` | PyTorch MLP wrapper (owns its scaler; save/load) |
| `experiment1_baseline.py` | train/eval, avg RMSE over 10/50/100 runs, slice plot |
| `experiment2_analytics.py` | distance stats + model-error-vs-spacing analysis |
| `DATASETS.md` | catalogue of all selectable datasets (links, descriptions, linear-fit) |
| `IMAGES.md` | guide to every produced figure (embeds each image + what it captures) |

## Datasets
Both programs accept `--dataset <key>` (default `california`). See **[DATASETS.md](DATASETS.md)**
for the full catalogue with links and descriptions. Quick list:
```bash
wsl ~/regexp-venv/bin/python experiment1_baseline.py --list-datasets
```
Keys (linear-fit): `california`, `ccpp`, `energy`, `autompg`, `realestate`, `mortality`, `bloodfat`.
Keys (UCI time-series, uneven spacing): `metro`, `beijing`, `airquality`.
Keys (Kaggle): `nyse` (next-day volume; data under `datasets/nyse/`), `nyse_log` (log-target control), `stock` (needs a local CSV).
Each dataset has its own fixed split, saved models, and analytics output folder.

## Experiment 1 — baseline
```bash
# smoke test
wsl ~/regexp-venv/bin/python experiment1_baseline.py --quick --runs 2

# full: cumulative avg RMSE at 10/50/100 runs, save models, plot first 50 test pts
wsl ~/regexp-venv/bin/python experiment1_baseline.py --runs 10 50 100 --save-model --slice 0:50

# reload saved models and redraw a different slice (no retraining)
wsl ~/regexp-venv/bin/python experiment1_baseline.py --load-model --slice 100:160
```
`max(runs)` models are trained once; the mean ± std is reported cumulatively at
each requested checkpoint (identical statistics, far cheaper than 10+50+100).

## Experiment 2 — analytics
Run experiment 1 with `--save-model` first, then:
```bash
wsl ~/regexp-venv/bin/python experiment2_analytics.py --slice 0:50 --buckets 8 --space both
```
Produces: descriptive distance table (train/test/whole); every spacing
definition in standardized-8D and PCA-2D; per-spacing diagnostic panels over the
slice; per-model RMSE-vs-spacing-bucket curves; and spacing↔error correlations.

## Spacing definitions (all computed; pick later)
- `nn_to_train_k1` — distance to nearest training point
- `knn_to_train_k5` — mean distance to 5 nearest training points
- `local_density_k5` — mean distance to 5 nearest points in the whole set
- `nn_to_test_k1` — distance to nearest other test point
