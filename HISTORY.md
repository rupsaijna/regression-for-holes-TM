# Project History

How the `RegressionExperiment` study evolved — the questions asked, what was built
to answer them, and what each step revealed. Newest phase last.

The recurring object of study is **RegressionTM (a Tsetlin Machine regressor) vs a
PyTorch MLP** on tabular regression, and — increasingly — **how the geometry of the
test points relates to where each model makes its errors**.

---

## Phase 0 — Single-dataset baseline
**Question:** can a RegressionTM match a neural net on a real regression task?

- Set up `experiment1_baseline.py`: train RegressionTM and an MLP on **California
  Housing**, report RMSE. Fixed, persisted train/test split so results are comparable
  across runs; cumulative mean ± std reported at 10 / 50 / 100 runs from a single
  training pass; `--save-model` / `--load-model`; prediction-slice plots.
- **Environment reality:** `tmu` builds C extensions and will not install on the host
  Windows Python, so everything runs in a **WSL Ubuntu venv** (`~/regexp-venv`, Python
  3.10) pinned to **numpy<2** (tmu raises `OverflowError` on numpy 2.x). pycuda-missing
  tracebacks at startup are harmless CPU-fallback noise.

**Critical fix discovered here:** the NN must **standardize the target `y`**. Without it
the MLP collapsed on large-target datasets (e.g. power output ~450, mortality ~1000).

---

## Phase 1 — Many datasets, chosen for linearity
**Question:** does the TM-vs-NN comparison hold across datasets, not just California?

- Built a dataset **registry** (`common/datasets.py`) with `--dataset <key>`, each set
  cached to `datasets/<key>.npz`, each with its own split / saved models / output dir.
- Seven datasets, all chosen for an approximately **linear** target relationship:
  `california`, `ccpp`, `energy`, `autompg`, `realestate`, `mortality`, `bloodfat`
  (sklearn + UCI via `ucimlrepo` + Sutanoy textbook sets). Documented in `DATASETS.md`.
- **Takeaway:** on small textbook sets (`mortality` n=60, `bloodfat` n=25) the TM has too
  few samples to shine; the larger sets (`ccpp`, `california`, `energy`) are where the
  comparison is meaningful.

---

## Phase 2 — The central question: does *spacing* explain error?
**Question:** are model errors larger for test points that sit far from other data
(in sparse regions of feature space)?

- Built `experiment2_analytics.py`: for each test point, compute four **spacing**
  definitions — `nn_to_train_k1`, `knn_to_train_k5`, `local_density_k5`,
  `nn_to_test_k1` — in both standardized and PCA-2D space, then correlate each with the
  point's |error| (Pearson/Spearman), with diagnostic panels and RMSE-vs-spacing bucket
  curves (`IMAGES.md`).
- Built `spacing_distribution.py` (how spacing is distributed per dataset: CV, skew,
  normalised entropy) and `correlation_summary.py` (the spacing-vs-error correlation
  across *all* datasets, with `|error|` averaged over N runs to denoise).
- **First cross-dataset result (runs=10, the original 7 datasets):** on the two
  statistically reliable sets the relationship was essentially **noise** — `california`
  ≤ 0.10, `ccpp` ≈ 0. A faint hint that the TM tracked spacing slightly more than the NN
  did **not** survive averaging. Small-n sets (`bloodfat`, `mortality`) gave large but
  meaningless numbers.

---

## Phase 3 — Stress-test with *unevenly-spaced* data
**Question:** the linear sets are fairly uniform; would the spacing→error effect appear
on data that is genuinely **unevenly spaced**?

- Added the UCI **time-series regression** subset, ranked by spacing `norm_entropy`
  (lower = more uneven): `metro` (Metro Interstate Traffic — the most uneven in the whole
  registry, norm-entropy ≈ 0), `beijing` (PM2.5), `airquality` (benzene). Each loader
  filters sentinels / missing markers; documented in `DATASETS.md`.
- Registered a Kaggle `stock` placeholder (competition data can't be fetched headlessly;
  needs a local CSV).

---

## Phase 4 — A finance task: NYSE next-day volume
**Question (user-driven):** fold real market data in, with the objective of
**predicting the next day's trading volume**.

- Added the `nyse` dataset from the Kaggle `dgawlik/nyse` daily price history
  (`datasets/nyse/`). The per-symbol time series is turned into a supervised table: one
  row per symbol-day; features = today's OHLCV + intraday return & range + yesterday's
  volume + 5-day average volume; **target = the same symbol's volume the next trading
  day**. Pooled ~850k rows, strided to ~50k for CPU-TM tractability. Target is heavy-tailed
  (volume spans 10²–10⁸ shares).

---

## Phase 5 — The two-regime finding (11 datasets)
With `metro`, `beijing`, `airquality`, `nyse` run fully through the pipeline, the
cross-dataset spacing-vs-error table (`outputs/spacing_error_correlation.md`, runs=10)
was refreshed to all 11 datasets and a clear picture emerged:

- **Regime 1 — spacing does NOT predict error** (|corr| ≲ 0.1): `california`, `ccpp`,
  `metro`, `beijing`. Strikingly, **`metro` is the most unevenly-spaced dataset yet shows
  ~0 correlation** — so uneven spacing *by itself* is not the driver.
- **Regime 2 — spacing STRONGLY predicts error** (|corr| ≈ 0.4–0.7): `nyse`
  (TM Pearson 0.65–0.69 > NN 0.57) and `airquality` (NN 0.54–0.69 > TM 0.40–0.47) — both
  **heavy-tailed targets**.

**Conclusion:** the discriminator is not how uneven the spacing is, but whether the
**target itself is hard / heavy-tailed in sparse regions**. Which model is more
spacing-sensitive is dataset-dependent (TM>NN on nyse, NN>TM on airquality) — there is
**no universal "TM tracks spacing more than NN" rule**.

**Heavy-tail test (resolved in Phase 6.5).** Added `nyse_log` — the identical dataset with
a log-transformed target — and re-ran the magnitude correlation at runs=10. It **collapsed**
from TM Pearson ≈ 0.65 to **0.26 (Spearman 0.04)** and NN ≈ 0. So the strong raw-`nyse`
signal was **largely a heavy-tail artefact**: far points erred mainly because extreme-volume
outliers are also geometric outliers. The real discriminator is a hard/heavy-tailed target
whose extremes coincide with sparse regions — not spacing magnitude or unevenness per se.

---

## Phase 6 — From spacing *magnitude* to spacing *variance* (current)
**Question (user-driven):** we have tested whether *larger* spacing means worse error.
Now test whether **uneven** spacing does — i.e. "if some data is close together and some
far apart, does performance suffer?"

- Added per-point **local-dispersion** metrics to `common/metrics.py`: `knn_cv_k5`
  (std/mean of the 5 nearest-train distances), `knn_ratio_k5` (d₅/d₁), `knn_std_k5`.
- Built `variance_summary.py` testing the hypothesis at two scales:
  **(B)** per-point local dispersion vs |error| within each dataset; and
  **(A)** dataset-level heterogeneity (`het_global_cv` = spacing varies between points;
  `het_local_cv` = typical within-neighbourhood unevenness) vs normalised RMSE, across
  datasets. Output: `outputs/spacing_variance_correlation.{csv,md}`.
- Ran the quick (single-model, runs=1) pass across all 11 datasets.

**Result — the hypothesis splits by scale:**
- **(B) Per-point: NO.** The scale-free dispersion metrics (`knn_cv_k5`, `knn_ratio_k5`)
  are ≈ 0 on every reliable dataset — local unevenness does not identify which points err.
  The only one that lights up, `knn_std_k5` on `nyse`/`airquality`, is the *raw* std, i.e.
  spacing **magnitude** leaking back in (std and mean neighbour-distance both grow as a
  point gets isolated). So the Phase-5 effect was about points being **far**, not their
  neighbourhoods being **uneven**.
- **(A) Per-dataset: suggestively YES, by rank.** Across the 7 reliable datasets,
  dataset-level heterogeneity `het_global_cv` tracks normalised RMSE with **Spearman ≈
  0.86–0.89** for both models (Pearson is inflated by the `metro` outlier — trust the rank
  version). Heavily caveated: only 7 datasets and confounded with general "messiness"
  (heavy-tailed / outlier-rich datasets have both uneven spacing and high error).

**Conclusion:** spacing *variance* per se does not predict error **within** a dataset; the
weak "uneven → harder" signal lives only **between** datasets and is likely a proxy for
outlier-heavy difficulty. **Confirmed at runs=10** (denoised): scale-free dispersion ≈ 0 on
all reliable sets; the only positive (`knn_std_k5` on nyse) is magnitude leaking back in and
itself collapses on `nyse_log`. Telling detail: `nyse` and `nyse_log` share identical spacing
but `nyse_log` has lower error — so heterogeneity alone doesn't drive performance, the target
tail does. Full numbers in `outputs/spacing_variance_correlation.md`.

---

## Phase 7 — masking experiments (complete)
**Question (user-driven):** stop *observing* spacing and *intervene* — remove training points
stepwise to increase unevenness, and test whether fresh models perform worse and/or need more
epochs/iterations/time to match the original-data performance.

Design (see `PLAN_masking.md`): at each removal fraction, mask training data two ways —
**uniform** (control: sparser but even) and **structured** (treatment: carve holes → uneven);
the structured−uniform gap isolates the pure unevenness penalty. Outcomes: final RMSE vs `p`,
and epochs/iterations/wall-time to reach the full-data target. The "masked test set" is a
region-split of the fixed test set into in-hole (support removed) vs surviving points.

Built `common/masking.py`, `fit_with_history` hooks on both models, `experiment4_masking.py`,
and `viz_masking.py` (per-snapshot 3×3 picture: original/uniform/structured data, the test
error map, and predicted-vs-actual — `outputs/masking/viz/`). Ran energy (prototype), then
**ccpp, california, nyse at confirmatory R=3, full fractions** (initial R=2 pass archived in
`outputs/masking/_r2_backup/`); R=3 reproduced every R=2 conclusion with near-identical numbers.

**Result (full reading in `outputs/masking_summary.md`):** on **clean targets the hypothesis
holds** — random (uniform) removal barely hurts (ccpp NN RMSE 3.80→3.92 at 60 % removed),
but the *same count* removed as holes degrades clearly (3.78→4.34), the structured−uniform gap
grows with `p`, structured-masked NN often **never** recovers the original target (censored
from p=0.2 while uniform reaches it to ~p=0.3), and error concentrates **in the holes**
(ccpp p=0.6 in-hole 4.75 vs surviving 3.58; the `viz` panels show it spatially). On the
**heavy-tailed target (nyse) it breaks down**: the region split doesn't track error (structured
in-hole *consistently better* than surviving, e.g. NN p=0.6 in-hole 5.2 M vs surviving 7.1 M —
holes carve out the easy low-volume corner while the hard tail sits in the dense survivors) and
effort is uninformative — error is governed by the target tail, not geometry, exactly as
Phases 5–6 predicted. Caveat: TM's effort metric is weak (loose 30-epoch baseline target vs
100-epoch masked budget); NN effort is the informative one. **Follow-up done**
(`effort_recovery.py` → `outputs/effort_recovery.md`): re-targeting each model's *own*
extended-budget best repairs TM (ccpp TM then reaches it ~60–75 ep under uniform but is
censored under structured from p=0.2 — the asymmetry NN already showed) but over-tightens NN
(it censors everywhere, having already trained near its own best). No single target is fair to
both → use NN@default-target + TM@extended-target, or a `best+tolerance` bar (future). Also
added `viz_masking.py` TM snapshots (ccpp/california/nyse) alongside the NN ones.

**Overarching arc:** across all phases the through-line is that **feature-space geometry
(spacing, unevenness) predicts/affects error only when the target is well-behaved enough for
geometry to govern error; whenever the target is heavy-tailed, the tail dominates and geometry
becomes a red herring.**

---

### Artifact map
| file | role |
|---|---|
| `experiment1_baseline.py` | TM vs NN baseline RMSE, per dataset |
| `experiment2_analytics.py` | per-point spacing vs error, one dataset (figures) |
| `spacing_distribution.py` | how spacing is distributed per dataset |
| `correlation_summary.py` | spacing **magnitude** vs error, all datasets |
| `variance_summary.py` | spacing **variance** vs error/performance, all datasets |
| `experiment4_masking.py` | Phase 7 masking: RMSE/effort vs `p`, uniform vs structured |
| `viz_masking.py` | Phase 7 snapshot picture: data + error map + fit, orig vs masked |
| `effort_recovery.py` | Phase 7 follow-up: fair effort vs own extended-budget-best target |
| `common/masking.py` | uniform/structured masks + in-hole/surviving region split |
| `common/datasets.py` | dataset registry + loaders |
| `common/metrics.py` | RMSE, spacing definitions, dispersion metrics |
| `DATASETS.md` / `IMAGES.md` / `README.md` | catalogue / figure guide / overview |
| `outputs/spacing_error_correlation.{csv,md}` | Phase 5 result table |
| `outputs/spacing_variance_correlation.{csv,md}` | Phase 6 result table |
| `outputs/masking_summary.md` + `outputs/masking/` | Phase 7 reading, curves, viz |
| `outputs/effort_recovery.md` + `outputs/masking/effort/` | Phase 7 fair-effort follow-up |
