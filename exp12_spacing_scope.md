# Experiment 12 — data spacing / density as the governing variable (SCOPE)

## Why this experiment
The whole E8–E11 distillation arc kept colliding with the same wall:
- Teacher **accuracy does not predict** whether distillation helps (energy: biggest teacher edge,
  worst benefit). `teacher-edge-doesnt-predict-benefit`.
- **More data is not always better** (airquality: full-data M00 1.745 > 20%-hole M20 1.657 >
  30%-hole student 1.736 — a "less-is-more" dataset; structured holes *removed* harmful points).
- Every mechanism we tried (adaptive trigger E8, opposite-side overrides E9/E10, teacher-bounded
  magnitude E11) is a **sharpener, not a fixer** — it scales whatever signal the teacher carries.

The variable that keeps implicating itself is **data spacing / density** — which is exactly the
question PROMPT 1 set out to answer: *"see if greater spacing between data points leads to poorer
learning of the regression line."* The distillation work gave us better modes and a clean
per-region probe (the structured holes ARE controlled sparse regions). E12 uses them to answer it.

## What already exists (reuse, don't rebuild)
- `density_metrics.py` → `outputs/distillation/density_metrics.csv`: per-dataset density battery
  for 12 datasets (spread, nn1_mean, knn5_mean/median, het_global_cv, knn_cv, norm_entropy,
  min_pairwise) + Phase 5/6 spacing↔error scores (spc_err_nn_k1, spc_err_strength, var_err_strength).
- `correlate_eff_density.py` → `eff_density_summary.md` / `eff_density_correlations.csv` / heatmap:
  per-dataset teacher-effectiveness (from experiment5 sweeps) × density. Prior top hit:
  `eff_vs_teacher × spc_err_nn_k1` Spearman **−0.81 (p=0.001)** and `eff_winrate × var_err_strength`
  −0.64 — i.e. datasets where spacing predicts error MORE strongly get LESS teacher benefit. BUT:
  single config, single (basic) mode, structured-only, tiny-n datasets included.
- `common/metrics.py`: spacing primitives (spacing_nn_to_self, spacing_local_density,
  spacing_knn_cv/ratio, min_pairwise_distance, spread).
- Trained-model harness from experiment11_tmag.py (unguided / guided modes / M20 / M00).
- 12 dataset sweeps already on disk (sweep__<ds>__structured.csv).

## The gap E12 fills
1. The per-dataset correlation used the OLD basic mode. Redo it with the E8–E11 BEST modes
   (force_ii, teacher_sign_tmag, force_ii_tmag, sameside_tmag) and the new M00 "less-is-more"
   signal, over the full 12-dataset panel.
2. Nobody has gone PER-REGION: does the teacher (and each model) actually learn worse in
   high-spacing regions, and does distillation's help concentrate there? That is the founding
   question and the mechanism that would EXPLAIN the dataset-level correlation.

---

## Tier A — per-dataset: does spacing/density predict WHICH datasets benefit? (broad, reuse-heavy)
**Hypothesis A.** Distillation benefit is governed by a spacing/density descriptor, and the
relationship is stronger / cleaner than teacher accuracy (which we showed is ~uncorrelated).

**Build.**
- A1. Run `experiment11_tmag.py --datasets <all 12> --seeds 6` (or a chosen subset of modes) to get
  `impr_<mode>`, win-rate, vs-M20, vs-M00 per dataset for the full panel. Also dumps M00 so the
  **"less-is-more" flag** `(rmse_M00 > rmse_unguided)` is computable per dataset (airquality flips it).
  *Compute note:* nyse/metro/beijing/california are large (16k–42k rows) × 8 models × 6 seeds × 100ep
  — STAGE it: small/mid datasets first (energy, autompg, realestate, bloodfat, mortality, ccpp,
  airquality), then the heavy ones; consider --seeds 4 or a subsample cap for nyse/metro/beijing.
- A2. Refresh `density_metrics.csv` if any descriptor is missing (it already covers all 12).
- A3. New/adapted correlator (extend `correlate_eff_density.py` or a sibling) that joins the E12
  effectiveness (per mode) to the density battery and reports Spearman/Pearson per (mode ×
  effectiveness-metric × density-metric), with the n≥100 subset (drop bloodfat/mortality).

**Deliverable.** `outputs/distillation/exp12/spacing_predicts_benefit.{csv,md,png}` — does a spacing
metric predict benefit across datasets, per mode? Does it beat teacher-edge as a predictor
(compare to `teacher_edge_corr`)? Is the "less-is-more" set explained by a density descriptor?

## Tier B — per-region: does learning degrade with spacing, and does the teacher fill the gaps? (NEW, the crux)
This is the direct PROMPT-1 test, and the mechanism behind Tier A.

**Hypotheses.**
- **B1 (founding question).** Per model (teacher M20, unguided student, best guided student),
  per-test-point error rises with the point's local spacing (distance to the nearest TRAIN point).
- **B2 (mechanism).** Distillation's benefit `|err_unguided| − |err_guided|` concentrates in
  HIGH-spacing test regions (the sparse / masked-hole regions) — OR, consistent with E8–E11, it
  does NOT, which would explain why teacher accuracy fails to predict benefit.
- **B3.** The masked structured holes are the controlled sparse regions: error/benefit measured as
  a function of "distance into the hole" isolates spacing from confounds.

**Build (`spacing_buckets.py`, ~1 new script).** Per dataset × seed:
- Train M20, unguided M30, best guided (e.g. force_ii and teacher_sign_tmag) — same harness as E11.
- For each TEST point: spacing = nn-distance to the kept (30%-hole) train set in standardized
  feature space (reuse `metrics.spacing_nn_to_self`/local_density); also a "hole-proximity" score
  = distance to nearest REMOVED train point.
- Record per-test-point: spacing, hole-proximity, and signed/abs error for each model.
- Aggregate into spacing quantile buckets (e.g. deciles): per bucket, mean |error| per model, and
  teacher benefit per bucket. Pool across seeds; also across datasets (normalized).

**Deliverable.** `outputs/distillation/exp12/spacing_buckets.{csv,md}` + per-dataset and pooled plots
`error_vs_spacing__<ds>.png`, `benefit_vs_spacing__<ds>.png`:
- error-vs-spacing curve per model (tests B1),
- teacher-benefit-vs-spacing curve (tests B2),
- split by in-hole vs out-of-hole (tests B3).

---

## Datasets
Full panel of 12 (drop `stock`; `nyse`/`nyse_log` overlap — keep `nyse_log`). For Tier B, start
with the E8–E11 four (ccpp, airquality, california, energy) since they anchor the distillation
findings and span good/weak/neutral/bad teacher + the less-is-more case, then broaden.

## Recommended order (smallest, most-novel first)
1. **Tier B on the 4 anchor datasets** — cheapest (3 models/seed), most novel, directly answers
   PROMPT 1 and explains airquality's "less-is-more". `spacing_buckets.py`.
2. **Tier A staged broad sweep** — reuses experiment11 + correlate; the compute-heavy part.
3. Synthesize: does per-region (B) mechanism explain per-dataset (A) correlation? Update
   `explanations_of_runs.py` and write the E12 verdict.

## Risks / notes
- n=12 datasets → dataset-level correlations are illustrative (the prior −0.81 had p=0.001 but on
  12 pts with tiny-n members); Tier B's per-point analysis has far more statistical power and is
  the stronger evidence.
- Spacing in standardized feature space ignores the y-direction; consider also input-space vs
  joint (x,y) spacing.
- Keep the held-out TEST set fixed and unmasked throughout (as in E8–E11).
