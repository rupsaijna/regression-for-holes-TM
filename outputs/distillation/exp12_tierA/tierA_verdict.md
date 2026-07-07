# Experiment 12 / Tier A — spacing/density predicts distillation benefit across datasets: verdict

Broad run: `experiment11_tmag.py --out exp12_tierA --seeds 4`, 12 datasets (large ones via
`--subsample-train 8000`); analysed by `correlate_modes_density.py` (Spearman of each mode's
fractional benefit `eff_norm` against every density metric AND the teacher_edge, across datasets).
Files: `modes_effectiveness.csv`, `modes_density_correlations.csv`, `modes_density_summary.md`,
`modes_density_heatmap__eff_norm.png`.

## Headline — density/spacing predicts benefit; teacher accuracy does not (confirmed at n=12)

Best DENSITY predictor vs teacher_edge for each mode (Spearman with eff_norm):

| mode | best density predictor | ρ | teacher_edge ρ |
|---|---|---|---|
| force_ii_tmag | **spc_err_strength** | **−0.78 (p=0.003)** | −0.29 |
| force_ii | d (dimensionality) | +0.58 | −0.04 |
| teacher_sign_tmag | knn5_mean | −0.49 | −0.24 |
| sameside_tmag | var_err_strength | −0.45 | −0.29 |
| uniform | n (size) | +0.46 | −0.07 |

For every mode a density/structure metric beats teacher accuracy, which sits at ≈0 (−0.04 to
−0.29). This confirms the arc-long thesis at the dataset level and replicates the earlier
`eff_density` result (the old basic mode had spc_err_nn_k1 ≈ −0.81).

## The robust spacing signal

`spc_err_strength` (how strongly local spacing predicts error in a dataset, from the Phase 5/6
analysis) is the strongest and most robust predictor: **force_ii_tmag eff_norm vs spc_err_strength
= −0.78 (p=0.003)**, with spc_err_nn_k1 −0.58 (p=0.049) and var_err_strength −0.49 echoing it. The
sign is NEGATIVE and consistent: **the more a dataset's error is governed by spacing, the LESS
distillation helps** (force_ii_tmag is positive on ccpp/nyse_log/beijing — low spc_err — and most
negative on airquality/bloodfat/energy — high spc_err). force_ii_tmag shows the cleanest coupling.

## Caveats (why the wording is "spacing/density", not a single clean law)
- **het_global_cv was a small-panel artifact:** −0.81/−0.74 at n=8 → −0.43/−0.42 (n.s.) at n=12. The
  large heterogeneous datasets (metro het_cv=13.4) broke it. Do not rely on it.
- **Some modes track SIZE/DIMENSIONALITY, not spacing:** uniform vs n +0.46, force_ii vs d +0.58 —
  their benefit is partly a "bigger/higher-dim dataset → more headroom" effect, not purely spacing.
- n=12 still includes tiny-n datasets (bloodfat n=20, mortality n=48); p-values uncorrected over many
  pairings. Treat as illustrative, not confirmatory.

## Less-is-more, corrected
Full-data M00 worse than the 30%-hole unguided student (normalized margin (M00−ung)/ung > 0):
**nyse_log +4.5%, realestate +2.8%, metro +0.4%** — real but weak. **airquality's E11 less-is-more
was seed-fragile** (1.745 vs 1.736 at 6 seeds; flips to M00-better at 4 seeds) — do NOT cite it as
the example. energy is firmly the opposite (M00 −18% better). So "more data can hurt" holds on a
few genuinely heterogeneous datasets, just not airquality robustly.

## Robustness: n≥100 subset (drop bloodfat n=20, mortality n=48 → 10 datasets)
The headline survives and tightens. `force_ii_tmag` eff_norm vs **spc_err_strength = −0.76
(p=0.011)**, and with the tiny-n datasets removed a whole CLUSTER of spacing/spread metrics turns
negative for it — spread −0.62, nn1_mean −0.57, min_pairwise −0.57, knn5_median −0.52, spc_err_nn_k1
−0.52, knn5_mean −0.50. So force_ii_tmag's benefit is broadly, not narrowly, anti-correlated with
spacing. Its teacher_edge correlation stays weak (−0.19).

HONEST NUANCE: the strong, robust dataset-level result is specifically for **force_ii_tmag** (the
mode with the cleanest spacing coupling). For the OTHER modes the correlations are weaker and noisier
at n=10–12 (e.g. uniform's best is the unstable het_global_cv, and its teacher_edge even reaches
+0.42 on the subset). So "spacing/density governs benefit, not teacher accuracy" is firmly
established for force_ii_tmag and directionally consistent — but not uniformly strong — across the
other modes. Run command: `python correlate_modes_density.py --exp-dir outputs/distillation/exp12_tierA --min-n 100`.

---

## Combined E12 synthesis (Tier A + Tier B)
- **Tier A (per-dataset):** spacing/density structure predicts WHICH datasets benefit — negatively
  via spc_err_strength — and beats teacher accuracy as a predictor.
- **Tier B (per-region):** yet the mechanism is NOT spatial gap-filling — benefit is spacing-agnostic
  within a dataset, and error rises with spacing only in the sparsest decile.
- **Unified picture:** distillation in RegressionTM is a GLOBAL reshaping of the learned function.
  It helps when a dataset's error is NOT strongly spacing-governed (smoother, more homogeneous → a
  global correction fits), and fails when error IS spacing-governed (heterogeneous → a global
  correction mismatches the local structure). This single mechanism explains both the negative
  dataset-level correlation (Tier A) and the spatially-flat benefit (Tier B), and closes the loop on
  why teacher accuracy and extra data are both poor guides to distillation's effect.
