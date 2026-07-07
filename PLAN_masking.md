# Phase 7 plan — masking experiments (data removal → unevenness → performance & effort)

**Status:** planned, not yet implemented. Author it as `experiment4_masking.py` (+ a
masking utility and small per-epoch hooks on the models). Builds on the Phase 5/6 finding
that spacing *magnitude* matters on hard targets but spacing *variance* per se does not
predict per-point error — here we move from *observing* spacing to *intervening* on it.

## Objective
Remove training points stepwise and test whether freshly-trained models (a) perform
**worse** and/or (b) need **more epochs / iterations / wall-time** to reach the
original-data performance — and crucially whether that is due to **less data** or to the
**unevenness** the removal creates.

## Central design idea (decided with user)
At each removal fraction `p`, mask the **training** set two ways:
- **uniform** (control): drop `p·n` points uniformly at random → sparser but still even.
- **structured** (treatment): drop `p·n` points by carving **holes** (random seed points +
  their nearest neighbours) → genuinely uneven (dense regions + empty regions).

Train fresh TM + NN on each. The **structured − uniform gap at the same `p`** is the *pure
unevenness penalty* with data quantity held constant. (Random removal alone increases
sparsity, not unevenness — hence the need for both arms.)

## Hypotheses
- H1 (data quantity): RMSE rises with `p` under **both** modes (less data ⇒ worse).
- H2 (unevenness): at equal `p`, **structured** degrades more than **uniform** (the gap > 0).
- H3 (effort): masked models need more epochs/iterations/time to hit the original target,
  more so under structured masking; some high-`p` settings never reach it (censored).
- H4 (where it hurts): degradation concentrates on test points whose local training
  support was removed (the "in-hole" region split below).

## Factors & grid
- `p` (removal fraction): **0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6** (stepwise; 0 = baseline).
- `mode`: {uniform, structured}. (`p=0` is shared — train once, reuse for both.)
- `repeats R`: **3** per (p, mode), different seeds → average over *which* points are removed
  (essential because removal is random). Report mean ± std.
- `model`: {TM, NN}, freshly trained each time.

## Masking utility (`common/masking.py`)
- `mask_uniform(idx, p, seed) -> kept_idx` : random subset.
- `mask_structured(X_std, idx, p, seed, hole_frac=0.05) -> (kept_idx, removed_idx)` :
  repeatedly pick a random surviving seed, remove it + its `hole_frac·n` nearest surviving
  neighbours, until `≈ p·n` removed. Carves several holes (count ≈ `p/hole_frac`). Deterministic
  per seed. `hole_frac` is the one knob trading "few big holes" vs "many small holes".
- Both operate on **standardized** train coordinates (consistent with all spacing work).

## Outcome A — final performance
For each (dataset, p, mode, repeat, model): RMSE on
1. **original test set** (fixed, never masked) — the honest generalization measure (primary);
2. **region-split of the original test set** (see below) — *where* the damage lands.

Plot RMSE vs `p`, uniform vs structured, per dataset/model; also plot the structured−uniform
gap (the unevenness penalty). Normalised RMSE (`/std(y)`) for cross-dataset comparability.

### Region split (the "masked test set")
After structured masking, label each **original** test point by how much of its local
training support was removed: `in_hole = (fraction of its k=5 nearest ORIGINAL-train
neighbours that were removed) ≥ 0.5`, else `surviving`. Report RMSE on `in_hole` vs
`surviving` separately. This is the meaningful "masked test set" — it isolates the points
sitting in the carved-out sparse regions. (Computed for uniform mode too, for symmetry, but
there the removed points are scattered so `in_hole` stays small — expected.)

## Outcome B — convergence effort
- **Target** per (dataset, model): the `p=0` model's RMSE on the original test set.
- Track **per-epoch** test RMSE during training; record the **first epoch** whose test RMSE
  ≤ target (`epochs_to_target`); **censored = NaN** if never reached within an extended budget.
- Also record **minibatch iterations** (NN: epochs × ⌈n/batch⌉) and **wall-time to target**.
- Give masked models a fair chance: raise the max-epoch budget (NN ~300, TM ~100) so "needs
  more epochs to reach the same performance" is observable rather than clipped.
- Plot `epochs_to_target` (and wall-time) vs `p`, uniform vs structured.

### Required model hooks (small)
Add `fit_with_history(X, y, X_eval, y_eval) -> list[per-epoch test RMSE]` to both wrappers:
- **NNModel**: inside the existing `for epoch` loop, after each epoch `predict(X_eval)` and
  append RMSE. (Net trains in place — trivial.)
- **TMModel**: the loop already calls `self.tm.fit(Xb, y)` once per epoch (tmu training is
  incremental), so `predict(X_eval)` between calls yields the curve. (No retraining cost
  beyond the eval prediction.)
No change to the default `fit()` path; the hook is opt-in for this experiment.

## Datasets (staged, to bound compute)
The grid is large (datasets × 7 `p` × 2 modes × 3 repeats × 2 models, fresh training each,
extended epochs). Stage it:
1. **Prototype/validate** on fast sets: `energy` (n≈768) and `ccpp` (clean linear, ~30 s/train).
2. **Main run** add `california` (moderate) and one hard/heavy-tail set — `nyse` (or `metro`).
Recommended primary set: **ccpp, california, nyse** (one per regime), `energy` for quick iterate.
Rough cost: `nyse` ≈ 2 min/train × (6 p × 2 modes × 3 repeats + 1 baseline) ≈ 37 trains × 2
models ≈ 2–2.5 h for nyse alone — so run as a background job per dataset, or trim to R=2 /
fewer `p` steps for a first look.

## Outputs / deliverables
- `experiment4_masking.py` (+ `common/masking.py`, model `fit_with_history` hooks).
- `outputs/masking/<dataset>__curve.csv` : rows (p, mode, repeat, model, rmse_orig,
  rmse_inhole, rmse_surviving, epochs_to_target, iters_to_target, secs_to_target, n_removed).
- `outputs/masking/<dataset>__rmse_vs_p.png`, `__gap_vs_p.png`, `__effort_vs_p.png`.
- `outputs/masking_summary.md` : the H1–H4 reading across datasets.
- Update `HISTORY.md` (Phase 7) and `README.md`/`IMAGES.md`.

## Open knobs (sensible defaults chosen; easy to change)
- `hole_frac` (hole size), `k=5` for the region-split, `R=3`, `p` grid max 0.6, extended
  epoch budgets, dataset list. All surfaced as CLI args.

## Not doing (explicit scope)
- No feature masking (only row/point removal). No re-tuning hyperparameters per `p` (we hold
  architecture/HP fixed so the effect is attributable to the data, not the model search).
