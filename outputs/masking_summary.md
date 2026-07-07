# Phase 7 — masking experiments: reading

Intervention test: remove training points stepwise, two ways — **uniform** (random,
sparser-but-even, the control) vs **structured** (carve holes, genuinely uneven, the
treatment) — and train fresh TM + NN at each removal fraction `p`. The
**structured − uniform gap at equal `p`** is the *pure unevenness penalty* with data
quantity held constant. Data: `outputs/masking/<ds>__curve.csv` + `__{rmse,gap,effort}_vs_p.png`;
per-snapshot pictures in `outputs/masking/viz/<ds>__<model>__p<p>__viz.png`.

**Run grid: confirmatory R=3, full fractions `p∈{0,.1,.2,.3,.4,.5,.6}` for ccpp, california,
nyse** (energy = earlier prototype). This supersedes the initial R=2 pass (curves archived in
`outputs/masking/_r2_backup/`); **R=3 reproduces every R=2 conclusion with near-identical
magnitudes** — the readings below are the R=3 numbers.

## Headline
On **clean targets the hypothesis holds clearly**: random data loss barely hurts, but the
*same amount* removed as holes (unevenness) degrades performance, slows/prevents convergence
to the original target, and the damage lands exactly in the carved-out regions. On the
**heavy-tailed target (nyse) the clean story breaks down** — geometry-based masking doesn't
track error, because nyse error is governed by the target tail, not feature-space geometry
(consistent with the Phase 5/6 heavy-tail-artefact finding).

## By hypothesis (clean datasets: ccpp, california)

**H1/H2 — unevenness hurts, pure sparsity barely does.** NN RMSE on **ccpp** (target 3.79):
uniform stays ~flat across removal — **3.80 → 3.92 even at 60 % removed** — while structured
rises monotonically **3.78 → 4.34**. The structured−uniform gap grows with `p`
(~0 at p=0.1 → ~+0.16 at p=0.3 → ~+0.42 at p=0.6). **california** NN (target 0.519): uniform
0.52 → 0.55, structured 0.54 → 0.62; the gap opens from p≥0.3. So removing *random* points
is largely absorbed; removing *clustered* points (creating holes) is what costs accuracy.

**H3 — effort to recover the original performance.** Structured-masked NN frequently **never
reaches** the full-data target within the extended 300-epoch budget (censored). On **ccpp**
structured NN is censored (3/3 repeats) from **p=0.2 onward**, whereas uniform NN still reaches
the target up to ~p=0.3 (first censored at p=0.4). On **california**, structured is censored
(3/3) from **p=0.3**, uniform from p=0.4. Unevenness doesn't just cost final accuracy, it makes
the original performance *unrecoverable* at a lower removal fraction than random loss does.

**H4 — where the damage lands.** On clean datasets, error concentrates in the holes:
structured `rmse_inhole ≫ rmse_surviving` (ccpp p=0.6: in-hole **4.75** vs surviving **3.58**;
california p=0.6: in-hole 0.65 vs surviving 0.57). The surviving (still-dense) region is often
**as good as baseline** — the model is fine where it kept support and bad only where support
was carved away. The `viz` snapshots make this visible: the structured DATA panel shows the
carved grey hole, and the in-hole test points (green rings) light up with error exactly there.

## The heavy-tailed regime (nyse) — the clean story inverts
nyse RMSE is in millions and dominated by the volume tail. Structured masking still raises RMSE
more than uniform (TM ≈10.0 M → **13.8 M** at p=0.6 vs uniform ~10.7 M; NN ~5.6 M → 6.1 M vs
uniform ~5.6 M), but the *diagnostic structure inverts*:
- The **region split does not track error**: across every structured `p`, `rmse_inhole` is
  *lower* than `rmse_surviving` (e.g. NN p=0.6: in-hole **5.2 M** vs surviving **7.1 M**;
  TM p=0.3: in-hole 9.6 M vs surviving 11.0 M). The holes happened to remove easy low-volume
  regions; the hard high-volume outliers sit in the dense survivors. Geometry ≠ difficulty here.
- **Effort is uninformative**: with target standardization the NN reaches its (tail-dominated)
  target in ~1–4 epochs regardless of masking (essentially never censored). The heavy tail, not
  data geometry, sets the error.

This is the same lesson as Phases 5–6: on nyse, geometry is a red herring; the target tail rules.

## Model note / caveats
- **TM effort is weak**: the target is the baseline at the default **30** TM epochs, but masked
  TM gets 100 — and the 30-epoch target is loose vs TM's continued convergence — so masked TM
  "reaches target" in 2–8 epochs trivially. The **NN** effort metric (100-epoch baseline vs
  300-epoch extended budget) is the informative one. The RMSE and H4 readings hold for both models.
- **Noise**: R=3 (up from R=2); TM final RMSE is still noisy run-to-run on clean sets, but the
  uniform-vs-structured ordering and the H1–H4 readings are stable across both passes.
- A cleaner effort comparison sets each model's target at *its own extended-budget* full-data
  RMSE (not the default-epoch one) — done in `outputs/effort_recovery.md` (`effort_recovery.py`).
  Result: the fair bar **repairs the TM metric** (on ccpp, TM now reaches it in ~60–75 epochs
  under uniform but is censored under structured from p=0.2 — the same asymmetry NN shows),
  but **over-tightens NN** (NN already trains near its own best, so once any data is removed it
  censors everywhere). No single target is fair to both; the informative pairing is NN@default-
  target + TM@extended-target, and both agree structured removal fails to recover sooner than
  uniform. A `best + small tolerance` target would restore the gradient for both (future work).

## Bottom line
Increasing unevenness by carving holes in the training data **does** make fresh models worse
and harder to converge — and measurably more than removing the same number of points at random
— **but only when the target is well-behaved enough that feature-space geometry governs error.**
On a heavy-tailed target the effect is swamped by the tail, exactly as the correlation phases
predicted. Confirmed at R=3.
