# Experiment 12 / Tier B — per-region spacing analysis: verdict

`spacing_buckets.py`, 4 anchor datasets, 6 seeds, held-out TEST points pooled. spacing =
distance from each test point to its nearest SURVIVING (30%-hole) train point, z-scored feature
space. Models: teacher M20, unguided 30%-hole student, guided force_ii & teacher_sign_tmag.
B1 = does error rise with spacing? B2 = does distillation benefit concentrate where it's sparse?

## B1 — error vs spacing: YES, but as a SPARSEST-DECILE THRESHOLD, not a smooth gradient

Spearman(spacing, |error|) over points is weak (unguided): ccpp +0.02, california +0.06,
airquality +0.14, energy +0.20. But the decile curves show why it's weak — error is roughly FLAT
across the bulk and jumps only in the sparsest bucket(s):

| dataset | err_unguided densest→sparsest decile | sparse penalty |
|---|---|---|
| ccpp | 4.09 (flat ~4.0–4.3) → **4.58** | +12% (only bucket 9) |
| airquality | 1.07 → 1.19 → … → **1.80** | **+69%** (steady, monotonic) |
| california | 0.44 → … → **0.55** | +25% (gentle, monotonic) |
| energy | 0.77 → … → **1.73** | +124% (degenerate spacing*) |

So the founding PROMPT-1 hypothesis — *greater spacing → poorer learning* — **holds at the
extreme**: the sparsest test points are clearly the hardest, for the teacher too (M20 error also
peaks in bucket 9 everywhere). It is **weak/absent through the bulk** of the spacing range. The
effect is strongest on airquality & energy (steady rise) and nearly confined to the last decile
on ccpp (a dense, smooth dataset). *energy spacing is partly DEGENERATE (discrete features →
median=min=0.645, deciles collapse to ~5 buckets) — read its number cautiously.

## B2 — benefit vs spacing: NO. Distillation does NOT concentrate help in sparse regions

Spearman(spacing, benefit) ≈ 0 everywhere (force_ii: ccpp +0.04, california +0.03, energy +0.02,
airquality −0.02). The decile curves confirm the benefit's SIGN is set by the dataset/teacher and
is spatially DIFFUSE, not localized to the holes:

- **ccpp**: benefit positive in EVERY bucket (force_ii +0.07…+0.50), only mildly higher in the
  sparsest. Distillation helps roughly everywhere.
- **airquality**: benefit positive everywhere but actually LOWEST in the sparsest bucket
  (force_ii +0.066 at bucket 9 vs +0.2–0.3 in the bulk) — the OPPOSITE of "fills the holes."
- **california / energy**: benefit NEGATIVE in every bucket (it hurts uniformly), marginally less
  negative in sparse regions.

## B3 — in-hole vs out-of-hole benefit: a GOOD teacher DOES fill its holes (continuous spacing missed this)

A test point is "in-hole" if its nearest REMOVED train point is closer than its nearest SURVIVING
one — i.e. its specific local training support was carved out (in-hole fraction ≈0.28–0.32 across
datasets). Mean benefit, in-hole / out-of-hole:

| dataset | teacher | force_ii (in / out) | teacher_sign_tmag (in / out) |
|---|---|---|---|
| **ccpp** | good | **+0.397 / +0.194** (≈2.0×) | **+0.458 / +0.312** (≈1.5×) |
| airquality | weak | +0.164 / +0.222 (in < out) | +0.126 / +0.109 (≈) |
| california | neutral | −0.025 / −0.016 (in worse) | −0.031 / −0.021 (in worse) |
| energy | bad | −0.192 / −0.179 (≈) | −0.095 / −0.124 (in less harm) |

**On the GOOD teacher (ccpp), distillation helps ~1.5–2× MORE in the carved-out regions** — a
genuine localized gap-filling effect that the B2 spacing-correlation (≈0) did not show. The two are
not contradictory: the structured holes carve k-NN neighbourhoods out of DENSE regions, so in-hole
test points sit at MODERATE distance-to-surviving-data, not in the sparse tail — B2's continuous
spacing axis doesn't separate them, but the binary "was your local support removed?" does. For the
weak/neutral/bad teachers the effect is absent: in-hole benefit is equal-to-slightly-worse than
out-of-hole. So localized hole-filling is real but **conditional on a good teacher**.

## Verdict — the mechanism behind the whole arc

**Distillation here is PREDOMINANTLY a global reshaping of the fit, with a localized hole-filling
component that appears only for a good teacher.** Whether it helps or hurts a point is governed
mainly by dataset/teacher quality, not by how sparse the point's neighbourhood is (B2 ≈ 0). BUT the
in/out-hole split (B3) shows a good teacher (ccpp) helps ~1.5–2× more in the specific regions whose
training support was carved out — it does transfer information about exactly those gaps. This
localized effect is absent for weak/neutral/bad teachers (airquality helps *least* in-hole;
california/energy equal-or-worse in-hole). So the teacher partially repairs its holes ONLY when it
is good there; otherwise the benefit (or harm) is spatially diffuse.

This finally explains the recurring E8–E11 puzzles with one mechanism:
- **Why teacher accuracy doesn't predict benefit** ([[teacher-edge-doesnt-predict-benefit]]): the
  teacher having more data in the holes is irrelevant, because its correction doesn't act in the
  holes — it acts globally.
- **Why "more data" reasoning fails / airquality is "less-is-more":** learning quality is dominated
  by the dense bulk (where most points and most error mass live), not the sparse tail. Adding data
  back (full M00) can re-introduce harmful bulk points without fixing the sparse tail that the
  models already half-ignore.
- **Why every mode is a "sharpener, not a fixer" (E8/E11):** a global reshaping scales whatever
  global signal the teacher carries; it has no spatial mechanism to selectively repair weak regions.

So spacing matters for ABSOLUTE difficulty (B1: sparse points are hardest) but NOT for WHERE
distillation acts (B2: benefit is spacing-agnostic). The two are decoupled.

## Caveats & next
- n=4 datasets; energy spacing degenerate. Tier A (broad 12-dataset, per-dataset density→benefit
  correlation) is the complementary test of B1's *dataset-level* version.
- `hole_dist` (distance to nearest REMOVED point) is recorded per point in the CSVs but not yet
  analysed — a direct "in-hole vs out-of-hole" benefit split would sharpen B2/B3. Worth a follow-up.
- Plots: `error_vs_spacing__<ds>.png`, `benefit_vs_spacing__<ds>.png`.
