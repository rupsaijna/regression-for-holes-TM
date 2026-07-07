# Experiment 11 — teacher-bounded magnitude generalized: verdict

Config: teacher 20% → student 30% mask, structured, f=0.75, p=0.5, 100 epochs, 6 seeds,
held-out TEST, paired by seed (`vanilla_regressor_v4.py`). Modes: `uniform` (baseline,
opposite ignored), `force_ii` (E9 ref), **`force_ii_tmag`** (NEW = force_ii + teacher-bounded
magnitude), `teacher_sign_tmag` (E10 winner ref), **`sameside_tmag`** (NEW = opposite untouched,
SAME-side updates take the teacher's magnitude). `impr` = rmse(unguided) − rmse(guided);
`gain` = mode − uniform. A full-data model `M00` (0% hole) was added as the ceiling via
`full_baseline.py` (the E11 run predated that edit).

**Controls reproduce E8/E9/E10 exactly** (uniform, force_ii, teacher_sign_tmag all match) → the
new-mode deltas are real.

| dataset | mode | mean impr | t | std | gain vs uniform |
|---|---|---|---|---|---|
| **ccpp** | uniform | +0.360 | +2.57 | 0.34 | — |
| | force_ii | +0.251 | +1.39 | 0.44 | −0.109 |
| | **force_ii_tmag** | **+0.431** | +3.52 | 0.30 | +0.071 |
| | teacher_sign_tmag | +0.367 | +2.72 | 0.33 | +0.007 |
| | sameside_tmag | +0.392 | +3.54 | 0.27 | +0.033 |
| **airquality** | uniform | +0.078 | +1.02 | 0.19 | — |
| | **force_ii** | **+0.175** | +2.33 | 0.18 | +0.098 |
| | force_ii_tmag | **−0.140** | −1.30 | 0.26 | −0.218 |
| | teacher_sign_tmag | +0.107 | +1.07 | 0.25 | +0.030 |
| | sameside_tmag | +0.072 | +1.05 | 0.17 | −0.006 |
| **california** | uniform | −0.013 | −3.01 | 0.01 | — |
| | sameside_tmag | −0.029 | −5.18 | 0.01 | −0.016 |
| | *(others)* | −0.015..−0.019 | | | ~uniform |
| **energy** | uniform | −0.062 | −1.12 | 0.13 | — |
| | force_ii | −0.237 | −4.53 | 0.13 | −0.175 |
| | force_ii_tmag | −0.116 | −1.71 | 0.17 | −0.054 |
| | teacher_sign_tmag | −0.152 | −3.37 | 0.11 | −0.090 |
| | **sameside_tmag** | **−0.013** | −0.51 | 0.06 | +0.048 |

## Verdict 1 — bounded magnitude is a SHARPENER, not a universal lever (E10 insight partly refuted)

The cleanest A/B is `force_ii` → `force_ii_tmag` (same forced Type-II, only the magnitude source
changes from student to teacher):

| dataset | force_ii | force_ii_tmag | effect of teacher magnitude |
|---|---|---|---|
| ccpp (good) | +0.251 | **+0.431** (beats no-teacher 6/6) | ✅ much better |
| airquality (weak) | **+0.175** | **−0.140** | ❌ **win destroyed** |
| california | −0.015 | −0.017 | ~ same |
| energy (bad) | −0.237 | −0.116 | ✅ harm ≈ halved |

E10 concluded teacher-bounded magnitude was *the* transferable lever. E11 shows that was too
strong. Bounded magnitude **amplifies a good teacher and caps a bad teacher's harm — but
backfires on a marginal teacher.** `force_ii`'s airquality win came specifically from using the
*student's* magnitude on disagreement samples; replacing it with the teacher's magnitude
over-trusts a teacher that is only weakly better, and +0.175 flips to −0.140. `sameside_tmag` in
isolation tells the same story: best-of-arc gentleness on energy (−0.013 ≈ no-teacher) and a small
ccpp gain, but it *worsens* the neutral california case. This is the same **sharpener-not-fixer**
behaviour E8 found for adaptive triggering: concentrating the teacher's action (here by trusting
its error magnitude) scales whatever signal it carries — helpful on a good teacher, harmful on a
weak/bad one.

**Best good-teacher mode of the whole arc:** `force_ii_tmag` on ccpp (+0.431, t=3.52, 6/6 seeds).
**Gentlest bad-teacher mode of the whole arc:** `sameside_tmag` on energy (−0.013, essentially
no harm). No mode is both.

## Verdict 2 — does any guided student beat the FULL-data model (M00)?

M00 (full, 0% hole) RMSE: ccpp 4.758, airquality **1.745**, california 0.612, energy 0.959.

- **ccpp — essentially a TIE.** `force_ii_tmag` reaches 4.772 vs M00 4.758 — within noise
  (−0.014 mean, **wins 3/6 seeds**). A student trained on 70% of the data + distillation nearly
  matches the full-data model. It does not *beat* it, but it closes almost the entire gap.
- **airquality — YES, and it's the surprise.** The full model is *anomalously bad* here
  (M00 1.745 > M20 1.657 AND > the 30% student 1.736): more data makes it WORSE. The structured
  holes happen to remove points that hurt generalization. So `force_ii` beats full by **+0.185
  (5/6 seeds)**, and uniform / teacher_sign_tmag / sameside_tmag beat it too. Caveat: this is
  "full is a weak baseline on this dataset," not "distillation beat an honest ceiling." It is a
  direct example of the project's core hypothesis — that on some datasets less, better-spaced data
  generalizes better — and airquality is a "less-is-more" dataset (RMSE is monotonic in the WRONG
  direction: full > 30% > 20%).
- **california / energy — NO.** Full is the genuine ceiling and wins comfortably (california gap
  ~0.02–0.04; energy gap large, M00 0.959 vs best guided ~1.24). No mode beats M00 (0–1/6 seeds).

## Student-beats-teacher (vs M20), final
- ccpp: `force_ii_tmag` strongest (+0.107, 4/6) — beats the teacher it learned from.
- airquality: `force_ii` best (+0.097, 4/6); `force_ii_tmag` does NOT (it lost the win).
- No single mode convincingly beats M20 on BOTH productive datasets; `teacher_sign_tmag` is the
  only one positive on both, but marginally (3/6 each).

## Teacher-edge correlation (refresh)
Across the 4 datasets, edge↔benefit Spearman ρ ≈ +0.40–0.80 depending on mode (sameside_tmag
0.80), Pearson +0.20–0.78 — still weak, still n=4, energy still the off-trend point. Teacher
accuracy remains a poor predictor of benefit. See `teacher_edge_corr.png`.

## Overall takeaways
1. There is **no single best mode.** The right correction depends on teacher quality, which is a
   per-dataset property: `force_ii_tmag`/`sameside_tmag` for a known-good teacher, plain `force_ii`
   for a marginal one, `sameside_tmag`/`uniform` to minimize harm when the teacher may be bad.
2. **Bounded magnitude ≠ free lunch** — it sharpens, helping good/bad extremes but hurting the
   marginal middle. Revises E10.
3. Distillation can bring a 70%-data student to **parity with a full-data model** on a good-teacher
   dataset (ccpp), and can **beat** a full-data model where extra data hurts (airquality) — but
   cannot manufacture signal where the teacher is bad (energy).
4. The thread that keeps recurring — teacher accuracy doesn't predict benefit, more data isn't
   always better (airquality) — points back to **data spacing/density** as the variable actually
   governing all of this. That is the natural next investigation.
