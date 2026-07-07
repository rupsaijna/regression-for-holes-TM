# Experiment 9 — opposite-side feedback override: verdict

Config: teacher 20% → student 30% mask, structured, f=0.75, p=0.5, 100 epochs, 6 seeds,
held-out TEST, paired by seed. On OPPOSITE-side teacher samples (teacher & student errors
on opposite sides of encoded_Y) the feedback TYPE is overridden while the student's error
MAGNITUDE (`update_p`) is kept (`vanilla_regressor_v2.py`, `opposite_mode`):
`uniform` = v1 baseline (opposite blended by f_opposite=0 → teacher ignored there);
`force_ii` = always Type-II; `teacher_sign` = route by the teacher's error sign (Type-II if
teacher over-predicts, Type-I if under). `impr` = rmse(unguided) − rmse(guided) (>0 helps);
`gain` = mode − uniform (paired; >0 = beats the uniform baseline).

**Sanity check:** the `uniform` arm reproduces E8's uniform numbers exactly on all four
datasets (ccpp +0.360/t=2.57, california −0.013/t=−3.01, energy −0.062/t=−1.12,
airquality +0.078/t=1.02) → `opposite_mode=None` is bit-for-bit v1, so the gains below are real.

| dataset | teacher? | method | mean impr | t | gain vs uniform | gain t |
|---|---|---|---|---|---|---|
| **ccpp** | helps | uniform | +0.360 | +2.57 | — | — |
| | | force_ii | +0.251 | +1.39 | −0.109 | −0.83 |
| | | **teacher_sign** | **+0.529** | **+3.09** | +0.170 | +1.32 |
| **airquality** | helps (weak) | uniform | +0.078 | +1.02 | — | — |
| | | **force_ii** | **+0.175** | **+2.33** | **+0.098** | **+2.27** |
| | | teacher_sign | **−18.29** | −1.57 | −18.36 | −1.58 |
| **california** | neutral/hurts | uniform | −0.013 | −3.01 | — | — |
| | | force_ii | −0.015 | −7.58 | −0.001 | −0.28 |
| | | teacher_sign | −0.009 | −2.69 | +0.004 | +0.80 |
| **energy** | hurts | uniform | −0.062 | −1.12 | — | — |
| | | force_ii | −0.237 | −4.53 | −0.175 | −2.86 |
| | | teacher_sign | −0.278 | −2.95 | −0.217 | −2.21 |

## Verdict
**`force_ii` is the safe, mildly-useful mode; `teacher_sign` is higher-ceiling but unstable to
the point of being unusable as-is. Neither rescues a bad teacher.**

- **`force_ii` (always Type-II on opposite-side) — one clean win, no blow-ups.** It turns
  airquality — the *weak*-teacher dataset where the uniform blend barely helped (t=1.02) — into a
  significant win: **+0.175 (t=2.33), gain +0.098 (t=2.27)**. It recovers value the uniform path
  left on the table by simply *acting* on opposite-side samples instead of ignoring them. Cost:
  it amplifies harm where the teacher is bad (energy gain −0.175, t=−2.86) and is a wash on
  ccpp/california. Critically it never explodes — Type-II only *erodes* clauses (predictions can
  only fall, bounded at 0), so the update is self-limiting.

- **`teacher_sign` (route by teacher sign) — best single result AND a catastrophe.** Best on
  ccpp (+0.529, t=3.09, > uniform's +0.360) but it **blows up on airquality: −18.3 mean, with
  individual seeds at −5, −21, −80.** Mechanism: on an opposite-side sample where the student
  *over*-predicts (error>0) but the teacher *under*-predicts (pe_teacher<0), `teacher_sign` fires
  **Type-I** — which pushes that already-too-high prediction *higher*. That is a positive-feedback
  loop with no upper bound, so it occasionally runs away. `force_ii` is immune because it never
  picks the runaway (raising) direction.

- **Both still hurt on energy (bad teacher), significantly.** Consistent with E8: the mechanism
  isn't the limiting factor — teacher *quality* on the dataset is. Changing how the teacher acts
  on opposite-side points can't manufacture signal a bad teacher doesn't have, and concentrating
  action there (as force_ii/teacher_sign do) amplifies the harm.

## Implication / next step
- **`force_ii` is a reasonable, stable default for the opposite-side branch** when you don't know
  the teacher is bad — it's the first lever in this arc that *adds* a significant win
  (airquality) without a corresponding blow-up. Pair it with a good-teacher gate to avoid the
  energy-style harm.
- **`teacher_sign` needs a stability fix before it's usable.** The runaway is the wrong-direction
  Type-I. Two candidate fixes worth an E10: (a) restrict teacher_sign to fire Type-II only and
  drop the Type-I branch (≈ force_ii but only when teacher over-predicts); or (b) drive the
  magnitude from the *teacher's* error (clipped) instead of the student's — note pure
  teacher-magnitude + teacher-sign collapses into the existing `f_opposite=1.0` hard-replace, so
  a clipped/partial blend is the interesting middle ground.
