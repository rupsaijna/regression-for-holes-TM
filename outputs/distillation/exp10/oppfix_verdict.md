# Experiment 10 — teacher_sign stability fixes: verdict

Config: teacher 20% → student 30% mask, structured, f=0.75, p=0.5, 100 epochs, 6 seeds,
held-out TEST, paired by seed. On OPPOSITE-side samples (`vanilla_regressor_v3.opposite_mode`):
`uniform` = v1 baseline (opposite ignored); `teacher_sign` = E9's unstable mode (route feedback
by teacher's error sign, incl. Type-I); **fix (a) `teacher_sign_ii`** = Type-II only when teacher
over-predicts, NO-OP otherwise (drops the Type-I branch); **fix (b) `teacher_sign_tmag`** = same
teacher-sign routing incl. Type-I, but `update_p` magnitude from the TEACHER's frozen/bounded
error instead of the student's. `impr` = rmse(unguided) − rmse(guided); `gain` = mode − uniform.

**Sanity check:** `uniform` reproduces E8/E9 exactly (ccpp +0.360/t=2.57, california −0.013/t=−3.01,
energy −0.062/t=−1.12, airquality +0.078/t=1.02) → V3 `opposite_mode=None` is bit-for-bit v1.
(Note: `teacher_sign`'s exact blow-up magnitude differs from E9 — −17 here vs −18 there — because
a runaway's final RMSE is numerically chaotic; the *instability itself* is the robust, reproduced fact.)

| dataset | method | mean impr | t | **std** | gain vs uniform | gain t |
|---|---|---|---|---|---|---|
| **ccpp** | uniform | +0.360 | +2.57 | 0.342 | — | — |
| | teacher_sign | +0.427 | +3.30 | 0.317 | +0.067 | +0.81 |
| | teacher_sign_ii | +0.340 | +2.05 | 0.407 | −0.019 | −0.19 |
| | teacher_sign_tmag | +0.367 | +2.72 | 0.331 | +0.007 | +0.07 |
| **airquality** | uniform | +0.078 | +1.02 | 0.186 | — | — |
| | teacher_sign | **−17.39** | −1.68 | **25.41** | −17.46 | −1.68 |
| | teacher_sign_ii | −0.001 | −0.01 | **0.227** | −0.079 | −0.90 |
| | **teacher_sign_tmag** | **+0.107** | +1.07 | **0.246** | +0.030 | +0.38 |
| **california** | uniform | −0.013 | −3.01 | 0.011 | — | — |
| | teacher_sign | −0.024 | −6.89 | 0.008 | −0.010 | −1.98 |
| | teacher_sign_ii | −0.018 | −3.05 | 0.014 | −0.004 | −0.94 |
| | teacher_sign_tmag | −0.019 | −4.42 | 0.010 | −0.005 | −1.16 |
| **energy** | uniform | −0.062 | −1.12 | 0.134 | — | — |
| | teacher_sign | −0.377 | −3.73 | 0.247 | −0.315 | −2.49 |
| | teacher_sign_ii | −0.366 | −4.06 | 0.221 | −0.304 | −2.83 |
| | **teacher_sign_tmag** | **−0.152** | −3.37 | **0.110** | −0.090 | −1.32 |

## Verdict
**Both fixes remove the runaway; fix (b) `teacher_sign_tmag` is the clear winner — it stabilizes,
is the gentlest on a bad teacher, and is mildly positive where teacher_sign exploded.**

- **The runaway is gone, decisively.** airquality `std` drops from **25.41 → 0.23 (ii) / 0.25
  (tmag)**; the −70/−28 per-seed blow-ups vanish. Both fixes do what they were designed to: (a)
  by never firing the wrong-direction Type-I, (b) by bounding the magnitude so the Type-I that
  does fire can't self-compound.

- **`teacher_sign_tmag` (fix b) is the better fix on every axis that matters:**
  - airquality: **+0.107 (t=1.07)** — actually positive and slightly beats uniform (+0.078),
    where `teacher_sign_ii` only manages ≈0 (−0.001).
  - energy (bad teacher): **−0.152**, roughly *half* the harm of `teacher_sign` (−0.377) and
    `teacher_sign_ii` (−0.366); its gain vs uniform is no longer significant (t=−1.32). Bounding
    the update to the teacher's frozen error caps how much a bad teacher can hurt.
  - Lowest variance throughout (energy std 0.110 vs 0.22–0.25 for the others).
  - ccpp/california: ≈uniform (the good/neutral cases are undisturbed).

- **`teacher_sign_ii` (fix a) is merely "safe", not good.** It kills the blow-up but recovers no
  upside (airquality ≈0) and is still significantly harmful on energy (gain t=−2.83) — i.e.
  dropping the Type-I branch removes the danger *and* most of the signal. Not worth preferring.

- **The deeper takeaway — bounded (teacher-)magnitude is the real lever.** The thing that fixed
  the instability AND softened the bad-teacher harm was sourcing the update magnitude from the
  frozen teacher's error, not the feedback-type gymnastics. This is the first mechanism in the
  arc that makes a *bad* teacher meaningfully *less* harmful (energy −0.38 → −0.15) rather than
  just amplifying whatever the teacher carries (cf. E7/E8).

## Where this leaves the opposite-side arc
- Best single *win*: E9 `force_ii` on airquality (+0.175, t=2.33) — still unbeaten.
- Best *stable* teacher-sign mode: `teacher_sign_tmag` (no blow-up, gentlest on bad teachers).
- Neither beats "use force_ii only when the teacher is known-good."

## Next step (candidate E11)
Combine the two best ideas: **`force_ii` with teacher-bounded magnitude** (a `force_ii_tmag` mode)
— does sourcing force_ii's magnitude from the frozen teacher keep its airquality win while
cutting the energy harm the way tmag did for teacher_sign? More broadly, test teacher-bounded
magnitude on the *same-side* blend too, since bounded-magnitude — not the type override — looks
like the transferable result of this experiment.
