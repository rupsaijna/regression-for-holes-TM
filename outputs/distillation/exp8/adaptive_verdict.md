# Experiment 8 — adaptive teacher trigger: verdict

Config: teacher 20% → student 30% mask, structured, f=0.75, p=0.5, 100 epochs, 6 seeds,
held-out TEST, paired by seed. Adaptive = per-epoch budget-matched trigger
`p_e = clip(p·d_e/mean(d_e), 0, 1)`; `d_e` = teacher↔student disagreement (`adapt_disagree`)
or student error (`adapt_error`). `impr` = rmse(unguided) − rmse(guided) (>0 helps);
`gain` = adaptive − uniform (paired; >0 = adaptive beats the uniform trigger).

| dataset | teacher? | method | mean impr | t | gain vs uniform | gain t |
|---|---|---|---|---|---|---|
| **ccpp** | helps | uniform | +0.360 | +2.57 | — | — |
| | | adapt_disagree | +0.322 | +2.11 | −0.038 | −0.29 |
| | | **adapt_error** | **+0.496** | **+4.64** | **+0.137** | +1.72 |
| **airquality** | helps (weak) | uniform | +0.078 | +1.02 | — | — |
| | | adapt_disagree | +0.096 | +1.08 | +0.019 | +0.36 |
| | | adapt_error | +0.147 | +1.51 | +0.069 | +1.01 |
| **california** | neutral/hurts | uniform | −0.013 | −3.01 | — | — |
| | | adapt_disagree | −0.029 | −5.95 | −0.015 | −2.37 |
| | | adapt_error | −0.062 | −7.08 | −0.049 | −4.61 |
| **energy** | hurts | uniform | −0.062 | −1.12 | — | — |
| | | adapt_disagree | −0.277 | −3.11 | −0.216 | −2.26 |
| | | adapt_error | −0.372 | −5.34 | −0.310 | −6.25 |

## Verdict
**Adaptive triggering AMPLIFIES the teacher's existing effect in both directions — it is a
gain-multiplier (sharpener), not a fixer.**

- **Where the teacher helps it helps MORE:** ccpp `adapt_error` +0.496 (t=4.64) vs uniform
  +0.360 — the single best distillation result in the whole arc (gain +0.137, t=1.72).
  airquality trends the same way.
- **Where the teacher hurts it hurts MORE, significantly:** california and energy get worse
  under both adaptive modes (gain t = −2.3 to −6.3). The hoped-for "concentrate on
  disagreement → avoid hurting easy points → stop hurting" did NOT happen — on a bad-teacher
  dataset the disagreement/error is exactly where the teacher is most wrong, so focusing there
  amplifies the harm.
- **`adapt_error` is the stronger amplifier than `adapt_disagree`** in both directions
  (biggest help on ccpp, biggest harm on energy).

## Implication
Adaptive (esp. error-gated) is worth using **only when you already know the teacher is good
for that dataset** — then it meaningfully boosts the benefit. It is not a safe default: it
sharpens bad teachers into worse ones. This is consistent with E7 (the teacher is an
error-targeted corrector): concentrating its action by error magnitude scales whatever signal
the teacher carries — helpful or harmful.
