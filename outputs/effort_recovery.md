# Phase 7 follow-up — a fair effort-to-recover metric

`experiment4_masking.py` measured "epochs for a masked model to reach the full-data
target", where the target was each model's RMSE at its **default** epoch budget
(TM=30, NN=100). The caveat flagged in `masking_summary.md`: that bar is **loose for
TM** — TM keeps improving well past 30 epochs, so a masked TM clears the 30-epoch RMSE
in a couple of epochs and the metric looks flat/uninformative.

`effort_recovery.py` recomputes effort against a **fairer bar: each model's own *best*
RMSE within the extended budget on full data** (min over its extended-budget curve).
Run: ccpp + california, full fractions, R=2 (heavy-tailed nyse excluded — effort is
uninformative there regardless: the tail, not geometry, sets the error). Data in
`outputs/masking/effort/<ds>__effort.csv`, plots `__effort_vs_p.png`.

## What the fair target fixes — and what it breaks

**It repairs the TM metric (the goal).** On **ccpp** the fair TM target is **4.54**
(vs the loose default **5.42**). Against the loose bar, masked TM was *never* censored and
"reached target" in 2–8 epochs at every `p` — no signal. Against the fair bar the expected
asymmetry appears: **uniform**-masked TM reaches it in ~60–75 epochs up to p=0.5 (censored
only at p=0.6), while **structured**-masked TM is censored from **p=0.2** (2/2 repeats) and
elsewhere needs ~95 epochs. So with an honest target, TM reproduces the same
*uniform-recoverable / structured-unrecoverable-sooner* story the NN default-target metric
showed — unevenness makes the original performance unrecoverable at a far lower removal
fraction than random loss does.

**It over-tightens the NN metric (the cost).** NN already trains close to its own best within
the default budget, so its "extended-budget best" bar (ccpp 3.66, california 0.504) sits
essentially at the full-data frontier. Remove *any* data — even 10 % uniformly — and a fresh
NN can no longer reach that frontier, so **every masked NN config is censored** (both datasets,
both modes, all `p`). The NN effort panel is intentionally empty: the bar is unreachable once
any data is removed, which erases the gradient that made the default-target NN metric useful.

## Takeaway / recommendation
There is **no single target that is fair to both models**: the default budget is too loose for
TM, the extended-budget best is too strict for NN. The informative pairing is therefore
**NN against the default-budget target** (experiment4) **and TM against the extended-budget
target** (this script) — and both, read on their fair bar, agree that **structured (uneven)
removal makes recovery fail at a lower `p` than uniform removal**. A cleaner single metric for a
follow-up would target *full-data best + a small tolerance* (e.g. +2–5 %, or "within 1 epoch-std
of best") so the bar is reachable for a slightly-reduced dataset and the epochs-to-reach gradient
survives for both models.

(california is consistent but less crisp: its TM extended best 0.606 is only just below the
default 0.619, so masked TM uniform also censors out — ccpp is the clean demonstration.)
