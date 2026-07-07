# E12 Tier B — error & distillation benefit vs local spacing

Spearman over held-out test points (pooled across seeds). spacing = distance to nearest surviving (30%-hole) train point, z-scored feature space.

B1: does error rise with spacing? (rho_err > 0 = sparser regions learned worse)

| dataset | n | err unguided | err M20 | err force_ii | err tsign_tmag |
|---|---|---|---|---|---|
| ccpp | 11484 | +0.02 | +0.02 | -0.01 | +0.00 |
| airquality | 8334 | +0.14 | +0.10 | +0.16 | +0.12 |
| california | 24768 | +0.06 | +0.06 | +0.05 | +0.05 |
| energy | 924 | +0.20 | +0.11 | +0.15 | +0.17 |

B2: does benefit concentrate where it's sparse? (rho_benefit > 0 = helps more in sparse regions)

| dataset | benefit force_ii | benefit tsign_tmag |
|---|---|---|
| ccpp | +0.04 | +0.02 |
| airquality | -0.02 | +0.01 |
| california | +0.03 | +0.03 |
| energy | +0.02 | +0.01 |

B3: benefit IN-hole vs OUT-of-hole (a test point is in-hole if its nearest REMOVED train point is closer than its nearest surviving one). If distillation fills holes, in-hole benefit > out-of-hole.

| dataset | frac in-hole | force_ii in/out | tsign_tmag in/out |
|---|---|---|---|
| ccpp | 0.305 | +0.397 / +0.194 | +0.458 / +0.312 |
| airquality | 0.316 | +0.164 / +0.222 | +0.126 / +0.109 |
| california | 0.313 | -0.025 / -0.016 | -0.031 / -0.021 |
| energy | 0.275 | -0.192 / -0.179 | -0.095 / -0.124 |
