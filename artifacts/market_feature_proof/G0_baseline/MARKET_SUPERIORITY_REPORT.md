# Market-superiority proof report

- Props passing all three gates: **0/4**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **2000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| ast | G0_current | 291 | 0.03066 [NA, NA] | 0.01142 [NA, NA] | -0.02816 [NA, NA] | **INSUFFICIENT** |
| fg3m | G0_current | 296 | 0.04765 [NA, NA] | 0.01760 [NA, NA] | -0.05415 [NA, NA] | **INSUFFICIENT** |
| pts | G0_current | 545 | 0.05739 [0.01350, 0.10754] | 0.02154 [0.00410, 0.04185] | 0.00650 [-0.06077, 0.06822] | **FAIL** |
| reb | G0_current | 434 | 0.06259 [0.02361, 0.10446] | 0.02339 [0.00786, 0.03971] | -0.05495 [-0.11229, 0.00011] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
