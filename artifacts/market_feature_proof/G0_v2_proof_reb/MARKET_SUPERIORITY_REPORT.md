# Market-superiority proof report

- Props passing all three gates: **0/1**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **3000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| reb | C4_blend | 419 | -0.00426 [-0.01206, 0.00204] | -0.00204 [-0.00589, 0.00107] | -0.00023 [-0.02093, 0.02228] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
