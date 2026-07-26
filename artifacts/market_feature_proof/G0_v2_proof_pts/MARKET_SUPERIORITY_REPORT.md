# Market-superiority proof report

- Props passing all three gates: **0/1**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **3000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| pts | C6_market_residual | 485 | 0.00690 [-0.00359, 0.01732] | 0.00340 [-0.00178, 0.00855] | -0.01674 [-0.05841, 0.02409] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
