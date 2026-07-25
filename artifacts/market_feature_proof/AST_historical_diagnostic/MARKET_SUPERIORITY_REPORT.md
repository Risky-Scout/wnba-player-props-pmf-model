# Market-superiority proof report

- Props passing all three gates: **0/1**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **5000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| ast | C1_platt | 325 | -0.00374 [-0.01530, 0.00808] | -0.00187 [-0.00758, 0.00396] | -0.02955 [-0.09476, 0.03921] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
