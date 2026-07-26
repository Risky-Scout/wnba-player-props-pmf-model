# Market-superiority proof report

- Props passing all three gates: **0/1**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **4000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| fg3m | C2_beta | 1029 | 0.03636 [0.00485, 0.06968] | 0.01682 [0.00188, 0.03209] | -0.09857 [-0.16154, -0.03037] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
