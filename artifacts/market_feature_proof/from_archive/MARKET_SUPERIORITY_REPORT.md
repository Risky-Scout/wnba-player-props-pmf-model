# Market-superiority proof report

- Props passing all three gates: **0/4**
- Settled-row minimum: **100**
- Cluster bootstrap replicates: **2000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| ast | production | 176 | 0.00567 [-0.02036, 0.03469] | 0.00315 [-0.00937, 0.01684] | -0.03096 [-0.10259, 0.03777] | **FAIL** |
| fg3m | production | 164 | 0.12959 [0.00694, 0.27743] | 0.02406 [0.00265, 0.04016] | -0.09829 [-0.20851, 0.02772] | **FAIL** |
| pts | production | 255 | 0.06715 [0.00720, 0.18176] | 0.01036 [0.00332, 0.01897] | -0.05475 [-0.12416, 0.04613] | **FAIL** |
| reb | production | 228 | 0.02118 [-0.00107, 0.04530] | 0.01030 [-0.00044, 0.02196] | -0.09099 [-0.18519, 0.00594] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
