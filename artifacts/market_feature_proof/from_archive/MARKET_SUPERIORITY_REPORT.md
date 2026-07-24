# Market-superiority proof report

- Props passing all three gates: **0/4**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **5000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| ast | production | 176 | 0.02900 [NA, NA] | 0.01084 [NA, NA] | -0.02748 [NA, NA] | **INSUFFICIENT** |
| fg3m | production | 164 | 0.08922 [NA, NA] | 0.02697 [NA, NA] | -0.10056 [NA, NA] | **INSUFFICIENT** |
| pts | production | 255 | 0.10133 [NA, NA] | 0.02555 [NA, NA] | 0.00969 [NA, NA] | **INSUFFICIENT** |
| reb | production | 228 | 0.10298 [NA, NA] | 0.02765 [NA, NA] | -0.05472 [NA, NA] | **INSUFFICIENT** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
