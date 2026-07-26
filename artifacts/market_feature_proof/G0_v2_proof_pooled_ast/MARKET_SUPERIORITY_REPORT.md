# Market-superiority proof report

- Props passing all three gates: **0/1**
- Settled-row minimum: **300**
- Cluster bootstrap replicates: **4000**
- Holm-adjusted one-sided alpha: **0.050**
- Delta signs: log loss/Brier negative is better; AUC positive is better.
- Pushes are excluded from binary metrics.

| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---:|
| ast | C4_blend | 914 | -0.00541 [-0.01746, 0.00729] | -0.00263 [-0.00856, 0.00362] | -0.00792 [-0.04461, 0.03007] | **FAIL** |

A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.
It is not a guarantee of future profitability.
