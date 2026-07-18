# P3 forecast gate — full 2026, committed split

Holdout 2026-06-20..2026-07-16 (25 dates); total dates 63

| stat | winner | n | dates | CRPS | dCRPS(base) | PIT-KS-p | forecast_allowed | reason |
|------|--------|---|-------|------|-------------|----------|------------------|--------|
| ast | challenger_A_locscale | 1315 | 25 | 0.898 | -0.256 | 0.0002 | False | randomized-PIT non-uniform (KS p=0.0002 < 0.01); 50% interval over-cov |
| blk | challenger_A_loc | 1315 | 25 | 0.268 | -0.026 | 0.0001 | False | randomized-PIT non-uniform (KS p=0.0001 < 0.01); 50% interval over-cov |
| fg3m | challenger_A_locscale | 1315 | 25 | 0.515 | -0.068 | 0.5226 | False | 50% interval over-covers (emp 0.831 vs 0.5; clustered CI excludes nomi |
| pts | challenger_A_locscale | 1315 | 25 | 3.261 | -0.936 | 0.0000 | False | randomized-PIT non-uniform (KS p=0.0000 < 0.01); 80% interval under-co |
| reb | challenger_A_locscale | 1315 | 25 | 1.301 | -0.362 | 0.0000 | False | randomized-PIT non-uniform (KS p=0.0000 < 0.01); 90% interval under-co |
| stl | challenger_A_locscale | 1315 | 25 | 0.436 | -0.057 | 0.1728 | False | 50% interval over-covers (emp 0.867 vs 0.5; clustered CI excludes nomi |
| turnover | challenger_A_locscale | 1315 | 25 | 0.648 | -0.076 | 0.2375 | False | 50% interval over-covers (emp 0.786 vs 0.5; clustered CI excludes nomi |

**Certified stats:** none
**Status:** BLOCKED_MODEL