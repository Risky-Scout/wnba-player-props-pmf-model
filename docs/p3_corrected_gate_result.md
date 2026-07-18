# P2 Forecasting Diagnostics (untouched holdout)

Holdout: 2026-07-02 → 2026-07-16 (5110 rows, 37 games)
Development: 2026-05-08 → 2026-06-30 (20090 rows)

Primary launch gate uses the fold-safe calibrated model.

## Raw OOF PMF

| stat | n | dates | bias | RMSE | CRPS | logS | PIT-KS-p | 50%cov | 80%cov | 90%cov | PASS |
|------|---|-------|------|------|------|------|----------|--------|--------|--------|------|
| ast | 730 | 15 | -0.39 | 1.74 | 0.972 | 1.980 | 0.0000 | 0.726✗ | 0.8904✗ | 0.9411✓ | NO |
| blk | 730 | 15 | -0.20 | 0.63 | 0.266 | 0.880 | 0.0000 | 0.7863✗ | 0.8575✓ | 0.9041✓ | NO |
| fg3m | 730 | 15 | +0.01 | 1.07 | 0.507 | 1.250 | 0.0085 | 0.826✗ | 0.9233✗ | 0.9493✓ | NO |
| pts | 730 | 15 | -0.85 | 5.90 | 3.319 | 3.068 | 0.0000 | 0.6425✗ | 0.8616✓ | 0.9411✓ | NO |
| reb | 730 | 15 | -0.59 | 2.40 | 1.354 | 2.279 | 0.0000 | 0.7055✗ | 0.9014✗ | 0.9521✓ | NO |
| stl | 730 | 15 | -0.45 | 1.02 | 0.544 | 1.383 | 0.0000 | 0.6438✗ | 0.8247✓ | 0.8959✓ | NO |
| turnover | 730 | 15 | -0.24 | 1.30 | 0.710 | 1.650 | 0.0000 | 0.7137✗ | 0.8904✗ | 0.9329✓ | NO |

## Fold-safe calibrated PMF (PRIMARY)

| stat | n | dates | bias | RMSE | CRPS | logS | PIT-KS-p | 50%cov | 80%cov | 90%cov | PASS |
|------|---|-------|------|------|------|------|----------|--------|--------|--------|------|
| ast | 730 | 15 | +0.12 | 1.70 | 0.901 | 1.836 | 0.1358 | 0.7836✗ | 0.9247✗ | 0.9603✓ | NO |
| blk | 730 | 15 | +0.07 | 0.60 | 0.246 | 0.729 | 0.0004 | 0.9178✗ | 0.963✗ | 0.974✗ | NO |
| fg3m | 730 | 15 | +0.11 | 1.08 | 0.502 | 1.211 | 0.1773 | 0.8493✗ | 0.9438✗ | 0.9699✓ | NO |
| pts | 730 | 15 | +0.89 | 5.91 | 3.284 | 7.181 | 0.0000 | 0.5534✓ | 0.7425✗ | 0.7904✗ | NO |
| reb | 730 | 15 | +0.22 | 2.34 | 1.293 | 2.322 | 0.0001 | 0.6425✗ | 0.8247✓ | 0.9068✓ | NO |
| stl | 730 | 15 | -0.04 | 0.90 | 0.442 | 1.134 | 0.5904 | 0.8233✗ | 0.926✗ | 0.963✓ | NO |
| turnover | 730 | 15 | +0.05 | 1.28 | 0.671 | 1.534 | 0.9932 | 0.7808✗ | 0.9342✗ | 0.963✓ | NO |

**Passed (launchable forecast):** none
**Suppressed:** ast, blk, fg3m, pts, reb, stl, turnover
- `ast`: insufficient coverage: 15 game-dates (<25); 50% interval over-covers (emp 0.784 vs 0.5; clustered CI excludes nominal); 80% interval over-covers (emp 0.925 vs 0.8; clustered CI excludes nominal)
- `blk`: insufficient coverage: 15 game-dates (<25); randomized-PIT non-uniform (KS p=0.0004 < 0.01); 50% interval over-covers (emp 0.918 vs 0.5; clustered CI excludes nominal); 80% interval over-covers (emp 0.963 vs 0.8; clustered CI excludes nominal); 90% interval over-covers (emp 0.974 vs 0.9; clustered CI excludes nominal)
- `fg3m`: insufficient coverage: 15 game-dates (<25); 50% interval over-covers (emp 0.849 vs 0.5; clustered CI excludes nominal); 80% interval over-covers (emp 0.944 vs 0.8; clustered CI excludes nominal)
- `pts`: insufficient coverage: 15 game-dates (<25); randomized-PIT non-uniform (KS p=0.0000 < 0.01); 80% interval under-covers (emp 0.743 vs 0.8; clustered CI excludes nominal); 90% interval under-covers (emp 0.790 vs 0.9; clustered CI excludes nominal)
- `reb`: insufficient coverage: 15 game-dates (<25); randomized-PIT non-uniform (KS p=0.0001 < 0.01); 50% interval over-covers (emp 0.642 vs 0.5; clustered CI excludes nominal)
- `stl`: insufficient coverage: 15 game-dates (<25); 50% interval over-covers (emp 0.823 vs 0.5; clustered CI excludes nominal); 80% interval over-covers (emp 0.926 vs 0.8; clustered CI excludes nominal)
- `turnover`: insufficient coverage: 15 game-dates (<25); 50% interval over-covers (emp 0.781 vs 0.5; clustered CI excludes nominal); 80% interval over-covers (emp 0.934 vs 0.8; clustered CI excludes nominal)