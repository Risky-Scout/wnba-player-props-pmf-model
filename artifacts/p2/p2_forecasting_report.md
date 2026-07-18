# P2 Forecasting Diagnostics (untouched holdout)

Holdout: 2026-07-02 → 2026-07-16 (5110 rows, 37 games)
Development: 2026-05-08 → 2026-06-30 (20090 rows)

Primary launch gate uses the fold-safe calibrated model.

## Raw OOF PMF

| stat | n | bias | MAE | RMSE | CRPS | PIT-ECE | 80%cov | 90%cov | calib-ECE | PASS |
|------|---|------|-----|------|------|---------|--------|--------|-----------|------|
| ast | 730 | -0.39 | 1.22 | 1.74 | 0.972 | 0.230 | 0.8904(X) | 0.9411(X) | 0.016 | NO |
| blk | 730 | -0.20 | 0.34 | 0.63 | 0.266 | 0.681 | 0.8575(X) | 0.9041(ok) | 0.020 | NO |
| fg3m | 730 | +0.01 | 0.77 | 1.07 | 0.507 | 0.266 | 0.9233(X) | 0.9493(X) | 0.006 | NO |
| pts | 730 | -0.85 | 4.40 | 5.90 | 3.319 | 0.123 | 0.8616(X) | 0.9411(X) | 0.014 | YES |
| reb | 730 | -0.59 | 1.78 | 2.40 | 1.354 | 0.179 | 0.9014(X) | 0.9521(X) | 0.020 | NO |
| stl | 730 | -0.45 | 0.70 | 1.02 | 0.544 | 0.463 | 0.8247(ok) | 0.8959(ok) | 0.045 | NO |
| turnover | 730 | -0.24 | 0.98 | 1.30 | 0.710 | 0.164 | 0.8904(X) | 0.9329(X) | 0.020 | NO |

## Fold-safe calibrated PMF (PRIMARY)

| stat | n | bias | MAE | RMSE | CRPS | PIT-ECE | 80%cov | 90%cov | calib-ECE | PASS |
|------|---|------|-----|------|------|---------|--------|--------|-----------|------|
| ast | 730 | +0.12 | 1.30 | 1.70 | 0.901 | 0.177 | 0.9247(X) | 0.9603(X) | 0.009 | NO |
| blk | 730 | +0.07 | 0.47 | 0.60 | 0.246 | 0.566 | 0.963(X) | 0.974(X) | 0.011 | NO |
| fg3m | 730 | +0.11 | 0.81 | 1.08 | 0.502 | 0.251 | 0.9438(X) | 0.9699(X) | 0.009 | NO |
| pts | 730 | +0.89 | 4.60 | 5.91 | 3.284 | 0.095 | 0.7425(X) | 0.7904(X) | 0.018 | NO |
| reb | 730 | +0.22 | 1.84 | 2.34 | 1.293 | 0.090 | 0.8247(ok) | 0.9068(ok) | 0.014 | YES |
| stl | 730 | -0.04 | 0.71 | 0.90 | 0.442 | 0.248 | 0.926(X) | 0.963(X) | 0.006 | NO |
| turnover | 730 | +0.05 | 1.01 | 1.28 | 0.671 | 0.123 | 0.9342(X) | 0.963(X) | 0.008 | YES |

**Passed (launchable forecast):** reb, turnover
**Suppressed:** ast, blk, fg3m, pts, stl
- `ast`: PIT non-uniformity 0.177 > 0.15 (wide-support stat)
- `blk`: PIT non-uniformity 0.566 > 0.15 (wide-support stat)
- `fg3m`: PIT non-uniformity 0.251 > 0.15 (wide-support stat)
- `pts`: 90% interval under-covers (0.790 < 0.85) — overconfident
- `stl`: PIT non-uniformity 0.248 > 0.15 (wide-support stat)