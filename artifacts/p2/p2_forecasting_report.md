# P2 Forecasting Diagnostics (untouched holdout)

Holdout: 2026-07-02 → 2026-07-16 (5110 rows, 37 games)
Development: 2026-05-08 → 2026-06-30 (20090 rows)

| stat | n | bias | MAE | RMSE | CRPS | PIT-ECE | 80%cov | 90%cov | calib-ECE | PASS |
|------|---|------|-----|------|------|---------|--------|--------|-----------|------|
| ast | 730 | -0.39 | 1.22 | 1.74 | 0.972 | 0.230 | 0.8904(X) | 0.9411(X) | 0.016 | NO |
| blk | 730 | -0.20 | 0.34 | 0.63 | 0.266 | 0.681 | 0.8575(X) | 0.9041(ok) | 0.020 | NO |
| fg3m | 730 | +0.01 | 0.77 | 1.07 | 0.507 | 0.266 | 0.9233(X) | 0.9493(X) | 0.006 | NO |
| pts | 730 | -0.85 | 4.40 | 5.90 | 3.319 | 0.123 | 0.8616(X) | 0.9411(X) | 0.014 | NO |
| reb | 730 | -0.59 | 1.78 | 2.40 | 1.354 | 0.179 | 0.9014(X) | 0.9521(X) | 0.020 | NO |
| stl | 730 | -0.45 | 0.70 | 1.02 | 0.544 | 0.463 | 0.8247(ok) | 0.8959(ok) | 0.045 | NO |
| turnover | 730 | -0.24 | 0.98 | 1.30 | 0.710 | 0.164 | 0.8904(X) | 0.9329(X) | 0.020 | NO |

**Passed (launchable forecast):** none
**Suppressed:** ast, blk, fg3m, pts, reb, stl, turnover
- `ast`: bias -0.39 not within 3.0·SE (0.063); 80% interval coverage incompatible with nominal; 90% interval coverage incompatible with nominal; PIT non-uniformity 0.230 > 0.06
- `blk`: bias -0.20 not within 3.0·SE (0.022); 80% interval coverage incompatible with nominal; PIT non-uniformity 0.681 > 0.06
- `fg3m`: 80% interval coverage incompatible with nominal; 90% interval coverage incompatible with nominal; PIT non-uniformity 0.266 > 0.06
- `pts`: bias -0.85 not within 3.0·SE (0.216); 80% interval coverage incompatible with nominal; 90% interval coverage incompatible with nominal; PIT non-uniformity 0.123 > 0.06
- `reb`: bias -0.59 not within 3.0·SE (0.086); 80% interval coverage incompatible with nominal; 90% interval coverage incompatible with nominal; PIT non-uniformity 0.179 > 0.06
- `stl`: bias -0.45 not within 3.0·SE (0.034); PIT non-uniformity 0.463 > 0.06
- `turnover`: bias -0.24 not within 3.0·SE (0.047); 80% interval coverage incompatible with nominal; 90% interval coverage incompatible with nominal; PIT non-uniformity 0.164 > 0.06