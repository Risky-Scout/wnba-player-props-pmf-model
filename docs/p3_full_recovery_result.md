# P3 full-distribution recovery — honest result (gates NOT weakened)

Strictly-prequential five-block evaluation on the existing 2026 OOF (holdout = latest 25
game-dates), candidates per block chosen ONLY on pre-block dates: raw, location,
location-and-scale, monotone PIT-CDF recalibration (+shrinkage), CDF+empirical mixture,
hurdle (sparse), and hierarchical empirical residual fallback. Combos built from the
calibrated components via the existing correlated combo code. Gate unchanged.

## Per-market result (OOS prequential ledger)
| market | chosen method | CRPS | ΔCRPS vs baseline | randomized-PIT p | forecast_allowed | blocking reason |
|--------|---------------|------|-------------------|------------------|------------------|-----------------|
| turnover | location (PR#60) | 0.648 | −0.076 | 0.24 | **TRUE** | — |
| stl | hierarchical | 0.443 | −0.050 | **0.31** | false | 90% interval residual CI excludes 0 by +0.010 (within 0.05 practical tol) |
| blk | cdf_mixture | 0.264 | −0.030 | 0.014 | false | 80% sharpness 1.44>1.15; interval-residual CIs |
| stocks | correlated combo | 0.586 | −0.082 | 0.018 | false | 50/80/90 interval-residual CIs exclude 0 (all within ~0.03) |
| pts | hierarchical | 3.18 | −0.94 | 0.000 | false | randomized-PIT (structural shape miscalibration) |
| reb | hierarchical | 1.27 | −0.36 | 0.000 | false | randomized-PIT |
| ast | hierarchical | 0.89 | −0.26 | 0.000 | false | randomized-PIT |
| fg3m | cdf_mixture | 0.52 | −0.07 | 0.001 | false | randomized-PIT |
| pts_reb, pts_ast, pts_reb_ast | correlated combo | — | — | — | false | inherit pts/reb/ast randomized-PIT failure |

## Conclusion
Every candidate beats the climatology baseline on CRPS, but only **turnover** clears ALL
committed gates. The wide-support stats (pts/reb/ast/fg3m) fail randomized-PIT — a
structural PMF shape defect that location/scale/CDF/mixture/hurdle/empirical calibration
cannot remove; fixing it requires Challenger B (minutes-distribution) / C (count-family)
RETRAINING, which is outside this bounded (calibration-only) mission. The sparse stats
(stl/blk/stocks) are borderline, failing only interval-residual significance or sharpness.

The gates were NOT weakened. The required all-ten forecast release is therefore NOT
achievable with bounded calibration on the current base model. Live state is unchanged:
turnover forecast-only, Edge abstaining.
