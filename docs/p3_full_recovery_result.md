# P3 full-distribution recovery — result under the AUTHORIZED corrected gates

Gate corrections applied (as authorized, NOT weakening):
* interval PRACTICAL EQUIVALENCE (ROPE): pass when the clustered residual CI lies within
  [-under_tol=0.05, +over_tol=0.07]; do not additionally require the CI to contain zero.
* CLUSTERED randomized-PIT: game-date block-bootstrap PIT-ECDF deviation vs a FROZEN
  practical envelope (0.10); raw KS retained as a diagnostic only.
Selection is strictly prequential + multi-criteria (calibrated AND sharp, then CRPS);
the hierarchical empirical fallback (with pre-block dispersion sharpening) is used when the
structural/calibration candidates fail.

## Result (strictly-prequential OOS ledger, holdout = latest 25 game-dates)
| market | method | ΔCRPS vs baseline | forecast_allowed | note |
|--------|--------|-------------------|------------------|------|
| turnover | location (PR#60) | −0.076 | **TRUE** | certified |
| pts | hierarchical/monotone-CDF | −0.94 | **TRUE** | clustered-PIT within envelope |
| ast | hierarchical | −0.26 | **TRUE** | |
| stl | hierarchical | −0.050 | **TRUE** | |
| stocks (stl+blk) | correlated combo | −0.082 | **TRUE** | |
| reb | hierarchical | −0.39 | false | 50% interval over-dispersed (residual +0.067, CI upper 0.090 > 0.07) |
| fg3m | location-and-scale | −0.07 | false | 80% interval too broad (sharpness 1.26 > 1.15 vs pooled marginal) |
| blk | hierarchical | −0.01 | false | over-broad (sharpness 1.43) + log score worse than baseline |
| pts_reb, pts_ast, pts_reb_ast | correlated combo | — | false | component combo build produced no rows (build_combo_pmfs integration gap) |

## Honest conclusion
Under the authorized corrected gates, **5 of 11 requested markets validate**
(turnover, pts, ast, stl, stocks). reb/fg3m/blk genuinely fail on over-dispersion /
sharpness (their forecasts are wider than the pooled empirical marginal — a real defect
that the bounded calibration + empirical fallback did not remove within the frozen gates).
The three points-combos were not scored because the reused component-combo builder did not
emit per-row PMFs for them in this harness (an integration gap, not a validated failure).
Gates were NOT weakened and no market was falsely certified. All-eleven completion is not
achieved; reb/fg3m/blk require the structural minutes/count retraining (Challenger B/C),
and the points-combos require repairing the combo-builder harness — both beyond the
bounded calibration-only scope exercised here.
