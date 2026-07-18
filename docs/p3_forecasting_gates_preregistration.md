# P3 — Preregistered forecasting & publication gates (v1)

Committed **before** final-holdout scoring. Any change requires a new version and a
new PR. Gates are evaluated on the exact production PMF artifact (artifact parity),
with game-date/game-clustered resampling for all uncertainty intervals.

## Definitions and status semantics
Each stat carries independent statuses (per-stat registry): forecast_publication,
market_comparison, betting_recommendation, validation_window, validation_metrics,
artifact_hashes, suppression_reason. One global allowlist is prohibited.

## Forecasting gates (per stat)

| gate | statistical test | practical tolerance | min sample | suppression rule |
|---|---|---|---|---|
| Material bias | mean(pred−actual); clustered 95% CI | \|bias\| ≤ 0.25·RMSE AND CI test | ≥ 300 rows, ≥ 25 game-dates | suppress if material or CI excludes a practically-null band |
| Two-sided coverage (50/80/90) | empirical vs nominal, clustered CI + coverage error | each interval within ±0.05 of nominal AND not overconfident (≥ nominal−0.05) AND not materially over-broad (≤ nominal+0.07) | ≥ 300 rows | suppress if any level materially mis-covers |
| Sharpness | mean/median interval width vs baseline (climatology/empirical marginal) | width ≤ 1.15× baseline at equal coverage | ≥ 300 rows | suppress if wide AND mis-covered |
| Discrete calibration | randomized PIT (seeded) KS/χ² vs Uniform; nonrandomized count-PIT vs reference | KS p ≥ 0.01 and no severe visual reliability break | ≥ 300 rows | suppress on severe non-uniformity |
| Proper scores | CRPS/RPS and log score vs predefined baseline | ≤ baseline (no material degradation) | ≥ 300 rows | suppress if worse than baseline |
| Line-level threshold calibration | Brier + log loss at REAL historical market lines; calibration slope/intercept; reliability by bucket | slope ∈ [0.8,1.25], intercept small; reliability ECE ≤ 0.06 computed at line level (one obs per line, NOT pooled per-k) | ≥ 150 lines | market_comparison suppressed if fails |

**Explicitly forbidden:** gating on ordinary midpoint-PIT uniformity; using pooled
per-threshold ECE (many dependent thresholds per observation) as the sole probability
calibration test. Turnovers must NOT pass on a 90% interval covering 96% while its 50%
interval covers ~78% — two-sided coverage must fail that.

## Promotion / suppression
- Promotion requires: all applicable forecasting gates PASS on development folds, then
  a single untouched-holdout confirmation, adequate independent game-date coverage,
  no severe role/minutes subgroup failure, and exact production-artifact parity.
- A stat that fails any gate is automatically suppressed (forecast and/or market).
- No automatic promotion; promotion requires a green validation artifact + PR.

## Edge (betting) gate — see Phase 8
Threshold selection runs a finite grid on chronological development folds, selected by
calibration + CLV stability + proper scores (NOT max ROI), frozen before the holdout,
and documented in `docs/p3_threshold_selection.md` (to be committed when Phase 8 runs).
Overs remain suppressed unless separate holdout evidence passes. No profitability claim
unless price-adjusted performance is positive with a game/date-clustered 95% CI
excluding zero.

## Baselines
Forecast baseline = seasonal per-player empirical marginal (climatology) with the same
minutes-availability handling; proper-score and sharpness comparisons are relative to it.
