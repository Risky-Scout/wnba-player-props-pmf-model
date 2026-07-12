# Calibration Methodology

## Overview

The model uses role-aware isotonic calibration (Stage 6) applied to OOF PMF predictions.

---

## Calibration Hierarchy (§8.2)

Calibration falls back through:

```
stat + role + minutes tier + line tier
stat + role + minutes tier
stat + role
stat
global
identity (raw predictions, labeled FALLBACK_IDENTITY)
```

The narrowest level with ≥150 effective observations and acceptable validation is used.

---

## Temporal Safety (§8.1)

**Critical rule:** A calibrator may not be applied to a prediction date that is before or equal to its training/calibration cutoff.

```
calibrator.fitted_at < prediction_date
```

If this condition is violated:
- `calibration_status = CAL_FUTURE_CUTOFF`
- Calibration is skipped
- Raw predictions are used and labeled accordingly

---

## Calibration Periods

Correct chronological split:

```
model-training period
→ calibration period
→ untouched evaluation period
```

All players from the same game remain in the same split (group by game_date, not individual rows).

---

## Push-Aware Calibration (§8.4)

For integer lines, the push probability is preserved:

```
p_over = (1 - p_push) * q_over
p_under = (1 - p_push) * (1 - q_over)
```

Always verify: `p_over + p_under + p_push = 1` within numerical tolerance.

---

## DNP Settlement (§8.5)

DNPs are NOT automatically treated as zero-stat losses. Vendor-specific settlement rules apply:
- `void_if_no_participation` (most common)
- `action_if_starts`

Availability probability affects whether a prediction is publishable, but does not affect the settled stat PMF unless the vendor's rules make the ticket actionable.

---

## Calibration Status Labels

| Status | Meaning |
|--------|---------|
| `PASS` | Calibrated successfully, cutoff check passed |
| `FAIL` | Calibration failed validation gate |
| `FALLBACK_IDENTITY` | No calibrator available, using raw predictions |
| `INSUFFICIENT_DATA` | Fewer than 150 independent observations |
| `STALE_ARTIFACT` | Calibrator is older than 90 days |
| `CUTOFF_AFTER_PREDICTION` | Calibrator trained after prediction date |
| `MISSING_ARTIFACT` | No calibration artifact found |

---

## Required Calibration Outputs

For each stat × role category:

- `sample_size`
- `independent_game_count`
- `calibrator_level_used`
- `calibration_status`
- `brier_score_raw` / `brier_score_calibrated`
- `log_loss_raw` / `log_loss_calibrated`
- `calibration_intercept` / `calibration_slope`
- `ece` / `ece_confidence_interval`
- `mean_prediction` / `observed_frequency`
- `pmf_mean_bias` / `pmf_variance_ratio`
- `randomized_pit_ks`
- `interval_50_coverage` / `interval_80_coverage` / `interval_90_coverage` / `interval_95_coverage`

---

## Publication Gates (§8.8)

| Gate | Threshold | Action on Fail |
|------|-----------|----------------|
| `settlement_ece` | < 0.05 | Category → INSUFFICIENT_DATA or SUPPRESSED |
| `ece_ucb95` | < 0.075 | Category → WATCHLIST |
| `calibration_slope` | 0.90–1.10 | Category → WATCHLIST |
| `mean_pmf_bias` | < 0.20 σ | Category → WATCHLIST |
| `sample_size` | ≥ 150 | Category → INSUFFICIENT_DATA |

Failing a gate never silently passes. All gate results are written to `calibration_by_stat_role.csv`.
