# Production Hardening — WNBA Player-Prop PMF Model

## Overview

This document describes the production-hardening changes implemented on this branch (`cursor/wnba-model-optimization-06e5`). These changes are **not merged to main** until they have been validated and approved.

---

## Priority 0: Immediate Integrity Fixes

### 1. Production-Safe Mode

Added `production` block to `config/wnba_model.yaml`:

```yaml
production:
  demo_safe_mode: true
  fail_on_feature_leakage: true
  fail_on_artifact_date_violation: true
  require_model_lineage: true
  disable_unvalidated_advanced_features: true
  disable_market_features_in_structural_model: true
  require_calibration_cutoff_before_prediction: true
  allow_identity_calibration_fallback: true
  suppress_failed_categories: true
  mark_insufficient_categories: true
  deterministic_seed: 20260712
  strip_market_prior_features_in_safe_mode: true
  min_calibration_obs: 150
  quote_stale_seconds: 7200
  min_publishable_edge: 0.04
```

Safe mode is loaded by `SafeModeConfig.from_yaml()` in `src/wnba_props_model/pipeline/safety.py`.

### 2. CLV Mislabeling Fixed

The `backtest_clv.py` script previously computed a metric called `clv_hit_rate` that was NOT CLV. It measured whether the model's edge direction agreed with line movement direction.

**Fix:** Renamed to `model_edge_vs_open_agreement_rate` with explicit `metric_note: "NOT CLV"` in all outputs.

**True CLV** is implemented in `src/wnba_props_model/evaluation/clv.py` and requires archived closing quotes.

### 3. Market-Prior Feature Stripping

The structural model previously had access to lagged market features (`player_market_p_over_prev`, `player_market_line_prev`, `player_line_movement_prev`) from the `market_prior` feature family. While these are lagged one game (no same-game leakage), they are market-derived signals that should not enter the structural outcome model.

**Fix:** In safe mode, these features are stripped before the HGB feature matrix is built. See `strip_market_prior_features()` in `safety.py`.

### 4. Calibration Temporal Safety

Calibrators may not be applied to prediction dates that are before or equal to the calibrator's training cutoff.

**Fix:** `validate_calibration_temporal_safety()` checks:
- `fitted_at > prediction_time` → `CAL_FUTURE_CUTOFF` → skip calibration
- `age_days > 90` → `CAL_STALE` → apply with warning
- Missing artifact → `CAL_MISSING` → use raw predictions

### 5. Quality Status Labels

All predictions now carry:
- `model_quality_status`: PUBLISHABLE | WATCHLIST | EXPERIMENTAL | SUPPRESSED | INSUFFICIENT_DATA
- `market_quality_status`: FRESH | AGING | STALE | MISSING_SIDE
- `data_quality_status`: FRESH | RECENT | STALE | UNKNOWN | MISSING

### 6. Model Lineage

Every prediction row now carries:
- `prediction_timestamp_utc`
- `feature_cutoff_utc`
- `training_cutoff_utc`
- `calibration_cutoff_utc`
- `model_version`
- `config_hash`
- `feature_manifest_hash`

### 7. Unified Prediction Pipeline

`src/wnba_props_model/pipeline/prediction_pipeline.py` implements `PredictionPipeline`, a unified class that:
- Is called by both OOF and live prediction paths
- Applies identical feature preprocessing
- Applies identical calibration with temporal safety
- Assigns identical quality statuses
- Tracks identical lineage

### 8. Demo Prediction Script

`scripts/prepare_demo_predictions.py` generates the July 13 presentation report with:

```bash
python scripts/prepare_demo_predictions.py \
  --date 2026-07-13 \
  --safe-mode \
  --output-dir reports/demo_2026-07-13
```

---

## Files Changed

| File | Change Type | Purpose |
|------|-------------|---------|
| `config/wnba_model.yaml` | Modified | Add production-safe mode block |
| `src/wnba_props_model/pipeline/safety.py` | Created | Safe-mode enforcement, quality statuses, lineage |
| `src/wnba_props_model/evaluation/clv.py` | Created | Correct CLV implementation |
| `src/wnba_props_model/pipeline/prediction_pipeline.py` | Created | Unified prediction pipeline |
| `scripts/prepare_demo_predictions.py` | Created | Demo report generator |
| `scripts/backtest_clv.py` | Modified | Rename false CLV metric |
| `tests/test_safety_module.py` | Created | Safety module tests |
| `tests/test_clv.py` | Created | CLV implementation tests |
| `tests/test_feature_temporal_safety.py` | Created | Feature point-in-time tests |
| `tests/test_minutes_coherence.py` | Created | Minutes model coherence tests |
| `tests/test_pmf_integrity.py` | Created | PMF invariant tests |
| `docs/clv_methodology.md` | Created | CLV methodology documentation |
| `docs/production_hardening.md` | Created | This document |
| `docs/calibration_methodology.md` | Created | Calibration methodology |
| `docs/feature_point_in_time_policy.md` | Created | Feature temporal policy |
| `docs/demo_readiness_2026-07-13.md` | Created | Demo readiness assessment |

---

## Tests Added

All new tests are in `tests/`:
- `test_safety_module.py` — 30+ tests for safety module
- `test_clv.py` — 25+ tests for CLV implementation (§4.7 checklist)
- `test_feature_temporal_safety.py` — temporal integrity tests
- `test_minutes_coherence.py` — minutes model coherence
- `test_pmf_integrity.py` — PMF invariant tests

---

## What Was NOT Changed

The following components were **not modified** to preserve the production pipeline:
- `scripts/predict_today.py` — production entrypoint unchanged
- `scripts/build_edge_report.py` — production edge report unchanged
- All GitHub Actions workflows — unchanged
- All existing model training scripts — unchanged
- `tools/odds-scanner/` — sportsodds page delivery unchanged
