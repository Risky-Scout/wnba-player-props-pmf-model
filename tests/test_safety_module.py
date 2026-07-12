"""Tests for production-safe mode, quality statuses, and temporal integrity."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.pipeline.safety import (
    CAL_FALLBACK,
    CAL_FUTURE_CUTOFF,
    CAL_MISSING,
    CAL_PASS,
    CAL_STALE,
    QUALITY_EXPERIMENTAL,
    QUALITY_INSUFFICIENT_DATA,
    QUALITY_PUBLISHABLE,
    QUALITY_SUPPRESSED,
    QUALITY_WATCHLIST,
    CalibrationArtifactInfo,
    ModelLineage,
    SafeModeConfig,
    american_to_implied,
    american_to_no_vig,
    assign_market_quality_status,
    assign_model_quality_status,
    compute_model_edge,
    fair_odds_american,
    no_vig_normalize,
    strip_market_prior_features,
    validate_calibration_temporal_safety,
    validate_feature_point_in_time,
)


# ---------------------------------------------------------------------------
# No-vig helpers (§4.5)
# ---------------------------------------------------------------------------

class TestAmericanToImplied:
    def test_positive_odds(self):
        # +150: 100/(150+100) = 0.4
        p = american_to_implied(150.0)
        assert abs(p - 100.0 / 250.0) < 1e-12

    def test_negative_odds(self):
        # -110: 110/(110+100) = 0.5238...
        p = american_to_implied(-110.0)
        assert abs(p - 110.0 / 210.0) < 1e-12

    def test_even_money(self):
        p = american_to_implied(100.0)
        assert abs(p - 0.5) < 1e-12

    def test_heavy_favorite(self):
        p = american_to_implied(-500.0)
        assert abs(p - 500.0 / 600.0) < 1e-12


class TestNoVigNormalize:
    def test_symmetric_market_sums_to_one(self):
        p_over, p_under = no_vig_normalize(0.5238, 0.5238)
        assert abs(p_over + p_under - 1.0) < 1e-12
        assert abs(p_over - 0.5) < 1e-12

    def test_asymmetric_market(self):
        p_over, p_under = no_vig_normalize(0.55, 0.50)
        assert abs(p_over + p_under - 1.0) < 1e-12
        assert p_over > 0.5

    def test_sums_to_one_with_juice(self):
        # -110 / -110 market
        raw_over = american_to_implied(-110.0)
        raw_under = american_to_implied(-110.0)
        p_over, p_under = no_vig_normalize(raw_over, raw_under)
        assert abs(p_over + p_under - 1.0) < 1e-12
        assert abs(p_over - 0.5) < 1e-12

    def test_degenerate_zero_returns_nan(self):
        p_over, p_under = no_vig_normalize(0.0, 0.0)
        assert math.isnan(p_over)
        assert math.isnan(p_under)


class TestAmericanToNoVig:
    def test_even_market(self):
        p_over, p_under = american_to_no_vig(-110.0, -110.0)
        assert abs(p_over - 0.5) < 1e-12
        assert abs(p_under - 0.5) < 1e-12

    def test_favorite_over(self):
        p_over, p_under = american_to_no_vig(-200.0, +165.0)
        assert p_over > 0.5
        assert abs(p_over + p_under - 1.0) < 1e-12


class TestFairOdds:
    def test_50pct(self):
        odds = fair_odds_american(0.5)
        assert abs(odds - (-100.0)) < 0.01

    def test_underdog(self):
        odds = fair_odds_american(0.4)
        assert odds > 0  # underdog = positive odds

    def test_favorite(self):
        odds = fair_odds_american(0.6)
        assert odds < 0  # favorite = negative odds

    def test_nan_extremes(self):
        assert math.isnan(fair_odds_american(0.0))
        assert math.isnan(fair_odds_american(1.0))


# ---------------------------------------------------------------------------
# Model edge (NOT CLV)
# ---------------------------------------------------------------------------

class TestComputeModelEdge:
    def test_positive_edge(self):
        edge = compute_model_edge(0.60, 0.50)
        assert abs(edge - 0.10) < 1e-12

    def test_negative_edge(self):
        edge = compute_model_edge(0.40, 0.50)
        assert abs(edge - (-0.10)) < 1e-12

    def test_nan_model_returns_nan(self):
        edge = compute_model_edge(float("nan"), 0.50)
        assert math.isnan(edge)

    def test_nan_market_returns_nan(self):
        edge = compute_model_edge(0.60, float("nan"))
        assert math.isnan(edge)


# ---------------------------------------------------------------------------
# Safe mode config
# ---------------------------------------------------------------------------

class TestSafeModeConfig:
    def test_default_is_safe(self):
        cfg = SafeModeConfig.default_safe()
        assert cfg.demo_safe_mode is True
        assert cfg.disable_market_features_in_structural_model is True
        assert cfg.require_calibration_cutoff_before_prediction is True

    def test_disabled_mode(self):
        cfg = SafeModeConfig.disabled()
        assert cfg.demo_safe_mode is False
        assert cfg.fail_on_feature_leakage is False

    def test_from_yaml_missing_file(self, tmp_path):
        cfg = SafeModeConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert cfg.demo_safe_mode is True  # defaults to safe

    def test_from_yaml_reads_production_block(self, tmp_path):
        yaml_content = """
production:
  demo_safe_mode: true
  deterministic_seed: 99999
"""
        cfg_file = tmp_path / "model.yaml"
        cfg_file.write_text(yaml_content)
        cfg = SafeModeConfig.from_yaml(cfg_file)
        assert cfg.deterministic_seed == 99999


# ---------------------------------------------------------------------------
# Market-prior feature stripping
# ---------------------------------------------------------------------------

class TestStripMarketPriorFeatures:
    def test_strips_known_market_prior_cols(self):
        df = pd.DataFrame({
            "player_id": [1, 2],
            "pts_per_min_roll5": [0.5, 0.4],
            "player_market_p_over_prev": [0.52, 0.48],
            "player_market_line_prev": [18.5, 16.5],
            "player_line_movement_prev": [0.5, -0.5],
        })
        clean, removed = strip_market_prior_features(df)
        assert "player_market_p_over_prev" not in clean.columns
        assert "player_market_line_prev" not in clean.columns
        assert "player_line_movement_prev" not in clean.columns
        assert "pts_per_min_roll5" in clean.columns  # basketball feature kept
        assert set(removed) == {"player_market_p_over_prev", "player_market_line_prev", "player_line_movement_prev"}

    def test_no_market_cols_returns_unchanged(self):
        df = pd.DataFrame({"player_id": [1], "pts_per_min_roll5": [0.5]})
        clean, removed = strip_market_prior_features(df)
        assert removed == []
        assert list(clean.columns) == list(df.columns)

    def test_market_contamination_test(self):
        """Structural model features must not include market signals."""
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES
        market_cols = [
            "market_line", "over_odds", "under_odds", "no_vig_prob_over",
            "market_prob_over", "clv", "closing_line",
        ]
        for col in market_cols:
            assert col in FORBIDDEN_MODEL_FEATURES, f"{col} should be in FORBIDDEN_MODEL_FEATURES"


# ---------------------------------------------------------------------------
# Calibration temporal safety
# ---------------------------------------------------------------------------

class TestCalibrationTemporalSafety:
    def _make_cal_info(self, fitted_at: datetime) -> CalibrationArtifactInfo:
        return CalibrationArtifactInfo(
            fitted_at=fitted_at,
            status=CAL_PASS,
        )

    def test_fresh_calibrator_passes(self):
        pred_dt = datetime(2026, 7, 13, tzinfo=timezone.utc)
        fitted_dt = datetime(2026, 7, 1, tzinfo=timezone.utc)
        cal_info = self._make_cal_info(fitted_dt)
        status = validate_calibration_temporal_safety(cal_info, pred_dt)
        assert status == CAL_PASS

    def test_future_calibrator_fails(self):
        pred_dt = datetime(2026, 7, 1, tzinfo=timezone.utc)
        fitted_dt = datetime(2026, 7, 13, tzinfo=timezone.utc)  # after prediction!
        cal_info = self._make_cal_info(fitted_dt)
        status = validate_calibration_temporal_safety(cal_info, pred_dt)
        assert status == CAL_FUTURE_CUTOFF

    def test_stale_calibrator_flagged(self):
        pred_dt = datetime(2026, 7, 13, tzinfo=timezone.utc)
        fitted_dt = datetime(2026, 3, 1, tzinfo=timezone.utc)  # 134 days ago
        cal_info = self._make_cal_info(fitted_dt)
        status = validate_calibration_temporal_safety(cal_info, pred_dt)
        assert status == CAL_STALE

    def test_missing_artifact_returns_missing(self):
        cal_info = CalibrationArtifactInfo(status=CAL_MISSING)
        pred_dt = datetime(2026, 7, 13, tzinfo=timezone.utc)
        status = validate_calibration_temporal_safety(cal_info, pred_dt)
        assert status == CAL_MISSING


# ---------------------------------------------------------------------------
# Quality status assignment
# ---------------------------------------------------------------------------

class TestAssignModelQualityStatus:
    def _base_row(self) -> dict:
        return {
            "stat": "pts",
            "calibration_status": CAL_PASS,
            "data_quality_status": "FRESH",
            "market_quality_status": "FRESH",
            "availability_probability": 0.95,
            "calibration_sample_size": 500,
            "model_edge_over": 0.06,
            "model_edge_under": -0.06,
            "warnings": "",
        }

    def test_publishable_with_good_inputs(self):
        row = self._base_row()
        status = assign_model_quality_status(row)
        assert status == QUALITY_PUBLISHABLE

    def test_suppressed_when_future_calibration(self):
        row = self._base_row()
        row["calibration_status"] = CAL_FUTURE_CUTOFF
        status = assign_model_quality_status(row)
        assert status == QUALITY_SUPPRESSED

    def test_suppressed_when_low_availability(self):
        row = self._base_row()
        row["availability_probability"] = 0.1
        status = assign_model_quality_status(row)
        assert status == QUALITY_SUPPRESSED

    def test_insufficient_when_low_calibration_sample(self):
        row = self._base_row()
        row["calibration_sample_size"] = 10
        status = assign_model_quality_status(row)
        assert status == QUALITY_INSUFFICIENT_DATA

    def test_experimental_for_combo_stat(self):
        row = self._base_row()
        row["stat"] = "pts_ast"
        status = assign_model_quality_status(row)
        assert status == QUALITY_EXPERIMENTAL

    def test_watchlist_when_stale_calibration(self):
        row = self._base_row()
        row["calibration_status"] = CAL_STALE
        status = assign_model_quality_status(row)
        assert status == QUALITY_WATCHLIST

    def test_watchlist_when_stale_market(self):
        row = self._base_row()
        row["market_quality_status"] = "STALE"
        status = assign_model_quality_status(row)
        assert status == QUALITY_WATCHLIST

    def test_suppressed_when_leakage_warning(self):
        row = self._base_row()
        row["warnings"] = "temporal_violation detected"
        status = assign_model_quality_status(row)
        assert status == QUALITY_SUPPRESSED


class TestAssignMarketQualityStatus:
    def test_fresh_with_valid_odds_and_age(self):
        row = {"over_odds": -110.0, "under_odds": -110.0, "quote_age_seconds": 100}
        status = assign_market_quality_status(row)
        assert status == "FRESH"

    def test_missing_side_no_under(self):
        row = {"over_odds": -110.0, "under_odds": None, "quote_age_seconds": 100}
        status = assign_market_quality_status(row)
        assert status == "MISSING_SIDE"

    def test_stale_beyond_threshold(self):
        row = {"over_odds": -110.0, "under_odds": -110.0, "quote_age_seconds": 10000}
        status = assign_market_quality_status(row)
        assert status == "STALE"


# ---------------------------------------------------------------------------
# Feature temporal validation
# ---------------------------------------------------------------------------

class TestFeatureTemporalValidation:
    def test_no_warnings_when_cutoff_before_prediction(self):
        df = pd.DataFrame({
            "feature_cutoff_utc": ["2026-07-12T10:00:00Z"],
            "scheduled_start_utc": ["2026-07-13T23:00:00Z"],
        })
        pred_time = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        warnings = validate_feature_point_in_time(df, pred_time)
        assert warnings == []

    def test_warning_when_cutoff_after_prediction(self):
        df = pd.DataFrame({
            "feature_cutoff_utc": ["2026-07-14T10:00:00Z"],  # future
        })
        pred_time = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
        warnings = validate_feature_point_in_time(df, pred_time)
        assert len(warnings) > 0
        assert any("leakage" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Model lineage
# ---------------------------------------------------------------------------

class TestModelLineage:
    def test_capture_returns_lineage_object(self):
        lineage = ModelLineage.capture(seed=12345)
        assert lineage.deterministic_seed == 12345
        assert lineage.model_version != ""

    def test_to_dict_is_serializable(self):
        lineage = ModelLineage.capture(seed=12345)
        d = lineage.to_dict()
        json_str = json.dumps(d)  # must not raise
        assert "model_version" in d

    def test_add_lineage_to_df(self):
        from wnba_props_model.pipeline.safety import add_lineage_to_df
        df = pd.DataFrame({"player_id": [1, 2], "stat": ["pts", "reb"]})
        lineage = ModelLineage(model_version="0.1.0", config_hash="abc123")
        out = add_lineage_to_df(df, lineage)
        assert "model_version" in out.columns
        assert "config_hash" in out.columns
        assert all(out["model_version"] == "0.1.0")
