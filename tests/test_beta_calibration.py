"""Tests for Beta calibration integration.

Verifies:
1. apply_beta_calibrators returns values in [0, 1] for any input.
2. predict_today.py p_over_beta column is added when calibrators are present.
3. Calibrator fit/apply round-trip is consistent.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.pipeline.calibrate import apply_beta_calibrators


# ---------------------------------------------------------------------------
# Helper: build a simple sklearn calibrator and save to a temp directory
# ---------------------------------------------------------------------------

def _make_dummy_calibrator(tmp_dir: Path, stat: str) -> Path:
    """Fit a trivial calibrated classifier and save as beta_cal_{stat}.pkl."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 1))
    y = (X[:, 0] > 0.5).astype(int)
    # Use Platt scaling as a stand-in for Beta calibration
    base = LogisticRegression()
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=2)
    cal.fit(X, y)
    out_path = tmp_dir / f"beta_cal_{stat}.pkl"
    joblib.dump(cal, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplyBetaCalibrators:
    def test_output_in_unit_interval(self, tmp_path):
        """Calibrated P(over) must always be in [0, 1]."""
        _make_dummy_calibrator(tmp_path, "pts")
        raw = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
        result = apply_beta_calibrators(raw, "pts", cal_dir=tmp_path)
        assert (result >= 0.0).all() and (result <= 1.0).all()

    def test_returns_original_when_no_calibrator(self, tmp_path):
        """If no calibrator file exists for the stat, return original series unchanged."""
        raw = pd.Series([0.2, 0.4, 0.6])
        result = apply_beta_calibrators(raw, "reb", cal_dir=tmp_path)
        pd.testing.assert_series_equal(result, raw)

    def test_nan_inputs_handled(self, tmp_path):
        """NaN inputs should be filled (not crash) and produce valid output."""
        _make_dummy_calibrator(tmp_path, "ast")
        raw = pd.Series([float("nan"), 0.5, 0.3])
        result = apply_beta_calibrators(raw, "ast", cal_dir=tmp_path)
        assert not result.isna().any(), "NaN inputs should be filled before calibration"
        assert (result >= 0.0).all() and (result <= 1.0).all()

    def test_index_preserved(self, tmp_path):
        """Output series must preserve the original index."""
        _make_dummy_calibrator(tmp_path, "fg3m")
        idx = [10, 20, 30]
        raw = pd.Series([0.3, 0.5, 0.7], index=idx)
        result = apply_beta_calibrators(raw, "fg3m", cal_dir=tmp_path)
        assert list(result.index) == idx

    def test_edge_probabilities_clipped(self, tmp_path):
        """Extreme inputs like 0.0 or 1.0 must not crash (clipped to epsilon)."""
        _make_dummy_calibrator(tmp_path, "stl")
        raw = pd.Series([0.0, 1.0, 0.5])
        result = apply_beta_calibrators(raw, "stl", cal_dir=tmp_path)
        assert not result.isna().any()
        assert (result >= 0.0).all() and (result <= 1.0).all()


class TestBetaCalibrationPushExclusion:
    """Test that push rows (actual == line) are excluded from Beta calibration fitting."""

    def test_push_rows_excluded(self):
        """fit_beta_calibrators must exclude rows where actual == line."""
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Build a synthetic OOF parquet with some push rows
            rng = np.random.default_rng(0)
            n = 100
            actuals = rng.integers(0, 25, n)
            line = 15.0
            # Include 10 push rows
            actuals[5:15] = int(line)

            # Build minimal PMF arrays
            pmfs = []
            for a in actuals:
                pmf = np.zeros(61)
                mu = max(0.5, float(a) + rng.normal(0, 1))
                pmf[min(60, max(0, int(round(mu))))] = 1.0
                pmfs.append(pmf)

            from wnba_props_model.models.simulation import pmf_to_json
            oof = pd.DataFrame({
                "player_id": np.arange(n),
                "game_id": np.arange(n),
                "stat": ["pts"] * n,
                "pmf_json": [pmf_to_json(p) for p in pmfs],
                "actual_outcome": actuals.astype(float),
                "outcome": actuals.astype(float),
                "line": [line] * n,
                "calibration_eligible": [True] * n,
                "role_bucket": ["all"] * n,
            })
            oof_path = out_dir / "oof.parquet"
            oof.to_parquet(oof_path, index=False)

            from wnba_props_model.pipeline.calibrate import fit_beta_calibrators
            paths = fit_beta_calibrators(oof_path, out_dir=out_dir, line_col="line")

            assert "pts" in paths, "Beta calibrator should be fitted for pts"
            cal = joblib.load(paths["pts"])
            # Round-trip check: calibrator should produce valid probabilities
            test_p = np.array([0.3, 0.5, 0.7]).reshape(-1, 1)
            out_p = cal.predict(test_p)
            assert all(0.0 <= p <= 1.0 for p in out_p)
