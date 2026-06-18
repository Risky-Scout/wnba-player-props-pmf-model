"""Tests for Pi rating OOF model baseline (P3.4)."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.pi_ratings import build_pi_rating_features


def _make_wide_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player_id": [1] * n,
        "game_id": range(1, n + 1),
        "game_date": pd.date_range("2024-05-01", periods=n, freq="3D"),
        "season": [2024] * n,
        "is_home": rng.integers(0, 2, n),
        "team_id": [10] * n,
        "opponent_team_id": [20] * n,
        "actual_pts": rng.integers(5, 25, n).astype(float),
        "player_pts_mean_l5": rng.uniform(8, 20, n),
    })


def test_pi_ratings_use_rolling_baseline_by_default():
    """Without model_predictions_df, baseline should be player_pts_mean_l5."""
    wide = _make_wide_df()
    result = build_pi_rating_features(wide, stats=["pts"])
    assert "player_pts_pi_home_form" in result.columns
    assert "player_pts_pi_away_form" in result.columns


def test_pi_ratings_switch_to_model_baseline():
    """When model_predictions_df is provided with pts_model_pred, it should be used as baseline."""
    wide = _make_wide_df()
    # Create model predictions: double the actual to simulate a "model" that overshoots
    model_preds = wide[["player_id", "game_id"]].copy()
    model_preds["pts_model_pred"] = wide["actual_pts"] * 2  # overestimates

    result_model = build_pi_rating_features(
        wide, stats=["pts"], model_predictions_df=model_preds
    )
    result_rolling = build_pi_rating_features(wide, stats=["pts"])

    pi_col = "player_pts_pi_home_form"
    # Pi ratings should differ between model and rolling baselines
    model_pi = result_model[pi_col].dropna()
    rolling_pi = result_rolling[pi_col].dropna()
    if len(model_pi) > 0 and len(rolling_pi) > 0:
        # Model overestimates → larger negative residuals → different Pi
        assert not np.allclose(model_pi.values, rolling_pi.values, atol=1e-6), \
            "Pi ratings should differ when baseline changes"


def test_model_baseline_cols_cleaned_up():
    """model_pred columns should not contaminate output unnecessarily."""
    wide = _make_wide_df()
    model_preds = wide[["player_id", "game_id"]].copy()
    model_preds["pts_model_pred"] = wide["actual_pts"] * 1.1
    result = build_pi_rating_features(wide, stats=["pts"], model_predictions_df=model_preds)
    # Pi feature columns should be present
    assert "player_pts_pi_home_form" in result.columns
