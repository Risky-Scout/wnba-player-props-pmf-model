"""Tests for shot profile features (P2.1)."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.build_features import _build_player_features


def _make_stats_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    games = pd.date_range("2024-05-01", periods=n, freq="3D")
    return pd.DataFrame({
        "player_id": [1] * n,
        "game_id": range(1, n + 1),
        "game_date": games,
        "season": [2024] * n,
        "did_play": [1] * n,
        "minutes": rng.uniform(20, 35, n),
        "pts": rng.integers(5, 25, n).astype(float),
        "reb": rng.integers(1, 10, n).astype(float),
        "ast": rng.integers(0, 8, n).astype(float),
        "fg3m": rng.integers(0, 5, n).astype(float),
        "stl": rng.integers(0, 3, n).astype(float),
        "blk": rng.integers(0, 2, n).astype(float),
        "turnover": rng.integers(0, 4, n).astype(float),
        "fga": rng.integers(5, 18, n).astype(float),
        "fg3a": rng.integers(0, 7, n).astype(float),
        "fta": rng.integers(0, 6, n).astype(float),
        "home_away": ["home"] * n,
        "is_home": [1] * n,
        "started_proxy": [1] * n,
        "team_id": [10] * n,
        "team_abbreviation": ["NYL"] * n,
        "opponent_team_id": [20] * n,
        "opponent_team_abbreviation": ["LVA"] * n,
        "position": ["G"] * n,
        "player_name": ["Test Player"] * n,
        "plus_minus": [0.0] * n,
        "pf": rng.integers(0, 4, n).astype(float),
    })


def _make_games_df() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": list(range(1, 21)),
        "game_date": pd.date_range("2024-05-01", periods=20, freq="3D"),
        "season": [2024] * 20,
        "home_team_id": [10] * 20,
        "visitor_team_id": [20] * 20,
        "home_team_abbreviation": ["NYL"] * 20,
        "visitor_team_abbreviation": ["LVA"] * 20,
        "total_score": [160.0] * 20,
    })


def test_fg3_attempt_rate_present():
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    assert "player_fg3_attempt_rate_l5" in result.columns
    assert "player_fg3_attempt_rate_l10" in result.columns
    assert "player_fg3_attempt_rate_season" in result.columns


def test_fg3_attempt_rate_bounded():
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    vals = result["player_fg3_attempt_rate_l5"].dropna()
    assert (vals >= 0).all(), "fg3_attempt_rate must be >= 0"
    assert (vals <= 1).all(), "fg3_attempt_rate must be <= 1"


def test_shot_zone_cols_present():
    """Shot zone cols should exist (NaN when shot_df unavailable)."""
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    for col in ["player_rim_freq_l5", "player_corner3_freq_l5", "player_above_break3_freq_l5"]:
        assert col in result.columns, f"Expected column: {col}"
        # Should be NaN since no shot_df provided
        assert result[col].isna().all(), f"{col} should be NaN without shot_df"


def test_shift1_no_leakage():
    """First row must have NaN fg3_attempt_rate (no prior data available)."""
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    first_row = result.sort_values("game_date").iloc[0]
    assert pd.isna(first_row["player_fg3_attempt_rate_l5"]), "First row must be NaN (no prior games)"
