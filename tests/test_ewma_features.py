"""Tests for EWMA rolling features (P2.4)."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.build_features import _build_player_features, STATS


def _make_stats_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player_id": [1] * n,
        "game_id": range(1, n + 1),
        "game_date": pd.date_range("2024-05-01", periods=n, freq="3D"),
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
        "home_away": ["home"] * n, "is_home": [1] * n, "started_proxy": [1] * n,
        "team_id": [10] * n, "team_abbreviation": ["NYL"] * n,
        "opponent_team_id": [20] * n, "opponent_team_abbreviation": ["LVA"] * n,
        "position": ["G"] * n, "player_name": ["Test"] * n,
        "plus_minus": [0.0] * n, "pf": [1.0] * n,
    })


def _make_games_df(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": range(1, n + 1),
        "game_date": pd.date_range("2024-05-01", periods=n, freq="3D"),
        "season": [2024] * n,
        "home_team_id": [10] * n, "visitor_team_id": [20] * n,
        "home_team_abbreviation": ["NYL"] * n, "visitor_team_abbreviation": ["LVA"] * n,
        "total_score": [160.0] * n,
    })


def test_ewma_cols_present_for_all_stats():
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    for stat in STATS:
        for hl in (3, 5):
            col = f"player_{stat}_ewma_halflife{hl}"
            assert col in result.columns, f"Expected EWMA col: {col}"


def test_ewma_minutes_cols_present():
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    assert "player_minutes_ewma_halflife3" in result.columns
    assert "player_minutes_ewma_halflife5" in result.columns


def test_ewma_shift1_safety():
    """First rows should be NaN (no prior games to form EWMA)."""
    stats = _make_stats_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    first_row = result.sort_values("game_date").iloc[0]
    assert pd.isna(first_row["player_pts_ewma_halflife3"]), "First row EWMA must be NaN"


def test_ewma_halflife3_weights_recent_more():
    """halflife3 should weight recent games more than halflife5 (faster decay)."""
    stats = _make_stats_df()
    # Set pts to 0 for first 10 games, then 20 for last 10 — halflife3 should be higher
    stats["pts"] = [0.0] * 10 + [20.0] * 10
    games = _make_games_df()
    result = _build_player_features(stats, games)
    # Compare last row: halflife3 should weigh recent 20s more
    last_row = result.sort_values("game_date").iloc[-1]
    hl3 = last_row["player_pts_ewma_halflife3"]
    hl5 = last_row["player_pts_ewma_halflife5"]
    if pd.notna(hl3) and pd.notna(hl5):
        assert hl3 >= hl5 - 0.5, "halflife3 should weight recent games ≥ halflife5 for rising trend"
