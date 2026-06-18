"""Tests for player-vs-specific-opponent matchup history features (P2.5)."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.build_features import _build_matchup_history_features


def _make_stats_df(seed: int = 42) -> pd.DataFrame:
    """Two players each playing 6 games against opponent 20 and 4 games vs opponent 30."""
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 1
    for player_id in (1, 2):
        for opp in (20, 30):
            n_games = 6 if opp == 20 else 4
            for i in range(n_games):
                rows.append({
                    "player_id": player_id,
                    "game_id": game_id,
                    "game_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=game_id),
                    "season": 2024,
                    "opponent_team_id": opp,
                    "pts": float(rng.integers(5, 25)),
                    "reb": float(rng.integers(1, 10)),
                    "ast": float(rng.integers(0, 8)),
                    "fg3m": float(rng.integers(0, 5)),
                    "stl": float(rng.integers(0, 3)),
                    "blk": float(rng.integers(0, 2)),
                    "turnover": float(rng.integers(0, 4)),
                    "minutes": float(rng.uniform(20, 35)),
                })
                game_id += 1
    return pd.DataFrame(rows)


def test_matchup_cols_present():
    stats = _make_stats_df()
    result = _build_matchup_history_features(stats)
    for stat in ("pts", "reb", "ast"):
        for suffix in ("_vs_opp_l3", "_vs_opp_career_mean", "_vs_opp_support"):
            col = f"player_{stat}{suffix}"
            assert col in result.columns, f"Expected column: {col}"


def test_matchup_minutes_cols_present():
    stats = _make_stats_df()
    result = _build_matchup_history_features(stats)
    assert "player_minutes_vs_opp_l3" in result.columns
    assert "player_minutes_vs_opp_career_mean" in result.columns


def test_matchup_shift1_no_leakage():
    """First game vs a given opponent must have NaN (no prior matchup)."""
    stats = _make_stats_df()
    result = _build_matchup_history_features(stats)
    # Each player's first game vs opponent 20 should be NaN
    first_vs_20 = (
        result.merge(stats[["player_id", "game_id", "opponent_team_id"]], on=["player_id", "game_id"])
        .query("player_id == 1 and opponent_team_id == 20")
        .sort_values("game_id")
        .iloc[0]
    )
    assert pd.isna(first_vs_20["player_pts_vs_opp_career_mean"]), "First matchup should be NaN"


def test_matchup_nan_when_support_lt_2():
    """career_mean should be NaN when only 1 prior game played against opponent."""
    stats = _make_stats_df()
    result = _build_matchup_history_features(stats)
    # Second game vs opp has support=1 → NaN for L3 and career_mean
    second_vs_20 = (
        result.merge(stats[["player_id", "game_id", "opponent_team_id"]], on=["player_id", "game_id"])
        .query("player_id == 1 and opponent_team_id == 20")
        .sort_values("game_id")
        .iloc[1]
    )
    assert pd.isna(second_vs_20["player_pts_vs_opp_career_mean"]), \
        "Second game (support=1) should have NaN career mean"


def test_matchup_support_grows():
    """Support count should increase as more games vs same opponent are played."""
    stats = _make_stats_df()
    result = _build_matchup_history_features(stats)
    vs_20 = (
        result.merge(stats[["player_id", "game_id", "opponent_team_id"]], on=["player_id", "game_id"])
        .query("player_id == 1 and opponent_team_id == 20")
        .sort_values("game_id")
    )
    supports = vs_20["player_pts_vs_opp_support"].fillna(0).values
    assert supports[-1] > supports[2], "Support should grow with more games"
