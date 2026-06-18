"""Tests for fatigue features (P2.3): 3-in-4, weekly load, cumulative minutes."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.build_features import _build_player_features


def _make_tight_schedule_df() -> pd.DataFrame:
    """4 games in 5 days — should trigger 3-in-4 flag on game 4."""
    dates = pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-05"])
    return pd.DataFrame({
        "player_id": [1, 1, 1, 1],
        "game_id": [1, 2, 3, 4],
        "game_date": dates,
        "season": [2024] * 4,
        "did_play": [1, 1, 1, 1],
        "minutes": [30.0, 32.0, 28.0, 31.0],
        "pts": [10.0, 12.0, 8.0, 15.0],
        "reb": [3.0] * 4, "ast": [2.0] * 4, "fg3m": [1.0] * 4,
        "stl": [1.0] * 4, "blk": [0.0] * 4, "turnover": [2.0] * 4,
        "fga": [8.0] * 4, "fg3a": [2.0] * 4, "fta": [2.0] * 4,
        "home_away": ["home"] * 4, "is_home": [1] * 4,
        "started_proxy": [1] * 4,
        "team_id": [10] * 4, "team_abbreviation": ["NYL"] * 4,
        "opponent_team_id": [20] * 4, "opponent_team_abbreviation": ["LVA"] * 4,
        "position": ["G"] * 4, "player_name": ["Test"] * 4,
        "plus_minus": [0.0] * 4, "pf": [1.0] * 4,
    })


def _make_games_df() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": [1, 2, 3, 4],
        "game_date": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-05"]),
        "season": [2024] * 4,
        "home_team_id": [10] * 4,
        "visitor_team_id": [20] * 4,
        "home_team_abbreviation": ["NYL"] * 4,
        "visitor_team_abbreviation": ["LVA"] * 4,
        "total_score": [160.0] * 4,
    })


def test_3in4_flag_triggered():
    stats = _make_tight_schedule_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    assert "player_3in4_flag" in result.columns
    # The 4th game is on May 5 and game 2 was May 2: May5-May2 = 3 days ≤ 3 → flag=1
    row4 = result.sort_values("game_date").iloc[3]
    assert row4["player_3in4_flag"] == 1, "4th game in tight schedule should have 3-in-4 flag"


def test_3in4_flag_not_triggered_rested():
    """7-day gap schedule should NOT trigger 3-in-4."""
    dates = pd.to_datetime(["2024-05-01", "2024-05-08", "2024-05-15", "2024-05-22"])
    stats = _make_tight_schedule_df().copy()
    stats["game_date"] = dates
    games = _make_games_df().copy()
    games["game_date"] = dates
    result = _build_player_features(stats, games)
    assert result["player_3in4_flag"].sum() == 0, "Rested schedule should not trigger 3-in-4"


def test_games_in_last_7_days():
    stats = _make_tight_schedule_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    assert "player_games_in_last_7_days" in result.columns
    # Game 4 (May 5): prior games May1, May2, May3 all within 7 days → count = 3
    row4 = result.sort_values("game_date").iloc[3]
    assert row4["player_games_in_last_7_days"] == 3


def test_cumulative_minutes_l3():
    stats = _make_tight_schedule_df()
    games = _make_games_df()
    result = _build_player_features(stats, games)
    assert "player_cumulative_minutes_l3" in result.columns
    # For game 4, prior 3 games had 30, 32, 28 minutes → sum = 90
    row4 = result.sort_values("game_date").iloc[3]
    assert abs(row4["player_cumulative_minutes_l3"] - 90.0) < 0.1
