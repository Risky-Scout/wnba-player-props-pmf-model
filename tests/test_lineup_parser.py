"""Tests for automated injury/lineup parser (P4.2)."""
import pandas as pd
import numpy as np
import pytest

from wnba_props_model.data.lineup_parser import LineupStatusParser, apply_lineup_overrides


def _make_injuries_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "game_id": [10, 10, 10, 20, 20],
        "status": ["out", "available", "questionable", "inactive", "probable"],
    })


def test_parse_out_player():
    parser = LineupStatusParser()
    inj = _make_injuries_df()
    result = parser.parse_injury_statuses(inj)
    out_row = result[result["player_id"] == 1].iloc[0]
    assert out_row["confirmed_starter"] == -1
    assert out_row["lineup_confirmed"] == 1
    assert out_row["inferred_out"] == 1


def test_parse_available_player():
    parser = LineupStatusParser()
    inj = _make_injuries_df()
    result = parser.parse_injury_statuses(inj)
    avail_row = result[result["player_id"] == 2].iloc[0]
    assert avail_row["confirmed_starter"] == 0
    assert avail_row["inferred_out"] == 0


def test_parse_questionable_player():
    parser = LineupStatusParser()
    inj = _make_injuries_df()
    result = parser.parse_injury_statuses(inj)
    q_row = result[result["player_id"] == 3].iloc[0]
    assert q_row["confirmed_starter"] == 0  # uncertain
    assert q_row["inferred_out"] == 0  # not confirmed out


def test_parse_inactive_player():
    parser = LineupStatusParser()
    inj = _make_injuries_df()
    result = parser.parse_injury_statuses(inj)
    inact_row = result[result["player_id"] == 4].iloc[0]
    assert inact_row["inferred_out"] == 1


def test_parse_empty_df():
    parser = LineupStatusParser()
    result = parser.parse_injury_statuses(pd.DataFrame())
    assert result.empty


def test_apply_lineup_overrides_sets_p_dnp_1_for_out():
    parser = LineupStatusParser()
    injuries = _make_injuries_df()
    wide = pd.DataFrame({
        "player_id": [1, 2, 3],
        "game_id": [10, 10, 10],
        "player_pts_mean_l5": [12.0, 15.0, 8.0],
    })
    result = apply_lineup_overrides(wide, injuries)
    out_row = result[result["player_id"] == 1].iloc[0]
    assert out_row["p_dnp_override"] == 1.0


def test_apply_lineup_overrides_nan_for_available():
    wide = pd.DataFrame({
        "player_id": [2],
        "game_id": [10],
        "player_pts_mean_l5": [15.0],
    })
    result = apply_lineup_overrides(wide, _make_injuries_df())
    avail_row = result[result["player_id"] == 2].iloc[0]
    assert pd.isna(avail_row["p_dnp_override"]), "Available player should have NaN override"


def test_infer_from_recent_starts():
    parser = LineupStatusParser()
    stats = pd.DataFrame({
        "player_id": [1] * 10,
        "game_id": range(1, 11),
        "game_date": pd.date_range("2024-01-01", periods=10),
        "started_proxy": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    })
    result = parser.infer_from_recent_starts(stats, window=5)
    assert "starter_rate_l5" in result.columns
    # First row should be NaN (no prior games)
    first = result.sort_values("game_id").iloc[0]
    assert pd.isna(first["starter_rate_l5"]), "First row must be NaN"


def test_leakage_contract_not_violated():
    """confirmed_starter must come from injury status, not box score columns."""
    parser = LineupStatusParser()
    inj = _make_injuries_df()
    result = parser.parse_injury_statuses(inj)
    # Ensure result has no box score columns
    box_score_cols = {"actual_minutes", "actual_pts", "did_play", "minutes"}
    assert not box_score_cols.intersection(result.columns), \
        "Lineup parser output must not contain box score columns"
