"""Unit tests for Enhancement 1: Usage Transfer Matrix."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.usage_transfer import (
    build_player_usage_map,
    build_wowy_splits,
    add_usage_transfer_features,
    POS_TRANSFER_WEIGHTS,
    _normalize_position,
)


def _make_stats_df() -> pd.DataFrame:
    """Minimal stats DataFrame with 3 players, 10 games each."""
    rng = np.random.default_rng(42)
    rows = []
    pids = [101, 102, 103]
    for pid in pids:
        for gn in range(10):
            rows.append({
                "player_id": pid,
                "game_id":   1000 + gn,
                "game_date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=gn),
                "team_id":   11,
                "position":  "G" if pid == 101 else ("F" if pid == 102 else "C"),
                "minutes":   float(rng.uniform(20, 35)),
                "fga":       int(rng.integers(5, 15)),
                "fta":       int(rng.integers(0, 8)),
                "turnover":  int(rng.integers(0, 4)),
                "pts":       int(rng.integers(5, 25)),
                "reb":       int(rng.integers(2, 12)),
                "ast":       int(rng.integers(0, 8)),
                "fg3m":      int(rng.integers(0, 4)),
                "did_play":  1,
            })
    return pd.DataFrame(rows)


def test_build_player_usage_map():
    stats = _make_stats_df()
    usage_map = build_player_usage_map(stats)
    assert len(usage_map) == 3
    for pid, info in usage_map.items():
        assert "usage_l5"    in info
        assert "usage_season" in info
        assert "position_group" in info
        assert info["usage_l5"] > 0
        assert info["usage_season"] > 0


def test_usage_map_cutoff():
    stats = _make_stats_df()
    cutoff = pd.Timestamp("2025-06-05")
    usage_map = build_player_usage_map(stats, cutoff_date=cutoff)
    # Should still find all players (they all have games before cutoff)
    assert len(usage_map) == 3


def test_normalize_position():
    assert _normalize_position("G") == "guard"
    assert _normalize_position("PG") == "guard"
    assert _normalize_position("SG") == "guard"
    assert _normalize_position("C") == "big"
    assert _normalize_position("F") == "wing"
    assert _normalize_position("SF") == "wing"
    assert _normalize_position("PF") == "wing"
    assert _normalize_position(None) == "wing"
    assert _normalize_position("") == "wing"


def test_pos_transfer_weights_sum():
    """Each beneficiary position should receive less than 100% of absent usage."""
    for absent_pos in ["guard", "wing", "big"]:
        total = sum(
            v for (ap, _bp), v in POS_TRANSFER_WEIGHTS.items() if ap == absent_pos
        )
        assert total < 1.01, f"Weights for absent pos={absent_pos} sum to {total}"


def test_add_usage_transfer_features_columns():
    stats = _make_stats_df()
    usage_map = build_player_usage_map(stats)
    wide = stats[["player_id", "game_id", "game_date", "team_id"]].drop_duplicates(
        subset=["player_id", "game_id"]
    ).copy()

    result = add_usage_transfer_features(wide, usage_map)

    assert "player_usage_rate_l5"    in result.columns
    assert "player_usage_rate_season" in result.columns
    assert "usage_shift"              in result.columns
    assert "usage_shift_abs"          in result.columns
    assert "projected_usage_given_absences" in result.columns
    assert "usage_transfer_delta"     in result.columns


def test_add_usage_transfer_features_values():
    stats = _make_stats_df()
    usage_map = build_player_usage_map(stats)
    wide = stats[["player_id", "game_id", "game_date", "team_id"]].drop_duplicates(
        subset=["player_id", "game_id"]
    ).copy()
    result = add_usage_transfer_features(wide, usage_map)

    # usage_shift_abs must be >= 0
    assert (result["usage_shift_abs"] >= 0).all()

    # projected_usage_given_absences >= player_usage_rate_season (absences can only add)
    assert (
        result["projected_usage_given_absences"] >= result["player_usage_rate_season"] - 1e-9
    ).all()


def test_build_wowy_splits():
    stats = _make_stats_df()
    usage_map = build_player_usage_map(stats)
    splits = build_wowy_splits(stats, usage_map)
    # At least some players should have splits
    assert isinstance(splits, dict)


def test_empty_usage_map():
    """add_usage_transfer_features should gracefully handle empty usage_map."""
    wide = pd.DataFrame({"player_id": [1, 2], "game_id": [10, 10]})
    result = add_usage_transfer_features(wide, {})
    # Should return original dataframe unchanged
    assert "player_id" in result.columns
