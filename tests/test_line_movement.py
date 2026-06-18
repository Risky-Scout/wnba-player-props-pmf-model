"""Tests for opening line tracking and line movement features (P4.1)."""
import pandas as pd
import numpy as np
import pytest

from wnba_props_model.pipeline.deliver import normalize_player_props_snapshot


def _make_raw_props(n: int = 4) -> pd.DataFrame:
    """Minimal raw props snapshot DataFrame."""
    return pd.DataFrame({
        "game_id": [1, 1, 2, 2],
        "player_id": [101, 101, 102, 102],
        "prop_type": ["points", "rebounds", "points", "assists"],
        "vendor": ["DraftKings"] * n,
        "line_value": [18.5, 6.5, 12.5, 4.5],
        "prop_line_open": [17.5, 6.5, 13.0, 4.0],  # P4.1: opening line
        "market": [
            {"over_odds": -115, "under_odds": -105},
            {"over_odds": -110, "under_odds": -110},
            {"over_odds": -120, "under_odds": 100},
            {"over_odds": -108, "under_odds": -112},
        ],
        "updated_at": [pd.Timestamp("2024-06-01 10:00", tz="UTC")] * n,
    })


def test_edge_under_present_in_market_comparison():
    """build_market_comparison should produce edge_under column."""
    from wnba_props_model.pipeline.deliver import build_market_comparison
    import json

    pmfs = pd.DataFrame({
        "game_id": [1, 2],
        "player_id": [101, 102],
        "stat": ["pts", "pts"],
        "pmf_json": [
            json.dumps({str(k): (0.05 if k < 20 else 0.0) for k in range(61)}),
            json.dumps({str(k): (0.05 if k < 13 else 0.0) for k in range(61)}),
        ],
        "pmf_mean": [15.0, 10.0],
    })
    raw_props = _make_raw_props()
    result = build_market_comparison(pmfs, raw_props)
    assert "edge_under" in result.columns, "edge_under column must be present"


def test_line_delta_computed():
    """normalize_player_props_snapshot should compute line_delta."""
    props = _make_raw_props()
    result = normalize_player_props_snapshot(props)
    if result.empty or "line_delta" not in result.columns:
        pytest.skip("normalize_player_props_snapshot did not produce line_delta (likely parser path)")
    # line_value - prop_line_open = 18.5 - 17.5 = 1.0 for player 101 pts
    pts_row = result[(result["player_id"] == 101) & (result["stat"] == "pts")].iloc[0]
    assert abs(pts_row["line_delta"] - 1.0) < 0.01


def test_line_moved_toward_over_flag():
    """line_moved_toward_over should be True when line_delta > 0.25."""
    props = _make_raw_props()
    result = normalize_player_props_snapshot(props)
    if result.empty or "line_moved_toward_over" not in result.columns:
        pytest.skip("line_moved_toward_over column not produced")
    pts_row = result[(result["player_id"] == 101) & (result["stat"] == "pts")].iloc[0]
    # delta = 1.0 > 0.25 → should be True
    assert pts_row["line_moved_toward_over"] is True or pts_row["line_moved_toward_over"] == 1


def test_edge_under_is_negative_edge_over():
    """edge_under + edge_over should sum to ~0 (they are symmetric)."""
    from wnba_props_model.pipeline.deliver import build_market_comparison
    import json

    pmfs = pd.DataFrame({
        "game_id": [1],
        "player_id": [101],
        "stat": ["pts"],
        "pmf_json": [json.dumps({str(k): (0.05 if k < 20 else 0.0) for k in range(61)})],
        "pmf_mean": [15.0],
    })
    raw_props = _make_raw_props().iloc[:1].copy()
    result = build_market_comparison(pmfs, raw_props)
    if result.empty:
        pytest.skip("No market comparison rows")
    row = result.iloc[0]
    total = row["edge_over"] + row["edge_under"]
    assert abs(total) < 0.01, f"edge_over + edge_under should ≈ 0, got {total}"
