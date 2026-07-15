"""Regression test for run 29429153433.

Fatal error: Column 'player_id' not found in quotes DataFrame.
The Odds API parquet has player_name + event_id but not player_id/game_id.
validate_player_identity_resolved and validate_game_identity_resolved must
accept these provider-native columns instead of failing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.pipeline.market_integrity import (
    UnmatchedIdentityError,
    validate_game_identity_resolved,
    validate_player_identity_resolved,
)


def _oddsapi_df(**overrides) -> pd.DataFrame:
    """Minimal Odds API parquet schema: vendor, event_id, player_name, stat, line."""
    rows = [
        {"vendor": "fanduel",    "event_id": "uuid-game1", "player_name": "Alyssa Thomas",
         "stat": "pts", "line": 18.5, "over_odds": -115.0, "under_odds": -105.0},
        {"vendor": "draftkings", "event_id": "uuid-game1", "player_name": "Alyssa Thomas",
         "stat": "reb", "line": 7.5, "over_odds": -110.0, "under_odds": -110.0},
        {"vendor": "fanduel",    "event_id": "uuid-game2", "player_name": "Sabrina Ionescu",
         "stat": "pts", "line": 22.5, "over_odds": -120.0, "under_odds": +100.0},
    ]
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


# ─── validate_player_identity_resolved ───────────────────────────────────────

def test_player_identity_accepts_player_name_when_no_player_id():
    """Odds API parquet has player_name — must not raise."""
    df = _oddsapi_df()
    assert "player_id" not in df.columns
    assert "player_name" in df.columns
    validate_player_identity_resolved(df)  # must not raise


def test_player_identity_prefers_player_id_when_present():
    """BDL parquet has player_id — must use it (original behaviour)."""
    df = _oddsapi_df()
    df["player_id"] = [735, 569, 612]
    validate_player_identity_resolved(df)  # must not raise


def test_player_identity_raises_when_player_name_blank():
    """Blank player_name is an unresolved identity."""
    df = _oddsapi_df()
    df.loc[0, "player_name"] = ""
    with pytest.raises(UnmatchedIdentityError, match="player_name"):
        validate_player_identity_resolved(df)


def test_player_identity_raises_when_player_name_null():
    """Null player_name is an unresolved identity."""
    df = _oddsapi_df()
    df.loc[1, "player_name"] = None
    with pytest.raises(UnmatchedIdentityError, match="player_name"):
        validate_player_identity_resolved(df)


def test_player_identity_raises_when_neither_column_present():
    """No player_id and no player_name → error."""
    df = pd.DataFrame([{"vendor": "fd", "stat": "pts", "line": 18.5}])
    with pytest.raises(UnmatchedIdentityError, match="Neither"):
        validate_player_identity_resolved(df)


def test_player_identity_raises_when_player_id_null_bdl():
    """BDL parquet with null player_id must still raise (original behaviour)."""
    df = _oddsapi_df()
    df["player_id"] = [735, None, 612]
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_player_identity_resolved(df)


# ─── validate_game_identity_resolved ─────────────────────────────────────────

def test_game_identity_accepts_event_id_when_no_game_id():
    """Odds API parquet has event_id — must not raise."""
    df = _oddsapi_df()
    assert "game_id" not in df.columns
    assert "event_id" in df.columns
    validate_game_identity_resolved(df)  # must not raise


def test_game_identity_prefers_game_id_when_present():
    """BDL parquet has game_id — must use it."""
    df = _oddsapi_df()
    df["game_id"] = [24931, 24931, 24932]
    validate_game_identity_resolved(df)  # must not raise


def test_game_identity_skips_when_neither_column_present():
    """Provider with no game context at all — skip silently."""
    df = pd.DataFrame([{"vendor": "fd", "player_name": "X", "stat": "pts", "line": 18.5}])
    assert "game_id" not in df.columns
    assert "event_id" not in df.columns
    validate_game_identity_resolved(df)  # must not raise (skip)


def test_game_identity_raises_when_event_id_blank():
    """Blank event_id is unresolved."""
    df = _oddsapi_df()
    df.loc[0, "event_id"] = ""
    with pytest.raises(UnmatchedIdentityError, match="event_id"):
        validate_game_identity_resolved(df)


# ─── Full validation pipeline (as called by build_edge_report.py) ─────────────

def test_full_oddsapi_validation_pipeline_passes():
    """All four validators pass on a realistic Odds API parquet."""
    from wnba_props_model.pipeline.market_integrity import (
        validate_no_duplicate_quotes,
        validate_odds_format,
    )
    df = _oddsapi_df()
    validate_no_duplicate_quotes(df)
    validate_player_identity_resolved(df)
    validate_game_identity_resolved(df)
    validate_odds_format(df)
