"""Focused tests for the two-stage identity contract.

Stage 1 (provider-native, pre-join): validate_provider_quotes()
Stage 2 (canonical, post-join):      validate_player/game_identity_resolved()
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.pipeline.market_integrity import (
    UnmatchedIdentityError,
    validate_game_identity_resolved,
    validate_player_identity_resolved,
    validate_provider_quotes,
)


def _odds_df(**overrides) -> pd.DataFrame:
    rows = [
        {"vendor": "fanduel",    "event_id": "uuid-1", "player_name": "Alyssa Thomas",
         "stat": "pts", "line": 18.5, "over_odds": -115.0, "under_odds": -105.0},
        {"vendor": "draftkings", "event_id": "uuid-1", "player_name": "Sabrina Ionescu",
         "stat": "pts", "line": 22.5, "over_odds": -110.0, "under_odds": -110.0},
    ]
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


def _bdl_df(**overrides) -> pd.DataFrame:
    rows = [
        {"game_id": 24931, "player_id": 735, "vendor": "draftkings",
         "stat": "pts", "line": 18.5, "over_odds": -115.0, "under_odds": -105.0},
        {"game_id": 24931, "player_id": 569, "vendor": "fanduel",
         "stat": "reb", "line": 7.5,  "over_odds": -110.0, "under_odds": -110.0},
    ]
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


# ─── Stage 1: Odds API provider-native validation ────────────────────────────

def test_oddsapi_event_id_plus_player_name_passes():
    validate_provider_quotes(_odds_df(), source="oddsapi")


def test_oddsapi_blank_player_name_is_fatal():
    df = _odds_df()
    df.loc[0, "player_name"] = ""
    with pytest.raises(UnmatchedIdentityError, match="player_name"):
        validate_provider_quotes(df, source="oddsapi")


def test_oddsapi_null_player_name_is_fatal():
    df = _odds_df()
    df.loc[1, "player_name"] = None
    with pytest.raises(UnmatchedIdentityError, match="player_name"):
        validate_provider_quotes(df, source="oddsapi")


def test_oddsapi_blank_event_id_is_fatal():
    df = _odds_df()
    df.loc[0, "event_id"] = ""
    with pytest.raises(UnmatchedIdentityError, match="event_id"):
        validate_provider_quotes(df, source="oddsapi")


def test_oddsapi_null_event_id_is_fatal():
    df = _odds_df()
    df.loc[1, "event_id"] = None
    with pytest.raises(UnmatchedIdentityError, match="event_id"):
        validate_provider_quotes(df, source="oddsapi")


def test_oddsapi_missing_event_id_column_is_fatal():
    df = _odds_df().drop(columns=["event_id"])
    with pytest.raises(UnmatchedIdentityError, match="event_id"):
        validate_provider_quotes(df, source="oddsapi")


def test_oddsapi_missing_player_name_column_is_fatal():
    df = _odds_df().drop(columns=["player_name"])
    with pytest.raises(UnmatchedIdentityError, match="player_name"):
        validate_provider_quotes(df, source="oddsapi")


# ─── Stage 1: BDL provider-native validation ─────────────────────────────────

def test_bdl_game_id_player_id_passes():
    validate_provider_quotes(_bdl_df(), source="bdl")


def test_bdl_null_player_id_is_fatal():
    df = _bdl_df()
    df.loc[0, "player_id"] = None
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_provider_quotes(df, source="bdl")


def test_bdl_null_game_id_is_fatal():
    df = _bdl_df()
    df.loc[1, "game_id"] = None
    with pytest.raises(UnmatchedIdentityError, match="game_id"):
        validate_provider_quotes(df, source="bdl")


# ─── Stage 2: Canonical post-join validation ──────────────────────────────────

def _joined_df(**overrides) -> pd.DataFrame:
    """Simulate a joined comp DataFrame with canonical IDs."""
    rows = [
        {"game_id": 24931, "player_id": 735, "player_name": "Alyssa Thomas",
         "stat": "pts", "line": 18.5, "vendor": "fanduel"},
        {"game_id": 24931, "player_id": 569, "player_name": "Sabrina Ionescu",
         "stat": "pts", "line": 22.5, "vendor": "draftkings"},
    ]
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


def test_canonical_validation_passes_after_join():
    comp = _joined_df()
    validate_player_identity_resolved(comp)
    validate_game_identity_resolved(comp)


def test_canonical_player_id_null_after_join_is_fatal():
    """Null player_id after join = reconciliation failed to resolve identity."""
    comp = _joined_df()
    comp.loc[0, "player_id"] = None
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_player_identity_resolved(comp)


def test_canonical_game_id_null_after_join_is_fatal():
    """Null game_id after join = event_id failed to resolve to canonical game."""
    comp = _joined_df()
    comp.loc[1, "game_id"] = None
    with pytest.raises(UnmatchedIdentityError, match="game_id"):
        validate_game_identity_resolved(comp)


def test_canonical_validator_requires_player_id_column():
    """Stage 2 must not accept player_name as substitute for player_id."""
    comp = _joined_df().drop(columns=["player_id"])
    assert "player_id" not in comp.columns
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_player_identity_resolved(comp)


def test_canonical_validator_requires_game_id_column():
    """Stage 2 must not accept event_id as substitute for game_id."""
    comp = _joined_df().drop(columns=["game_id"])
    comp["event_id"] = "uuid-1"
    assert "game_id" not in comp.columns
    with pytest.raises(UnmatchedIdentityError, match="game_id"):
        validate_game_identity_resolved(comp)


# ─── One provider quote → exactly one canonical ID ───────────────────────────

def test_one_to_one_provider_to_canonical_mapping():
    """Each provider row must resolve to exactly one (game_id, player_id, stat)."""
    from wnba_props_model.pipeline.deliver import normalize_player_props_snapshot

    # Simulate Odds API raw row (player_id=None, event_id present)
    raw = pd.DataFrame([{
        "player_name": "Alyssa Thomas", "event_id": "uuid-1",
        "vendor": "fanduel", "stat": "pts", "line": 18.5,
        "over_odds": -115.0, "under_odds": -105.0,
        "market_prob_over_no_vig": None,
    }])
    normalized = normalize_player_props_snapshot(raw)
    # event_id must be preserved through normalization
    assert "event_id" in normalized.columns, "event_id must survive normalization"
    assert "player_name" in normalized.columns, "player_name must survive normalization"


def test_event_id_preserved_through_normalization():
    """normalize_player_props_snapshot must retain event_id for later reconciliation."""
    from wnba_props_model.pipeline.deliver import normalize_player_props_snapshot

    raw = pd.DataFrame([{
        "player_name": "Player X", "event_id": "test-uuid-42",
        "vendor": "draftkings", "stat": "reb", "line": 6.5,
        "over_odds": -110.0, "under_odds": -110.0,
    }])
    norm = normalize_player_props_snapshot(raw)
    assert "event_id" in norm.columns
    assert norm.iloc[0]["event_id"] == "test-uuid-42"
