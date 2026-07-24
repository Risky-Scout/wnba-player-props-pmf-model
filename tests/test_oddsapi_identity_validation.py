"""Regression tests for Odds API provider identity validation.

Updated for the two-stage identity contract:
- Stage 1 uses validate_provider_quotes() (provider-native columns)
- Stage 2 uses validate_player/game_identity_resolved() (canonical, post-join only)
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


# ─── Stage 1 canonical validators now require canonical IDs ──────────────────

def test_validate_player_identity_requires_player_id_column():
    """Stage 2 canonical validator requires player_id — not player_name."""
    df = pd.DataFrame([{"player_name": "X", "stat": "pts"}])
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_player_identity_resolved(df)


def test_validate_player_identity_requires_nonblank_player_id():
    df = pd.DataFrame([{"player_id": None, "stat": "pts"}])
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_player_identity_resolved(df)


def test_validate_player_identity_passes_with_canonical_player_id():
    df = pd.DataFrame([{"player_id": 735, "stat": "pts"}])
    validate_player_identity_resolved(df)  # must not raise


def test_validate_game_identity_requires_game_id_column():
    """Stage 2 canonical validator requires game_id — not event_id."""
    df = pd.DataFrame([{"event_id": "uuid-1", "stat": "pts"}])
    with pytest.raises(UnmatchedIdentityError, match="game_id"):
        validate_game_identity_resolved(df)


def test_validate_game_identity_requires_nonblank_game_id():
    df = pd.DataFrame([{"game_id": None, "stat": "pts"}])
    with pytest.raises(UnmatchedIdentityError, match="game_id"):
        validate_game_identity_resolved(df)


def test_validate_game_identity_passes_with_canonical_game_id():
    df = pd.DataFrame([{"game_id": 24931, "stat": "pts"}])
    validate_game_identity_resolved(df)  # must not raise


# ─── Stage 1 provider-native validators ──────────────────────────────────────

def test_provider_quotes_oddsapi_passes():
    df = pd.DataFrame([{
        "vendor": "fanduel", "event_id": "uuid-1", "player_name": "Player A",
        "stat": "pts", "line": 18.5, "over_odds": -115.0, "under_odds": -105.0
    }])
    validate_provider_quotes(df, source="oddsapi")  # must not raise


def test_provider_quotes_oddsapi_blank_player_name_fatal():
    df = pd.DataFrame([{
        "vendor": "fanduel", "event_id": "uuid-1", "player_name": "",
        "stat": "pts", "line": 18.5
    }])
    with pytest.raises(UnmatchedIdentityError, match="player_name"):
        validate_provider_quotes(df, source="oddsapi")


def test_provider_quotes_oddsapi_blank_event_id_fatal():
    df = pd.DataFrame([{
        "vendor": "fanduel", "event_id": "", "player_name": "Player A",
        "stat": "pts", "line": 18.5
    }])
    with pytest.raises(UnmatchedIdentityError, match="event_id"):
        validate_provider_quotes(df, source="oddsapi")


def test_provider_quotes_bdl_passes():
    df = pd.DataFrame([{"game_id": 24931, "player_id": 735, "stat": "pts"}])
    validate_provider_quotes(df, source="bdl")  # must not raise


def test_provider_quotes_bdl_null_player_id_fatal():
    df = pd.DataFrame([{"game_id": 24931, "player_id": None, "stat": "pts"}])
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_provider_quotes(df, source="bdl")


# ─── Source-alias normalization (regression for the "odds_api" routing gap) ──────

@pytest.mark.parametrize("alias", ["odds_api", "odds_api_v4", "oddsapi"])
def test_provider_quotes_oddsapi_aliases_route_to_validation(alias):
    """`_load_props` returns "odds_api"; policies use "odds_api_v4". All Odds API
    aliases must reach provider-native validation (previously "odds_api" silently
    skipped it)."""
    df = pd.DataFrame([{
        "vendor": "fanduel", "event_id": "", "player_name": "Player A",
        "stat": "pts", "line": 18.5,
    }])
    with pytest.raises(UnmatchedIdentityError, match="event_id"):
        validate_provider_quotes(df, source=alias)


def test_provider_quotes_oddsapi_accepts_game_id_without_event_id():
    """Reconciled Odds API rows may carry game_id instead of event_id."""
    df = pd.DataFrame([{
        "vendor": "fanduel", "game_id": 24931, "player_name": "Player A",
        "stat": "pts", "line": 18.5,
    }])
    validate_provider_quotes(df, source="odds_api")  # must not raise


def test_provider_quotes_odds_api_then_bdl_routes_to_bdl():
    df = pd.DataFrame([{"game_id": 24931, "player_id": None, "stat": "pts"}])
    with pytest.raises(UnmatchedIdentityError, match="player_id"):
        validate_provider_quotes(df, source="odds_api_then_bdl")
