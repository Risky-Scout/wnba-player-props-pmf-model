"""LANE 2 W2/W3: tracking capability matrix + fail-closed identity crosswalk."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from wnba_props_model.data.identity_crosswalk import (
    CrosswalkCoverageError,
    build_game_crosswalk,
    build_player_crosswalk,
    normalize_name,
)

REPO = Path(__file__).resolve().parent.parent


def _cap():
    spec = importlib.util.spec_from_file_location(
        "cap", REPO / "scripts" / "build_tracking_capability_matrix.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- capability matrix

def test_capability_matrix_classifies_from_real_assets():
    cap = _cap()
    if not cap.TRACKING.exists():
        pytest.skip("tracking asset not fetched in this checkout")
    m = cap.build()
    # Rebound chances + touches + passes are directly available at player+game grain.
    assert m["features"]["reb_chances_total"]["status"] == cap.DIRECT
    assert m["features"]["touches"]["status"] == cap.DIRECT
    assert m["features"]["assist_opportunity_proxy"]["status"] == cap.DERIVABLE
    assert m["features"]["fg3m_attempts"]["status"] == cap.PROXY
    # Hustle is player-degenerate in this extract -> deferred/unavailable.
    assert m["hustle"]["deferred"] is True
    assert m["features"]["deflections"]["status"] == cap.UNAVAILABLE


# ---------------------------------------------------------------- game crosswalk

def _prov_games():
    return pd.DataFrame({
        "gameId": ["p1", "p2"], "season": ["2026", "2026"],
        "gdate": ["2026-07-20", "2026-07-21"],
        "home": ["Aces", "Liberty"], "away": ["Liberty", "Aces"]})


def _canon_games():
    return pd.DataFrame({
        "game_id": ["c1", "c2"], "season": ["2026", "2026"],
        "game_date": ["2026-07-20", "2026-07-21"],
        "home_team": ["Aces", "Liberty"], "away_team": ["Liberty", "Aces"]})


def test_game_crosswalk_exact_date_team():
    # Real matcher: exact (season, date, team-set) - NOT identical-id.
    out = build_game_crosswalk(
        _prov_games(), _canon_games(),
        provider_season_col="season", canonical_season_col="season",
        provider_date_col="gdate", canonical_date_col="game_date",
        provider_home_col="home", provider_away_col="away",
        canonical_home_col="home_team", canonical_away_col="away_team", min_coverage=0.98)
    assert (out["status"] == "RESOLVED").all()
    assert set(out["match_method"]) == {"exact_date_team"}
    assert dict(zip(out["provider_game_id"], out["canonical_game_id"])) == {"p1": "c1", "p2": "c2"}


def test_game_crosswalk_no_silent_identical_id_and_failclosed():
    # Identical ids are NOT auto-resolved without a reviewed map -> fails closed.
    with pytest.raises(CrosswalkCoverageError):
        build_game_crosswalk(pd.DataFrame({"gameId": ["1", "2"]}),
                             pd.DataFrame({"game_id": ["1", "2"]}), min_coverage=0.98)
    # A reviewed id map resolves them explicitly.
    out = build_game_crosswalk(pd.DataFrame({"gameId": ["1", "2"]}),
                               pd.DataFrame({"game_id": ["1", "2"]}),
                               reviewed_id_map={"1": "1", "2": "2"}, min_coverage=0.98)
    assert (out["status"] == "RESOLVED").all()
    assert set(out["match_method"]) == {"reviewed_id_map"}


def test_game_crosswalk_conflict_not_guessed():
    # Two canonical games with the same (date, team-set) -> CONFLICT, never guessed.
    prov = pd.DataFrame({"gameId": ["p1"], "season": ["2026"], "gdate": ["2026-07-20"],
                         "home": ["Aces"], "away": ["Liberty"]})
    canon = pd.DataFrame({"game_id": ["c1", "c2"], "season": ["2026", "2026"],
                          "game_date": ["2026-07-20", "2026-07-20"],
                          "home_team": ["Aces", "Aces"], "away_team": ["Liberty", "Liberty"]})
    out = build_game_crosswalk(
        prov, canon, provider_season_col="season", canonical_season_col="season",
        provider_date_col="gdate", canonical_date_col="game_date",
        provider_home_col="home", provider_away_col="away",
        canonical_home_col="home_team", canonical_away_col="away_team", min_coverage=0.0)
    assert out.iloc[0]["status"] == "CONFLICT_GAME"


# ---------------------------------------------------------------- player crosswalk

def test_player_crosswalk_exact_no_fuzzy():
    gcw = build_game_crosswalk(pd.DataFrame({"gameId": ["g1"]}),
                               pd.DataFrame({"game_id": ["g1"]}),
                               reviewed_id_map={"g1": "g1"}, min_coverage=0.5)
    prov = pd.DataFrame({"gameId": ["g1", "g1"], "personId": ["101", "102"],
                         "player_name": ["A'ja Wilson", "Kelsey Plum"]})
    canon = pd.DataFrame({"game_id": ["g1", "g1"], "player_id": ["c1", "c2"],
                          "player_name": ["Aja Wilson", "Kelsey Plum"]})
    out = build_player_crosswalk(prov, canon, game_crosswalk=gcw, min_coverage=0.5)
    resolved = out[out["status"] == "RESOLVED"]
    assert set(resolved["canonical_player_id"]) == {"c1", "c2"}   # exact normalized-name match


def test_player_crosswalk_ambiguous_is_not_guessed():
    gcw = build_game_crosswalk(pd.DataFrame({"gameId": ["g1"]}),
                               pd.DataFrame({"game_id": ["g1"]}),
                               reviewed_id_map={"g1": "g1"}, min_coverage=0.5)
    prov = pd.DataFrame({"gameId": ["g1"], "personId": ["101"], "player_name": ["Alyssa Thomas"]})
    # Two canonical players share the same normalized name in the same game -> AMBIGUOUS.
    canon = pd.DataFrame({"game_id": ["g1", "g1"], "player_id": ["c1", "c2"],
                          "player_name": ["Alyssa Thomas", "Alyssa Thomas"]})
    out = build_player_crosswalk(prov, canon, game_crosswalk=gcw, min_coverage=0.0)
    assert out.iloc[0]["status"] == "AMBIGUOUS_PLAYER"
    assert out.iloc[0]["canonical_player_id"] is None            # never auto-accepted


def test_player_crosswalk_fails_closed_on_low_coverage():
    gcw = build_game_crosswalk(pd.DataFrame({"gameId": ["g1"]}),
                               pd.DataFrame({"game_id": ["g1"]}),
                               reviewed_id_map={"g1": "g1"}, min_coverage=0.5)
    prov = pd.DataFrame({"gameId": ["g1", "g1"], "personId": ["1", "2"],
                         "player_name": ["Known Player", "Ghost Player"]})
    canon = pd.DataFrame({"game_id": ["g1"], "player_id": ["c1"], "player_name": ["Known Player"]})
    with pytest.raises(CrosswalkCoverageError):
        build_player_crosswalk(prov, canon, game_crosswalk=gcw, min_coverage=0.98)


def test_normalize_name_strips_accents_and_suffixes():
    assert normalize_name("A'ja Wilson") == "aja wilson"
    assert normalize_name("Nikola Jokić Jr.") == "nikola jokic"
