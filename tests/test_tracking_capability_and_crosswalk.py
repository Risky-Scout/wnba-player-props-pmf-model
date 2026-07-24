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

def test_game_crosswalk_exact_and_failclosed():
    prov = pd.DataFrame({"gameId": ["1", "2", "3"]})
    canon = pd.DataFrame({"game_id": ["1", "2", "3"]})
    out = build_game_crosswalk(prov, canon, min_coverage=0.98)
    assert (out["status"] == "RESOLVED").all()
    # One unmatched -> coverage 2/3 < 0.98 -> fail closed.
    with pytest.raises(CrosswalkCoverageError):
        build_game_crosswalk(pd.DataFrame({"gameId": ["1", "2", "x"]}), canon, min_coverage=0.98)


# ---------------------------------------------------------------- player crosswalk

def test_player_crosswalk_exact_no_fuzzy():
    gcw = build_game_crosswalk(pd.DataFrame({"gameId": ["g1"]}),
                               pd.DataFrame({"game_id": ["g1"]}), min_coverage=0.5)
    prov = pd.DataFrame({"gameId": ["g1", "g1"], "personId": ["101", "102"],
                         "player_name": ["A'ja Wilson", "Kelsey Plum"]})
    canon = pd.DataFrame({"game_id": ["g1", "g1"], "player_id": ["c1", "c2"],
                          "player_name": ["Aja Wilson", "Kelsey Plum"]})
    out = build_player_crosswalk(prov, canon, game_crosswalk=gcw, min_coverage=0.5)
    resolved = out[out["status"] == "RESOLVED"]
    assert set(resolved["canonical_player_id"]) == {"c1", "c2"}   # exact normalized-name match


def test_player_crosswalk_ambiguous_is_not_guessed():
    gcw = build_game_crosswalk(pd.DataFrame({"gameId": ["g1"]}),
                               pd.DataFrame({"game_id": ["g1"]}), min_coverage=0.5)
    prov = pd.DataFrame({"gameId": ["g1"], "personId": ["101"], "player_name": ["Alyssa Thomas"]})
    # Two canonical players share the same normalized name in the same game -> AMBIGUOUS.
    canon = pd.DataFrame({"game_id": ["g1", "g1"], "player_id": ["c1", "c2"],
                          "player_name": ["Alyssa Thomas", "Alyssa Thomas"]})
    out = build_player_crosswalk(prov, canon, game_crosswalk=gcw, min_coverage=0.0)
    assert out.iloc[0]["status"] == "AMBIGUOUS_PLAYER"
    assert out.iloc[0]["canonical_player_id"] is None            # never auto-accepted


def test_player_crosswalk_fails_closed_on_low_coverage():
    gcw = build_game_crosswalk(pd.DataFrame({"gameId": ["g1"]}),
                               pd.DataFrame({"game_id": ["g1"]}), min_coverage=0.5)
    prov = pd.DataFrame({"gameId": ["g1", "g1"], "personId": ["1", "2"],
                         "player_name": ["Known Player", "Ghost Player"]})
    canon = pd.DataFrame({"game_id": ["g1"], "player_id": ["c1"], "player_name": ["Known Player"]})
    with pytest.raises(CrosswalkCoverageError):
        build_player_crosswalk(prov, canon, game_crosswalk=gcw, min_coverage=0.98)


def test_normalize_name_strips_accents_and_suffixes():
    assert normalize_name("A'ja Wilson") == "aja wilson"
    assert normalize_name("Nikola Jokić Jr.") == "nikola jokic"
