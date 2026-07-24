"""A6: validated Over/Under quote pairs (same book/line/snapshot, skew cap, fail-closed)."""
from __future__ import annotations

import pandas as pd

from wnba_props_model.data.quote_pairs import (
    AFTER_DECISION_CUTOFF,
    AMBIGUOUS_PLAYER,
    AT_OR_AFTER_TIP,
    BLOCKED_INVALID_TIP,
    EXACT_PAIR,
    INVALID_ODDS,
    ONE_SIDED,
    SKEW_EXCEEDED,
    build_quote_pairs,
    quote_pair_id,
)

TIP = "2026-07-20T23:00:00Z"
DEC = "2026-07-20T22:00:00Z"


def _side(side, ts="2026-07-20T21:00:00Z", book="fanduel", line=18.5, odds=-110,
          player="p1", event="e1", tip=TIP, dec=DEC):
    return {"provider": "oddsapi", "sportsbook": book, "event_id": event, "player_id": player,
            "prop": "pts", "line": line, "side": side, "snapshot_timestamp": ts,
            "american_odds": odds, "scheduled_tip_utc": tip, "decision_timestamp_utc": dec}


def _status(rows):
    p = build_quote_pairs(pd.DataFrame(rows), max_skew_seconds=120)
    assert len(p) == 1
    return p.iloc[0]["quote_pair_status"]


def test_exact_pair_when_same_book_line_snapshot():
    assert _status([_side("over"), _side("under")]) == EXACT_PAIR


def test_pair_id_deterministic():
    a = quote_pair_id("oddsapi", "fanduel", "e1", "p1", "pts", 18.5, TIP)
    assert a == quote_pair_id("oddsapi", "fanduel", "e1", "p1", "pts", 18.5, TIP)
    assert a != quote_pair_id("oddsapi", "draftkings", "e1", "p1", "pts", 18.5, TIP)


def test_one_sided_rejected():
    assert _status([_side("over")]) == ONE_SIDED


def test_cross_book_never_forms_exact_pair():
    # Two different books -> two groups, each ONE_SIDED (a pair can never cross books).
    p = build_quote_pairs(pd.DataFrame([_side("over", book="fanduel"),
                                        _side("under", book="draftkings")]), max_skew_seconds=120)
    assert set(p["quote_pair_status"]) == {ONE_SIDED}


def test_cross_line_never_forms_exact_pair():
    p = build_quote_pairs(pd.DataFrame([_side("over", line=18.5),
                                        _side("under", line=19.5)]), max_skew_seconds=120)
    assert set(p["quote_pair_status"]) == {ONE_SIDED}


def test_invalid_odds_rejected():
    assert _status([_side("over", odds=0), _side("under")]) == INVALID_ODDS


def test_at_or_after_tip_rejected():
    assert _status([_side("over", ts=TIP), _side("under")]) == AT_OR_AFTER_TIP


def test_after_decision_cutoff_rejected():
    late = "2026-07-20T22:30:00Z"   # after decision cutoff (22:00) but before tip (23:00)
    assert _status([_side("over", ts=late), _side("under")]) == AFTER_DECISION_CUTOFF


def test_skew_exceeded_rejected():
    assert _status([_side("over", ts="2026-07-20T21:00:00Z"),
                    _side("under", ts="2026-07-20T21:10:00Z")]) == SKEW_EXCEEDED


def test_unparseable_tip_is_blocked_not_substituted():
    # No 23:00 UTC substitution: an unparseable tip -> BLOCKED_INVALID_TIP.
    assert _status([_side("over", tip="not-a-date"), _side("under", tip="not-a-date")]) == BLOCKED_INVALID_TIP
    assert _status([_side("over", tip=None), _side("under", tip=None)]) == BLOCKED_INVALID_TIP


def test_ambiguous_player_rejected():
    assert _status([_side("over", player=""), _side("under", player="")]) == AMBIGUOUS_PLAYER
