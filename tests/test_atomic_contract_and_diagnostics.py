"""Section 4/5 tests: canonical atomic contract (collector row -> pair builder with no manual
renaming) and the separate counterpart-rejection diagnostic (CROSS_BOOK/CROSS_LINE etc.)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from wnba_props_model.data.atomic_quotes import (
    AMBIGUOUS_PLAYER,
    CROSS_BOOK_COUNTERPART_ONLY,
    CROSS_LINE_COUNTERPART_ONLY,
    DUPLICATE_SIDE,
    HAS_EXACT_COUNTERPART,
    ONE_SIDED,
    counterpart_rejection_audit,
    to_raw_side_snapshots,
)
from wnba_props_model.data.quote_pairs import EXACT_PAIR, build_quote_pairs

_SPEC = importlib.util.spec_from_file_location(
    "backfill_hist", Path(__file__).resolve().parent.parent / "scripts" / "backfill_historical_quotes.py")
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)


def _odds_payload(mkt_last="2024-08-20T10:55:00Z"):
    """A minimal historical event-odds response with one book, one market, over+under."""
    return {
        "timestamp": "2024-08-20T10:57:00Z",
        "previous_timestamp": "2024-08-20T10:52:00Z",
        "next_timestamp": "2024-08-20T11:02:00Z",
        "data": {"bookmakers": [{
            "key": "draftkings", "last_update": mkt_last,
            "markets": [{"key": "player_points", "last_update": mkt_last, "outcomes": [
                {"name": "Over", "description": "A Player", "point": 15.5, "price": -110},
                {"name": "Under", "description": "A Player", "point": 15.5, "price": -110},
            ]}],
        }]},
    }


def test_collector_row_enters_pair_builder_without_manual_renaming():
    roster = pd.DataFrame([{"game_id": "g1", "player_id": 1, "player_name": "A Player"}])
    rows = backfill._parse_event_odds(
        _odds_payload(), requested_snap="2024-08-20T11:00:00Z", role="decision",
        decision_cut="2024-08-20T11:00:00Z", closing_cut="2024-08-20T22:55:00Z",
        tip_iso="2024-08-20T23:00:00Z", event_id="e1", gid="g1", roster_df=roster,
        collection_ts="2024-08-20T12:00:00Z")
    atomic = pd.DataFrame(rows)
    # both sides inherit the SAME market object's market_last_update
    assert set(atomic["market_last_update_utc"]) == {"2024-08-20T10:55:00Z"}
    assert (atomic["quote_timestamp_utc"] == "2024-08-20T10:55:00Z").all()
    # straight into the pair builder via the adapter -> EXACT_PAIR, no manual editing
    raw = to_raw_side_snapshots(atomic)
    pairs = build_quote_pairs(raw, snapshot_label="decision")
    assert (pairs["quote_pair_status"] == EXACT_PAIR).all()


def test_missing_market_timestamp_is_blocked_not_fabricated():
    roster = pd.DataFrame([{"game_id": "g1", "player_id": 1, "player_name": "A Player"}])
    rows = backfill._parse_event_odds(
        _odds_payload(mkt_last=None), requested_snap="2024-08-20T11:00:00Z", role="decision",
        decision_cut="2024-08-20T11:00:00Z", closing_cut="2024-08-20T22:55:00Z",
        tip_iso="2024-08-20T23:00:00Z", event_id="e1", gid="g1", roster_df=roster,
        collection_ts="2024-08-20T12:00:00Z")
    atomic = pd.DataFrame(rows)
    assert (atomic["quote_timestamp_source"] == "BLOCKED_NO_MARKET_TIMESTAMP").all()
    assert atomic["quote_timestamp_utc"].isna().all()
    assert (atomic["exact_quote_status"] == "BLOCKED_EXACT_QUOTES").all()
    # blocked rows never reach pairing
    assert len(to_raw_side_snapshots(atomic)) == 0


# --- counterpart-rejection diagnostic ---------------------------------------------
def _side(book, side, line, pid="p1"):
    return {"event_id": "e1", "sportsbook": book, "player_id": pid, "prop": "pts",
            "line": line, "side": side, "snapshot_role": "decision"}


def test_counterpart_audit_classifies_each_case():
    # each scenario uses a DISTINCT player so counterpart search cannot cross-contaminate
    df = pd.DataFrame([
        _side("dk", "over", 15.5, pid="p1"), _side("dk", "under", 15.5, pid="p1"),   # exact counterpart
        _side("fd", "over", 20.5, pid="p4"),                                          # one-sided (no under)
        _side("bm", "over", 10.5, pid="p5"), _side("cs", "under", 10.5, pid="p5"),    # cross-book only
        _side("dk", "over", 8.5, pid="p2"), _side("dk", "under", 9.5, pid="p2"),      # cross-line only
        _side("dk", "over", 5.5, pid="p3"), _side("dk", "over", 5.5, pid="p3"),       # duplicate side
        {**_side("dk", "over", 3.5, pid="p6"), "player_id": None},                    # ambiguous player
    ])
    out = counterpart_rejection_audit(df)
    by_player = dict(zip(out["player_id"].astype(str), out["counterpart_status"]))
    st = set(out["counterpart_status"])
    assert by_player["p1"] == HAS_EXACT_COUNTERPART
    assert by_player["p4"] == ONE_SIDED
    assert by_player["p5"] == CROSS_BOOK_COUNTERPART_ONLY
    assert by_player["p2"] == CROSS_LINE_COUNTERPART_ONLY
    assert {DUPLICATE_SIDE, AMBIGUOUS_PLAYER} <= st


def test_multiple_legitimate_lines_are_not_flagged_when_each_has_its_pair():
    # same player, two DIFFERENT lines, each fully paired at the same book -> both EXACT,
    # never a cross-line error.
    df = pd.DataFrame([
        _side("dk", "over", 15.5), _side("dk", "under", 15.5),
        _side("dk", "over", 16.5), _side("dk", "under", 16.5),
    ])
    out = counterpart_rejection_audit(df)
    assert (out["counterpart_status"] == HAS_EXACT_COUNTERPART).all()
