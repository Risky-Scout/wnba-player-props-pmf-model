"""W0.7: atomic quote store - append-only, no book averaging, BLOCKED_EXACT_QUOTES."""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.data.atomic_quotes import (
    ATOMIC_QUOTE_COLUMNS,
    BLOCKED_EXACT_QUOTES,
    EXACT,
    append_atomic_quotes,
    assert_no_book_averaging,
    atomic_quote_id,
)


def _row(book="fanduel", side="over", snap="2026-07-20T18:00:00Z", status=EXACT):
    line = 18.5
    return {c: None for c in ATOMIC_QUOTE_COLUMNS} | {
        "quote_id": atomic_quote_id(book, "evt1", "p1", "pts", line, side, snap),
        "sportsbook": book, "event_id": "evt1", "player_id": "p1", "prop": "pts",
        "line": line, "side": side, "american_odds": -110, "snapshot_label": "decision",
        "snapshot_time": snap, "exact_quote_status": status, "settlement_status": "pending",
    }


def test_quote_id_is_deterministic_and_book_specific():
    a = atomic_quote_id("fanduel", "e", "p", "pts", 18.5, "over", "t")
    assert a == atomic_quote_id("fanduel", "e", "p", "pts", 18.5, "over", "t")
    assert a != atomic_quote_id("draftkings", "e", "p", "pts", 18.5, "over", "t")  # per book


def test_no_book_averaging_rejects_consensus_rows():
    df = pd.DataFrame([_row(book="consensus")])
    with pytest.raises(ValueError):
        assert_no_book_averaging(df)
    df2 = pd.DataFrame([_row(book="")])
    with pytest.raises(ValueError):
        assert_no_book_averaging(df2)


def test_append_is_append_only_and_deduplicates(tmp_path):
    store = tmp_path / "atomic.parquet"
    df1 = pd.DataFrame([_row(book="fanduel"), _row(book="draftkings")])
    s1 = append_atomic_quotes(store, df1)
    assert s1["added"] == 2 and s1["total"] == 2
    # Re-append the same rows + one new -> only the new one is added; existing untouched.
    df2 = pd.DataFrame([_row(book="fanduel"), _row(book="betmgm")])
    s2 = append_atomic_quotes(store, df2)
    assert s2["added"] == 1 and s2["total"] == 3
    stored = pd.read_parquet(store)
    assert stored["quote_id"].is_unique
    assert set(stored["sportsbook"]) == {"fanduel", "draftkings", "betmgm"}


def test_existing_quotes_are_immutable(tmp_path):
    store = tmp_path / "atomic.parquet"
    append_atomic_quotes(store, pd.DataFrame([_row(book="fanduel")]))
    # A row with the SAME quote_id but a different price must NOT overwrite the original.
    dup = _row(book="fanduel"); dup["american_odds"] = 999
    append_atomic_quotes(store, pd.DataFrame([dup]))
    stored = pd.read_parquet(store)
    assert len(stored) == 1
    assert int(stored.iloc[0]["american_odds"]) == -110  # original preserved


def test_blocked_exact_quotes_status_round_trips(tmp_path):
    store = tmp_path / "atomic.parquet"
    append_atomic_quotes(store, pd.DataFrame([_row(status=BLOCKED_EXACT_QUOTES)]))
    stored = pd.read_parquet(store)
    assert (stored["exact_quote_status"] == BLOCKED_EXACT_QUOTES).all()
