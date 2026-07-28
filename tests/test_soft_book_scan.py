"""Deterministic unit tests for the soft-book +EV scan (Definition B).

Covers: American→decimal-profit conversion, the EV formula sign, end-to-end flagging
of a genuinely +EV soft-book price, and the "exclude the scored book from consensus"
rule (no self-reference).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.edge.soft_book_scan import (
    american_to_decimal_profit,
    ev_fraction,
    scan_soft_book_edges,
)
from wnba_props_model.models.market import shin_no_vig_two_way

FUTURE = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
PAST = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()


def _row(book, side, odds, *, player="A. Player", stat="pts", line=15.5,
         event="evt1", commence=FUTURE):
    return {
        "collected_utc": "2026-07-28T18:00:00+00:00",
        "event_id": event,
        "commence_time": commence,
        "home_team": "HOME",
        "away_team": "AWAY",
        "book": book,
        "book_last_update": "2026-07-28T17:59:00+00:00",
        "market_key": "player_points",
        "stat": stat,
        "player_name": player,
        "side": side,
        "line": line,
        "american_odds": odds,
    }


# --------------------------------------------------------------------------- #
# American -> decimal profit
# --------------------------------------------------------------------------- #
def test_american_to_decimal_profit_positive():
    assert american_to_decimal_profit(150) == pytest.approx(1.5)
    assert american_to_decimal_profit(100) == pytest.approx(1.0)


def test_american_to_decimal_profit_negative():
    assert american_to_decimal_profit(-120) == pytest.approx(100.0 / 120.0)
    assert american_to_decimal_profit(-110) == pytest.approx(100.0 / 110.0)


def test_american_to_decimal_profit_invalid():
    assert american_to_decimal_profit(None) is None
    assert american_to_decimal_profit(50) is None      # |odds| < 100
    assert american_to_decimal_profit(-99) is None
    assert american_to_decimal_profit(float("nan")) is None


# --------------------------------------------------------------------------- #
# EV formula sign
# --------------------------------------------------------------------------- #
def test_ev_fraction_positive_when_price_beats_fair():
    # Fair 50%, offered +110 -> EV = 0.5*1.1 - 0.5 = +0.05
    assert ev_fraction(0.5, 110) == pytest.approx(0.05)


def test_ev_fraction_negative_when_price_worse_than_fair():
    # Fair 50%, offered -110 -> EV = 0.5*(100/110) - 0.5 < 0
    ev = ev_fraction(0.5, -110)
    assert ev < 0
    assert ev == pytest.approx(0.5 * (100.0 / 110.0) - 0.5)


def test_ev_fraction_zero_at_fair_price():
    # Fair 50%, offered +100 (even money) -> EV = 0
    assert ev_fraction(0.5, 100) == pytest.approx(0.0)


def test_ev_fraction_invalid_inputs():
    assert ev_fraction(None, 110) is None
    assert ev_fraction(0.5, None) is None
    assert ev_fraction(1.5, 110) is None  # prob out of range


# --------------------------------------------------------------------------- #
# End-to-end: a genuinely +EV soft-book over price is flagged
# --------------------------------------------------------------------------- #
def test_soft_book_positive_ev_is_flagged():
    # 3 consensus books priced symmetric -110/-110 -> fair P(over) ~ 0.5 each.
    # 1 soft book offers over at +120 (generous) with a -140 under.
    rows = []
    for bk in ("pinnacle", "betonlineag", "draftkings"):
        rows.append(_row(bk, "over", -110))
        rows.append(_row(bk, "under", -110))
    rows.append(_row("softbook", "over", 120))
    rows.append(_row("softbook", "under", -140))
    board = scan_soft_book_edges(pd.DataFrame(rows), ev_threshold=0.025)

    soft_over = board[(board["book"] == "softbook") & (board["side"] == "over")]
    assert len(soft_over) == 1
    r = soft_over.iloc[0]
    assert r["qualified"]                       # clears 2.5%
    assert r["consensus_n_books"] == 3          # 3 OTHER books
    assert r["ev_frac"] > 0.05                  # ~0.10 given fair~0.5, +120
    # consensus for the soft book excludes itself -> ~0.5 from the three -110 books
    assert r["consensus_p_over"] == pytest.approx(0.5, abs=0.02)


def test_efficient_market_has_no_qualifying_edge():
    # Every book -110/-110: fair ~0.5 vs offered -110 -> EV ~ -4.5% -> nothing qualifies.
    rows = []
    for bk in ("pinnacle", "betonlineag", "draftkings", "fanduel"):
        rows.append(_row(bk, "over", -110))
        rows.append(_row(bk, "under", -110))
    board = scan_soft_book_edges(pd.DataFrame(rows), ev_threshold=0.025)
    assert not board["qualified"].any()


# --------------------------------------------------------------------------- #
# Self-exclusion: consensus for a book never includes that book
# --------------------------------------------------------------------------- #
def test_consensus_excludes_scored_book():
    # 3 books: two priced ~fair 0.5, one skewed toward the under (low fair P(over)).
    rows = [
        _row("book_a", "over", -110), _row("book_a", "under", -110),
        _row("book_b", "over", -110), _row("book_b", "under", -110),
        _row("book_c", "over", 200),  _row("book_c", "under", -260),  # low fair P(over)
    ]
    board = scan_soft_book_edges(pd.DataFrame(rows), ev_threshold=0.025,
                                 min_consensus_books=2)

    # Expected per-book fair P(over) from the same devig function used internally.
    fair = {
        "book_a": shin_no_vig_two_way(-110, -110)[0],
        "book_b": shin_no_vig_two_way(-110, -110)[0],
        "book_c": shin_no_vig_two_way(200, -260)[0],
    }

    # Every scored row must have exactly n_total - 1 consensus books.
    assert (board["consensus_n_books"] == 2).all()

    # When scoring book_c, consensus = median of the OTHER two books (a & b), not c.
    c_row = board[(board["book"] == "book_c") & (board["side"] == "over")].iloc[0]
    expected_c = float(np.median([fair["book_a"], fair["book_b"]]))
    assert c_row["consensus_p_over"] == pytest.approx(expected_c, abs=1e-6)

    # When scoring book_a, consensus = median of b & c (includes the skewed c),
    # which differs from book_c's consensus -> proves self is excluded, not a global mean.
    a_row = board[(board["book"] == "book_a") & (board["side"] == "over")].iloc[0]
    expected_a = float(np.median([fair["book_b"], fair["book_c"]]))
    assert a_row["consensus_p_over"] == pytest.approx(expected_a, abs=1e-6)
    assert a_row["consensus_p_over"] != pytest.approx(c_row["consensus_p_over"], abs=1e-3)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_min_consensus_books_guard_blocks_thin_markets():
    # Only 2 books total -> scoring a book leaves 1 in consensus < default 3 -> no qualify.
    rows = [
        _row("pinnacle", "over", -110), _row("pinnacle", "under", -110),
        _row("softbook", "over", 120),  _row("softbook", "under", -140),
    ]
    board = scan_soft_book_edges(pd.DataFrame(rows), ev_threshold=0.025,
                                 min_consensus_books=3)
    assert not board["qualified"].any()  # consensus_n_books == 1 < 3


def test_stale_events_are_dropped():
    rows = [
        _row("pinnacle", "over", -110, commence=PAST),
        _row("pinnacle", "under", -110, commence=PAST),
        _row("softbook", "over", 120, commence=PAST),
        _row("softbook", "under", -140, commence=PAST),
    ]
    board = scan_soft_book_edges(pd.DataFrame(rows), drop_stale=True)
    assert board.empty


def test_one_sided_book_is_excluded():
    # A book with only an over (no under) cannot be de-vigged and must be dropped.
    rows = [
        _row("book_a", "over", -110), _row("book_a", "under", -110),
        _row("book_b", "over", -110), _row("book_b", "under", -110),
        _row("book_c", "over", -110), _row("book_c", "under", -110),
        _row("softbook", "over", 120),  # only one side
    ]
    board = scan_soft_book_edges(pd.DataFrame(rows), min_consensus_books=2)
    assert "softbook" not in set(board["book"])


def test_empty_input_returns_typed_frame():
    board = scan_soft_book_edges(pd.DataFrame())
    assert board.empty
    assert "qualified" in board.columns
