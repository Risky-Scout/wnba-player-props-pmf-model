"""Soft-book +EV scan — Definition B (soft book vs sharp consensus).

This module implements pure market-vs-market line shopping. It does NOT use the
PMF model or any information edge. For each individual book's posted price it asks:
is this price better (for the bettor) than the no-vig fair probability implied by
the *consensus of the other books*? Positive expected value (EV) means yes.

Pipeline per (event, player, stat, line):
  1. For every book that posts BOTH the over and the under at that line, de-vig its
     two-sided price with Shin's method (``shin_no_vig_two_way``) to get that book's
     fair P(over)/P(under).
  2. Build a robust CONSENSUS fair P(over) = the MEDIAN of the per-book fair P(over)
     across the available books. When scoring a specific book, that same book is
     EXCLUDED from the consensus (no self-reference).
  3. For each book+side, EV = fair_p * decimal_profit - (1 - fair_p), where fair_p is
     the consensus fair probability for that side and decimal_profit is the net payout
     multiple of the book's offered American odds. Positive EV = the book's price beats
     the consensus fair line.

Guards (all configurable): require >= ``min_consensus_books`` books in the consensus
(after self-exclusion), require valid |American odds| >= 100, require both sides present
for the scored book, and drop stale rows (commence_time already in the past).

Consensus choice: the consensus is the median of ALL books that post a two-sided price,
not only the "sharp" books. Median-of-all is robust to a single mispriced outlier and
does not require us to hard-code which books are sharp. Known-sharper books (Pinnacle,
BetOnline, LowVig) are ANNOTATED (``is_sharp_book`` on the scored row, and a separate
``sharp_consensus_p_over`` column) so a human can prefer edges that also agree with the
sharp subset, but the qualifying EV is always computed against the median-of-all
consensus. This is documented in ``docs/SOFT_BOOK_EDGE.md`` and echoed into the board
artifact's ``method`` block.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from wnba_props_model.models.market import shin_no_vig_two_way

log = logging.getLogger(__name__)

# Known sharper books (annotation only — consensus is median of ALL books).
SHARP_BOOKS: frozenset[str] = frozenset({"pinnacle", "betonlineag", "lowvig"})

# Defaults
DEFAULT_EV_THRESHOLD = 0.025      # 2.5%
DEFAULT_MIN_CONSENSUS_BOOKS = 3   # books in consensus AFTER self-exclusion
MIN_ABS_AMERICAN_ODDS = 100       # |odds| must be >= 100 (valid American price)

_GROUP_KEYS = ["event_id", "player_name", "stat", "line"]


def american_to_decimal_profit(odds: float | int | None) -> float | None:
    """Net decimal profit multiple ``b`` for one unit staked at American ``odds``.

    +150 -> 1.5 (win 1.5 per unit), -120 -> 0.8333. This is decimal_odds - 1.
    Returns None for missing / invalid (|odds| < 100) prices.
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(o) or abs(o) < MIN_ABS_AMERICAN_ODDS:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def ev_fraction(fair_p: float | None, american_odds: float | int | None) -> float | None:
    """Expected value per unit staked given fair win prob and offered American odds.

    EV = fair_p * decimal_profit - (1 - fair_p).  Positive EV means the offered price
    pays more than the fair probability warrants. Returns None if inputs are invalid.
    """
    if fair_p is None:
        return None
    try:
        p = float(fair_p)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= p <= 1.0):
        return None
    b = american_to_decimal_profit(american_odds)
    if b is None:
        return None
    return p * b - (1.0 - p)


def _valid_odds(odds: float | int | None) -> bool:
    return american_to_decimal_profit(odds) is not None


def _median_excluding(values: dict[str, float], exclude_book: str) -> tuple[float | None, int]:
    """Median of dict values excluding one key. Returns (median, n_used)."""
    kept = [v for b, v in values.items() if b != exclude_book and v is not None]
    if not kept:
        return None, 0
    return float(np.median(kept)), len(kept)


def _per_book_two_sided(group: pd.DataFrame) -> dict[str, dict]:
    """For a (event,player,stat,line) group, collapse to per-book two-sided quotes.

    Returns {book: {"over_odds", "under_odds", "fair_over", "fair_under",
                    "over_last_update", "under_last_update"}} for books that post BOTH
    a valid over and a valid under at this line.
    """
    out: dict[str, dict] = {}
    for book, bdf in group.groupby("book", sort=False):
        over_rows = bdf[bdf["side"] == "over"]
        under_rows = bdf[bdf["side"] == "under"]
        if over_rows.empty or under_rows.empty:
            continue
        # If a book posted the same side twice (rare), take the most recent quote.
        over = over_rows.sort_values("collected_utc").iloc[-1]
        under = under_rows.sort_values("collected_utc").iloc[-1]
        oo, uo = over.get("american_odds"), under.get("american_odds")
        if not (_valid_odds(oo) and _valid_odds(uo)):
            continue
        fair_over, fair_under = shin_no_vig_two_way(oo, uo)
        if fair_over is None or fair_under is None:
            continue
        out[str(book)] = {
            "over_odds": float(oo),
            "under_odds": float(uo),
            "fair_over": float(fair_over),
            "fair_under": float(fair_under),
            "over_last_update": over.get("book_last_update"),
            "under_last_update": under.get("book_last_update"),
        }
    return out


def _best_available(per_book: dict[str, dict], side: str) -> tuple[str | None, float | None]:
    """Best (highest decimal-profit) offered odds for a side across books."""
    key = "over_odds" if side == "over" else "under_odds"
    best_book, best_odds, best_b = None, None, -math.inf
    for book, q in per_book.items():
        b = american_to_decimal_profit(q[key])
        if b is not None and b > best_b:
            best_b, best_book, best_odds = b, book, q[key]
    return best_book, best_odds


def _parse_commence(value) -> datetime | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def scan_soft_book_edges(
    quotes: pd.DataFrame,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    min_consensus_books: int = DEFAULT_MIN_CONSENSUS_BOOKS,
    sharp_books: Iterable[str] = SHARP_BOOKS,
    now: datetime | None = None,
    drop_stale: bool = True,
) -> pd.DataFrame:
    """Scan atomic two-sided prop quotes for soft-book +EV plays (Definition B).

    Parameters
    ----------
    quotes : DataFrame
        Atomic side rows with columns: event_id, commence_time, home_team, away_team,
        book, market_key, stat, player_name, side ('over'/'under'), line, american_odds,
        collected_utc, book_last_update. Extra columns are ignored. ``team`` is used if
        present, else left blank.
    ev_threshold : float
        Minimum EV (fraction) for a row to be flagged ``qualified`` (default 0.025 = 2.5%).
    min_consensus_books : int
        Minimum number of OTHER books required in the consensus (after self-exclusion).
    sharp_books : iterable of str
        Book keys treated as "sharp" for annotation only.
    now : datetime | None
        Reference UTC time for staleness (defaults to datetime.now(timezone.utc)).
    drop_stale : bool
        If True, drop rows whose commence_time is already in the past.

    Returns
    -------
    DataFrame
        One row per scored (event, player, stat, line, side, book) with columns:
        player_name, team, stat, market_key, line, side, book, offered_odds, fair_p,
        ev_pct, ev_frac, consensus_n_books, consensus_p_over, sharp_consensus_p_over,
        is_sharp_book, best_book, best_odds, event_id, home_team, away_team,
        commence_time, qualified. Sorted by ev_frac descending. Empty (typed) frame if
        no rows can be scored.
    """
    cols = [
        "player_name", "team", "stat", "market_key", "line", "side", "book",
        "offered_odds", "fair_p", "ev_pct", "ev_frac", "consensus_n_books",
        "consensus_p_over", "sharp_consensus_p_over", "is_sharp_book",
        "best_book", "best_odds", "event_id", "home_team", "away_team",
        "commence_time", "qualified",
    ]
    if quotes is None or len(quotes) == 0:
        return pd.DataFrame(columns=cols)

    df = quotes.copy()
    # Normalize side + required fields.
    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df = df[df["side"].isin(["over", "under"])]
    df = df[df["stat"].notna() & df["line"].notna() & df["player_name"].notna()]
    df = df[df["american_odds"].apply(_valid_odds)]
    if "collected_utc" not in df.columns:
        df["collected_utc"] = ""
    if "team" not in df.columns:
        df["team"] = ""
    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    ref_now = now or datetime.now(timezone.utc)
    sharp_set = {str(b).lower() for b in sharp_books}

    rows: list[dict] = []
    for (event_id, player_name, stat, line), group in df.groupby(_GROUP_KEYS, sort=False):
        commence = group["commence_time"].iloc[0] if "commence_time" in group.columns else None
        if drop_stale:
            tip = _parse_commence(commence)
            if tip is not None and tip <= ref_now:
                continue

        per_book = _per_book_two_sided(group)
        if len(per_book) < 2:
            # Need at least the scored book + 1 other; real qualification needs more.
            continue

        fair_over_by_book = {b: q["fair_over"] for b, q in per_book.items()}
        home_team = group["home_team"].iloc[0] if "home_team" in group.columns else None
        away_team = group["away_team"].iloc[0] if "away_team" in group.columns else None
        market_key = group["market_key"].iloc[0] if "market_key" in group.columns else None
        team = group["team"].iloc[0] if "team" in group.columns else ""

        # Sharp-only consensus (annotation): median fair P(over) over sharp books present.
        sharp_vals = [v for b, v in fair_over_by_book.items() if b in sharp_set]
        sharp_consensus = float(np.median(sharp_vals)) if sharp_vals else None

        for book, q in per_book.items():
            consensus_p_over, n_other = _median_excluding(fair_over_by_book, book)
            if consensus_p_over is None:
                continue
            for side in ("over", "under"):
                fair_p = consensus_p_over if side == "over" else (1.0 - consensus_p_over)
                offered = q["over_odds"] if side == "over" else q["under_odds"]
                ev = ev_fraction(fair_p, offered)
                if ev is None:
                    continue
                best_book, best_odds = _best_available(per_book, side)
                qualified = bool(ev >= ev_threshold and n_other >= min_consensus_books)
                rows.append({
                    "player_name": player_name,
                    "team": team,
                    "stat": stat,
                    "market_key": market_key,
                    "line": float(line),
                    "side": side,
                    "book": book,
                    "offered_odds": float(offered),
                    "fair_p": round(float(fair_p), 6),
                    "ev_pct": round(float(ev) * 100.0, 4),
                    "ev_frac": round(float(ev), 6),
                    "consensus_n_books": int(n_other),
                    "consensus_p_over": round(float(consensus_p_over), 6),
                    "sharp_consensus_p_over": (
                        round(float(sharp_consensus), 6) if sharp_consensus is not None else None
                    ),
                    "is_sharp_book": book in sharp_set,
                    "best_book": best_book,
                    "best_odds": best_odds,
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": commence,
                    "qualified": qualified,
                })

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    out = out.sort_values("ev_frac", ascending=False, kind="stable").reset_index(drop=True)
    return out
