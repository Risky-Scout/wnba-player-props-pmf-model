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
from collections.abc import Iterable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wnba_props_model.models.market import shin_no_vig_two_way

log = logging.getLogger(__name__)

# Known sharper books (annotation only — consensus is median of ALL books).
SHARP_BOOKS: frozenset[str] = frozenset({"pinnacle", "betonlineag", "lowvig"})

# Approved sharp / reference book set used for the consensus-quality check (req 5):
# we record whether the leave-one-out consensus actually included any of these, so a
# consensus made only of soft books is never mistaken for sharp truth.
REFERENCE_BOOKS: frozenset[str] = SHARP_BOOKS

# This scan is a MARKET DISLOCATION detector, not a model edge. Every emitted row is
# stamped with this and defaults to actionable=False.
SOURCE_TYPE_MARKET_DISLOCATION = "MARKET_DISLOCATION"

# Defaults
DEFAULT_EV_THRESHOLD = 0.025      # 2.5%
DEFAULT_MIN_CONSENSUS_BOOKS = 3   # books in consensus AFTER self-exclusion
MIN_ABS_AMERICAN_ODDS = 100       # |odds| must be >= 100 (valid American price)

# Rejection reason vocabulary (persisted for the acceptance-gate audit).
REJ_UNRESOLVED_IDENTITY = "unresolved_identity"
REJ_POST_TIP = "post_tip_or_stale_event"
REJ_STALE_QUOTE = "stale_quote_age"
REJ_MALFORMED_TIMESTAMP = "malformed_timestamp"
REJ_MISSING_OPPOSITE_SIDE = "missing_opposite_side_no_vig_fail_closed"
REJ_INVALID_ODDS = "invalid_american_odds"
REJ_DEVIG_FAILED = "no_vig_extraction_failed"
REJ_INSUFFICIENT_CONSENSUS = "insufficient_consensus_books"

# Atomic segregation: standard vs alternate markets must never be compared. We group on
# market_key (not just stat) so ``player_points`` and ``player_points_alternate`` at the
# same line are never mixed into one consensus.
_GROUP_KEYS = ["event_id", "player_name", "market_key", "line"]


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


def _per_book_two_sided(group: pd.DataFrame) -> tuple[dict[str, dict], list[dict]]:
    """For an atomic (event,player,market_key,line) group, collapse to per-book two-sided
    quotes and record why any book was excluded (no-vig fail-closed audit trail).

    Returns ``(per_book, rejections)`` where ``per_book`` is
    ``{book: {"over_odds", "under_odds", "fair_over", "fair_under",
              "over_last_update", "under_last_update"}}`` for books posting BOTH a valid
    over and a valid under at this line, and ``rejections`` is a list of
    ``{"book", "reason"}`` for books dropped (missing opposite side, invalid odds, or a
    failed no-vig extraction). Missing-opposite-side is a HARD fail-closed: we never
    normalize a one-sided quote.
    """
    out: dict[str, dict] = {}
    rejections: list[dict] = []
    for book, bdf in group.groupby("book", sort=False):
        over_rows = bdf[bdf["side"] == "over"]
        under_rows = bdf[bdf["side"] == "under"]
        if over_rows.empty or under_rows.empty:
            # Fail closed: a single-outcome prop is never de-vigged/normalized here.
            rejections.append({"book": str(book), "reason": REJ_MISSING_OPPOSITE_SIDE})
            continue
        # If a book posted the same side twice (rare), take the most recent quote.
        over = over_rows.sort_values("collected_utc").iloc[-1]
        under = under_rows.sort_values("collected_utc").iloc[-1]
        oo, uo = over.get("american_odds"), under.get("american_odds")
        if not (_valid_odds(oo) and _valid_odds(uo)):
            rejections.append({"book": str(book), "reason": REJ_INVALID_ODDS})
            continue
        fair_over, fair_under = shin_no_vig_two_way(oo, uo)
        if fair_over is None or fair_under is None:
            rejections.append({"book": str(book), "reason": REJ_DEVIG_FAILED})
            continue
        out[str(book)] = {
            "over_odds": float(oo),
            "under_odds": float(uo),
            "fair_over": float(fair_over),
            "fair_under": float(fair_under),
            "over_last_update": over.get("book_last_update"),
            "under_last_update": under.get("book_last_update"),
        }
    return out, rejections


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


# Timestamp parsing is identical for provider/ingestion/commence stamps.
_parse_ts = _parse_commence


def _quote_age_seconds(provider_ts, now: datetime) -> float | None:
    """Age in seconds of a provider quote timestamp vs the scan time.

    Returns None (malformed) if the timestamp cannot be parsed. Negative ages (a
    provider clock slightly ahead of ours) are clamped to 0.0.
    """
    dt = _parse_ts(provider_ts)
    if dt is None:
        return None
    return max(0.0, (now - dt).total_seconds())


def _dispersion(values: list[float]) -> tuple[float | None, float | None]:
    """Population stdev and inter-quartile range of the consensus fair-prob set."""
    if not values or len(values) < 2:
        return (0.0 if values else None), (0.0 if values else None)
    arr = np.asarray(values, dtype=float)
    stdev = float(np.std(arr))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    return stdev, iqr


def scan_soft_book_edges(
    quotes: pd.DataFrame,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    min_consensus_books: int = DEFAULT_MIN_CONSENSUS_BOOKS,
    sharp_books: Iterable[str] = SHARP_BOOKS,
    now: datetime | None = None,
    drop_stale: bool = True,
    *,
    reference_books: Iterable[str] | None = None,
    max_quote_age_seconds: float | None = None,
    identity_index: dict[str, set] | None = None,
    require_identity: bool = False,
    return_rejections: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
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
        If True, drop (and record as rejected) rows whose commence_time is past.
    reference_books : iterable of str | None
        Approved sharp/reference book set for the consensus-quality check (req 5). We
        record whether the leave-one-out consensus included any of these. Defaults to
        REFERENCE_BOOKS (== SHARP_BOOKS).
    max_quote_age_seconds : float | None
        Strict live-age threshold. Quotes whose provider timestamp is older than this
        are rejected (``stale_quote_age``). None disables the age gate (still records age).
    identity_index : dict[str, set] | None
        ``{normalized_name: {canonical_player_id}}`` used to resolve player_name -> a
        canonical player_id (exact, fail-closed). None => player_id is unresolved.
    require_identity : bool
        If True, groups whose player_name does not resolve to exactly one canonical id
        are rejected (``unresolved_identity``) and never scored.
    return_rejections : bool
        If True, return ``(board, rejections_df)`` where rejections_df carries one row per
        rejected atomic unit with a reason.

    Returns
    -------
    DataFrame (or (DataFrame, DataFrame) when return_rejections=True)
        One row per scored (event, player, market_key, line, side, book) with full
        MARKET_DISLOCATION provenance. Sorted by ev_frac descending. Empty typed frame if
        no rows can be scored. Every row carries source_type=MARKET_DISLOCATION and
        actionable=False.
    """
    cols = [
        "player_name", "team", "stat", "market_key", "is_alternate_market", "line",
        "side", "book", "bookmaker", "offered_odds", "displayed_odds", "player_id",
        "player_id_resolved", "fair_p", "reference_p", "ev_pct", "ev_frac",
        "theoretical_ev_pct", "theoretical_ev_frac", "executable_ev_pct",
        "price_survived_30s", "price_survived_60s", "consensus_n_books",
        "consensus_p_over", "consensus_books", "consensus_market_key",
        "consensus_dispersion_stdev", "consensus_dispersion_iqr",
        "consensus_includes_sharp", "self_excluded", "sharp_consensus_p_over",
        "is_sharp_book", "best_book", "best_odds", "event_id", "home_team", "away_team",
        "commence_time", "scheduled_tip", "provider_timestamp", "ingestion_timestamp",
        "scan_timestamp", "quote_age_seconds", "validation_status", "actionable",
        "rejection_reason", "warning_reason", "source_type", "qualified",
    ]
    rej_cols = [
        "event_id", "player_name", "player_id", "market_key", "line", "book", "side",
        "reason", "scan_timestamp",
    ]
    empty_board = pd.DataFrame(columns=cols)
    empty_rej = pd.DataFrame(columns=rej_cols)
    if quotes is None or len(quotes) == 0:
        return (empty_board, empty_rej) if return_rejections else empty_board

    ref_now = now or datetime.now(timezone.utc)
    scan_ts = ref_now.isoformat()
    sharp_set = {str(b).lower() for b in sharp_books}
    ref_set = {str(b).lower() for b in (reference_books if reference_books is not None
                                        else REFERENCE_BOOKS)}
    rejections: list[dict] = []

    def _reject(reason, *, event_id=None, player_name=None, player_id=None,
                market_key=None, line=None, book=None, side=None) -> None:
        rejections.append({
            "event_id": event_id, "player_name": player_name, "player_id": player_id,
            "market_key": market_key, "line": line, "book": book, "side": side,
            "reason": reason, "scan_timestamp": scan_ts,
        })

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
    if "market_key" not in df.columns:
        df["market_key"] = None
    if len(df) == 0:
        return (empty_board, empty_rej) if return_rejections else empty_board

    rows: list[dict] = []
    for (event_id, player_name, market_key, line), group in df.groupby(_GROUP_KEYS, sort=False):
        stat = group["stat"].iloc[0] if "stat" in group.columns else None
        commence = group["commence_time"].iloc[0] if "commence_time" in group.columns else None
        is_alt = bool(str(market_key or "").endswith("_alternate"))

        # Requirement 1: exact identity. Resolve name -> canonical player_id, fail closed.
        player_id, id_resolved = None, False
        if identity_index is not None:
            from wnba_props_model.edge.prop_identity import (  # noqa: PLC0415
                STATUS_RESOLVED,
                resolve_player_id,
            )
            player_id, id_status = resolve_player_id(player_name, identity_index)
            id_resolved = id_status == STATUS_RESOLVED
        if require_identity and not id_resolved:
            _reject(REJ_UNRESOLVED_IDENTITY, event_id=event_id, player_name=player_name,
                    market_key=market_key, line=line)
            continue

        # Requirement 4: timestamp integrity — reject post-tip / stale events.
        if drop_stale:
            tip = _parse_commence(commence)
            if tip is not None and tip <= ref_now:
                _reject(REJ_POST_TIP, event_id=event_id, player_name=player_name,
                        player_id=player_id, market_key=market_key, line=line)
                continue

        per_book, book_rejections = _per_book_two_sided(group)
        for br in book_rejections:
            _reject(br["reason"], event_id=event_id, player_name=player_name,
                    player_id=player_id, market_key=market_key, line=line,
                    book=br["book"])
        if len(per_book) < 2:
            # Need at least the scored book + 1 other; real qualification needs more.
            continue

        fair_over_by_book = {b: q["fair_over"] for b, q in per_book.items()}
        home_team = group["home_team"].iloc[0] if "home_team" in group.columns else None
        away_team = group["away_team"].iloc[0] if "away_team" in group.columns else None
        team = group["team"].iloc[0] if "team" in group.columns else ""

        # Sharp-only consensus (annotation): median fair P(over) over sharp books present.
        sharp_vals = [v for b, v in fair_over_by_book.items() if b in sharp_set]
        sharp_consensus = float(np.median(sharp_vals)) if sharp_vals else None

        for book, q in per_book.items():
            consensus_p_over, n_other = _median_excluding(fair_over_by_book, book)
            if consensus_p_over is None:
                continue
            other_books = [b for b in fair_over_by_book if b != book]
            other_vals = [fair_over_by_book[b] for b in other_books]
            disp_stdev, disp_iqr = _dispersion(other_vals)
            consensus_includes_sharp = any(b in ref_set for b in other_books)
            for side in ("over", "under"):
                fair_p = consensus_p_over if side == "over" else (1.0 - consensus_p_over)
                offered = q["over_odds"] if side == "over" else q["under_odds"]
                ev = ev_fraction(fair_p, offered)
                if ev is None:
                    continue
                best_book, best_odds = _best_available(per_book, side)
                provider_ts = (q["over_last_update"] if side == "over"
                               else q["under_last_update"])
                age = _quote_age_seconds(provider_ts, ref_now)

                warning_reason = None
                rejection_reason = None
                # Requirement 4: strict live-age gate.
                if provider_ts is None:
                    warning_reason = REJ_MALFORMED_TIMESTAMP
                elif age is None:
                    warning_reason = REJ_MALFORMED_TIMESTAMP
                elif max_quote_age_seconds is not None and age > max_quote_age_seconds:
                    rejection_reason = REJ_STALE_QUOTE
                # Requirement 5: consensus quality (min independent books after self-excl).
                if n_other < min_consensus_books and rejection_reason is None:
                    warning_reason = REJ_INSUFFICIENT_CONSENSUS

                if rejection_reason is not None:
                    _reject(rejection_reason, event_id=event_id, player_name=player_name,
                            player_id=player_id, market_key=market_key, line=line,
                            book=book, side=side)
                    validation_status = "REJECTED"
                else:
                    validation_status = "PENDING_VALIDATION"

                qualified = bool(
                    ev >= ev_threshold
                    and n_other >= min_consensus_books
                    and rejection_reason is None
                )
                ev_pct = round(float(ev) * 100.0, 4)
                row = {
                    "player_name": player_name,
                    "team": team,
                    "stat": stat,
                    "market_key": market_key,
                    "is_alternate_market": is_alt,
                    "line": float(line),
                    "side": side,
                    "book": book,
                    "bookmaker": book,
                    "offered_odds": float(offered),
                    "displayed_odds": float(offered),
                    "player_id": player_id,
                    "player_id_resolved": bool(id_resolved),
                    "fair_p": round(float(fair_p), 6),
                    "reference_p": round(float(fair_p), 6),
                    "ev_pct": ev_pct,
                    "ev_frac": round(float(ev), 6),
                    "theoretical_ev_pct": ev_pct,
                    "theoretical_ev_frac": round(float(ev), 6),
                    # EXECUTABLE_EV is unknown until a forward price-survival recheck runs.
                    "executable_ev_pct": None,
                    "price_survived_30s": None,
                    "price_survived_60s": None,
                    "consensus_n_books": int(n_other),
                    "consensus_p_over": round(float(consensus_p_over), 6),
                    "consensus_books": list(other_books),
                    "consensus_market_key": market_key,
                    "consensus_dispersion_stdev": (
                        round(disp_stdev, 6) if disp_stdev is not None else None
                    ),
                    "consensus_dispersion_iqr": (
                        round(disp_iqr, 6) if disp_iqr is not None else None
                    ),
                    "consensus_includes_sharp": bool(consensus_includes_sharp),
                    "self_excluded": True,
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
                    "scheduled_tip": commence,
                    "provider_timestamp": provider_ts,
                    "ingestion_timestamp": group["collected_utc"].iloc[0],
                    "scan_timestamp": scan_ts,
                    "quote_age_seconds": age,
                    "validation_status": validation_status,
                    # Requirement 8: actionable stays False during the validation period.
                    "actionable": False,
                    "rejection_reason": rejection_reason,
                    "warning_reason": warning_reason,
                    "source_type": SOURCE_TYPE_MARKET_DISLOCATION,
                    "qualified": qualified,
                }
                rows.append(row)

    rej_df = pd.DataFrame(rejections, columns=rej_cols)
    if not rows:
        return (empty_board, rej_df) if return_rejections else empty_board
    out = pd.DataFrame(rows, columns=cols)
    out = out.sort_values("ev_frac", ascending=False, kind="stable").reset_index(drop=True)
    return (out, rej_df) if return_rejections else out
