"""Closing Line Value (CLV) computation — correct implementation.

This module implements three distinct CLV concepts and explicitly separates
them from model-edge-at-entry metrics.

Terminology:
  model_edge_at_entry   — model_prob - entry_quote_no_vig_prob (NOT CLV)
  model_edge_vs_open    — model_prob - opening_quote_no_vig_prob (NOT CLV)
  same_line_price_clv   — closing_no_vig_prob - entry_no_vig_prob (true price CLV)
  line_clv              — directional line movement (over: close - entry, under: entry - close)
  ticket_ev_at_close    — ticket EV using closing probability distribution at entry odds

None of these metrics is available without an archived closing quote.
If no closing quote exists, return NOT_AVAILABLE.

References:
  - CLV is the difference between the closing-market assessment and the bet's entry price.
  - Line movement direction is only a proxy and must not be labeled as CLV.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.pipeline.safety import american_to_implied, no_vig_normalize

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "NOT_AVAILABLE"
STALE_CLOSE = "STALE_CLOSE"
AFTER_TIP = "AFTER_TIP"
VENDOR_MISMATCH = "VENDOR_MISMATCH"

# Maximum seconds between close snapshot and tip-off to accept as "closing" quote
CLOSE_STALENESS_THRESHOLD_SECONDS = 86400  # 24 hours


@dataclass
class CLVResult:
    """Results for a single bet's CLV analysis."""
    game_id: Any = None
    player_id: Any = None
    stat: str = ""
    market_type: str = ""
    vendor: str = ""
    side: str = ""  # "over" or "under"

    entry_line: float = float("nan")
    entry_over_odds: float = float("nan")
    entry_under_odds: float = float("nan")
    entry_no_vig_p_over: float = float("nan")
    entry_no_vig_p_under: float = float("nan")
    entry_pulled_at_utc: str = ""

    opening_line: float = float("nan")
    opening_over_odds: float = float("nan")
    opening_under_odds: float = float("nan")
    opening_no_vig_p_over: float = float("nan")
    opening_no_vig_p_under: float = float("nan")

    closing_line: float = float("nan")
    closing_over_odds: float = float("nan")
    closing_under_odds: float = float("nan")
    closing_no_vig_p_over: float = float("nan")
    closing_no_vig_p_under: float = float("nan")
    closing_pulled_at_utc: str = ""

    scheduled_start_utc: str = ""

    # True CLV (requires closing quote on same line)
    same_line_price_clv: float | str = NOT_AVAILABLE
    line_clv: float | str = NOT_AVAILABLE
    ticket_ev_at_close: float | str = NOT_AVAILABLE

    # Model edge (NOT CLV)
    model_edge_at_entry: float = float("nan")
    model_edge_vs_open: float = float("nan")

    # Diagnostics
    close_validation_status: str = ""
    is_hypothetical: bool = True


def compute_same_line_price_clv(
    entry_no_vig_p_selected: float,
    closing_no_vig_p_selected: float,
) -> float:
    """Compute same-line price CLV.

    Only valid when entry and close use the same line.
    Positive = closing market gives better odds for this bet than at entry.

    CLV = closing_no_vig_p_selected - entry_no_vig_p_selected

    Note: for the bettor, a higher p_selected at close means the market
    moved against them (their bet is worth less). We follow the sign convention
    where POSITIVE = bet became more valuable (market moved in bettor's favor):
      - For over: entry_p_over > closing_p_over means market moved toward under
        (closing is cheaper for over) → bettor got a better price = positive CLV.
    
    Wait — standard convention: CLV > 0 means you bet at better odds than close.
    This means you got LOWER implied probability than the closing market assigned.
    
    Standard: same_line_price_clv = closing_no_vig_p - entry_no_vig_p
    Positive means closing market thinks probability is HIGHER than entry implied,
    which means entry ODDS were higher (better for bettor) than closing odds.
    
    So: CLV = closing_no_vig_p_selected - entry_no_vig_p_selected
    Positive = closing market more confident, meaning entry odds were generous = value.
    """
    if np.isnan(entry_no_vig_p_selected) or np.isnan(closing_no_vig_p_selected):
        return float("nan")
    return closing_no_vig_p_selected - entry_no_vig_p_selected


def compute_line_clv(
    entry_line: float,
    closing_line: float,
    side: str,
) -> float:
    """Compute side-adjusted line CLV.

    For an over ticket: line_clv = closing_line - entry_line
      Positive = line moved up at close = bettor has a better number.

    For an under ticket: line_clv = entry_line - closing_line
      Positive = line moved down at close = bettor has a better number.
    """
    if np.isnan(entry_line) or np.isnan(closing_line):
        return float("nan")
    if side == "over":
        return closing_line - entry_line
    elif side == "under":
        return entry_line - closing_line
    return float("nan")


def validate_closing_quote(
    close_pulled_at: str | datetime | None,
    scheduled_start: str | datetime | None,
    close_source_updated_at: str | datetime | None = None,
) -> str:
    """Validate that a closing quote is acceptable.

    Returns: "valid", AFTER_TIP, STALE_CLOSE, or "unknown".
    """
    if close_pulled_at is None:
        return "unknown"
    if scheduled_start is None:
        return "unknown"

    try:
        if isinstance(close_pulled_at, str):
            close_dt = pd.to_datetime(close_pulled_at, utc=True)
        else:
            close_dt = close_pulled_at.replace(tzinfo=timezone.utc) if close_pulled_at.tzinfo is None else close_pulled_at

        if isinstance(scheduled_start, str):
            start_dt = pd.to_datetime(scheduled_start, utc=True)
        else:
            start_dt = scheduled_start.replace(tzinfo=timezone.utc) if scheduled_start.tzinfo is None else scheduled_start

        # Quote pulled after tip-off
        if close_dt > start_dt:
            return AFTER_TIP

        # Source timestamp check
        if close_source_updated_at is not None:
            if isinstance(close_source_updated_at, str):
                src_dt = pd.to_datetime(close_source_updated_at, utc=True)
            else:
                src_dt = close_source_updated_at
            if src_dt > start_dt:
                return AFTER_TIP

        # Staleness check
        time_before_tip = (start_dt - close_dt).total_seconds()
        if time_before_tip > CLOSE_STALENESS_THRESHOLD_SECONDS:
            return STALE_CLOSE

        return "valid"

    except Exception as exc:
        logger.warning("close validation failed: %s", exc)
        return "unknown"


def select_closing_quote(
    ledger: pd.DataFrame,
    *,
    game_id: Any,
    player_id: Any,
    stat: str,
    market_type: str,
    vendor: str,
    scheduled_start_utc: str | datetime,
) -> pd.Series | None:
    """Select the definitive closing quote from an append-only quote ledger.

    Closing quote = latest valid quote where:
      pulled_at_utc < scheduled_start_utc
      source_updated_at <= scheduled_start_utc (if column exists)

    Returns None if no valid closing quote exists.
    """
    if ledger is None or ledger.empty:
        return None

    mask = (
        (ledger["game_id"] == game_id)
        & (ledger["player_id"] == player_id)
        & (ledger["stat"] == stat)
        & (ledger["vendor"] == vendor)
    )
    if "market_type" in ledger.columns:
        mask &= ledger["market_type"] == market_type

    candidates = ledger[mask].copy()
    if candidates.empty:
        return None

    if isinstance(scheduled_start_utc, str):
        start_dt = pd.to_datetime(scheduled_start_utc, utc=True)
    else:
        start_dt = scheduled_start_utc

    candidates["_pulled_dt"] = pd.to_datetime(candidates["pulled_at_utc"], utc=True, errors="coerce")
    candidates = candidates[candidates["_pulled_dt"] < start_dt]

    if "source_updated_at" in candidates.columns:
        candidates["_src_dt"] = pd.to_datetime(candidates["source_updated_at"], utc=True, errors="coerce")
        candidates = candidates[candidates["_src_dt"] <= start_dt]

    if candidates.empty:
        return None

    return candidates.sort_values("_pulled_dt").iloc[-1]


def compute_clv_for_bet(
    entry_row: dict[str, Any],
    closing_row: dict[str, Any] | None,
    model_p_over: float,
    side: str,
    *,
    scheduled_start_utc: str | None = None,
) -> CLVResult:
    """Compute all CLV metrics for a single bet.

    Parameters
    ----------
    entry_row : dict with entry quote fields (line, over_odds, under_odds, etc.)
    closing_row : dict with closing quote fields, or None if unavailable
    model_p_over : structural model probability of over
    side : "over" or "under"
    scheduled_start_utc : ISO string of game start time
    """
    entry_line = float(entry_row.get("line", float("nan")))
    entry_over_odds = float(entry_row.get("over_odds", float("nan")))
    entry_under_odds = float(entry_row.get("under_odds", float("nan")))

    entry_nv_over, entry_nv_under = float("nan"), float("nan")
    if not (np.isnan(entry_over_odds) or np.isnan(entry_under_odds)):
        p_raw_over = american_to_implied(entry_over_odds)
        p_raw_under = american_to_implied(entry_under_odds)
        entry_nv_over, entry_nv_under = no_vig_normalize(p_raw_over, p_raw_under)

    entry_nv_selected = entry_nv_over if side == "over" else entry_nv_under

    # Model edge at entry (NOT CLV)
    model_edge_at_entry = float("nan")
    if not np.isnan(model_p_over) and not np.isnan(entry_nv_over):
        if side == "over":
            model_edge_at_entry = model_p_over - entry_nv_over
        else:
            model_edge_at_entry = (1.0 - model_p_over) - entry_nv_under

    result = CLVResult(
        game_id=entry_row.get("game_id"),
        player_id=entry_row.get("player_id"),
        stat=entry_row.get("stat", ""),
        market_type=entry_row.get("market_type", ""),
        vendor=entry_row.get("vendor", ""),
        side=side,
        entry_line=entry_line,
        entry_over_odds=entry_over_odds,
        entry_under_odds=entry_under_odds,
        entry_no_vig_p_over=entry_nv_over,
        entry_no_vig_p_under=entry_nv_under,
        entry_pulled_at_utc=str(entry_row.get("pulled_at_utc", "")),
        scheduled_start_utc=scheduled_start_utc or "",
        model_edge_at_entry=model_edge_at_entry,
        is_hypothetical=True,
        same_line_price_clv=NOT_AVAILABLE,
        line_clv=NOT_AVAILABLE,
        ticket_ev_at_close=NOT_AVAILABLE,
    )

    if closing_row is None:
        result.close_validation_status = "no_closing_quote"
        return result

    # Validate closing quote timing
    close_status = validate_closing_quote(
        closing_row.get("pulled_at_utc"),
        scheduled_start_utc,
        closing_row.get("source_updated_at"),
    )
    result.close_validation_status = close_status

    if close_status in (AFTER_TIP, STALE_CLOSE):
        return result

    close_line = float(closing_row.get("line", float("nan")))
    close_over_odds = float(closing_row.get("over_odds", float("nan")))
    close_under_odds = float(closing_row.get("under_odds", float("nan")))

    close_nv_over, close_nv_under = float("nan"), float("nan")
    if not (np.isnan(close_over_odds) or np.isnan(close_under_odds)):
        p_raw_over = american_to_implied(close_over_odds)
        p_raw_under = american_to_implied(close_under_odds)
        close_nv_over, close_nv_under = no_vig_normalize(p_raw_over, p_raw_under)

    result.closing_line = close_line
    result.closing_over_odds = close_over_odds
    result.closing_under_odds = close_under_odds
    result.closing_no_vig_p_over = close_nv_over
    result.closing_no_vig_p_under = close_nv_under
    result.closing_pulled_at_utc = str(closing_row.get("pulled_at_utc", ""))

    # Same-line price CLV (only when lines match)
    close_nv_selected = close_nv_over if side == "over" else close_nv_under
    if not np.isnan(close_line) and not np.isnan(entry_line) and abs(close_line - entry_line) < 0.01:
        result.same_line_price_clv = compute_same_line_price_clv(entry_nv_selected, close_nv_selected)
    else:
        result.same_line_price_clv = NOT_AVAILABLE

    # Line CLV
    if not np.isnan(close_line) and not np.isnan(entry_line):
        result.line_clv = compute_line_clv(entry_line, close_line, side)

    # ticket_ev_at_close requires monotonic closing probability curve — NOT_AVAILABLE
    # without sufficient alternate-line or cross-market data
    result.ticket_ev_at_close = NOT_AVAILABLE

    return result


# ---------------------------------------------------------------------------
# Quote ledger schema (append-only)
# ---------------------------------------------------------------------------

QUOTE_LEDGER_SCHEMA = {
    "snapshot_id": str,         # UUID or hash
    "game_id": "Int64",
    "player_id": "Int64",
    "stat": str,
    "market_type": str,
    "vendor": str,
    "line": float,
    "over_odds": float,
    "under_odds": float,
    "source_updated_at": str,   # ISO UTC from source
    "pulled_at_utc": str,       # when we pulled it
    "scheduled_start_utc": str,
    "is_opening_snapshot": bool,
    "is_current_snapshot": bool,
    "raw_source_reference": str,
}


def create_empty_ledger() -> pd.DataFrame:
    """Create an empty quote ledger with correct schema."""
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in QUOTE_LEDGER_SCHEMA.items()})


def append_to_ledger(
    ledger: pd.DataFrame,
    new_quotes: pd.DataFrame,
    *,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Append new quotes to the ledger.

    Only deduplicates EXACT duplicate observations (same source_updated_at,
    same game/player/stat/vendor/line/odds). Does NOT discard changed prices.
    """
    if new_quotes.empty:
        return ledger

    combined = pd.concat([ledger, new_quotes], ignore_index=True)

    if deduplicate:
        dedup_keys = ["game_id", "player_id", "stat", "vendor", "line",
                      "over_odds", "under_odds", "source_updated_at", "pulled_at_utc"]
        dedup_keys = [k for k in dedup_keys if k in combined.columns]
        combined = combined.drop_duplicates(subset=dedup_keys, keep="first")

    return combined.reset_index(drop=True)
