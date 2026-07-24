"""A6 - validated Over/Under quote PAIRS built from immutable raw side snapshots.

A raw side snapshot is one book's single side (over OR under) at one provider timestamp. A
canonical PAIR joins the two sides of the SAME market and is EXACT_PAIR only when both sides
share provider, sportsbook, event, player, prop, line, and a permitted snapshot window within
a configurable maximum timestamp skew. Everything else is rejected with an explicit status.

We NEVER infer a tip time: if the scheduled tip cannot be parsed the pair is BLOCKED_INVALID_TIP
(no 23:00 UTC substitution).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Raw side key (one immutable snapshot of a single side).
RAW_SIDE_KEY = ["provider", "sportsbook", "event_id", "player_id", "prop", "line", "side",
                "snapshot_timestamp"]

PAIR_COLUMNS = [
    "quote_pair_id", "provider", "sportsbook", "event_id", "player_id", "prop", "line",
    "over_odds", "under_odds", "over_timestamp", "under_timestamp", "pair_timestamp",
    "pair_skew_seconds", "scheduled_tip_utc", "decision_timestamp_utc", "snapshot_label",
    "quote_pair_status",
]

EXACT_PAIR = "EXACT_PAIR"
ONE_SIDED = "ONE_SIDED"
CROSS_BOOK = "CROSS_BOOK"
CROSS_LINE = "CROSS_LINE"
INVALID_ODDS = "INVALID_ODDS"
DUPLICATE_SIDE = "DUPLICATE_SIDE"
AMBIGUOUS_PLAYER = "AMBIGUOUS_PLAYER"
AMBIGUOUS_GAME = "AMBIGUOUS_GAME"
AFTER_DECISION_CUTOFF = "AFTER_DECISION_CUTOFF"
AT_OR_AFTER_TIP = "AT_OR_AFTER_TIP"
BLOCKED_INVALID_TIP = "BLOCKED_INVALID_TIP"
SKEW_EXCEEDED = "SKEW_EXCEEDED"

_PAIR_GROUP = ["provider", "sportsbook", "event_id", "player_id", "prop", "line"]


def quote_pair_id(provider, sportsbook, event_id, player_id, prop, line,
                  pair_snapshot_timestamp) -> str:
    payload = "|".join(str(x) for x in (
        provider, sportsbook, event_id, player_id, prop, line, pair_snapshot_timestamp))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _ts(x):
    if x is None:
        return None
    try:
        t = pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if t is pd.NaT or (isinstance(t, float) and np.isnan(t)):
        return None
    return None if pd.isna(t) else t.to_pydatetime()


def _valid_american(o) -> bool:
    try:
        v = float(o)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(v):
        return False
    # American odds are >= +100 or <= -100 and never 0.
    return v >= 100.0 or v <= -100.0


def _is_ambiguous(v) -> bool:
    s = str(v).strip().lower()
    return v is None or s in ("", "nan", "none", "ambiguous", "unknown")


def build_quote_pairs(raw: pd.DataFrame, *, max_skew_seconds: int = 120,
                      snapshot_label: str = "decision") -> pd.DataFrame:
    """Pair raw single-side snapshots into canonical Over/Under pairs with an explicit status.

    A pair is EXACT_PAIR only when both sides share book/event/player/prop/line, both lie at or
    before the decision cutoff and strictly before tip, and their timestamps are within
    ``max_skew_seconds``. Tip must be parseable (else BLOCKED_INVALID_TIP)."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    missing = [c for c in RAW_SIDE_KEY if c not in raw.columns]
    if missing:
        raise ValueError(f"raw side snapshots missing required columns: {missing}")

    rows = []
    for key, g in raw.groupby(_PAIR_GROUP, dropna=False):
        provider, sportsbook, event_id, player_id, prop, line = key
        overs = g[g["side"].astype(str).str.lower() == "over"]
        unders = g[g["side"].astype(str).str.lower() == "under"]
        tip = _ts(g["scheduled_tip_utc"].iloc[0] if "scheduled_tip_utc" in g.columns else None)
        decision = _ts(g["decision_timestamp_utc"].iloc[0]
                       if "decision_timestamp_utc" in g.columns else None)
        over_ts = _ts(overs["snapshot_timestamp"].iloc[0]) if len(overs) else None
        under_ts = _ts(unders["snapshot_timestamp"].iloc[0]) if len(unders) else None
        pair_ts = max([t for t in (over_ts, under_ts) if t is not None], default=None)
        over_odds = overs["american_odds"].iloc[0] if ("american_odds" in overs.columns and len(overs)) else None
        under_odds = unders["american_odds"].iloc[0] if ("american_odds" in unders.columns and len(unders)) else None
        skew = (abs((over_ts - under_ts).total_seconds())
                if (over_ts is not None and under_ts is not None) else None)

        def _rec(status):
            return {
                "quote_pair_id": quote_pair_id(provider, sportsbook, event_id, player_id, prop,
                                               line, pair_ts.isoformat() if pair_ts else "NA"),
                "provider": provider, "sportsbook": sportsbook, "event_id": event_id,
                "player_id": player_id, "prop": prop, "line": line,
                "over_odds": over_odds, "under_odds": under_odds,
                "over_timestamp": over_ts.isoformat() if over_ts else None,
                "under_timestamp": under_ts.isoformat() if under_ts else None,
                "pair_timestamp": pair_ts.isoformat() if pair_ts else None,
                "pair_skew_seconds": skew, "scheduled_tip_utc": tip.isoformat() if tip else None,
                "decision_timestamp_utc": decision.isoformat() if decision else None,
                "snapshot_label": snapshot_label, "quote_pair_status": status,
            }

        # Fail-closed ordering: identity/tip first, then structure, then timing, then skew.
        if _is_ambiguous(player_id):
            rows.append(_rec(AMBIGUOUS_PLAYER)); continue
        if _is_ambiguous(event_id):
            rows.append(_rec(AMBIGUOUS_GAME)); continue
        if tip is None:
            rows.append(_rec(BLOCKED_INVALID_TIP)); continue      # never substitute a tip time
        if len(overs) > 1 or len(unders) > 1:
            rows.append(_rec(DUPLICATE_SIDE)); continue
        if len(overs) == 0 or len(unders) == 0:
            rows.append(_rec(ONE_SIDED)); continue
        if not (_valid_american(over_odds) and _valid_american(under_odds)):
            rows.append(_rec(INVALID_ODDS)); continue
        if any(t >= tip for t in (over_ts, under_ts)):
            rows.append(_rec(AT_OR_AFTER_TIP)); continue
        if decision is not None and any(t > decision for t in (over_ts, under_ts)):
            rows.append(_rec(AFTER_DECISION_CUTOFF)); continue
        if skew is not None and skew > max_skew_seconds:
            rows.append(_rec(SKEW_EXCEEDED)); continue
        rows.append(_rec(EXACT_PAIR))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)
