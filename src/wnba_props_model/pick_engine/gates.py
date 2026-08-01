"""Validity gates for pick-engine candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from wnba_props_model.pick_engine.constants import (
    ABSTAIN_IDENTITY,
    ABSTAIN_INVALID_PMF,
    ABSTAIN_MISSING_PURE_PROBABILITY,
    ABSTAIN_OOD,
    ABSTAIN_PLAYER_OUT,
    ABSTAIN_POST_TIP,
    ABSTAIN_STALE_AVAILABILITY,
    ABSTAIN_STALE_QUOTE,
    ABSTAIN_UNSUPPORTED_TARGET,
    DEFAULT_AVAILABILITY_FRESHNESS_HOURS,
    DEFAULT_QUOTE_FRESHNESS_HOURS,
    EXCLUDED_COMBO_MARKET_KEYS,
    EXCLUDED_COMBO_STATS,
    MARKET_KEY_TO_STAT,
    SUPPORTED_MARKET_KEYS,
    SUPPORTED_STATS,
)
from wnba_props_model.pick_engine.probabilities import validate_pmf_mass


def _parse_ts(value) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str = ""


def normalize_stat(stat_or_market: str) -> str | None:
    s = str(stat_or_market).strip().lower()
    if s in SUPPORTED_STATS:
        return s
    if s in MARKET_KEY_TO_STAT:
        return MARKET_KEY_TO_STAT[s]
    # Common aliases
    aliases = {
        "points": "pts",
        "rebounds": "reb",
        "assists": "ast",
        "threes": "fg3m",
        "three_pointers": "fg3m",
        "steals": "stl",
        "blocks": "blk",
        "turnovers": "turnover",
        "to": "turnover",
    }
    return aliases.get(s)


def is_combination_target(stat: str | None, market_key: str | None = None) -> bool:
    if stat and str(stat).lower() in EXCLUDED_COMBO_STATS:
        return True
    if market_key and str(market_key).lower() in EXCLUDED_COMBO_MARKET_KEYS:
        return True
    if market_key and str(market_key).lower() not in SUPPORTED_MARKET_KEYS:
        # Unsupported non-combo markets also blocked via unsupported target.
        return False
    return False


def evaluate_gates(row: dict[str, Any]) -> GateResult:
    """Evaluate all validity gates; return first failing abstention reason."""
    market_key = row.get("market_key")
    stat = normalize_stat(row.get("stat") or market_key or "")
    if stat is None or (market_key and str(market_key) not in SUPPORTED_MARKET_KEYS and stat not in SUPPORTED_STATS):
        if is_combination_target(row.get("stat"), market_key):
            return GateResult(False, ABSTAIN_UNSUPPORTED_TARGET)
        return GateResult(False, ABSTAIN_UNSUPPORTED_TARGET)
    if is_combination_target(stat, market_key):
        return GateResult(False, ABSTAIN_UNSUPPORTED_TARGET)

    if not row.get("game_id_valid", True) or row.get("canonical_game_id") in (None, "", "ambiguous"):
        return GateResult(False, ABSTAIN_IDENTITY)
    if not row.get("player_id_valid", True) or row.get("canonical_player_id") in (None, "", "ambiguous"):
        return GateResult(False, ABSTAIN_IDENTITY)
    if row.get("identity_rejected"):
        return GateResult(False, ABSTAIN_IDENTITY)
    if row.get("team_mismatch"):
        return GateResult(False, ABSTAIN_IDENTITY)
    if not row.get("current_team_valid", True):
        return GateResult(False, ABSTAIN_IDENTITY)

    status = str(row.get("availability_status") or row.get("injury_status") or "").upper()
    if status in {"OUT", "CONFIRMED_OUT"} or row.get("confirmed_out"):
        return GateResult(False, ABSTAIN_PLAYER_OUT)

    tip = _parse_ts(row.get("scheduled_tip_utc"))
    pred_ts = _parse_ts(row.get("prediction_timestamp"))
    quote_ts = _parse_ts(row.get("provider_quote_timestamp") or row.get("quote_timestamp"))
    asof = _parse_ts(row.get("asof_timestamp")) or pred_ts
    avail_ts = _parse_ts(row.get("availability_timestamp"))

    if tip and asof and asof >= tip:
        return GateResult(False, ABSTAIN_POST_TIP)
    if tip and pred_ts and pred_ts >= tip:
        return GateResult(False, ABSTAIN_POST_TIP)
    if tip and quote_ts and quote_ts >= tip:
        return GateResult(False, ABSTAIN_POST_TIP)
    cutoff = _parse_ts(row.get("prediction_cutoff")) or pred_ts
    if cutoff and quote_ts and quote_ts > cutoff:
        return GateResult(False, ABSTAIN_STALE_QUOTE)

    freshness_h = float(row.get("quote_freshness_hours") or DEFAULT_QUOTE_FRESHNESS_HOURS)
    if quote_ts and asof:
        age_h = (asof - quote_ts).total_seconds() / 3600.0
        if age_h > freshness_h:
            return GateResult(False, ABSTAIN_STALE_QUOTE)
    elif row.get("quote_stale"):
        return GateResult(False, ABSTAIN_STALE_QUOTE)

    avail_fresh_h = float(
        row.get("availability_freshness_hours") or DEFAULT_AVAILABILITY_FRESHNESS_HOURS
    )
    if avail_ts and asof:
        if (asof - avail_ts).total_seconds() / 3600.0 > avail_fresh_h:
            return GateResult(False, ABSTAIN_STALE_AVAILABILITY)
    elif row.get("availability_stale"):
        return GateResult(False, ABSTAIN_STALE_AVAILABILITY)

    if row.get("ood_flag"):
        return GateResult(False, ABSTAIN_OOD)

    pmf = row.get("active_pmf") or row.get("active_pmf_json")
    if pmf is None:
        return GateResult(False, ABSTAIN_INVALID_PMF)
    ok_mass, _, _mass_reason = validate_pmf_mass(pmf)
    if not ok_mass:
        return GateResult(False, ABSTAIN_INVALID_PMF)

    pure_p = row.get("pure_probability")
    if pure_p is None or (isinstance(pure_p, float) and (math.isnan(pure_p) or not math.isfinite(pure_p))):
        return GateResult(False, ABSTAIN_MISSING_PURE_PROBABILITY)
    if float(pure_p) < 0 or float(pure_p) > 1:
        return GateResult(False, ABSTAIN_MISSING_PURE_PROBABILITY)

    # Never silently fill missing probabilities with 0.5.
    for key in ("reference_market_probability", "production_probability", "pick_probability"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, float) and (math.isnan(val) or not math.isfinite(val)):
            return GateResult(False, ABSTAIN_MISSING_PURE_PROBABILITY)

    if not row.get("executable_price_available", True):
        return GateResult(False, ABSTAIN_STALE_QUOTE)

    if row.get("period") not in (None, "", "game", "full_game", "full"):
        # Q1 / period markets without matching PMFs are unsupported in v1.
        return GateResult(False, ABSTAIN_UNSUPPORTED_TARGET)

    return GateResult(True, "")
