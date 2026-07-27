"""Data contracts (schemas, enums, priors) for Opportunity V2.

Single source of truth for the shape of every snapshot / label table and for the canonical
prediction-cutoff identity columns. Kept dependency-light (pandas only) so every other module and
every script can import it without cycles.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


class ContractError(ValueError):
    """Raised when a frame violates an Opportunity V2 data contract."""


# ---------------------------------------------------------------------------
# Canonical identity / cutoff columns (section 7)
# ---------------------------------------------------------------------------
CUTOFF_REQUIRED_COLUMNS: tuple[str, ...] = (
    "prediction_cutoff_utc",
    "scheduled_tip_utc",
    "game_id",
    "player_id",
    "team_id",
    "opponent_team_id",
)

CUTOFF_SOURCE_EXACT = "exact_quote_timestamp"
CUTOFF_SOURCE_FALLBACK = "scheduled_tip_minus_90m"

STARTER_LABEL_OFFICIAL = "official"
STARTER_LABEL_PROXY = "minutes_proxy"

# ---------------------------------------------------------------------------
# Data tiers (section 8.4)
# ---------------------------------------------------------------------------
DATA_TIER_BOX = 0       # box score only
DATA_TIER_PBP = 1       # play-by-play derived
DATA_TIER_TRACKING = 2  # tracking derived

# ---------------------------------------------------------------------------
# Availability status normalization (section 8.1)
# ---------------------------------------------------------------------------
AVAILABILITY_STATUS_NORMALIZED: frozenset[str] = frozenset({
    "out", "doubtful", "questionable", "probable", "available",
    "suspended", "not_with_team", "personal", "minutes_limit", "unknown",
})

# Fallback priors ONLY for missing-model / diagnostic situations. NOT trained probabilities.
STATUS_PRIOR: dict[str, float] = {
    "out": 0.001,
    "suspended": 0.001,
    "not_with_team": 0.001,
    "doubtful": 0.15,
    "questionable": 0.55,
    "minutes_limit": 0.90,
    "probable": 0.93,
    "available": 0.995,
    "personal": 0.50,
    "unknown": 0.80,
}

LINEUP_STATUS_ALLOWED: frozenset[str] = frozenset({
    "confirmed_starter", "projected_starter", "projected_bench",
    "confirmed_bench", "unknown",
})

# ---------------------------------------------------------------------------
# Table schemas: column -> dtype kind. Dtype kinds are validated leniently (see
# ``validate_frame_schema``); UTC columns are checked by ``snapshot_store.canonicalize_utc``.
# ---------------------------------------------------------------------------
_UTC = "datetime64[ns, UTC]"

AVAILABILITY_SNAPSHOT_SCHEMA: dict[str, str] = {
    "snapshot_id": "string",
    "source": "string",
    "source_record_id": "string",
    "pulled_at_utc": _UTC,
    "available_at_utc": _UTC,
    "effective_at_utc": _UTC,
    "snapshot_date_utc": "date",
    "game_id": "Int64",
    "scheduled_tip_utc": _UTC,
    "player_id": "Int64",
    "team_id": "Int64",
    "status_raw": "string",
    "status_normalized": "string",
    "status_reason": "string",
    "minutes_limit_reported": "float64",
    "is_expected_available_raw": "boolean",
    "source_url_hash": "string",
    "payload_sha256": "string",
}
AVAILABILITY_REQUIRED: tuple[str, ...] = (
    "snapshot_id", "source", "pulled_at_utc", "available_at_utc", "snapshot_date_utc",
    "player_id", "team_id", "status_raw", "status_normalized", "payload_sha256",
)
AVAILABILITY_UNIQUE: tuple[str, ...] = ("snapshot_id",)

LINEUP_SNAPSHOT_SCHEMA: dict[str, str] = {
    "snapshot_id": "string",
    "source": "string",
    "pulled_at_utc": _UTC,
    "available_at_utc": _UTC,
    "snapshot_date_utc": "date",
    "game_id": "Int64",
    "scheduled_tip_utc": _UTC,
    "team_id": "Int64",
    "player_id": "Int64",
    "lineup_status": "string",
    "is_projected_starter": "boolean",
    "is_confirmed_starter": "boolean",
    "position_slot": "string",
    "lineup_order": "Int64",
    "payload_sha256": "string",
}
LINEUP_REQUIRED: tuple[str, ...] = (
    "snapshot_id", "source", "pulled_at_utc", "available_at_utc", "snapshot_date_utc",
    "game_id", "team_id", "player_id", "lineup_status", "payload_sha256",
)
LINEUP_UNIQUE: tuple[str, ...] = ("snapshot_id",)

ROSTER_INTERVAL_SCHEMA: dict[str, str] = {
    "player_id": "Int64",
    "team_id": "Int64",
    "valid_from_utc": _UTC,
    "valid_to_utc": _UTC,
    "transaction_type": "string",
    "source": "string",
    "available_at_utc": _UTC,
}
ROSTER_REQUIRED: tuple[str, ...] = (
    "player_id", "team_id", "valid_from_utc", "transaction_type", "source", "available_at_utc",
)

PLAYER_TRACKING_SCHEMA: dict[str, str] = {
    "game_id": "Int64", "game_date": "date", "scheduled_tip_utc": _UTC,
    "player_id": "Int64", "team_id": "Int64", "opponent_team_id": "Int64",
    "actual_minutes": "float64", "actual_started": "boolean",
    "touches": "float64", "passes_made": "float64", "passes_received": "float64",
    "potential_assists": "float64", "time_of_possession_seconds": "float64",
    "drives": "float64", "paint_touches": "float64", "frontcourt_touches": "float64",
    "rebound_chances": "float64", "oreb_chances": "float64", "dreb_chances": "float64",
    "contested_rebound_chances": "float64", "uncontested_rebound_chances": "float64",
    "catch_shoot_3pa": "float64", "pullup_3pa": "float64", "wide_open_3pa": "float64",
    "open_3pa": "float64", "contested_3pa": "float64",
    "rim_attempts_defended": "float64", "shot_contests": "float64",
    "defensive_matchup_possessions": "float64",
    "primary_ballhandler_matchup_possessions": "float64",
    "source": "string", "source_available_at_utc": _UTC, "data_tier": "Int64",
}
PLAYER_TRACKING_REQUIRED: tuple[str, ...] = (
    "game_id", "player_id", "team_id", "actual_minutes", "source",
    "source_available_at_utc", "data_tier",
)

TEAM_TRACKING_SCHEMA: dict[str, str] = {
    "game_id": "Int64", "game_date": "date", "team_id": "Int64", "opponent_team_id": "Int64",
    "possessions": "float64", "fga": "float64", "fg2a": "float64", "fg3a": "float64",
    "fta": "float64", "fg_misses": "float64", "fg2_misses": "float64", "fg3_misses": "float64",
    "turnovers": "float64", "live_ball_turnovers": "float64", "rim_attempts": "float64",
    "potential_assists": "float64", "rebound_chances": "float64", "touches": "float64",
    "passes": "float64", "source": "string", "source_available_at_utc": _UTC,
    "data_tier": "Int64",
}
TEAM_TRACKING_REQUIRED: tuple[str, ...] = (
    "game_id", "team_id", "opponent_team_id", "possessions", "source",
    "source_available_at_utc", "data_tier",
)


def validate_frame_schema(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    name: str,
) -> None:
    """Raise ``ContractError`` when a required column is missing.

    Extra columns are permitted (append-only evolution). Dtype coercion is the responsibility of
    the writer (``snapshot_store``); this guards presence + non-emptiness of the identity contract.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"{name}: expected a DataFrame, got {type(frame)!r}")
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise ContractError(f"{name}: missing required column(s): {missing}")


def normalize_availability_status(raw: str | None) -> str:
    """Map a raw injury/availability string to a canonical status. Unknown NEVER becomes available.

    A missing / unrecognized status resolves to ``"unknown"`` (prior 0.80), never ``"available"``.
    """
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    if not s:
        return "unknown"
    # Direct canonical hits.
    collapsed = s.replace(" ", "_")
    if collapsed in AVAILABILITY_STATUS_NORMALIZED:
        return collapsed
    # Heuristic mapping of common provider strings.
    if any(k in s for k in ("out for season", "season", "acl", "torn")):
        return "out"
    if "out" in s:
        return "out"
    if "doubtful" in s:
        return "doubtful"
    if "question" in s or "gtd" in s or "game time" in s:
        return "questionable"
    if "probable" in s:
        return "probable"
    if "minutes" in s and ("limit" in s or "restrict" in s):
        return "minutes_limit"
    if "suspend" in s:
        return "suspended"
    if "not with team" in s or "g league" in s or "g_league" in s:
        return "not_with_team"
    if "personal" in s or "bereavement" in s or "family" in s:
        return "personal"
    if any(k in s for k in ("available", "active", "healthy", "cleared", "probable")):
        return "available"
    return "unknown"


def status_prior(status_normalized: str | None) -> float:
    """Fallback active-probability prior for a normalized status (diagnostic use only)."""
    return STATUS_PRIOR.get(str(status_normalized), STATUS_PRIOR["unknown"])


def forbidden_market_columns(columns: Sequence[str]) -> list[str]:
    """Return any columns that look like market/odds/sportsbook signals (forbidden as inputs)."""
    banned_tokens = (
        "market", "odds", "line", "price", "spread", "total", "consensus", "clv",
        "vig", "implied", "book", "sportsbook", "moneyline", "american_odds", "decimal_odds",
        "closing", "opening_line", "over_under", "handle", "prob_over", "prob_under",
    )
    found = []
    for c in columns:
        cl = str(c).lower()
        if any(tok in cl for tok in banned_tokens):
            found.append(c)
    return found


def as_mapping(row: Mapping[str, object]) -> dict[str, object]:
    """Coerce a mapping/row to a plain dict (helper for payload hashing)."""
    return {str(k): v for k, v in dict(row).items()}
