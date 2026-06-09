"""Feature contract: permitted model features and leakage guards.

Rules:
  1. Only features in FEATURE_FAMILIES may enter model training.
  2. Any column in FORBIDDEN_MODEL_FEATURES must never appear in a training
     feature list.
  3. Market / evaluation columns are catalogued but blocked from models.
  4. assert_no_forbidden_features() is the authoritative leakage gate.
"""
from __future__ import annotations

from wnba_props_model.constants import FORBIDDEN_MARKET_COLUMNS

# ---------------------------------------------------------------------------
# Permitted feature families
# ---------------------------------------------------------------------------

IDENTITY_FEATURES = [
    "player_id_code",
    "team_id_code",
    "opponent_team_id_code",
    "is_home",
    "position_G",
    "position_F",
    "position_C",
]

SCHEDULE_FEATURES = [
    "rest_days",
    "is_b2b",
    "is_3in4",
    "travel_proxy",
    "postseason",
]

MINUTES_ROLE_FEATURES = [
    "minutes_lag1",
    "minutes_roll3",
    "minutes_roll5",
    "minutes_roll10",
    "minutes_std10",
    "start_proxy_lag1",
    "recent_starter_rate5",
    "p_inactive",
    "pred_minutes_mean",
    "pred_minutes_q25",
    "role_bucket_code",
]

PLAYER_RATE_FEATURES = [
    "pts_per_min_roll5",
    "reb_per_min_roll5",
    "ast_per_min_roll5",
    "fg3m_per_min_roll5",
    "tov_per_min_roll5",
    "stl_per_min_roll10",
    "blk_per_min_roll10",
    "usage_proxy_roll5",
    "fga_per_min_roll5",
    "fta_per_min_roll5",
]

TEAM_CONTEXT_FEATURES = [
    "team_pts_roll5",
    "team_pace_proxy_roll5",
    "team_ast_roll5",
    "team_reb_roll5",
    "opp_pts_allowed_roll5",
    "opp_reb_allowed_roll5",
    "opp_ast_allowed_roll5",
    "opp_tov_rate_roll5",
    "opp_rim_pressure_proxy_roll5",
]

INJURY_AVAILABILITY_FEATURES = [
    "team_out_count",
    "team_questionable_count",
    "usage_vacated_proxy",
    "rebound_vacated_proxy",
    "assist_vacated_proxy",
]

LINEUP_FALLBACK_FEATURES = [
    # WNBA BDL has no documented lineup endpoint at build time.
    # These are estimated pre-tip from recent games/injury status.
    "expected_starter",
    "expected_bench",
    "team_expected_starters_count",
    "lineup_confirmed",
    "confirmed_starter",
]

SPARSE_EVENT_FEATURES = [
    "stl_opp_tov_rate",
    "stl_opp_pass_risk",
    "blk_opp_rim_att",
    "defender_role_code",
]

FEATURE_FAMILIES: dict[str, list[str]] = {
    "identity": IDENTITY_FEATURES,
    "schedule": SCHEDULE_FEATURES,
    "minutes_role": MINUTES_ROLE_FEATURES,
    "player_rates": PLAYER_RATE_FEATURES,
    "team_context": TEAM_CONTEXT_FEATURES,
    "injury_availability": INJURY_AVAILABILITY_FEATURES,
    "lineup_fallback": LINEUP_FALLBACK_FEATURES,
    "sparse_event": SPARSE_EVENT_FEATURES,
}

MODEL_FEATURES: list[str] = [f for family in FEATURE_FAMILIES.values() for f in family]

# ---------------------------------------------------------------------------
# Forbidden columns  (market / post-game leakage)
# ---------------------------------------------------------------------------

# Same-game box score leakage
_BOX_SCORE_LEAKAGE = frozenset({
    "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "tov",
    "min", "plus_minus", "actual_starter", "actual_minutes",
    "final_score", "home_score", "away_score",
    "home_team_score", "visitor_team_score", "total_score",
})

# Market data (full list from constants + legacy names)
_MARKET_LEAKAGE = FORBIDDEN_MARKET_COLUMNS

# Post-game / outcome leakage
_OUTCOME_LEAKAGE = frozenset({
    "outcome", "hit_result", "result", "settled",
    "over_hit", "under_hit", "push",
})

FORBIDDEN_MODEL_FEATURES: frozenset[str] = (
    _BOX_SCORE_LEAKAGE | _MARKET_LEAKAGE | _OUTCOME_LEAKAGE
)


def feature_families() -> dict[str, list[str]]:
    return FEATURE_FAMILIES.copy()


def assert_no_forbidden_features(features: "list[str] | pd.DataFrame") -> None:
    """Raise ValueError if any feature/column is in the forbidden set.

    Accepts either a list of column names or a pandas DataFrame (checks .columns).
    """
    import pandas as pd  # local import to avoid circular dependency at module level
    if isinstance(features, pd.DataFrame):
        cols: list[str] = list(features.columns)
    else:
        cols = list(features)
    overlap = sorted(set(cols) & FORBIDDEN_MODEL_FEATURES)
    if overlap:
        raise ValueError(
            f"Forbidden leakage features in training feature list: {overlap}"
        )


def assert_no_market_columns(columns: list[str]) -> None:
    """Raise ValueError if any market/evaluation-only column is present."""
    overlap = sorted(set(columns) & _MARKET_LEAKAGE)
    if overlap:
        raise ValueError(
            f"Market-only (evaluation) columns found in model feature list: {overlap}"
        )
