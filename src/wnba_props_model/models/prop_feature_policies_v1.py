"""Per-stat feature policies (v1) — one explicit, causally-valid feature contract per prop.

Each prop gets its OWN contract instead of the shared 128-column matrix. Required columns
are the Tier-A mandatory causal core (opportunity + exposure); optional columns are Tier-B
lagged context admitted only when present. Every column is resolved against the real feature
manifest (``feature_contract.MODEL_FEATURES``) and every ``pure_forecast`` policy is verified
market-free at construction (:class:`PropFeaturePolicy` enforces this).

Candidate families follow the directive's fixed matrix:

* ``B0_BASE_RATE``            intercept / structured base rate (no columns).
* ``B2_COMPACT_CAUSAL_CORE``  Tier-A pure core (required) + Tier-B optional.
* ``B4_FULL_379_CONTROL``     legacy full matrix — non-certifiable comparison only.
* ``B6_INTERNAL_GAME_CONTEXT`` / ``B7_EXTERNAL_MARKET_CONTEXT`` for points.

These are *candidate* contracts to be measured by nested chronological PMF OOF; none is
certified for production until it passes the advancement + market-superiority gates.
"""
from __future__ import annotations

from wnba_props_model.models.prop_feature_policy import PropFeaturePolicy

# Tier-A mandatory causal core (required) — pure, strictly-lagged, always-present columns.
_CORE_REQUIRED: dict[str, tuple[str, ...]] = {
    "minutes": ("minutes_roll5", "minutes_roll10", "minutes_std10", "minutes_lag1",
                "recent_starter_rate5"),
    "pts": ("pred_minutes_mean", "pred_minutes_q25", "fga_per_min_roll5",
            "fta_per_min_roll5", "fg3m_per_min_roll5", "usage_proxy_roll5"),
    "reb": ("pred_minutes_mean", "pred_minutes_q25", "reb_per_min_roll5"),
    "ast": ("pred_minutes_mean", "pred_minutes_q25", "ast_per_min_roll5", "usage_proxy_roll5"),
    "fg3m": ("pred_minutes_mean", "fg3m_per_min_roll5", "fga_per_min_roll5"),
    "stl": ("pred_minutes_mean", "stl_per_min_roll10"),
    "blk": ("pred_minutes_mean", "blk_per_min_roll10"),
    "turnover": ("pred_minutes_mean", "tov_per_min_roll5", "usage_proxy_roll5"),
}

# Tier-B optional lagged context — admitted only when present in the matrix.
_CORE_OPTIONAL: dict[str, tuple[str, ...]] = {
    "minutes": ("p_inactive", "player_games_in_last_7d", "rest_advantage", "rest_days",
                "role_bucket_code", "pred_minutes_q25", "minutes_roll3"),
    "pts": ("recent_starter_rate5", "minutes_roll5", "minutes_std10", "player_usage_pct_ewma10",
            "opp_pts_allowed_roll5", "team_pace_proxy_roll5", "rest_advantage", "player_load_index"),
    "reb": ("recent_starter_rate5", "opp_reb_allowed_roll5", "opp_rim_pressure_proxy_roll5",
            "team_pace_proxy_roll5", "role_bucket_code", "minutes_roll5", "minutes_std10"),
    "ast": ("player_usage_pct_ewma10", "opp_ast_allowed_roll5", "team_pace_proxy_roll5",
            "recent_starter_rate5", "player_games_in_last_7d", "minutes_roll5"),
    "fg3m": ("pred_minutes_q25", "usage_proxy_roll5", "player_usage_pct_ewma10",
             "opp_pts_allowed_roll5", "team_pace_proxy_roll5", "recent_starter_rate5"),
    "stl": ("stl_opp_tov_rate", "stl_opp_pass_risk", "opp_tov_rate_roll5",
            "team_pace_proxy_roll5", "defender_role_code", "recent_starter_rate5"),
    "blk": ("blk_opp_rim_att", "opp_rim_pressure_proxy_roll5", "defender_role_code",
            "role_bucket_code", "recent_starter_rate5"),
    "turnover": ("fga_per_min_roll5", "player_usage_pct_ewma10", "opp_tov_rate_roll5",
                 "team_pace_proxy_roll5", "recent_starter_rate5"),
}

STATS = tuple(_CORE_REQUIRED.keys())


def compact_causal_core(stat: str) -> PropFeaturePolicy:
    """B2: pure Tier-A core (+ Tier-B optional) for a stat."""
    return PropFeaturePolicy(
        stat=stat, mode="explicit",
        required_columns=_CORE_REQUIRED[stat],
        optional_columns=_CORE_OPTIONAL.get(stat, ()),
        information_contract="pure_forecast",
        missing_required_policy="raise",
        feature_set_id=f"{stat}_B2_COMPACT_CAUSAL_CORE_v1",
    )


def base_rate(stat: str) -> PropFeaturePolicy:
    """B0: intercept / structured base-rate candidate (no columns)."""
    return PropFeaturePolicy(
        stat=stat, mode="base_rate", required_columns=(), optional_columns=(),
        information_contract="pure_forecast", missing_required_policy="raise",
        feature_set_id=f"{stat}_B0_BASE_RATE_v1",
    )


def full_control(stat: str) -> PropFeaturePolicy:
    """B4: frozen full-matrix control (NON-certifiable)."""
    return PropFeaturePolicy(
        stat=stat, mode="legacy_full_diagnostic", required_columns=(), optional_columns=(),
        information_contract="external_market_anchored", missing_required_policy="raise",
        feature_set_id=f"{stat}_B4_FULL_CONTROL_v1",
    )


def internal_game_context(stat: str) -> PropFeaturePolicy:
    """B6: compact core + internal (net-rating) game context. Only the internal projections,
    never the sportsbook total."""
    return PropFeaturePolicy(
        stat=stat, mode="explicit",
        required_columns=_CORE_REQUIRED[stat],
        optional_columns=_CORE_OPTIONAL.get(stat, ()) + ("game_pace_predicted",),
        information_contract="internal_game_context",
        missing_required_policy="raise",
        feature_set_id=f"{stat}_B6_INTERNAL_GAME_CONTEXT_v1",
    )


def external_market_context(stat: str) -> PropFeaturePolicy:
    """B7: compact core + current-game Vegas total/spread (external market anchored)."""
    return PropFeaturePolicy(
        stat=stat, mode="explicit",
        required_columns=_CORE_REQUIRED[stat],
        optional_columns=_CORE_OPTIONAL.get(stat, ()) + ("game_total", "game_spread_home"),
        information_contract="external_market_anchored",
        missing_required_policy="raise",
        feature_set_id=f"{stat}_B7_EXTERNAL_MARKET_CONTEXT_v1",
    )


def candidate_matrix(stat: str) -> dict[str, PropFeaturePolicy]:
    """The frozen candidate family for a stat (excludes data-driven B3/B5 which are produced
    by the nested selection script)."""
    cands = {
        "B0_BASE_RATE": base_rate(stat),
        "B2_COMPACT_CAUSAL_CORE": compact_causal_core(stat),
        "B4_FULL_379_CONTROL": full_control(stat),
        "B6_INTERNAL_GAME_CONTEXT": internal_game_context(stat),
    }
    if stat == "pts":
        cands["B7_EXTERNAL_MARKET_CONTEXT"] = external_market_context(stat)
    return cands
