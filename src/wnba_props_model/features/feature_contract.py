from __future__ import annotations

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

FEATURE_FAMILIES = {
    "identity": IDENTITY_FEATURES,
    "schedule": SCHEDULE_FEATURES,
    "minutes_role": MINUTES_ROLE_FEATURES,
    "player_rates": PLAYER_RATE_FEATURES,
    "team_context": TEAM_CONTEXT_FEATURES,
    "injury_availability": INJURY_AVAILABILITY_FEATURES,
    "lineup_fallback": LINEUP_FALLBACK_FEATURES,
    "sparse_event": SPARSE_EVENT_FEATURES,
}

MODEL_FEATURES = [
    f for family in FEATURE_FAMILIES.values() for f in family
]

MARKET_FEATURES_LEAKAGE_TAGGED = [
    "market_line",
    "market_prob_over_no_vig",
    "consensus_total",
    "consensus_spread",
]

FORBIDDEN_TARGET_LEAKAGE = {
    "same_game_box_score": ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "min", "plus_minus"],
    "market": MARKET_FEATURES_LEAKAGE_TAGGED,
    "post_tip": ["actual_starter", "actual_minutes", "final_score", "home_score", "away_score"],
}


def feature_families() -> dict[str, list[str]]:
    return FEATURE_FAMILIES.copy()


def assert_no_forbidden_features(features: list[str]) -> None:
    forbidden = {x for xs in FORBIDDEN_TARGET_LEAKAGE.values() for x in xs}
    overlap = sorted(set(features) & forbidden)
    if overlap:
        raise ValueError(f"Forbidden leakage features in training feature list: {overlap}")
