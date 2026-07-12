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

# ---------------------------------------------------------------------------
# Advanced features (Item 2 — 24 new columns from extended BDL endpoints)
# ---------------------------------------------------------------------------

ADVANCED_USAGE_FEATURES: list[str] = [
    "player_usage_pct",           # Season USG% from player_season_advanced_stats
    "player_usage_pct_ewma10",    # 10-game EWMA of game-level USG%
    "player_usage_pct_vs_avg",    # Current USG% minus season average
]

ADVANCED_SHOT_QUALITY_FEATURES: list[str] = [
    "player_pct_fg_restricted",   # % FGA from restricted area
    "player_pct_fg_corner3",      # % FGA from corner 3
    "player_pct_fg_midrange",     # % FGA from mid-range
    "player_fg_pct_restricted",   # FG% from restricted area
    "shot_quality_score",         # Weighted shot quality index
]

ADVANCED_INJURY_FEATURES: list[str] = [
    "teammate_out_count",                    # # rotation teammates ruled out
    "teammate_questionable_count",           # # teammates questionable
    "team_total_usage_of_out_players",       # Redistributable USG%
]

ADVANCED_OPPONENT_FEATURES: list[str] = [
    "opp_def_rating_ewma10",   # Opponent EWMA defensive rating
    "opp_pace_ewma10",         # Opponent EWMA pace
    "game_pace_predicted",     # Expected total possessions
]

ADVANCED_STANDINGS_FEATURES: list[str] = [
    "team_playoff_seed",    # Playoff seeding (numeric)
    "team_games_behind",    # Games behind first (numeric)
    # season_phase is a categorical string ('early'/'mid'/'late'/'playoff')
    # stored in ROLE_BUCKET_COLS in build_features.py and one-hot encoded
    # separately.  It must NOT appear here or it will pass the FEATURE_FAMILIES
    # allowlist and crash HGBR with 'could not convert string to float: early'.
]

ADVANCED_FOUR_FACTORS_FEATURES: list[str] = [
    "player_efg_pct",   # Effective field goal percentage
    "player_ft_rate",   # Free throw rate
    "player_tov_pct",   # Turnover percentage
]

# SVD player quality embeddings (Enhancement 12a — self-supervised)
# 8 latent dimensions from TruncatedSVD on player × season-avg rolling stats.
# Column names player_svd_dim_{0..7} — captured by "player_" safe prefix.
PLAYER_EMBEDDING_FEATURES: list[str] = [
    f"player_svd_dim_{i}" for i in range(8)
]

# ---------------------------------------------------------------------------
# Usage Transfer Matrix features (Enhancement 1)
# Dynamic teammate-specific columns (teammate_{pid}_is_out etc.) are matched
# by the "teammate_" and "without_" safe prefixes in build_features.py.
# ---------------------------------------------------------------------------
USAGE_TRANSFER_FEATURES: list[str] = [
    "player_usage_rate_l5",           # Player's last-5-game usage rate
    "player_usage_rate_season",        # Player's season-average usage rate
    "usage_shift",                     # l5 - season (detects role changes)
    "usage_shift_abs",                 # |usage_shift| (magnitude of role change)
    "projected_usage_given_absences",  # UTM-projected usage with absences
    "usage_transfer_delta",            # Projected - base usage (pure lineup effect)
]

# Extended schedule fatigue features (Enhancement 3)
FATIGUE_FEATURES: list[str] = [
    "is_4_in_5",                # 4 games in 5 days
    "is_5_in_7",                # 5 games in 7 days
    "cumulative_minutes_l7",    # Total minutes played in last 7 days
    "altitude_flag",            # Playing at altitude (Denver/Salt Lake)
    "schedule_fatigue_index",   # Composite fatigue score
    "rest_interaction_high_usage",  # Fatigue × high usage interaction
]

# Shot quality / efficiency regression features (Enhancement 4)
SHOT_QUALITY_FEATURES: list[str] = [
    "shot_quality_delta_l10",   # eFG% l10 vs season (running hot/cold proxy)
    "is_running_hot",           # shot_quality_delta > 0.05
    "is_running_cold",          # shot_quality_delta < -0.05
]

# ---------------------------------------------------------------------------
# Game script / blowout probability features (Enhancement 5)
# Built by _build_game_script_features() in build_features.py.
# Names do NOT start with safe prefixes (player_/team_/opp_) so they must
# be registered here explicitly to pass the FEATURE_FAMILIES allowlist check.
# ---------------------------------------------------------------------------
GAME_SCRIPT_FEATURES: list[str] = [
    "pregame_win_probability",       # P(player's team wins) — Normal CDF from net rating spread
    "blowout_probability",           # P(margin > 15 either way) — reduces starters' minutes
    "close_game_probability",        # P(|margin| < 5) — increases starters' minutes
    "expected_minutes_given_script", # Base minutes adjusted for game script
    "minutes_upside",                # Additional minute potential (e.g. in close games for stars)
    # Season stage
    "game_number_in_season",         # Ordinal game number (1-40 regular season)
    "season_completion_pct",         # Games played / 40 (0→1 over season)
    "is_playoff_game",               # 1 if playoff, 0 otherwise
    # Schedule/travel (rest_days, is_b2b, is_3in4, travel_proxy omitted here —
    # they are already in SCHEDULE_FEATURES; duplicating inflates MODEL_FEATURES count)
    "team_timezone_diff",            # |home_tz - away_tz| proxy for travel fatigue
    "team_3in4_flag",                # Player's team on 3-in-4 schedule
    "opp_3in4_flag",                 # Opponent on 3-in-4 schedule
]

# Parts B+F: Prior-game market features — lagged by 1 game to avoid leakage.
# Same-day market data is forbidden; prior-day closing line is legal as a feature.
# player_market_p_over_prev: prior game's closing no-vig P(over) for this stat
# player_market_line_prev:   prior game's closing line value
# player_line_movement_prev: prior game's (closing_line - opening_line) movement delta
#
# STRUCTURAL MODEL WARNING:
# These features are market-derived signals. While they do not cause same-game
# temporal leakage (they are lagged one game), they must NOT enter the structural
# outcome model. In production-safe mode they are stripped before the HGB feature
# matrix is built (see pipeline/safety.py:strip_market_prior_features).
# They may be used only in an optional market-aware layer separate from the
# structural model.
MARKET_PRIOR_FEATURES: list[str] = [
    "player_market_p_over_prev",    # prior closing P(over) — aggregated sharp signal
    "player_market_line_prev",      # prior closing line — market consensus benchmark
    "player_line_movement_prev",    # prior (close - open) — sharp money direction signal
]

# Market-prior features must not contaminate the structural model.
# strip_market_prior_features() in pipeline/safety.py enforces this at runtime.
STRUCTURAL_MODEL_FORBIDDEN_FEATURES: frozenset[str] = frozenset(MARKET_PRIOR_FEATURES)

FEATURE_FAMILIES: dict[str, list[str]] = {
    "identity": IDENTITY_FEATURES,
    "schedule": SCHEDULE_FEATURES,
    "minutes_role": MINUTES_ROLE_FEATURES,
    "player_rates": PLAYER_RATE_FEATURES,
    "team_context": TEAM_CONTEXT_FEATURES,
    "injury_availability": INJURY_AVAILABILITY_FEATURES,
    "lineup_fallback": LINEUP_FALLBACK_FEATURES,
    "sparse_event": SPARSE_EVENT_FEATURES,
    # Advanced features from Item 2
    "advanced_usage": ADVANCED_USAGE_FEATURES,
    "advanced_shot_quality": ADVANCED_SHOT_QUALITY_FEATURES,
    "advanced_injury": ADVANCED_INJURY_FEATURES,
    "advanced_opponent": ADVANCED_OPPONENT_FEATURES,
    "advanced_standings": ADVANCED_STANDINGS_FEATURES,
    "advanced_four_factors": ADVANCED_FOUR_FACTORS_FEATURES,
    # SVD player quality embeddings (Enhancement 12a — latent skill-mix profile)
    "player_embeddings": PLAYER_EMBEDDING_FEATURES,
    # Usage Transfer Matrix (Enhancement 1 — previously excluded by prefix gate)
    "usage_transfer": USAGE_TRANSFER_FEATURES,
    # Schedule fatigue (Enhancement 3 — previously excluded by prefix gate)
    "fatigue": FATIGUE_FEATURES,
    # Shot quality (Enhancement 4 — previously excluded by prefix gate)
    "shot_quality": SHOT_QUALITY_FEATURES,
    # Game script / blowout probability (Enhancement 5 — previously excluded)
    "game_script": GAME_SCRIPT_FEATURES,
    # Parts B+F: prior-game market features (safe — lagged 1 game, no leakage)
    "market_prior": MARKET_PRIOR_FEATURES,
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

# Constant/degenerate features: always the same value across all players/games,
# providing zero predictive signal and potentially harming model training.
_CONSTANT_DEGENERATE_FEATURES = frozenset({
    # rotation_minutes_bimodal is always True (the rotation model always uses
    # bimodal distributions); a constant feature carries no information.
    "rotation_minutes_bimodal",
    # opp_aggression_index saturates at its clip ceiling (10.0) for all WNBA
    # matchups because WNBA team TOV/steal rates exceed the formula's per-poss
    # scaling.  opp_switch_rate_proxy is derived from it and is equally useless.
    "opp_aggression_index",
    "opp_switch_rate_proxy",
    # All rotation_minutes_* distribution features are near-constant across
    # players because the rotation model outputs population-level distributions
    # rather than player-specific ones (all players get ~22-min league average,
    # std < 0.6 minutes vs actual range of 0-40).  They add noise and actively
    # push elite starters' predictions toward bench-player averages.
    "rotation_minutes_mean",
    "rotation_minutes_p_over_30",
    "rotation_minutes_p_under_25",
    "rotation_minutes_q10",
    "rotation_minutes_q25",
    "rotation_minutes_q50",
    "rotation_minutes_q75",
    "rotation_minutes_q90",
    "rotation_minutes_std",
})

# Position-level opponent defensive stats (e.g. how many pts does this team
# allow to Centers per game) use the WRONG granularity for individual player
# prediction.  They measure performance across all opposing players at a
# position, which systematically underestimates elite starters (who score far
# above position average) and overestimates bench players.  Player-vs-opponent
# career features (player_pts_vs_opp_*) capture the correct individual signal.
_WRONG_GRANULARITY_FEATURES = frozenset({
    "opp_ast_vs_C_allowed_l5",
    "opp_ast_vs_F_allowed_l5",
    "opp_ast_vs_G_allowed_l5",
    "opp_blk_vs_C_allowed_l5",
    "opp_blk_vs_F_allowed_l5",
    "opp_blk_vs_G_allowed_l5",
    "opp_fg3m_vs_C_allowed_l5",
    "opp_fg3m_vs_F_allowed_l5",
    "opp_fg3m_vs_G_allowed_l5",
    "opp_pts_vs_C_allowed_l5",
    "opp_pts_vs_F_allowed_l5",
    "opp_pts_vs_G_allowed_l5",
    "opp_reb_vs_C_allowed_l5",
    "opp_reb_vs_F_allowed_l5",
    "opp_reb_vs_G_allowed_l5",
    "opp_stl_vs_C_allowed_l5",
    "opp_stl_vs_F_allowed_l5",
    "opp_stl_vs_G_allowed_l5",
    "opp_turnover_vs_C_allowed_l5",
    "opp_turnover_vs_F_allowed_l5",
    "opp_turnover_vs_G_allowed_l5",
})

FORBIDDEN_MODEL_FEATURES: frozenset[str] = (
    _BOX_SCORE_LEAKAGE
    | _MARKET_LEAKAGE
    | _OUTCOME_LEAKAGE
    | _CONSTANT_DEGENERATE_FEATURES
    | _WRONG_GRANULARITY_FEATURES
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


def audit_feature_variance(
    features_path: str,
    manifest_path: str,
    min_std: float = 0.05,
) -> list[str]:
    """Return list of model features whose cross-row std is below min_std.

    Used as a CI gate to catch constant/near-constant features before they
    corrupt HGB tree splits (e.g. rotation_minutes_p_over_30 std=0.004).

    Args:
        features_path: Path to the wide features parquet file.
        manifest_path: Path to the feature_schema_manifest.json file.
        min_std: Minimum acceptable standard deviation (default 0.05).

    Returns:
        List of violation strings in the format "feature_name (std=X.XXXX)".
        Empty list means all features pass.
    """
    import json
    from pathlib import Path
    import pandas as pd

    wide = pd.read_parquet(features_path)
    manifest = json.loads(Path(manifest_path).read_text())
    cols: list[str] = manifest.get("model_feature_columns", [])

    violations: list[str] = []
    for c in cols:
        if c not in wide.columns:
            continue
        if not pd.api.types.is_numeric_dtype(wide[c]):
            continue
        std = float(wide[c].std(skipna=True))
        if std < min_std:
            violations.append(f"{c} (std={std:.4f})")

    if violations:
        print(f"FAIL: {len(violations)} features with std < {min_std}:")
        for v in violations:
            print(f"  - {v}")
    else:
        print(f"PASS: All {len(cols)} model features have std >= {min_std}")
    return violations


if __name__ == "__main__":
    import argparse as _ap
    import sys as _sys

    _p = _ap.ArgumentParser(description="Audit model feature variance")
    _sub = _p.add_subparsers(dest="cmd")
    _va = _sub.add_parser("audit-variance")
    _va.add_argument("--features-path", required=True)
    _va.add_argument("--manifest-path", required=True)
    _va.add_argument("--min-std", type=float, default=0.05)
    _va.add_argument("--fail-on-violation", action="store_true")
    _args = _p.parse_args()

    if _args.cmd == "audit-variance":
        _violations = audit_feature_variance(
            _args.features_path, _args.manifest_path, _args.min_std
        )
        if _args.fail_on_violation and _violations:
            _sys.exit(1)
