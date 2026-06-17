from __future__ import annotations

# ---------------------------------------------------------------------------
# Stat naming
# ---------------------------------------------------------------------------

# Internal stat keys used across model code (raw parquet layer uses these).
DIRECT_STATS = ("pts", "reb", "ast", "fg3m", "turnover", "stl", "blk")
COMBO_STATS = ("stocks", "pa", "pr", "ra", "pra")
SUPPORTED_STATS = DIRECT_STATS + COMBO_STATS

# Canonical stat names used in processed / canonical tables.
# "tov" → "turnover"; combo short-names get underscored canonical forms.
CANONICAL_STAT_NAMES = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
    "tov": "turnover",
    "stl": "stl",
    "blk": "blk",
    "stocks": "stocks",
    "pa": "pts_ast",
    "pr": "pts_reb",
    "ra": "reb_ast",
    "pra": "pts_reb_ast",
}

# BDL raw market stat name → canonical stat key
# Aliased as BDL_PROP_TO_STAT for use in deliver.py and other pipeline code
PROP_STAT_NAME_MAP: dict[str, str] = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "three_pointers_made": "fg3m",
    "fg3m": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
    "turnover": "turnover",
    "tov": "turnover",
    "points_assists": "pts_ast",
    "pts_ast": "pts_ast",
    "points_rebounds": "pts_reb",
    "pts_reb": "pts_reb",
    "rebounds_assists": "reb_ast",
    "reb_ast": "reb_ast",
    "points_rebounds_assists": "pts_reb_ast",
    "pts_reb_ast": "pts_reb_ast",
    "steals_blocks": "stocks",
    "stocks": "stocks",
    "double_double": "double_double",
    "triple_double": "triple_double",
}

# Alias used in pipeline/deliver.py
BDL_PROP_TO_STAT = PROP_STAT_NAME_MAP

# BDL raw column name → internal stat key (used in flatten_player_stat_row)
STAT_TO_BDL_COL = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
    "tov": "turnover",  # BDL field "turnover" → raw parquet col "tov" → canonical renames to "turnover"
    "stl": "stl",
    "blk": "blk",
}

DOMAIN_MAX = {
    "pts": 60,
    "reb": 30,
    "ast": 20,
    "fg3m": 12,
    "tov": 12,
    "turnover": 12,
    "stl": 10,
    "blk": 10,
    "stocks": 20,
    "pa": 80,
    "pts_ast": 80,
    "pr": 90,
    "pts_reb": 90,
    "ra": 45,
    "reb_ast": 45,
    "pra": 105,
    "pts_reb_ast": 105,
    "team_total": 130,
    "game_total": 260,
}

# ---------------------------------------------------------------------------
# Injury status normalization
# ---------------------------------------------------------------------------

# Raw BDL injury status (lowercased, stripped) → normalized status.
# normalized values: available, probable, questionable, doubtful, out, inactive, unknown
INJURY_STATUS_MAP: dict[str, str] = {
    "active": "available",
    "available": "available",
    "probable": "probable",
    "questionable": "questionable",
    "doubtful": "doubtful",
    "gtd": "questionable",
    "day-to-day": "questionable",
    "day to day": "questionable",
    "out": "out",
    "week-to-week": "out",
    "week to week": "out",
    "ir": "out",
    "injured reserve": "out",
    "suspension": "out",
    "suspended": "out",
    "inactive": "inactive",
    "not with team": "inactive",
    "nwt": "inactive",
    "dnp": "inactive",
    "did not play": "inactive",
}

# ---------------------------------------------------------------------------
# Forbidden market / leakage columns (must never appear in model features)
# ---------------------------------------------------------------------------

FORBIDDEN_MARKET_COLUMNS: frozenset[str] = frozenset({
    # Prop line and odds
    "line",
    "over_odds",
    "under_odds",
    "market_id",
    "odds_id",
    "book",
    "sportsbook",
    "vendor",               # canonical sportsbook identifier
    "prop_type_raw",        # evaluation-only metadata
    # Derived market probabilities
    "market_prob_over",
    "market_prob_under",
    "no_vig_prob_over",
    "no_vig_prob_under",
    # Evaluation / post-game
    "edge",
    "clv",
    "closing_line",
    "closing_odds",
    "hit_result",
    "outcome",
    # Legacy names used in earlier feature contract
    "market_line",
    "market_prob_over_no_vig",
    "consensus_total",
    "consensus_spread",
    # Game odds columns — actual BDL WNBA flat response field names
    "spread_home_value",
    "spread_home_odds",
    "spread_away_value",
    "spread_away_odds",
    "moneyline_home_odds",
    "moneyline_away_odds",
    "total_value",
    "total_over_odds",
    "total_under_odds",
    # Legacy / alternate column names
    "spread_value",
    "spread_visitor_odds",
    "moneyline_visitor_odds",
    "snapshot_timestamp_utc",
    "updated_at",
})

# ---------------------------------------------------------------------------
# Model calibration / quantile settings
# ---------------------------------------------------------------------------

QUANTILES = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
)

ROLE_BUCKETS = ("inactive_risk", "fringe", "bench", "rotation", "core", "starter")

ROLE_MIN_ROWS = {
    "inactive_risk": 10**12,  # force global-only
    "fringe": 350,
    "bench": 350,
    "rotation": 400,
    "core": 450,
    "starter": 500,
}

ROLE_GLOBAL_ONLY_BUCKETS = {"inactive_risk"}

CALIBRATION_GATES = {
    "pit_ks_max": 0.15,          # year-1; tighten to 0.10 in year 2
    "ece_max": 0.10,              # year-1; tighten to 0.05 in year 2
    "mean_error_abs_max": 1.00,  # year-1; tighten to 0.50 in year 2
    "variance_error_abs_max": 0.20,
}

MARKET_SUPERIORITY_GATES = {
    "ucb95_logloss_max": -0.0025,
    "ucb95_brier_max": -0.0010,
    "min_rows": 100,
    "bootstrap_reps": 2000,
    "bootstrap_seed": 20260512,
}
