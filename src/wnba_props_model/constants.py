from __future__ import annotations

DIRECT_STATS = ("pts", "reb", "ast", "fg3m", "tov", "stl", "blk")
COMBO_STATS = ("stocks", "pa", "pr", "ra", "pra")
SUPPORTED_STATS = DIRECT_STATS + COMBO_STATS

BDL_PROP_TO_STAT = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "points_assists": "pa",
    "points_rebounds": "pr",
    "rebounds_assists": "ra",
    "points_rebounds_assists": "pra",
    # WNBA BDL currently lists milestone markets for these, not full PMF stat events.
    "double_double": "double_double",
    "triple_double": "triple_double",
}

STAT_TO_BDL_COL = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
    "tov": "turnover",
    "stl": "stl",
    "blk": "blk",
}

DOMAIN_MAX = {
    "pts": 60,
    "reb": 30,
    "ast": 20,
    "fg3m": 12,
    "tov": 12,
    "stl": 10,
    "blk": 10,
    "stocks": 20,
    "pa": 80,
    "pr": 90,
    "ra": 45,
    "pra": 105,
    "team_total": 130,
    "game_total": 260,
}

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
    "pit_ks_max": 0.075,
    "ece_max": 0.025,
    "mean_error_abs_max": 0.15,
    "variance_error_abs_max": 0.20,
}

MARKET_SUPERIORITY_GATES = {
    "ucb95_logloss_max": -0.0025,
    "ucb95_brier_max": -0.0010,
    "min_rows": 100,
    "bootstrap_reps": 2000,
    "bootstrap_seed": 20260512,
}
