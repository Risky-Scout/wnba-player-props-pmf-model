"""Pick-engine constants: markets, statuses, abstention reasons."""

from __future__ import annotations

# Initial supported full-game markets (Odds API keys -> internal stat keys).
MARKET_KEY_TO_STAT: dict[str, str] = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
    "player_steals": "stl",
    "player_blocks": "blk",
    "player_turnovers": "turnover",
}

STAT_TO_MARKET_KEY: dict[str, str] = {v: k for k, v in MARKET_KEY_TO_STAT.items()}

SUPPORTED_MARKET_KEYS: frozenset[str] = frozenset(MARKET_KEY_TO_STAT)
SUPPORTED_STATS: frozenset[str] = frozenset(MARKET_KEY_TO_STAT.values())

# Combination / surrogate markets excluded until a fitted joint model exists.
EXCLUDED_COMBO_STATS: frozenset[str] = frozenset(
    {
        "pts_ast",
        "pts_reb",
        "reb_ast",
        "pts_reb_ast",
        "stocks",
        "pra",
        "pr",
        "pa",
        "ra",
    }
)
EXCLUDED_COMBO_MARKET_KEYS: frozenset[str] = frozenset(
    {
        "player_points_rebounds_assists",
        "player_points_rebounds",
        "player_points_assists",
        "player_rebounds_assists",
        "player_stocks",
        "player_points_alternate",
    }
)

# Selection statuses
DAILY_RANKED_SELECTION = "DAILY_RANKED_SELECTION"
PROVISIONAL_MODEL_PICK = "PROVISIONAL_MODEL_PICK"
CERTIFIED_MODEL_PICK = "CERTIFIED_MODEL_PICK"
NO_POSITIVE_CONSERVATIVE_EV = "NO_POSITIVE_CONSERVATIVE_EV"

SELECTION_STATUSES = frozenset(
    {
        DAILY_RANKED_SELECTION,
        PROVISIONAL_MODEL_PICK,
        CERTIFIED_MODEL_PICK,
        NO_POSITIVE_CONSERVATIVE_EV,
    }
)

# Explicit abstention reasons
ABSTAIN_PLAYER_OUT = "ABSTAIN_PLAYER_OUT"
ABSTAIN_IDENTITY = "ABSTAIN_IDENTITY"
ABSTAIN_STALE_AVAILABILITY = "ABSTAIN_STALE_AVAILABILITY"
ABSTAIN_STALE_QUOTE = "ABSTAIN_STALE_QUOTE"
ABSTAIN_INVALID_PMF = "ABSTAIN_INVALID_PMF"
ABSTAIN_MISSING_PURE_PROBABILITY = "ABSTAIN_MISSING_PURE_PROBABILITY"
ABSTAIN_UNSUPPORTED_TARGET = "ABSTAIN_UNSUPPORTED_TARGET"
ABSTAIN_POST_TIP = "ABSTAIN_POST_TIP"
ABSTAIN_OOD = "ABSTAIN_OOD"

ABSTAIN_REASONS = frozenset(
    {
        ABSTAIN_PLAYER_OUT,
        ABSTAIN_IDENTITY,
        ABSTAIN_STALE_AVAILABILITY,
        ABSTAIN_STALE_QUOTE,
        ABSTAIN_INVALID_PMF,
        ABSTAIN_MISSING_PURE_PROBABILITY,
        ABSTAIN_UNSUPPORTED_TARGET,
        ABSTAIN_POST_TIP,
        ABSTAIN_OOD,
    }
)

RETROSPECTIVE_LABEL = "RETROSPECTIVE_PICK_ENGINE_REPLAY"

DEFAULT_QUOTE_FRESHNESS_HOURS = 6.0
DEFAULT_AVAILABILITY_FRESHNESS_HOURS = 12.0
DEFAULT_MIN_REFERENCE_BOOKS = 2
DEFAULT_TOP_N = 10

PROB_EPS = 1e-6
MASS_TOL = 1e-6
