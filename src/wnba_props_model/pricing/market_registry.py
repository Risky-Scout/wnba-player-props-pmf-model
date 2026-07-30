"""Canonical WNBA player-prop market registry (single source of truth for pricing v1).

Every offered market maps to an internal outcome key + settlement contract. Alternates and
milestone prices settle from the SAME underlying distribution as their base market — there is
never a separate binary model per offered line. Fantasy scoring requires an explicit operator
scoring-rule id (no universal formula assumed).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSpec:
    provider_market_key: str
    internal_outcome_key: str
    market_family: str            # over_under_count | over_under_combo | over_under_q1 |
                                  # event_yes_no | event_categorical | alternate | fantasy
    settlement_type: str          # count_over_under | yes_no | categorical
    source_distribution: str      # which model distribution settles it
    support_requirements: str
    push_rule: str                # integer_line_push | half_point_no_push | not_applicable
    dnp_rule: str                 # void_on_dnp | active_conditional | not_applicable
    required_scoring_config: str = ""   # e.g. fantasy scoring-rule id
    release_status: str = "IMPLEMENTED"
    base_market: str = ""         # for alternates: the base provider market key


# ---- direct full-game counts ----
_DIRECT = {
    "player_points": ("pts", "points", "convolved 2PM/3PM/FTM component construction"),
    "player_rebounds": ("reb", "rebounds", "OREB+DREB convolution"),
    "player_assists": ("ast", "assists", "overdispersed count"),
    "player_threes": ("fg3m", "three_pointers_made", "attempt x conversion beta-binomial"),
    "player_blocks": ("blk", "blocks", "possession-exposure hurdle count"),
    "player_steals": ("stl", "steals", "possession-exposure hurdle count"),
    "player_turnovers": ("turnover", "turnovers", "touch-exposure hurdle count"),
    "player_field_goals": ("fgm", "field_goals_made", "2PM+3PM derived"),
    "player_frees_made": ("ftm", "free_throws_made", "FTA x FT% beta-binomial"),
    "player_frees_attempts": ("fta", "free_throw_attempts", "FTA opportunity count"),
}

# ---- first-quarter counts ----
_Q1 = {
    "player_points_q1": ("pts_q1", "points_q1"),
    "player_rebounds_q1": ("reb_q1", "rebounds_q1"),
    "player_assists_q1": ("ast_q1", "assists_q1"),
}

# ---- joint / combination counts ----
_COMBO = {
    "player_blocks_steals": ("stocks", ("blk", "stl")),
    "player_points_rebounds": ("pts_reb", ("pts", "reb")),
    "player_points_assists": ("pts_ast", ("pts", "ast")),
    "player_rebounds_assists": ("reb_ast", ("reb", "ast")),
    "player_points_rebounds_assists": ("pts_reb_ast", ("pts", "reb", "ast")),
}

# ---- event markets ----
_EVENTS = {
    "player_double_double": ("double_double", "event_yes_no", "yes_no",
                             "joint(pts,reb,ast,stl,blk) >=10 in >=2 categories", "not_applicable"),
    "player_triple_double": ("triple_double", "event_yes_no", "yes_no",
                             "joint(pts,reb,ast,stl,blk) >=10 in >=3 categories", "not_applicable"),
    "player_first_basket": ("first_basket", "event_categorical", "categorical",
                            "event-level competing-risk first-score hazard", "not_applicable"),
    "player_first_team_basket": ("first_team_basket", "event_categorical", "categorical",
                                 "team-scoped competing-risk first-score hazard", "not_applicable"),
    "player_method_of_first_basket": ("method_first_basket", "event_categorical", "categorical",
                                      "conditional categorical method | first basket", "not_applicable"),
}

# ---- alternates (settle from the SAME base distribution) ----
_ALTERNATES = {
    "player_points_alternate": "player_points",
    "player_rebounds_alternate": "player_rebounds",
    "player_assists_alternate": "player_assists",
    "player_threes_alternate": "player_threes",
    "player_blocks_alternate": "player_blocks",
    "player_steals_alternate": "player_steals",
    "player_points_rebounds_alternate": "player_points_rebounds",
    "player_points_assists_alternate": "player_points_assists",
    "player_rebounds_assists_alternate": "player_rebounds_assists",
    "player_points_rebounds_assists_alternate": "player_points_rebounds_assists",
}


def _build() -> dict[str, MarketSpec]:
    reg: dict[str, MarketSpec] = {}
    for k, (okey, _name, src) in _DIRECT.items():
        reg[k] = MarketSpec(k, okey, "over_under_count", "count_over_under", src,
                            "adaptive to tail<1e-8", "integer_line_push", "void_on_dnp",
                            release_status="IMPLEMENTED")
    for k, (okey, _name) in _Q1.items():
        reg[k] = MarketSpec(k, okey, "over_under_q1", "count_over_under",
                            "separate Q1 minutes+opportunity layer", "adaptive to tail<1e-8",
                            "integer_line_push", "void_on_dnp", release_status="IMPLEMENTED")
    for k, (okey, comps) in _COMBO.items():
        reg[k] = MarketSpec(k, okey, "over_under_combo", "count_over_under",
                            f"joint-dependence convolution of {comps}", "adaptive to tail<1e-8",
                            "integer_line_push", "void_on_dnp", release_status="IMPLEMENTED")
    for k, (okey, fam, settle, src, push) in _EVENTS.items():
        reg[k] = MarketSpec(k, okey, fam, settle, src, "event probability", push,
                            "active_conditional", release_status="IMPLEMENTED")
    for k, base in _ALTERNATES.items():
        b = reg[base]
        reg[k] = MarketSpec(k, b.internal_outcome_key, "alternate", "count_over_under",
                            b.source_distribution, b.support_requirements, "integer_line_push",
                            "void_on_dnp", release_status="IMPLEMENTED", base_market=base)
    # fantasy requires an explicit scoring-rule id (configuration-dependent)
    reg["player_fantasy_points"] = MarketSpec(
        "player_fantasy_points", "fantasy_points", "fantasy", "count_over_under",
        "linear combination of component distributions per scoring rule", "adaptive to tail<1e-8",
        "half_point_no_push", "void_on_dnp",
        required_scoring_config="REQUIRES_OPERATOR_SCORING_RULE_ID", release_status="CONFIG_REQUIRED")
    return reg


MARKET_REGISTRY: dict[str, MarketSpec] = _build()

DIRECT_COUNT_MARKETS = [k for k, v in MARKET_REGISTRY.items() if v.market_family == "over_under_count"]
COMBO_MARKETS = [k for k, v in MARKET_REGISTRY.items() if v.market_family == "over_under_combo"]
Q1_MARKETS = [k for k, v in MARKET_REGISTRY.items() if v.market_family == "over_under_q1"]
EVENT_MARKETS = [k for k, v in MARKET_REGISTRY.items() if v.settlement_type in ("yes_no", "categorical")]
ALTERNATE_MARKETS = [k for k, v in MARKET_REGISTRY.items() if v.market_family == "alternate"]


def get(market_key: str) -> MarketSpec:
    if market_key not in MARKET_REGISTRY:
        raise KeyError(f"unknown market key: {market_key}")
    return MARKET_REGISTRY[market_key]


def registry_as_records() -> list[dict]:
    return [v.__dict__.copy() for v in MARKET_REGISTRY.values()]
