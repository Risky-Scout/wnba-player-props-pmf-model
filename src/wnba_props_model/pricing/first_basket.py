"""First-basket event pricing (Section 5): competing-risk over eligible players.

player_first_basket: every eligible player + a residual/unresolved state, total probability = 1.
player_first_team_basket: probabilities within each team sum to 1.
player_method_of_first_basket: a conditional categorical over provider-supported methods.

Hazards come from first-stint starter probability x initial minutes exposure x first-shot usage
x historical first-score hazard. Where evidence is insufficient a market-anchored prior may be
supplied and the price is flagged MARKET_ANCHORED_UNCERTIFIED (never a fabricated pure result).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FirstBasketHazard:
    player_id: str
    team_id: str
    hazard: float                       # relative first-score hazard weight (>=0)
    method_mix: dict[str, float] = field(default_factory=dict)   # conditional method probabilities
    certified: bool = False


def _norm(d: dict[str, float]) -> dict[str, float]:
    s = sum(max(v, 0.0) for v in d.values())
    return {k: max(v, 0.0) / s for k, v in d.items()} if s > 0 else d


def price_first_basket(hazards: list[FirstBasketHazard], residual_hazard: float = 0.0) -> dict:
    """Whole-event competing risk. Returns per-player probabilities + a residual state summing
    to 1 across the complete event."""
    weights = {h.player_id: max(h.hazard, 0.0) for h in hazards}
    total = sum(weights.values()) + max(residual_hazard, 0.0)
    if total <= 0:
        raise ValueError("first-basket hazards sum to non-positive")
    probs = {pid: w / total for pid, w in weights.items()}
    probs["__RESIDUAL_UNRESOLVED__"] = max(residual_hazard, 0.0) / total
    status = "PRICED" if all(h.certified for h in hazards) else "MARKET_ANCHORED_UNCERTIFIED"
    return {"market_key": "player_first_basket", "probabilities": probs,
            "normalized_sum": float(sum(probs.values())), "pricing_status": status}


def price_first_team_basket(hazards: list[FirstBasketHazard]) -> dict:
    """Per-team competing risk. Probabilities within EACH team sum to 1."""
    by_team: dict[str, dict[str, float]] = {}
    for h in hazards:
        by_team.setdefault(h.team_id, {})[h.player_id] = max(h.hazard, 0.0)
    out = {team: _norm(players) for team, players in by_team.items()}
    sums = {team: float(sum(p.values())) for team, p in out.items()}
    status = "PRICED" if all(h.certified for h in hazards) else "MARKET_ANCHORED_UNCERTIFIED"
    return {"market_key": "player_first_team_basket", "by_team": out,
            "per_team_normalized_sums": sums, "pricing_status": status}


def price_method_of_first_basket(hazard: FirstBasketHazard) -> dict:
    """Conditional categorical over methods (two_point_make, three_point_make, free_throw, ...)
    for a player, GIVEN they score the first basket. Methods are NOT independent binaries."""
    methods = _norm(hazard.method_mix) if hazard.method_mix else {}
    if not methods:
        return {"market_key": "player_method_of_first_basket", "player_id": hazard.player_id,
                "pricing_status": "NO_EVIDENCE", "methods": {}}
    return {"market_key": "player_method_of_first_basket", "player_id": hazard.player_id,
            "methods": methods, "normalized_sum": float(sum(methods.values())),
            "pricing_status": "PRICED" if hazard.certified else "MARKET_ANCHORED_UNCERTIFIED"}
