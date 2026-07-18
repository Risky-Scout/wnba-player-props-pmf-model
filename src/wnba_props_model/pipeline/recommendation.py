"""Single shared recommendation-selection policy.

Used by BOTH the production edge report (build_edge_report.py) and the P1
historical replay so the replay reproduces production exactly (no simplified
imitation). Pure and deterministic: given a model P(over), the two market prices
for one book/line, the stat, and the policy thresholds, it returns the side,
edge, no-vig market probability, eligibility and selection — using ONLY the
information passed in (never a realized outcome).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    stat: str
    model_prob_over: float
    market_prob_over_no_vig: float
    edge_over: float          # model_prob_over - market_prob_over_no_vig
    edge_under: float         # -(edge_over)
    side: str                 # 'over' | 'under'
    abs_edge: float
    shin_z: float | None
    eligible: bool
    selected: bool
    reason: str = ""


def select_recommendation(
    *,
    model_prob_over: float,
    over_odds: float | int | None,
    under_odds: float | int | None,
    stat: str,
    no_vig_fn,
    publishable_stats,
    edge_threshold: float,
    min_market_prob: float = 0.05,
    max_shin_z: float = 0.15,
) -> Recommendation | None:
    """Reproduce production edge selection for one (stat, line, book) quote.

    Returns None when there is no usable paired market (no-vig prob undefined).
    ``no_vig_fn(over_odds, under_odds) -> (p_over, p_under, shin_z)``.
    Edge/side formulas are identical to build_edge_report.py:
      edge_over = model_prob_over - market_prob_over_no_vig; side = over iff edge_over > 0.
    Eligibility mirrors production: market prob within [min_market_prob, 1-min],
    stat in the publishable set, and shin_z <= max_shin_z. Selected = eligible and
    |edge| >= edge_threshold.
    """
    p_over, _p_under, shin_z = no_vig_fn(over_odds, under_odds)
    if p_over is None:
        return None
    p_over = float(p_over)
    edge_over = float(model_prob_over) - p_over
    edge_under = -edge_over
    side = "over" if edge_over > 0 else "under"
    abs_edge = abs(edge_over)

    reasons = []
    eligible = True
    if not (min_market_prob <= p_over <= 1.0 - min_market_prob):
        eligible = False; reasons.append("market_prob_out_of_range")
    if stat not in publishable_stats:
        eligible = False; reasons.append("stat_not_publishable")
    if shin_z is not None and float(shin_z) > max_shin_z:
        eligible = False; reasons.append("shin_z_above_max")

    selected = bool(eligible and abs_edge >= edge_threshold)
    return Recommendation(
        stat=stat, model_prob_over=float(model_prob_over),
        market_prob_over_no_vig=p_over, edge_over=edge_over, edge_under=edge_under,
        side=side, abs_edge=abs_edge, shin_z=(None if shin_z is None else float(shin_z)),
        eligible=eligible, selected=selected, reason=";".join(reasons),
    )
