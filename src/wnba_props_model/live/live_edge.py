"""Live edge calculator for WNBA player props.

Compares live model P(over) against live prop lines from BDL
/wnba/v1/odds/player_props?game_id=X (real-time, not historical).

Edge = model_p_over - vig_free_implied_p_over
If |edge| >= min_edge (default 4pp = 0.04): bettable edge exists.

Uses Shin's no-vig method for implied probability extraction (consistent with
the pre-game edge calculator in market.py).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from wnba_props_model.models.market import shin_no_vig_two_way  # noqa: PLC0415

log = logging.getLogger(__name__)

# BDL prop_type to internal stat name mapping
PROP_TYPE_TO_STAT: dict[str, str] = {
    "pts": "pts",
    "points": "pts",
    "player_points": "pts",
    "reb": "reb",
    "rebounds": "reb",
    "player_rebounds": "reb",
    "ast": "ast",
    "assists": "ast",
    "player_assists": "ast",
    "fg3m": "fg3m",
    "three_pointers": "fg3m",
    "player_threes": "fg3m",
    "stl": "stl",
    "steals": "stl",
    "blk": "blk",
    "blocks": "blk",
    "turnover": "turnover",
    "turnovers": "turnover",
    "pra": "pra",
    "pts_reb_ast": "pra",
    "stocks": "stocks",
}


class LiveEdgeCalculator:
    """Compare live model P(over) against live prop lines from BDL.

    Live props come from /wnba/v1/odds/player_props?game_id=X
    (returned in real-time, not stored historically).

    Edge = model_p_over - implied_p_over_from_odds (vig-free)

    If edge > 4pp (0.04): bettable edge exists on the over
    If edge < -4pp: bettable edge exists on the under
    """

    def __init__(self, min_edge: float = 0.04) -> None:
        self.min_edge = min_edge

    # Combo stat components for live projection aggregation
    LIVE_COMBO_STATS: dict[str, list[str]] = {
        "pra": ["pts", "reb", "ast"],
        "pa": ["pts", "ast"],
        "pr": ["pts", "reb"],
        "ra": ["reb", "ast"],
        "stocks": ["stl", "blk"],
    }

    def _get_combo_p_over(
        self,
        live_predictions: dict[int, dict[str, dict]],
        player_id,
        combo_stat: str,
        line: float,
    ) -> Optional[float]:
        """Estimate P(over line) for a combo stat by summing component projected means.

        Uses Poisson convolution approximation: sum of independent Poisson means is
        Poisson with μ = Σμᵢ. P(sum > line) is then computed from the Poisson CDF.
        """
        import math  # noqa: PLC0415
        components = self.LIVE_COMBO_STATS.get(combo_stat, [])
        if not components:
            return None
        player_data = live_predictions.get(player_id, {})
        if not all(c in player_data for c in components):
            return None
        # Sum projected totals (Bayesian posterior means)
        combo_mean = sum(
            float(player_data[c].get("projected_total", player_data[c].get("mean", 0.0)))
            for c in components
        )
        if combo_mean <= 0:
            return None
        # Poisson CDF P(X > line) = 1 - P(X <= floor(line))
        k_floor = int(math.floor(line))
        # Use scipy if available, else Poisson approximation via log-gamma
        try:
            from scipy.stats import poisson  # noqa: PLC0415
            return float(1.0 - poisson.cdf(k_floor, combo_mean))
        except ImportError:
            # Manual Poisson CDF
            log_lam = math.log(combo_mean)
            log_p = 0.0
            cum = 0.0
            for k in range(k_floor + 1):
                if k > 0:
                    log_p += log_lam - math.log(k)
                else:
                    log_p = -combo_mean
                cum += math.exp(log_p - combo_mean + log_lam * k) if k == 0 else math.exp(log_p)
            return float(max(0.0, 1.0 - cum))

    def compute_live_edges(
        self,
        live_predictions: dict[int, dict[str, dict]],
        live_props: pd.DataFrame,
    ) -> list[dict]:
        """Compute edges for all live props.

        Args:
            live_predictions: {player_id: {stat: {p_over, p_under, projected_total, ...}}}
            live_props: DataFrame from /wnba/v1/odds/player_props with columns:
                player_id, prop_type, line_value, over_odds, under_odds, vendor, updated_at

        Returns:
            list of edges sorted by |edge| descending:
              [{player_id, stat, line, model_p_over, market_p_over, edge, bettable, direction, over_odds}]
        """
        if live_props.empty:
            return []

        edges: list[dict] = []
        for _, prop in live_props.iterrows():
            pid = prop.get("player_id")
            prop_type = str(prop.get("prop_type", ""))
            stat = PROP_TYPE_TO_STAT.get(prop_type.lower(), prop_type.lower())
            line_raw = prop.get("line_value")

            if line_raw is None:
                continue
            line = float(line_raw)

            if pid not in live_predictions:
                continue

            if stat in live_predictions[pid]:
                model_result = live_predictions[pid][stat]
                model_p_over = float(model_result.get("p_over", 0.5))
            elif stat in self.LIVE_COMBO_STATS:
                # Compute combo P(over) from component stats
                combo_p_over = self._get_combo_p_over(live_predictions, pid, stat, line)
                if combo_p_over is None:
                    continue
                model_result = {"p_over": combo_p_over, "projected_total": None,
                                "observed_count": None, "elapsed_minutes": None}
                model_p_over = combo_p_over
            else:
                continue

            over_odds = prop.get("over_odds")
            under_odds = prop.get("under_odds")
            if over_odds is None or under_odds is None:
                continue

            # Vig-free using Shin (1993) — consistent with pre-game edge calculator.
            # shin_no_vig_two_way takes raw American odds and applies the Shin
            # informed-bettor model; falls back to multiplicative if penaltyblog
            # is unavailable.
            shin_over, shin_under = shin_no_vig_two_way(over_odds, under_odds)
            if shin_over is not None:
                vig_free_over = shin_over
            else:
                # Last-resort fallback: raw implied / sum
                imp_over = self._american_to_implied(over_odds)
                imp_under = self._american_to_implied(under_odds)
                total_imp = imp_over + imp_under
                vig_free_over = imp_over / total_imp if total_imp > 0 else 0.5

            edge = model_p_over - vig_free_over

            edges.append({
                "player_id": pid,
                "stat": stat,
                "prop_type": prop_type,
                "line": line,
                "model_p_over": round(model_p_over, 4),
                "market_p_over": round(vig_free_over, 4),
                "edge": round(edge, 4),
                "edge_pp": round(edge * 100, 2),
                "bettable": abs(edge) >= self.min_edge,
                "direction": "over" if edge > 0 else "under",
                "over_odds": over_odds,
                "under_odds": under_odds,
                "vendor": prop.get("vendor"),
                "projected_total": model_result.get("projected_total"),
                "observed_count": model_result.get("observed_count"),
                "elapsed_minutes": model_result.get("elapsed_minutes"),
            })

        # Sort by |edge| descending — largest edges first
        return sorted(edges, key=lambda x: abs(x["edge"]), reverse=True)

    def filter_bettable(self, edges: list[dict]) -> list[dict]:
        """Return only edges that exceed the minimum threshold."""
        return [e for e in edges if e["bettable"]]

    def edge_summary(self, edges: list[dict]) -> dict:
        """Summarize edges by stat."""
        if not edges:
            return {"n_total": 0, "n_bettable": 0, "by_stat": {}}
        bettable = self.filter_bettable(edges)
        by_stat: dict[str, dict] = {}
        for e in bettable:
            stat = e["stat"]
            if stat not in by_stat:
                by_stat[stat] = {"n": 0, "mean_edge_pp": 0.0, "best": 0.0}
            by_stat[stat]["n"] += 1
            by_stat[stat]["mean_edge_pp"] = (
                by_stat[stat]["mean_edge_pp"] * (by_stat[stat]["n"] - 1) + e["edge_pp"]
            ) / by_stat[stat]["n"]
            by_stat[stat]["best"] = max(by_stat[stat]["best"], abs(e["edge_pp"]))
        return {
            "n_total": len(edges),
            "n_bettable": len(bettable),
            "pct_bettable": round(len(bettable) / len(edges) * 100, 1),
            "by_stat": by_stat,
        }

    @staticmethod
    def _american_to_implied(odds: float | int | str) -> float:
        """Convert American odds to raw implied probability."""
        try:
            odds_int = int(float(odds))
        except (TypeError, ValueError):
            return 0.5
        if odds_int > 0:
            return 100.0 / (100.0 + odds_int)
        elif odds_int < 0:
            return abs(odds_int) / (100.0 + abs(odds_int))
        return 0.5
