"""Live edge calculator for WNBA player props.

Compares live model P(over) against live prop lines from BDL
/wnba/v1/odds/player_props?game_id=X (real-time, not historical).

Edge = model_p_over - vig_free_implied_p_over
If |edge| >= min_edge (default 4pp = 0.04): bettable edge exists.

Uses Shin's no-vig method for implied probability extraction.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

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

            if pid not in live_predictions or stat not in live_predictions[pid]:
                continue
            if line_raw is None:
                continue

            line = float(line_raw)
            model_result = live_predictions[pid][stat]
            model_p_over = float(model_result.get("p_over", 0.5))

            over_odds = prop.get("over_odds")
            under_odds = prop.get("under_odds")
            if over_odds is None or under_odds is None:
                continue

            # Convert American odds to implied probability
            imp_over = self._american_to_implied(over_odds)
            imp_under = self._american_to_implied(under_odds)

            # Vig-free (Shin's method)
            total_imp = imp_over + imp_under
            if total_imp > 0:
                vig_free_over = imp_over / total_imp
            else:
                vig_free_over = 0.5

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
