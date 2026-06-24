"""Game Total Coherence Anchoring.

The game total market is the most efficiently priced WNBA market.
If the model's implied total (sum of player point projections) diverges from
the market total by more than 3 points, the model is wrong — not the market.

This module scales player point projections to be coherent with the
efficiently-priced game total, while partially scaling correlated stats
(assists) and leaving uncorrelated stats (rebounds) untouched.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_PARTIAL_SCALE_STATS = {"ast", "ast_mean", "fg3m", "fg3m_mean"}
_FULL_SCALE_STATS = {"pts", "pts_mean"}
_NO_SCALE_STATS = {"reb", "reb_mean", "stl", "stl_mean", "blk", "blk_mean", "turnover"}


class GameTotalAnchoring:
    """Scale player point projections to match efficiently-priced game total.

    Uses market total from /wnba/v1/odds (total_value field).
    Only applies correction when divergence > threshold.

    Args:
        threshold: Minimum divergence before correction is applied (default 3 pts)
        max_scale: Maximum scaling factor (prevents wild adjustments, default 1.15)
    """

    def __init__(self, threshold: float = 3.0, max_scale: float = 1.15) -> None:
        self.threshold = threshold
        self.max_scale = max_scale

    def anchor(
        self,
        player_projections: list[dict],
        market_total: float,
    ) -> list[dict]:
        """Apply proportional scaling per team to match market total.

        Algorithm:
          1. Compute model_implied_home = sum(pts_mean for home players)
          2. Compute model_implied_away = sum(pts_mean for away players)
          3. Compute each team's share of model total
          4. Scale each team's projections proportionally toward market total

        Args:
            player_projections: list of dicts with keys: team ("home"/"away"),
                pts_mean, and optionally ast_mean, reb_mean, etc.
            market_total: market-implied game total (from wnba/v1/odds)

        Returns:
            player_projections with anchored columns added (_anchored suffix)
        """
        home_total = sum(
            p.get("pts_mean", p.get("pts", 0.0))
            for p in player_projections
            if p.get("team") == "home"
        )
        away_total = sum(
            p.get("pts_mean", p.get("pts", 0.0))
            for p in player_projections
            if p.get("team") == "away"
        )
        model_total = home_total + away_total

        divergence = abs(model_total - market_total)
        log.debug(
            "GameTotalAnchoring: model=%.1f market=%.1f divergence=%.1f",
            model_total, market_total, divergence,
        )

        if divergence < self.threshold or model_total < 1.0:
            # Within acceptable divergence — add anchored columns = raw values
            for p in player_projections:
                p["pts_mean_anchored"] = p.get("pts_mean", p.get("pts", 0.0))
                p["ast_mean_anchored"] = p.get("ast_mean", p.get("ast", 0.0))
                p["reb_mean_anchored"] = p.get("reb_mean", p.get("reb", 0.0))
            return player_projections

        # Proportional scaling by team share
        home_share = home_total / max(model_total, 1.0)
        home_target = market_total * home_share
        away_target = market_total * (1.0 - home_share)

        home_scale = float(np.clip(
            home_target / max(home_total, 1.0),
            1.0 / self.max_scale, self.max_scale,
        ))
        away_scale = float(np.clip(
            away_target / max(away_total, 1.0),
            1.0 / self.max_scale, self.max_scale,
        ))

        for p in player_projections:
            scale = home_scale if p.get("team") == "home" else away_scale
            pts_raw = p.get("pts_mean", p.get("pts", 0.0))
            ast_raw = p.get("ast_mean", p.get("ast", 0.0))
            reb_raw = p.get("reb_mean", p.get("reb", 0.0))

            p["pts_mean_anchored"] = pts_raw * scale
            # Assists scale weakly with pace/possessions
            p["ast_mean_anchored"] = ast_raw * (1.0 + 0.3 * (scale - 1.0))
            # Rebounds do not scale with pace
            p["reb_mean_anchored"] = reb_raw
            p["anchor_scale_factor"] = scale

        log.info(
            "GameTotalAnchoring: home_scale=%.3f away_scale=%.3f (model=%.1f→market=%.1f)",
            home_scale, away_scale, model_total, market_total,
        )
        return player_projections

    def get_market_total(self, odds_df: "pd.DataFrame") -> Optional[float]:
        """Extract the best (consensus) market total from an odds DataFrame.

        Prefers the median across vendors to reduce outlier noise.
        """
        import pandas as pd  # noqa: PLC0415
        if odds_df is None or odds_df.empty:
            return None
        for col in ["total_value", "total", "game_total"]:
            if col in odds_df.columns:
                vals = pd.to_numeric(odds_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    return float(vals.median())
        return None

    def anchor_from_odds_df(
        self,
        player_projections: list[dict],
        odds_df: "pd.DataFrame",
    ) -> list[dict]:
        """Convenience wrapper: extract market total from odds DF then anchor."""
        market_total = self.get_market_total(odds_df)
        if market_total is None:
            log.warning("GameTotalAnchoring: no market total found in odds DF — skipping")
            return player_projections
        return self.anchor(player_projections, market_total)
