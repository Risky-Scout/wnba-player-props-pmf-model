"""Possession-Level Monte Carlo Game Simulation (Enhancement 14).

Simulates a complete WNBA game possession-by-possession, producing:
1. Full JOINT distribution of all player stats — no Gaussian copula needed.
   All cross-stat and cross-player correlations are endogenous to the simulation.
2. Game total, team totals, and final margin — coherent by construction.
3. Quarter-by-quarter scoring for conditional/live props.
4. Blowout / garbage-time detection with roster swaps.

Theory
------
Each possession has one discrete outcome (made_2pt, made_3pt, turnover, etc.)
and that outcome simultaneously updates ALL relevant player stats.  This is the
EPV (Expected Possession Value) framework (Cervone et al.; Terner & Franks 2020)
applied as a simulation rather than an analytical model.

Advantages over Gaussian copula
- Nonlinear dependencies: in a 130-pt game ALL starters' points are elevated.
- Game-total coherence: sum(player_pts) == game_total by construction.
- Blowout dynamics: late-game margin triggers bench substitutions, naturally
  creating negative star/bench correlations without hand-coded copula weights.

References
----------
Terner & Franks (2020). Modeling Player and Team Performance in Basketball.
Annual Review of Statistics and Its Application.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# WNBA-calibrated defaults
WNBA_AVG_POSSESSIONS_PER_GAME = 78.0
WNBA_QUARTER_DURATION_MINS = 10.0
WNBA_MAX_PLAYER_MINUTES = 40.0

POSSESSION_OUTCOMES = [
    "made_2pt",
    "missed_2pt",
    "made_3pt",
    "missed_3pt",
    "turnover",
    "foul_drawn",
    "pass",  # ball moves to a teammate; re-simulated
]


@dataclass
class PlayerRates:
    """Per-possession rates for one player, WNBA-calibrated defaults."""
    player_id:        int
    usage:            float = 0.15
    fg2_pct:          float = 0.45
    fg3_pct:          float = 0.33
    ft_pct:           float = 0.80
    shoot_2_rate:     float = 0.25    # P(shoot 2pt | possession)
    shoot_3_rate:     float = 0.20    # P(shoot 3pt | possession)
    turnover_rate:    float = 0.12
    foul_drawn_rate:  float = 0.08
    ast_rate:         float = 0.15
    oreb_rate:        float = 0.05
    dreb_rate:        float = 0.15
    stl_rate:         float = 0.03
    blk_rate:         float = 0.02

    @property
    def pass_rate(self) -> float:
        total = self.shoot_2_rate + self.shoot_3_rate + self.turnover_rate + self.foul_drawn_rate
        return max(0.0, 1.0 - total)

    @classmethod
    def from_feature_row(cls, player_id: int, row: dict[str, Any]) -> "PlayerRates":
        """Build from a feature-table row or projection dict."""
        def g(k: str, default: float) -> float:
            v = row.get(k, row.get(k.replace("player_", ""), default))
            return float(v) if v is not None and not np.isnan(float(v)) else default

        usage = g("player_usage_rate_l5", g("player_usage_rate_season", 0.15))
        usage = np.clip(usage, 0.05, 0.40)

        # Derive approximate per-possession rates from per-game rates / pace
        pace = g("team_pace", WNBA_AVG_POSSESSIONS_PER_GAME)
        minutes = g("projected_minutes", g("player_minutes_l5", 20.0))

        poss_per_game = usage * pace
        poss_played = poss_per_game if poss_per_game > 0 else 1.0

        fg2a = g("player_fg2a_l5", poss_played * 0.25)
        fg3a = g("player_fg3a_l5", poss_played * 0.20)
        fta  = g("player_fta_l5",  poss_played * 0.08)
        tov  = g("player_turnover_l5", poss_played * 0.12)

        shoot2 = np.clip(fg2a / max(poss_played, 1), 0.0, 0.45)
        shoot3 = np.clip(fg3a / max(poss_played, 1), 0.0, 0.35)
        tov_r  = np.clip(tov  / max(poss_played, 1), 0.0, 0.20)
        foul_r = np.clip(fta  / max(poss_played, 1) / 2.0, 0.0, 0.15)

        return cls(
            player_id=player_id,
            usage=usage,
            fg2_pct=np.clip(g("player_fg2_pct_l5", 0.45), 0.25, 0.75),
            fg3_pct=np.clip(g("player_fg3_pct_l5", 0.33), 0.15, 0.60),
            ft_pct=np.clip(g("player_ft_pct_l5", 0.80), 0.40, 1.0),
            shoot_2_rate=shoot2,
            shoot_3_rate=shoot3,
            turnover_rate=tov_r,
            foul_drawn_rate=foul_r,
            ast_rate=np.clip(g("player_ast_per_min_l5", 0.15), 0.0, 0.40),
            oreb_rate=np.clip(g("player_oreb_rate_l5", 0.05), 0.0, 0.15),
            dreb_rate=np.clip(g("player_dreb_rate_l5", 0.15), 0.0, 0.30),
            stl_rate=np.clip(g("player_stl_per_min_l5", 0.03) * 0.4, 0.0, 0.10),
            blk_rate=np.clip(g("player_blk_per_min_l5", 0.02) * 0.4, 0.0, 0.10),
        )


@dataclass
class PossessionSimulator:
    """Simulate a complete WNBA game at the possession level.

    Key insight: each possession updates ALL stats simultaneously, so
    cross-player and cross-stat correlations are endogenous.  No copula needed.

    Parameters
    ----------
    home_rates, away_rates : dict {player_id: PlayerRates}
    home_minutes, away_minutes : dict {player_id: projected_minutes}
    team_pace : expected total possessions per game (each team's)
    n_simulations : Monte Carlo replications
    """
    home_rates:    dict[int, PlayerRates]
    away_rates:    dict[int, PlayerRates]
    home_minutes:  dict[int, float]
    away_minutes:  dict[int, float]
    team_pace:     float = WNBA_AVG_POSSESSIONS_PER_GAME
    n_simulations: int = 5_000

    # ── Public API ───────────────────────────────────────────────────────────

    def simulate_game(self) -> list[dict]:
        """Run n_simulations complete game simulations.

        Returns list of dicts, each containing::
            {
              "player_stats": {player_id: {stat: value}},
              "home_score": int,
              "away_score": int,
              "margin": int,
              "possessions": int,
            }
        """
        return [self._simulate_single_game() for _ in range(self.n_simulations)]

    def compute_prop_pmf(
        self,
        player_id: int,
        stat: str,
        line: float,
        n_sims: int | None = None,
    ) -> dict[str, Any]:
        """Compute over/under probabilities from possession-level simulations.

        This replaces the marginal NegBin PMF + copula approach.

        Returns
        -------
        dict with keys: pmf, p_over, p_under, p_push, mean, std, percentiles
        """
        sims = self.simulate_game() if n_sims is None else [self._simulate_single_game() for _ in range(n_sims)]
        values = np.array([
            r["player_stats"].get(player_id, {}).get(stat, 0)
            for r in sims
        ])

        unique, counts = np.unique(values, return_counts=True)
        pmf = {int(v): float(c / len(values)) for v, c in zip(unique, counts)}

        p_over  = float(np.mean(values > line))
        p_push  = float(np.mean(values == line))
        p_under = float(np.mean(values < line))

        return {
            "pmf": pmf,
            "p_over": p_over,
            "p_under": p_under,
            "p_push": p_push,
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)),
            "percentiles": {
                p: float(np.percentile(values, p)) for p in [5, 10, 25, 50, 75, 90, 95]
            },
        }

    def compute_joint_pmf(
        self,
        player_a: int,
        stat_a: str,
        line_a: float,
        player_b: int,
        stat_b: str,
        line_b: float,
    ) -> dict[str, float]:
        """Compute joint over/over probability for a combo prop."""
        sims = self.simulate_game()
        a_vals = np.array([r["player_stats"].get(player_a, {}).get(stat_a, 0) for r in sims])
        b_vals = np.array([r["player_stats"].get(player_b, {}).get(stat_b, 0) for r in sims])
        return {
            "p_both_over":  float(np.mean((a_vals > line_a) & (b_vals > line_b))),
            "p_a_over_b_under": float(np.mean((a_vals > line_a) & (b_vals <= line_b))),
            "p_a_under_b_over": float(np.mean((a_vals <= line_a) & (b_vals > line_b))),
            "p_both_under": float(np.mean((a_vals <= line_a) & (b_vals <= line_b))),
        }

    def compute_game_total_pmf(self, line: float) -> dict[str, Any]:
        """Compute game-total over/under probability from simulations."""
        sims = self.simulate_game()
        totals = np.array([r["home_score"] + r["away_score"] for r in sims])
        return {
            "p_over":  float(np.mean(totals > line)),
            "p_under": float(np.mean(totals <= line)),
            "mean":    float(np.mean(totals)),
            "std":     float(np.std(totals)),
        }

    # ── Core simulation ──────────────────────────────────────────────────────

    def _simulate_single_game(self) -> dict:
        """Simulate one complete game possession by possession."""
        player_stats = {
            pid: {s: 0 for s in ["pts", "reb", "ast", "stl", "blk",
                                   "turnover", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]}
            for pid in list(self.home_rates) + list(self.away_rates)
        }

        home_score = 0
        away_score = 0
        possession_num = 0
        total_poss = int(np.random.poisson(self.team_pace))

        for poss_idx in range(total_poss * 2):  # both teams possess
            is_home = poss_idx % 2 == 0
            poss_num = poss_idx // 2
            quarter = min(1 + poss_num * 4 // max(total_poss, 1), 4)
            margin = home_score - away_score

            # Rotation: determine who is active
            active = self._active_players(
                poss_num, total_poss, quarter, margin, is_home
            )
            if not active:
                continue

            rates_map = self.home_rates if is_home else self.away_rates
            active = [p for p in active if p in rates_map]
            if not active:
                continue

            # Ball handler by usage weight
            handler = self._weighted_choice(active, rates_map, key="usage")
            if handler is None:
                continue

            # Simulate possession (recursive for pass)
            outcome, scorer = self._simulate_possession(handler, active, rates_map, depth=0)
            primary = scorer if scorer is not None else handler

            # Stat updates
            if outcome == "made_2pt":
                player_stats[primary]["pts"] += 2
                player_stats[primary]["fgm"] += 1
                player_stats[primary]["fga"] += 1
                if is_home:
                    home_score += 2
                else:
                    away_score += 2
                # Assist (40% of made shots)
                if np.random.random() < 0.40:
                    ast = self._weighted_choice(
                        [p for p in active if p != primary], rates_map, key="ast_rate"
                    )
                    if ast:
                        player_stats[ast]["ast"] += 1
                # Rebound (offensive if missed — not applicable; defensive)
                def_team = self.away_rates if is_home else self.home_rates
                reb = self._weighted_choice(list(def_team.keys()), def_team, key="dreb_rate")
                if reb:
                    player_stats[reb]["reb"] += 1

            elif outcome == "missed_2pt":
                player_stats[primary]["fga"] += 1
                self._rebound(player_stats, primary, active,
                               rates_map, is_home, total_poss, poss_num, margin)

            elif outcome == "made_3pt":
                player_stats[primary]["pts"] += 3
                player_stats[primary]["fgm"] += 1
                player_stats[primary]["fga"] += 1
                player_stats[primary]["fg3m"] += 1
                player_stats[primary]["fg3a"] += 1
                if is_home:
                    home_score += 3
                else:
                    away_score += 3
                if np.random.random() < 0.30:
                    ast = self._weighted_choice(
                        [p for p in active if p != primary], rates_map, key="ast_rate"
                    )
                    if ast:
                        player_stats[ast]["ast"] += 1

            elif outcome == "missed_3pt":
                player_stats[primary]["fga"] += 1
                player_stats[primary]["fg3a"] += 1
                self._rebound(player_stats, primary, active,
                               rates_map, is_home, total_poss, poss_num, margin)

            elif outcome == "turnover":
                player_stats[primary]["turnover"] += 1
                # Steal: 15% of turnovers
                opp_map = self.away_rates if is_home else self.home_rates
                if np.random.random() < 0.15:
                    stealer = self._weighted_choice(list(opp_map.keys()), opp_map, key="stl_rate")
                    if stealer:
                        player_stats[stealer]["stl"] += 1

            elif outcome == "foul_drawn":
                ft_pct = rates_map[primary].ft_pct
                ftm = int(np.random.binomial(2, ft_pct))
                player_stats[primary]["pts"] += ftm
                player_stats[primary]["ftm"] += ftm
                player_stats[primary]["fta"] += 2
                if is_home:
                    home_score += ftm
                else:
                    away_score += ftm

            possession_num += 1

        return {
            "player_stats": player_stats,
            "home_score":   home_score,
            "away_score":   away_score,
            "margin":       home_score - away_score,
            "possessions":  possession_num,
        }

    def _simulate_possession(
        self,
        handler: int,
        active: list[int],
        rates_map: dict[int, PlayerRates],
        depth: int = 0,
    ) -> tuple[str, int | None]:
        """Return (outcome, primary_scorer)."""
        if depth > 3 or handler not in rates_map:
            return "turnover", handler

        r = rates_map[handler]
        rand = np.random.random()
        cum = 0.0

        cum += r.shoot_2_rate
        if rand < cum:
            made = np.random.random() < r.fg2_pct
            return ("made_2pt" if made else "missed_2pt"), handler

        cum += r.shoot_3_rate
        if rand < cum:
            made = np.random.random() < r.fg3_pct
            return ("made_3pt" if made else "missed_3pt"), handler

        cum += r.turnover_rate
        if rand < cum:
            return "turnover", handler

        cum += r.foul_drawn_rate
        if rand < cum:
            return "foul_drawn", handler

        # Pass to teammate
        others = [p for p in active if p != handler]
        if others:
            secondary = self._weighted_choice(others, rates_map, key="usage")
            if secondary:
                return self._simulate_possession(secondary, active, rates_map, depth + 1)
        return "turnover", handler

    def _active_players(
        self, poss_num: int, total_poss: int, quarter: int, margin: int, is_home: bool
    ) -> list[int]:
        """Return active player IDs for the current game state (WNBA rotation)."""
        minutes_map = self.home_minutes if is_home else self.away_minutes
        rates_map   = self.home_rates   if is_home else self.away_rates

        frac = poss_num / max(total_poss, 1)
        starters = sorted(minutes_map, key=lambda p: -minutes_map.get(p, 0))[:5]
        bench    = [p for p in minutes_map if p not in starters]

        if frac < 0.25:          # Q1: starting unit
            return [p for p in starters if p in rates_map]
        elif frac < 0.50:        # Q2: mixed unit
            return [p for p in (starters[:2] + bench[:3]) if p in rates_map]
        elif frac < 0.75:        # Q3: starting unit
            return [p for p in starters if p in rates_map]
        else:                    # Q4: game-script dependent
            if abs(margin) > 20:  # blowout → bench unit
                return [p for p in bench[:5] if p in rates_map] or [p for p in starters if p in rates_map]
            return [p for p in starters if p in rates_map]

    def _rebound(
        self,
        player_stats: dict,
        shooter: int,
        active: list[int],
        rates_map: dict[int, PlayerRates],
        is_home: bool,
        total_poss: int,
        poss_num: int,
        margin: int,
    ) -> None:
        """Handle rebound assignment after a miss."""
        opp_map = self.away_rates if is_home else self.home_rates
        opp_active = self._active_players(poss_num, total_poss,
                                          min(1 + poss_num * 4 // max(total_poss, 1), 4),
                                          margin, not is_home)
        if np.random.random() < 0.70:  # Defensive rebound
            reb = self._weighted_choice(
                [p for p in opp_active if p in opp_map], opp_map, key="dreb_rate"
            )
        else:                          # Offensive rebound
            reb = self._weighted_choice(
                [p for p in active if p != shooter], rates_map, key="oreb_rate"
            )
        if reb:
            player_stats[reb]["reb"] += 1

    @staticmethod
    def _weighted_choice(
        players: list[int],
        rates_map: dict[int, PlayerRates],
        key: str,
    ) -> int | None:
        """Sample one player weighted by a PlayerRates attribute."""
        if not players:
            return None
        weights = np.array([getattr(rates_map.get(p, PlayerRates(p)), key, 0.1) for p in players], dtype=float)
        weights = np.clip(weights, 1e-9, None)
        weights /= weights.sum()
        return int(np.random.choice(players, p=weights))

    # ── Factory helpers ──────────────────────────────────────────────────────

    @classmethod
    def from_projections(
        cls,
        projections: list[dict],
        n_simulations: int = 5_000,
    ) -> "PossessionSimulator":
        """Build a PossessionSimulator from a list of player projection dicts.

        Each dict should contain at minimum:
            player_id, team (home/away), projected_minutes, and optional
            per-possession rates (usage, fg2_pct, fg3_pct, etc.)
        """
        home_rates:   dict[int, PlayerRates] = {}
        away_rates:   dict[int, PlayerRates] = {}
        home_minutes: dict[int, float] = {}
        away_minutes: dict[int, float] = {}
        pace = WNBA_AVG_POSSESSIONS_PER_GAME

        for proj in projections:
            pid  = int(proj["player_id"])
            team = proj.get("team", "home").lower()
            mins = float(proj.get("projected_minutes", proj.get("player_minutes_l5", 20.0)))
            pr   = PlayerRates.from_feature_row(pid, proj)
            if "team_pace" in proj:
                pace = float(proj["team_pace"])
            if team == "home":
                home_rates[pid]   = pr
                home_minutes[pid] = mins
            else:
                away_rates[pid]   = pr
                away_minutes[pid] = mins

        return cls(
            home_rates=home_rates,
            away_rates=away_rates,
            home_minutes=home_minutes,
            away_minutes=away_minutes,
            team_pace=pace,
            n_simulations=n_simulations,
        )


def build_simulator_from_features(
    player_feats:   dict[int, dict],
    n_simulations:  int = 5_000,
    rng_seed:       int | None = None,
) -> PossessionSimulator:
    """Build a PossessionSimulator from a lightweight player-feature dict.

    Parameters
    ----------
    player_feats : {player_id: {usage, proj_minutes, fg2_pct, fg3_pct, ...}}
        The first half of players (by player_id order) are assigned to "home",
        the second half to "away" — suitable for simulating both teams.
    n_simulations : Monte Carlo replications
    rng_seed : optional numpy random seed for reproducibility

    Returns
    -------
    Configured PossessionSimulator ready to call .simulate_game()
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    projections = []
    player_ids = sorted(player_feats.keys())
    n_home = max(1, len(player_ids) // 2)
    for i, pid in enumerate(player_ids):
        team = "home" if i < n_home else "away"
        row = dict(player_feats[pid])
        row["player_id"] = pid
        row["team"] = team
        # Map 'proj_minutes' → 'projected_minutes'
        if "proj_minutes" in row and "projected_minutes" not in row:
            row["projected_minutes"] = row["proj_minutes"]
        projections.append(row)

    return PossessionSimulator.from_projections(projections, n_simulations=n_simulations)
