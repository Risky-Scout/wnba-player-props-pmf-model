"""Live In-Play Bayesian Engine for WNBA player props.

Enhancement 6: Live In-Play Bayesian Engine (NEW MODEL)
Priority: CRITICAL — opens entirely new profit stream

Architecture:
  1. Initialize from pre-game PMF (Gamma-Poisson conjugate prior)
  2. On each play-by-play event:
     a. Update observed stat counts for affected players
     b. Recompute Gamma posterior parameters  (O(1) per event)
     c. Estimate remaining minutes with foul / blowout adjustments
  3. Compute live posterior predictive PMF for each prop
  4. Output live P(over) / P(under) against market lines

The pre-game and live models share the same data (player ratings, features)
but use entirely different algorithms:
  Pre-game  → HistGradientBoosting batch training (off-line)
  Live      → Gamma-Poisson conjugate updating    (real-time, O(1))

Reference:
  Maddox, Sides & Harvill (2022). Bayesian estimation of in-game home team
  win probability for college basketball. JQAS.
  https://doi.org/10.1515/jqas-2021-0086
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

GAME_DURATION: float = 40.0   # WNBA regulation: 4 × 10-minute quarters
BLOWOUT_MARGIN: int  = 20     # Points margin that triggers blowout sub model
LEAGUE_AVG_PPP: float = 161.0 / GAME_DURATION  # ~4.025 PPP per minute

STATS_TRACKED = ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "ftm")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LivePlayerState:
    """Per-player live game state."""
    player_id: int
    team_id: int
    position_group: str = "wing"
    is_active: bool = True
    fouls: int = 0
    elapsed_minutes: float = 0.0

    # Observed stat counts this game
    observed: Dict[str, int] = field(default_factory=dict)

    # Gamma-Poisson prior (from pre-game model)
    alpha_prior: Dict[str, float] = field(default_factory=dict)
    beta_prior:  Dict[str, float] = field(default_factory=dict)

    # Posterior (updated on each event)
    alpha_post: Dict[str, float] = field(default_factory=dict)
    beta_post:  Dict[str, float] = field(default_factory=dict)

    # Pre-game projections (frozen at game start)
    pregame_projection: Dict[str, float] = field(default_factory=dict)
    pregame_var:        Dict[str, float] = field(default_factory=dict)
    pregame_minutes:    float = 25.0


@dataclass
class GameState:
    """Live game state."""
    home_team_id:  Optional[int] = None
    away_team_id:  Optional[int] = None
    home_score:    int   = 0
    away_score:    int   = 0
    period:        int   = 1
    elapsed_minutes: float = 0.0

    @property
    def margin(self) -> int:
        return abs(self.home_score - self.away_score)

    @property
    def time_remaining(self) -> float:
        return max(0.0, GAME_DURATION - self.elapsed_minutes)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveEngine:
    """Processes play-by-play events and updates all prop probabilities.

    Usage::

        engine = LiveEngine(pregame_ratings)
        engine.initialize_game(game_info)

        for event in pbp_stream:
            engine.process_event(event)
            for pid, stat in props:
                p_over, p_push = engine.compute_live_over_probability(pid, stat, line)
                proj, var       = engine.compute_live_projection(pid, stat)

    Parameters
    ----------
    pregame_ratings : {player_id: {stat: {"projection": float, "variance": float,
                                          "minutes": float}, ...,
                                   "team_id": int, "position_group": str}}
    """

    def __init__(self, pregame_ratings: Dict[int, dict], hmm=None):
        self.pregame = pregame_ratings
        self.player_states: Dict[int, LivePlayerState] = {}
        self.game_state = GameState()
        self._initialized = False
        # Optional HMM for regime-aware rate adjustment (Enhancement 22)
        self._hmm = hmm
        self._recent_events: List[dict] = []  # sliding window for HMM inference

    # ── Initialization ──────────────────────────────────────────────────────

    def initialize_game(self, game_info: dict) -> None:
        """Set up live engine with pre-game projections as Gamma-Poisson priors."""
        self.game_state = GameState(
            home_team_id=game_info.get("home_team_id"),
            away_team_id=game_info.get("away_team_id"),
        )

        for pid, ratings in self.pregame.items():
            s = LivePlayerState(
                player_id=pid,
                team_id=ratings.get("team_id", 0),
                position_group=ratings.get("position_group", "wing"),
                pregame_minutes=float(ratings.get("minutes", 25.0)),
            )

            for stat in STATS_TRACKED:
                info = ratings.get(stat)
                if not isinstance(info, dict):
                    # Try flat key format: "pts_projection"
                    proj = ratings.get(f"{stat}_projection", None)
                    var  = ratings.get(f"{stat}_variance",  None)
                else:
                    proj = info.get("projection")
                    var  = info.get("variance")

                if proj is None:
                    continue

                proj = float(proj)
                var  = float(var) if var is not None else max(proj * 1.5, 1.0)
                mins = float(ratings.get("minutes", 25.0))

                # ── Gamma-Poisson conjugate prior ─────────────────────────
                # rate_per_min = projection / minutes
                # Parameterize Gamma so E[λ] = rate/min, Var[λ] = σ² / mins²
                rate_per_min = proj / max(mins, 1.0)
                sigma = math.sqrt(max(var, 0.01))
                # Method-of-moments: α = (μ/σ)², β = μ/σ²
                alpha = (rate_per_min / sigma) ** 2
                beta  = rate_per_min / (sigma ** 2)
                alpha = max(alpha, 0.01)
                beta  = max(beta,  0.01)

                s.alpha_prior[stat] = alpha
                s.beta_prior[stat]  = beta
                s.alpha_post[stat]  = alpha
                s.beta_post[stat]   = beta
                s.pregame_projection[stat] = proj
                s.pregame_var[stat]        = var
                s.observed[stat]           = 0

            self.player_states[pid] = s

        self._initialized = True
        logger.info(
            "LiveEngine.initialize_game: %d players initialized for game %s",
            len(self.player_states),
            game_info.get("game_id", "?"),
        )

    # ── Event processing ────────────────────────────────────────────────────

    def process_event(self, event: dict) -> None:
        """Process a single PBP event and update all player posterior states."""
        if not self._initialized:
            logger.warning("LiveEngine.process_event called before initialize_game")
            return

        # Update game clock
        self.game_state.home_score = int(event.get("home_score", self.game_state.home_score))
        self.game_state.away_score = int(event.get("away_score", self.game_state.away_score))
        self.game_state.period     = int(event.get("period", self.game_state.period))
        clock_str = event.get("clock", event.get("game_clock", ""))
        self.game_state.elapsed_minutes = self._clock_to_elapsed(
            str(clock_str), self.game_state.period
        )

        # Update player observed stat counts
        for pid_raw, credits in event.get("stat_credits", {}).items():
            pid = int(pid_raw)
            if pid not in self.player_states:
                continue
            for stat, inc in credits.items():
                if stat in STATS_TRACKED:
                    self.player_states[pid].observed[stat] = (
                        self.player_states[pid].observed.get(stat, 0) + int(inc)
                    )

        # Track fouls
        etype = event.get("event_type", "")
        if etype == "foul":
            pid = event.get("primary_player_id")
            if pid and int(pid) in self.player_states:
                self.player_states[int(pid)].fouls += 1

        # Track substitutions
        if etype == "substitution":
            pid_out = event.get("secondary_player_id")
            pid_in  = event.get("primary_player_id")
            if pid_out and int(pid_out) in self.player_states:
                self.player_states[int(pid_out)].is_active = False
            if pid_in and int(pid_in) in self.player_states:
                self.player_states[int(pid_in)].is_active = True

        # Track recent events for HMM regime inference (Enhancement 22)
        margin = self.game_state.home_score - self.game_state.away_score
        elapsed = self.game_state.elapsed_minutes
        all_pts = sum(
            s.observed.get("pts", 0) for s in self.player_states.values()
        )
        recent_pts_rate = (all_pts / max(elapsed, 1.0)) / 4.0  # normalise
        self._recent_events.append({
            "margin":              margin,
            "clock_remaining_secs": max(0.0, (GAME_DURATION - elapsed) * 60),
            "cumulative_pts_rate": recent_pts_rate,
            "cumulative_reb_rate": 0.5,
            "possession_frac":     elapsed / GAME_DURATION,
            "timeout_flag":        1 if event.get("event_type") == "timeout" else 0,
            "recent_pts_rate":     recent_pts_rate,
        })
        # Keep only last 30 events for memory efficiency
        if len(self._recent_events) > 30:
            self._recent_events = self._recent_events[-30:]

        # Bayesian posterior update
        self._bayesian_update_all()

    def _bayesian_update_all(self) -> None:
        """Update Gamma-Poisson posteriors for all player-stat pairs.

        Conjugate update rule:
            α_post ← α_prior + observed_count
            β_post ← β_prior + expected_count_from_elapsed_time

        This is O(1) per player-stat pair and can process 25fps event streams.
        """
        t = self.game_state.elapsed_minutes
        t_frac = t / GAME_DURATION

        for s in self.player_states.values():
            for stat, X_t in s.observed.items():
                # Expected count if projecting from pre-game rate
                f_t = s.pregame_projection.get(stat, 0.0) * t_frac
                s.alpha_post[stat] = s.alpha_prior[stat] + float(X_t)
                s.beta_post[stat]  = s.beta_prior[stat]  + max(f_t, 0.001)

    # ── Projections ─────────────────────────────────────────────────────────

    def compute_live_projection(
        self, player_id: int, stat: str
    ) -> Tuple[float, float]:
        """Live posterior mean and variance for total stat by game end.

        Returns (projection, variance).
        """
        s = self.player_states.get(player_id)
        if not s:
            return 0.0, 0.0

        t = self.game_state.elapsed_minutes
        t_rem = max(0.0, GAME_DURATION - t)
        X_t = float(s.observed.get(stat, 0))

        alpha = s.alpha_post.get(stat, s.alpha_prior.get(stat, 1.0))
        beta  = s.beta_post.get(stat,  s.beta_prior.get(stat,  1.0))
        # post_rate = alpha/beta = posterior per-minute rate (pts/min, reb/min, etc.)
        post_rate = alpha / beta if beta > 0 else 0.0

        m_rem = self._estimate_remaining_minutes(s, t_rem)
        w_gs  = self._game_state_adj(s)

        # Enhancement 22: regime-aware rate adjustment via HMM
        if self._hmm is not None and self._recent_events:
            try:
                post_rate, _state_name, _state_prob = self._hmm.adjust_live_rate(
                    stat, post_rate, self._recent_events, blend_weight=0.70
                )
            except Exception:
                pass  # silently fall back to unadjusted rate

        # remaining = per-minute rate × remaining minutes
        remaining = post_rate * m_rem * w_gs
        proj = X_t + remaining

        # Posterior variance of lambda (per-minute rate), scaled to remaining minutes²
        post_var = alpha / (beta ** 2) if alpha > 0 and beta > 0 else 1.0
        var = post_var * (m_rem ** 2) + 0.05 * max(proj, 1.0) ** 2

        return float(proj), float(var)

    def compute_live_over_probability(
        self, player_id: int, stat: str, line: float
    ) -> Tuple[float, float]:
        """Compute P(final_stat > line) and P(final_stat == line) for live prop.

        Uses Negative Binomial posterior predictive (exact conjugate).

        Returns (p_over, p_push).
        """
        s = self.player_states.get(player_id)
        if not s:
            return 0.5, 0.0

        X_t = float(s.observed.get(stat, 0))
        rem_line = line - X_t

        if rem_line < 0:
            return 0.999, 0.001
        if rem_line == 0:
            return 0.99, 0.01

        alpha = s.alpha_post.get(stat, s.alpha_prior.get(stat, 1.0))
        beta  = s.beta_post.get(stat,  s.beta_prior.get(stat,  1.0))

        t_rem = max(0.0, GAME_DURATION - self.game_state.elapsed_minutes)
        m_rem = self._estimate_remaining_minutes(s, t_rem)
        min_frac = m_rem / max(s.pregame_minutes, 1.0)

        # Scale posterior for remaining time
        r_post = max(alpha * min_frac, 0.001)
        p_nb   = beta / (beta + 1.0 / max(min_frac, 0.01))
        p_nb   = float(np.clip(p_nb, 1e-6, 1 - 1e-6))

        try:
            p_at_most = float(sp_stats.nbinom.cdf(
                int(math.floor(rem_line)), n=r_post, p=p_nb
            ))
            p_over = 1.0 - p_at_most
            p_push = (
                float(sp_stats.nbinom.pmf(int(rem_line), n=r_post, p=p_nb))
                if rem_line == int(rem_line)
                else 0.0
            )
        except Exception:
            p_over, p_push = 0.5, 0.0

        return float(np.clip(p_over, 0.001, 0.999)), float(p_push)

    def compute_game_total_live(
        self, market_line: float
    ) -> Tuple[float, float]:
        """Live game total projection and P(over market_line).

        Returns (projected_total, p_over).
        """
        obs_total = self.game_state.home_score + self.game_state.away_score
        t     = self.game_state.elapsed_minutes
        t_rem = max(0.0, GAME_DURATION - t)

        # Observed pace factor vs league average
        if t > 0:
            pace_factor = (obs_total / t) / LEAGUE_AVG_PPP
        else:
            pace_factor = 1.0

        remaining = pace_factor * LEAGUE_AVG_PPP * GAME_DURATION * (t_rem / GAME_DURATION)
        proj = obs_total + remaining

        # Uncertainty shrinks as game progresses (√ time remaining)
        sigma = 11.0 * math.sqrt(t_rem / GAME_DURATION) * math.sqrt(2)
        z = (market_line - proj) / max(sigma, 0.1)
        p_over = float(np.clip(1.0 - sp_stats.norm.cdf(z), 0.001, 0.999))

        return float(proj), p_over

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the current live state."""
        return {
            "elapsed_minutes": self.game_state.elapsed_minutes,
            "period": self.game_state.period,
            "home_score": self.game_state.home_score,
            "away_score": self.game_state.away_score,
            "players": {
                str(pid): {
                    "observed": s.observed,
                    "fouls": s.fouls,
                    "is_active": s.is_active,
                    "alpha_post": s.alpha_post,
                    "beta_post": s.beta_post,
                }
                for pid, s in self.player_states.items()
            },
        }

    def get_all_live_probabilities(
        self, props: List[Tuple[int, str, float]]
    ) -> List[dict]:
        """Batch-compute live probabilities for a list of props.

        Parameters
        ----------
        props : [(player_id, stat, line), ...]

        Returns
        -------
        [{"player_id", "stat", "line", "projection", "variance",
          "p_over", "p_push", "p_under", "elapsed_minutes"}, ...]
        """
        results = []
        for pid, stat, line in props:
            proj, var = self.compute_live_projection(pid, stat)
            p_over, p_push = self.compute_live_over_probability(pid, stat, line)
            results.append({
                "player_id": pid,
                "stat": stat,
                "line": line,
                "projection": round(proj, 3),
                "variance": round(var, 3),
                "p_over": round(p_over, 4),
                "p_push": round(p_push, 4),
                "p_under": round(max(0.001, 1.0 - p_over - p_push), 4),
                "elapsed_minutes": round(self.game_state.elapsed_minutes, 2),
                "period": self.game_state.period,
            })
        return results

    # ── Internal helpers ────────────────────────────────────────────────────

    def _estimate_remaining_minutes(
        self, s: LivePlayerState, t_rem: float
    ) -> float:
        """Estimate remaining playing time with foul and blowout adjustments."""
        if not s.is_active:
            return 0.0

        base = s.pregame_minutes * (t_rem / GAME_DURATION)

        # Foul trouble (WNBA: 6 fouls = foul out)
        if s.fouls >= 6:
            base *= 0.20   # nearly fouled out
        elif s.fouls >= 5:
            base *= 0.40   # in serious foul trouble
        elif s.fouls >= 4:
            base *= 0.80   # coach manages minutes

        # Blowout substitution model
        margin = self.game_state.margin
        t_rem_clk = t_rem   # remaining game minutes
        if margin > BLOWOUT_MARGIN and t_rem_clk < 8.0:
            if s.pregame_minutes > 28:
                base *= 0.30   # stars sit in blowouts
            else:
                base *= 1.40   # bench gets extended run

        # Close-game extension for stars
        if margin < 5 and t_rem_clk < 5.0 and s.pregame_minutes > 28:
            base *= 1.05

        return max(0.0, base)

    def _game_state_adj(self, s: LivePlayerState) -> float:
        """Minute-rate adjustment for game state (garbage time, foul trouble)."""
        margin = self.game_state.margin
        t_rem  = self.game_state.time_remaining
        if margin > BLOWOUT_MARGIN and t_rem < 8.0:
            return 0.90   # counting stats reduced in garbage time
        return 1.0

    @staticmethod
    def _clock_to_elapsed(clock_str: str, period: int) -> float:
        """Convert 'MM:SS' clock string to total elapsed game minutes."""
        if not clock_str or ":" not in clock_str:
            return min((period - 1) * 10.0, GAME_DURATION)
        try:
            parts = clock_str.split(":")
            mins  = int(parts[0])
            secs  = int(parts[1]) if len(parts) > 1 else 0
            remaining_in_period = mins + secs / 60.0
            elapsed = (period - 1) * 10.0 + 10.0 - remaining_in_period
            return float(max(0.0, min(elapsed, GAME_DURATION)))
        except (ValueError, IndexError):
            return min((period - 1) * 10.0, GAME_DURATION)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_pregame_ratings_from_pmfs(
    pmfs_long: "pd.DataFrame",   # noqa: F821
    feature_wide: "pd.DataFrame",  # noqa: F821
) -> dict[int, dict]:
    """Build the pregame_ratings dict expected by LiveEngine from pipeline outputs.

    Parameters
    ----------
    pmfs_long : long-format PMF DataFrame (player_id, stat, projection, variance, ...)
    feature_wide : wide feature DataFrame (player_id, projected_minutes_proxy, position, ...)

    Returns
    -------
    {player_id: {stat: {"projection": float, "variance": float}, ...,
                 "minutes": float, "team_id": int, "position_group": str}}
    """
    import pandas as pd  # noqa: PLC0415

    ratings: dict[int, dict] = {}
    proj_col = "projection" if "projection" in pmfs_long.columns else "pred_mean"
    var_col  = "variance"   if "variance"   in pmfs_long.columns else None

    # Build minutes / team / position lookup from wide table
    meta_cols = ["player_id"]
    for c in ["projected_minutes_proxy", "team_id", "position"]:
        if c in feature_wide.columns:
            meta_cols.append(c)
    meta = feature_wide[meta_cols].drop_duplicates("player_id").set_index("player_id")

    for pid, grp in pmfs_long.groupby("player_id"):
        pid = int(pid)
        info: dict = {}
        for _, row in grp.iterrows():
            stat = row.get("stat")
            if stat not in STATS_TRACKED:
                continue
            proj = float(row.get(proj_col, 0))
            var  = float(row.get(var_col, max(proj * 1.5, 1.0))) if var_col else max(proj * 1.5, 1.0)
            info[stat] = {"projection": proj, "variance": var}

        if pid in meta.index:
            row_meta = meta.loc[pid]
            info["minutes"] = float(row_meta.get("projected_minutes_proxy", 25.0) or 25.0)
            info["team_id"] = int(row_meta.get("team_id", 0))
            pos_raw = row_meta.get("position", "F")
            from wnba_props_model.models.usage_transfer import _normalize_position  # noqa: PLC0415
            info["position_group"] = _normalize_position(pos_raw)
        else:
            info["minutes"] = 25.0
            info["team_id"] = 0
            info["position_group"] = "wing"

        if info:
            ratings[pid] = info

    return ratings
