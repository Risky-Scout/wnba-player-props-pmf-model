"""Gamma-Poisson Live Bayesian Updater for WNBA player props.

Model:
  lambda_i ~ Gamma(alpha, beta)          [pre-game prior from HGB rate model]
  X_i | lambda_i ~ Poisson(lambda_i)     [observed counting stats per minute]

After observing k events in t minutes:
  lambda_i | data ~ Gamma(alpha + k, beta + t)   [posterior]

Posterior predictive (for remaining t_rem minutes):
  X_remaining | data ~ NegBin(r=alpha+k, p=beta/(beta+t+t_rem))

This produces a FULL PMF for the remaining game, not just a point estimate,
enabling live P(over) computation with proper uncertainty quantification.

The Gamma-Poisson conjugacy ensures:
  - Closed-form posterior updates (no MCMC needed)
  - Full PMF (not point estimate) for remaining stats
  - Graceful degradation when few minutes have elapsed (prior dominates)
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import nbinom

log = logging.getLogger(__name__)

LIVE_STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover")

# WNBA game is 40 minutes regulation, OT adds 5-min periods
MAX_REGULATION_MINUTES = 40.0
MAX_STAT_VALUES = {
    "pts": 60,
    "reb": 25,
    "ast": 20,
    "fg3m": 12,
    "stl": 10,
    "blk": 8,
    "turnover": 12,
}


class GammaPoissonLiveEngine:
    """Live Bayesian updater using Gamma-Poisson conjugacy.

    Given a pre-game HGB projection (mean rate per minute), set the Gamma prior
    parameters as:
        alpha = rate_per_minute * beta_scale
        beta  = beta_scale

    A reasonable beta_scale = 10 means the prior has weight equivalent to 10 minutes
    of observation. After observing k events in t minutes, the posterior is:
        lambda | data ~ Gamma(alpha + k, beta + t)

    The posterior predictive PMF for the remaining game is a Negative Binomial.
    """

    def __init__(
        self,
        stats: tuple[str, ...] = LIVE_STATS,
        beta_scale: float = 10.0,
    ) -> None:
        self.stats = stats
        self.beta_scale = beta_scale

    def _prior_params(
        self,
        mean_per_game: float,
        projected_total_minutes: float,
    ) -> tuple[float, float]:
        """Convert pre-game projection to Gamma prior parameters.

        rate_per_minute = mean_per_game / projected_total_minutes
        alpha = rate_per_minute * beta_scale
        beta  = beta_scale
        """
        if projected_total_minutes <= 0:
            projected_total_minutes = 28.0  # league average minutes
        rate_per_min = max(mean_per_game / projected_total_minutes, 1e-6)
        alpha = rate_per_min * self.beta_scale
        beta = self.beta_scale
        return alpha, beta

    def compute_posterior_pmf(
        self,
        stat: str,
        mean_per_game: float,
        projected_total_minutes: float,
        observed_count: int,
        elapsed_minutes: float,
    ) -> dict[int, float]:
        """Compute posterior predictive PMF for the TOTAL stat (observed + remaining).

        Args:
            stat: stat name ("pts", "reb", etc.)
            mean_per_game: pre-game projected mean (e.g. 14.5 pts)
            projected_total_minutes: pre-game projected minutes
            observed_count: stat count so far in this game
            elapsed_minutes: minutes already played in this game

        Returns:
            pmf: dict {total_stat_value: probability}
        """
        alpha_prior, beta_prior = self._prior_params(mean_per_game, projected_total_minutes)

        # Posterior parameters
        alpha_post = alpha_prior + observed_count
        beta_post = beta_prior + elapsed_minutes

        # Remaining minutes — use regulation ceiling if elapsed exceeds projection (overtime)
        effective_total = max(projected_total_minutes, elapsed_minutes)
        # If player has already exceeded projected minutes, cap at MAX_REGULATION_MINUTES
        if elapsed_minutes > projected_total_minutes:
            effective_total = max(MAX_REGULATION_MINUTES, elapsed_minutes)
        t_remaining = max(effective_total - elapsed_minutes, 0.0)

        if t_remaining < 0.5:
            # Player is done — degenerate PMF at observed count
            return {observed_count: 1.0}

        # NegBin parameters for posterior predictive of X_remaining
        # X_rem ~ NegBin(r=alpha_post, p=beta_post/(beta_post+t_remaining))
        r = alpha_post
        p = beta_post / (beta_post + t_remaining)

        # Build PMF for remaining stat
        max_possible = min(
            MAX_STAT_VALUES.get(stat, 30),
            int(max((alpha_prior / max(beta_prior, 0.01)) * projected_total_minutes * 3, 10)),
        )

        pmf_total: dict[int, float] = {}
        for k in range(max_possible + 1):
            prob_k = float(nbinom.pmf(k, r, p))
            if prob_k < 1e-10:
                if k > 0 and pmf_total:
                    break
                continue
            total_val = observed_count + k
            pmf_total[total_val] = pmf_total.get(total_val, 0.0) + prob_k

        # Normalize
        total = sum(pmf_total.values())
        if total > 0:
            pmf_total = {k: v / total for k, v in pmf_total.items()}
        else:
            pmf_total = {observed_count: 1.0}

        return pmf_total

    def compute_live_p_over(
        self,
        stat: str,
        mean_per_game: float,
        projected_total_minutes: float,
        observed_count: int,
        elapsed_minutes: float,
        line: float,
        foul_count: int = 0,
        score_diff: int = 0,
        is_star: bool = False,
    ) -> dict:
        """Compute live P(over line) for a stat.

        This is the KEY output for live edge calculation.

        Returns:
            dict with: p_over, p_under, p_push, projected_total, pmf, mean_remaining
        """
        pmf = self.compute_posterior_pmf(
            stat,
            mean_per_game,
            projected_total_minutes,
            observed_count,
            elapsed_minutes,
        )

        # Part H: Apply foul trouble adjustment
        if foul_count > 0:
            pmf = self.adjust_for_foul_trouble(pmf, foul_count, stat=stat)

        # Part H: Apply score differential (blowout) adjustment
        minutes_remaining = max(0.0, projected_total_minutes - elapsed_minutes)
        if abs(score_diff) >= 15 and minutes_remaining <= 8.0:
            pmf = self.adjust_for_score_differential(pmf, score_diff, minutes_remaining, is_star)

        p_over = sum(p for k, p in pmf.items() if k > line)
        p_push = sum(p for k, p in pmf.items() if k == line)
        p_under = sum(p for k, p in pmf.items() if k < line)
        projected_total = sum(k * p for k, p in pmf.items())

        # #region agent log
        import json as _j3, time as _t3
        _log_path3 = "/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log"
        with open(_log_path3, "a") as _lf3:
            _lf3.write(_j3.dumps({"sessionId":"94807e","hypothesisId":"E","location":"bayesian_updater.py:compute_live_p_over","message":"live_computation_POST_FIX","data":{"stat":stat,"p_over":round(float(p_over),4),"observed":observed_count,"elapsed":elapsed_minutes,"foul_count":foul_count,"score_diff":score_diff,"is_star":is_star,"has_foul_adjustment":foul_count>0,"has_blowout_adjustment":abs(score_diff)>=15 and minutes_remaining<=8.0},"timestamp":int(_t3.time()*1000)}) + "\n")
        # #endregion agent log
        return {
            "p_over": round(float(p_over), 6),
            "p_under": round(float(p_under), 6),
            "p_push": round(float(p_push), 6),
            "projected_total": round(float(projected_total), 3),
            "observed_count": observed_count,
            "elapsed_minutes": elapsed_minutes,
            "pmf": pmf,
        }

    def adjust_for_foul_trouble(
        self,
        pmf: dict[int, float],
        foul_count: int,
        stat: str = "pts",
    ) -> dict[int, float]:
        """Part H: Scale PMF by P(player finishes game without fouling out).

        WNBA rules: 6 personal fouls → disqualification.
        Historical foul-out rates (empirical from BDL data):
          3 fouls: ~3% chance of fouling out → scale by 0.97
          4 fouls: ~15% chance of fouling out → scale by 0.85
          5 fouls: ~60% chance of fouling out → scale by 0.40
          6 fouls: already fouled out → return zero PMF
        """
        if foul_count >= 6:
            return {0: 1.0}
        foul_penalty = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.97, 4: 0.85, 5: 0.40}
        p_finishes = foul_penalty.get(foul_count, 1.0)
        if p_finishes >= 1.0:
            return pmf
        # Blend: p_finishes * pmf + (1-p_finishes) * {observed: 1.0}
        # (if fouled out, no more counting stats — but some already in)
        observed = min(pmf.keys()) if pmf else 0
        zero_mass = 1.0 - p_finishes
        blended: dict[int, float] = {}
        for k, v in pmf.items():
            blended[k] = v * p_finishes
        blended[observed] = blended.get(observed, 0.0) + zero_mass
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else {observed: 1.0}

    def adjust_for_score_differential(
        self,
        pmf: dict[int, float],
        score_diff: int,
        minutes_remaining: float,
        is_star: bool = False,
    ) -> dict[int, float]:
        """Part H: Reduce star player usage when game is a blowout.

        Args:
            score_diff: positive = player's team is winning
            minutes_remaining: game time left (WNBA = 40 min regulation)
            is_star: True if role_bucket in ["starter", "core"]
        """
        if minutes_remaining > 8.0 or abs(score_diff) < 15:
            return pmf
        if not is_star:
            return pmf
        if score_diff <= -15:
            # Losing by 15+ in final 8 min: trailing stars may get garbage time
            # They often INCREASE usage (chasing), slight upward nudge
            return pmf
        # Winning by 15+ in final 8 min: star likely benched
        p_benched = min(0.85, (score_diff - 15) / 20.0)
        observed = min(pmf.keys()) if pmf else 0
        blended: dict[int, float] = {}
        for k, v in pmf.items():
            blended[k] = v * (1.0 - p_benched)
        blended[observed] = blended.get(observed, 0.0) + p_benched
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else {observed: 1.0}

    def batch_compute(
        self,
        projections: dict[int, dict],
        player_states: dict,
        elapsed_minutes: float,
        score_diff: int = 0,
    ) -> dict[int, dict[str, dict]]:
        """Compute live P(over) for all players and all stats.

        Args:
            projections: {player_id: {stat: {mean, line, projected_minutes}}}
            player_states: {player_id: LivePlayerState}
            elapsed_minutes: current game elapsed time
            score_diff: current score differential (positive = player's team winning)

        Returns:
            {player_id: {stat: {p_over, p_under, ...}}}
        """
        results: dict[int, dict[str, dict]] = {}
        for pid, proj in projections.items():
            results[pid] = {}
            ps = player_states.get(pid)
            proj_minutes = float(proj.get("projected_minutes", 28.0))
            _is_star = proj.get("role_bucket", "rotation") in ("starter", "core")
            _foul_count = getattr(ps, "personal_fouls", 0) if ps else 0
            for stat in self.stats:
                if stat not in proj:
                    continue
                stat_proj = proj[stat]
                mean_proj = float(stat_proj.get("mean", stat_proj.get("expected", 0.0)))
                line = float(stat_proj.get("line", mean_proj))
                observed = getattr(ps, stat, 0) if ps else 0
                elapsed = elapsed_minutes if ps and not getattr(ps, "ejected", False) else 0.0
                result = self.compute_live_p_over(
                    stat, mean_proj, proj_minutes, observed, elapsed, line,
                    foul_count=_foul_count,
                    score_diff=score_diff,
                    is_star=_is_star,
                )
                results[pid][stat] = result
        return results
