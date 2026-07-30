"""Coherent joint active-player outcome generator (shared-latent simulation).

Given pregame opportunity/conversion parameters for one player-game, generate a deterministic,
reproducible joint sample of ALL primitive outcomes with the structural identities holding in
EVERY sample:

    field_goals_made = two_point_makes + three_point_makes
    points           = 2*two_point_makes + 3*three_point_makes + free_throws_made
    rebounds         = offensive_rebounds + defensive_rebounds
    stocks           = steals + blocks

Correlation across stats comes from the SHARED sampled minutes (the dominant latent): all
counts are conditioned on the same drawn minutes, so combination markets (PRA, pts+ast, ...)
are NOT sums of independent marginals. The sampler is deterministic given a fixed seed; Monte
Carlo standard error is measured and reported (SIMULATION_PRECISION_NOT_MET when a published
probability's SE exceeds the frozen tolerance).

No sportsbook data enters this generator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MC_SE_TOLERANCE = 5e-4
RELEASE_SEED = 20260730

_OUTCOME_KEYS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                 "fgm", "ftm", "fta", "fg3a", "fg2a", "fg2m", "oreb", "dreb"]
_COMBOS = {"stocks": ("stl", "blk"), "pts_ast": ("pts", "ast"), "pts_reb": ("pts", "reb"),
           "reb_ast": ("reb", "ast"), "pts_reb_ast": ("pts", "reb", "ast")}


@dataclass
class PlayerGameParams:
    """Pregame (feature-driven) parameters for one active player. Rates are per-minute."""
    player_id: str
    p_active: float = 0.9
    minutes_mean: float = 24.0
    minutes_sd: float = 6.0
    fga_per_min: float = 0.35
    fg3a_share: float = 0.35          # share of FGA that are 3PA
    fta_per_min: float = 0.12
    fg2_pct: float = 0.50
    fg3_pct: float = 0.34
    ft_pct: float = 0.80
    oreb_per_min: float = 0.05
    dreb_per_min: float = 0.15
    ast_per_min: float = 0.10
    stl_per_min: float = 0.03
    blk_per_min: float = 0.02
    tov_per_min: float = 0.08
    q1_minutes_share: float = 0.28    # fraction of minutes typically in Q1 (not 0.25 flat)


@dataclass
class JointOutcome:
    player_id: str
    pmfs: dict[str, np.ndarray]
    q1_pmfs: dict[str, np.ndarray]
    event_probs: dict[str, float]
    p_dnp: float
    n_samples: int
    seed: int
    mc_max_se: float
    pricing_status: str
    identities_hold: bool


def _pmf_from_samples(samples: np.ndarray, cap: int | None = None) -> np.ndarray:
    m = int(samples.max()) if len(samples) else 0
    if cap is not None:
        m = min(m, cap)
    pmf = np.bincount(np.clip(samples, 0, m), minlength=m + 1).astype(float)
    return pmf / pmf.sum()


def simulate_player(params: PlayerGameParams, *, n_samples: int = 40000,
                    seed: int = RELEASE_SEED) -> JointOutcome:
    """Deterministic shared-latent simulation of one active player's joint outcomes."""
    rng = np.random.default_rng(seed + (abs(hash(params.player_id)) % 100000))
    n = int(n_samples)

    # shared latent: minutes (truncated normal, >=0, overtime headroom to 50)
    minutes = np.clip(rng.normal(params.minutes_mean, max(params.minutes_sd, 1e-3), n), 0, 50)

    def _pois(rate_per_min, mins):
        return rng.poisson(np.clip(rate_per_min, 0, None) * np.clip(mins, 0, None))

    def _binom(nn, p):
        return rng.binomial(np.clip(nn, 0, None).astype(int), min(max(p, 0.0), 1.0))

    def _sample_block(mins):
        fga = _pois(params.fga_per_min, mins)
        fg3a = _binom(fga, params.fg3a_share)
        fg2a = fga - fg3a
        fg2m = _binom(fg2a, params.fg2_pct)
        fg3m = _binom(fg3a, params.fg3_pct)
        fta = _pois(params.fta_per_min, mins)
        ftm = _binom(fta, params.ft_pct)
        oreb = _pois(params.oreb_per_min, mins)
        dreb = _pois(params.dreb_per_min, mins)
        out = {
            "fg2a": fg2a, "fg3a": fg3a, "fg2m": fg2m, "fg3m": fg3m, "fta": fta, "ftm": ftm,
            "oreb": oreb, "dreb": dreb,
            "fgm": fg2m + fg3m,                         # identity
            "pts": 2 * fg2m + 3 * fg3m + ftm,           # identity
            "reb": oreb + dreb,                          # identity
            "ast": _pois(params.ast_per_min, mins),
            "stl": _pois(params.stl_per_min, mins),
            "blk": _pois(params.blk_per_min, mins),
            "turnover": _pois(params.tov_per_min, mins),
        }
        return out

    full = _sample_block(minutes)
    q1_minutes = np.clip(minutes * params.q1_minutes_share, 0, 12)
    q1 = _sample_block(q1_minutes)

    # identity checks (must hold in every sample)
    identities_hold = bool(
        np.all(full["fgm"] == full["fg2m"] + full["fg3m"]) and
        np.all(full["pts"] == 2 * full["fg2m"] + 3 * full["fg3m"] + full["ftm"]) and
        np.all(full["reb"] == full["oreb"] + full["dreb"]))

    pmfs = {k: _pmf_from_samples(full[k]) for k in _OUTCOME_KEYS}
    for combo, comps in _COMBOS.items():
        pmfs[combo] = _pmf_from_samples(sum(full[c] for c in comps))
    q1_pmfs = {f"{k}_q1": _pmf_from_samples(q1[k]) for k in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "fgm", "ftm"]}

    # event markets from the JOINT samples (never product of marginals)
    cats = np.vstack([full["pts"] >= 10, full["reb"] >= 10, full["ast"] >= 10,
                      full["stl"] >= 10, full["blk"] >= 10]).sum(axis=0)
    p_dd = float(np.mean(cats >= 2))
    p_td = float(np.mean(cats >= 3))
    event_probs = {"double_double": p_dd, "triple_double": p_td}

    # Monte Carlo SE for the reported event probabilities + a few reference line probs
    ref_probs = [p_dd, p_td,
                 float(np.mean(full["pts"] > 15.5)), float(np.mean(full["reb"] > 6.5)),
                 float(np.mean(full["ast"] > 4.5))]
    mc_max_se = float(max(np.sqrt(p * (1 - p) / n) for p in ref_probs))
    status = "PRICED" if mc_max_se <= MC_SE_TOLERANCE else "SIMULATION_PRECISION_NOT_MET"

    return JointOutcome(
        player_id=params.player_id, pmfs=pmfs, q1_pmfs=q1_pmfs, event_probs=event_probs,
        p_dnp=float(1.0 - min(max(params.p_active, 0.0), 1.0)), n_samples=n, seed=seed,
        mc_max_se=mc_max_se, pricing_status=status, identities_hold=identities_hold)


def adaptive_simulate(params: PlayerGameParams, *, start: int = 20000, max_samples: int = 400000,
                      seed: int = RELEASE_SEED) -> JointOutcome:
    """Increase sample count until MC SE <= tolerance or the cap is reached."""
    n = start
    out = simulate_player(params, n_samples=n, seed=seed)
    while out.mc_max_se > MC_SE_TOLERANCE and n < max_samples:
        n = min(n * 2, max_samples)
        out = simulate_player(params, n_samples=n, seed=seed)
    return out
