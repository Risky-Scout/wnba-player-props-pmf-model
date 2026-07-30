"""Sharp v4 core: explicit feature contracts, join-cardinality guard, exact-tail count PMFs
(no outcome clipping), adaptive support + overflow, and hierarchical dispersion.

Sportsbook data never enters the PURE feature path.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from wnba_props_model.sharp_v3.core import (  # noqa: F401
    ID_COLS,
    LABEL_COLS,
    TIER_A,
    american_to_prob,
    ece,
    load_verified,
    no_vig_over,
)

SEED = 20260730
TAIL_TOL = 1e-6
# Emergency computational caps (match frozen config/pmf_support_v4.yaml). They bound only the
# STORED atom array length; NLL/CRPS/pricing use exact analytic tails (CountPMF.logpmf/cdf) that
# are independent of the cap, so the cap never clips an outcome.
EMERGENCY_CAP = {"pts": 80, "reb": 40, "ast": 30, "fg3m": 18, "stl": 15, "blk": 15,
                 "turnover": 18, "fgm": 35, "ftm": 35, "fta": 40, "oreb": 25, "dreb": 30,
                 "fg2a": 40, "fg3a": 25, "fg2m": 30, "minutes": 48}

# ---- A. Explicit ordered feature contracts (anchored patterns, NOT `if stat in col`) ----
# Each component maps to anchored regex patterns; the resolved ordered column list is the exact,
# hashed contract. Anchors (^player_<stat>_, ^opp_<stat>_allowed) prevent unrelated columns.
_COMMON = [r"^player_minutes_", r"^player_cumulative_minutes", r"^cumulative_minutes_",
           r"^player_rest_days$", r"^opp_rest_days$", r"^rest_advantage$",
           r"^days_since_", r"^game_number_in_season$", r"^is_home$", r"^is_starter_prior",
           r"^player_usage_proxy_"]
_FAMILIES: dict[str, list[str]] = {
    "participation": [r"^player_minutes_", r"^player_cumulative_minutes", r"^cumulative_minutes_",
                      r"^player_rest_days$", r"^opp_rest_days$", r"^rest_advantage$",
                      r"^days_since_", r"^game_number_in_season$", r"^player_games_played",
                      r"^player_.*_season_zscore$"],
    "minutes": _COMMON + [r"^player_minutes_.*_(mean|std|form)", r"^player_.*_season_zscore$"],
    "pts": _COMMON + [r"^player_pts_", r"^opp_pts_allowed", r"^opp_pos_pts_allowed",
                      r"^player_fga_", r"^player_fg3a_", r"^player_fta_"],
    "reb": _COMMON + [r"^player_reb_", r"^player_oreb_", r"^player_dreb_", r"^opp_reb_allowed"],
    "ast": _COMMON + [r"^player_ast_", r"^opp_ast_allowed", r"^player_usage_proxy_"],
    "fg3m": _COMMON + [r"^player_fg3m_", r"^player_fg3a_", r"^opp_fg3m_allowed", r"^opp_fg3a_allowed"],
    "stl": _COMMON + [r"^player_stl_", r"^opp_stl_", r"^opp_turnover_forced"],
    "blk": _COMMON + [r"^player_blk_", r"^opp_blk_"],
    "turnover": _COMMON + [r"^player_turnover_", r"^opp_turnover_forced"],
    "fg2a": _COMMON + [r"^player_fga_", r"^player_fg3a_", r"^opp_pts_allowed"],
    "fg3a": _COMMON + [r"^player_fg3a_", r"^opp_fg3a_allowed"],
    "fta": _COMMON + [r"^player_fta_", r"^player_ftr", r"^opp_pts_allowed"],
    "oreb": _COMMON + [r"^player_oreb_", r"^player_reb_", r"^opp_reb_allowed"],
    "dreb": _COMMON + [r"^player_dreb_", r"^player_reb_", r"^opp_reb_allowed"],
}


def resolve_contract(component: str, all_cols: list[str]) -> list[str]:
    """Resolve a component's anchored patterns to an EXACT ORDERED column list, excluding all
    id/label columns. Deterministic and hashable."""
    pats = [re.compile(p) for p in _FAMILIES[component]]
    forbidden = set(ID_COLS) | set(LABEL_COLS)
    out: list[str] = []
    for c in all_cols:
        if c in forbidden or c.endswith("_tgt"):
            continue
        if any(p.search(c) for p in pats) and c not in out:
            out.append(c)
    return out


def contract_hash(cols: list[str]) -> str:
    return hashlib.sha256("\n".join(cols).encode()).hexdigest()[:16]


def build_all_contracts(all_cols: list[str]) -> dict[str, dict]:
    contracts = {}
    for comp in _FAMILIES:
        cols = resolve_contract(comp, all_cols)
        contracts[comp] = {"n": len(cols), "schema_hash": contract_hash(cols),
                           "features": cols, "missingness": "HGB native NaN handling",
                           "provenance": "pregame T-1.2 lagged features (recovered_v2)"}
    return contracts


# ---- B. Join cardinality guard ----
def assert_unique_keys(df: pd.DataFrame, keys: list[str], name: str = "frame") -> None:
    dup = int(df.duplicated(subset=keys).sum())
    if dup:
        raise ValueError(f"JOIN CARDINALITY: {name} has {dup} duplicate rows on keys {keys}")


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], how: str = "inner") -> pd.DataFrame:
    assert_unique_keys(left, keys, "left"); assert_unique_keys(right, keys, "right")
    out = left.merge(right, on=keys, how=how, validate="one_to_one")
    return out


# ---- C+D. Exact-tail count PMF (NEVER clips outcomes) ----
@dataclass
class CountPMF:
    mu: float
    r: float | None            # None => Poisson
    support_max: int
    atoms: np.ndarray          # P(Y=0..support_max)
    overflow: float            # exact P(Y > support_max)
    tail_method: str
    normalization_error: float

    def logpmf(self, y: int) -> float:
        """Exact log P(Y=y) for ANY y (no clipping)."""
        y = int(max(y, 0))
        if self.r is None:
            return float(poisson.logpmf(y, max(self.mu, 1e-9)))
        p = self.r / (self.r + max(self.mu, 1e-9))
        return float(nbinom.logpmf(y, self.r, p))

    def cdf(self, k: float) -> float:
        k = int(np.floor(k))
        if k < 0:
            return 0.0
        if self.r is None:
            return float(poisson.cdf(k, max(self.mu, 1e-9)))
        p = self.r / (self.r + max(self.mu, 1e-9))
        return float(nbinom.cdf(k, self.r, p))

    def prob_over(self, line: float) -> float:
        return float(max(0.0, 1.0 - self.cdf(np.floor(line))))    # exact survival incl. tail

    def prob_under(self, line: float) -> float:
        return float(self.cdf(np.ceil(line) - 1))

    def prob_push(self, line: float) -> float:
        if not float(line).is_integer():
            return 0.0
        return float(np.exp(self.logpmf(int(line))))

    def tail_upper_bound(self) -> float:
        return self.overflow


def build_count_pmf(mu: float, r: float | None, stat: str) -> CountPMF:
    """Adaptive support until exact survival < TAIL_TOL (or emergency cap). Overflow retained."""
    mu = max(float(mu), 1e-9)
    cap = EMERGENCY_CAP.get(stat, 60)
    method = "poisson_sf" if r is None else "nbinom_sf"
    K = max(int(mu + 6 * np.sqrt(mu)), 5)
    for _ in range(24):
        if r is None:
            sf = float(poisson.sf(K, mu)); atoms = poisson.pmf(np.arange(K + 1), mu)
        else:
            p = r / (r + mu); sf = float(nbinom.sf(K, r, p)); atoms = nbinom.pmf(np.arange(K + 1), r, p)
        if sf < TAIL_TOL or K >= cap:
            break
        K = min(int(K * 1.6) + 2, cap)
    overflow = float(sf)
    norm_err = float(abs(atoms.sum() + overflow - 1.0))
    return CountPMF(mu=mu, r=r, support_max=int(K), atoms=np.asarray(atoms, float),
                    overflow=overflow, tail_method=method, normalization_error=norm_err)


# ---- E. Hierarchical (partially-pooled) dispersion ----
def hierarchical_dispersion(y: np.ndarray, mu: np.ndarray, group: np.ndarray,
                            shrink: float = 50.0) -> dict:
    """NB2 dispersion r shrunk group -> global via a pseudo-count. Returns {group_value: r} plus
    '__global__'. r from conditional residuals: phi = E[(y-mu)^2 - mu]/E[mu^2], r = 1/phi."""
    def _r(yy, mm):
        num = float(np.mean((yy - mm) ** 2 - mm)); den = float(np.mean(mm ** 2))
        if den <= 1e-9 or num <= 1e-9:
            return None
        return float(np.clip(den / num, 0.3, 500.0))
    g_global = _r(y, mu)
    phi_global = (1.0 / g_global) if g_global else 0.0
    out = {"__global__": g_global}
    for gv in np.unique(group):
        m = group == gv
        if m.sum() < 5:
            out[gv] = g_global; continue
        rg = _r(y[m], mu[m])
        phi_g = (1.0 / rg) if rg else phi_global
        n = int(m.sum())
        phi_shrunk = (n * phi_g + shrink * phi_global) / (n + shrink)   # partial pooling in phi
        out[gv] = (1.0 / phi_shrunk) if phi_shrunk > 1e-9 else g_global
    return out


# ---- exact-tail metrics (shared support semantics) ----
def market_consistent_atoms(atoms: np.ndarray, line: float, target_over: float) -> np.ndarray:
    """Minimum-KL projection of a pure PMF onto the constraint P(Y>line)=target_over via a single
    exponential tilt on the over-region indicator (closed form). Preserves shape within each region,
    nonnegativity, and unit mass. This is the market-consistent atom distribution for one line."""
    a = np.clip(np.asarray(atoms, float), 0.0, None)
    a = a / a.sum()
    k = np.arange(a.size)
    over = k > line
    S = float(a[over].sum())
    q = float(min(max(target_over, 1e-6), 1 - 1e-6))
    if S <= 1e-9 or S >= 1 - 1e-9:
        return a
    # e^theta = q(1-S) / ((1-q)S)
    w = (q * (1 - S)) / ((1 - q) * S)
    tilted = a.copy()
    tilted[over] = a[over] * w
    tilted = tilted / tilted.sum()
    return tilted


def nll_exact(pmfs: list[CountPMF], y: np.ndarray) -> float:
    return float(-np.mean([p.logpmf(int(yi)) for p, yi in zip(pmfs, y)]))


def crps_exact(pmfs: list[CountPMF], y: np.ndarray) -> float:
    vals = []
    for p, yi in zip(pmfs, y):
        kmax = int(max(p.support_max, yi) + 5)
        ks = np.arange(kmax + 1)
        cdf = np.array([p.cdf(k) for k in ks])
        heavi = (ks >= yi).astype(float)
        vals.append(float(np.sum((cdf - heavi) ** 2)))
    return float(np.mean(vals))


def pit_ks(pmfs: list[CountPMF], y: np.ndarray, rng: np.random.Generator) -> float:
    u = []
    for p, yi in zip(pmfs, y):
        lo = p.cdf(yi - 1)
        u.append(lo + rng.random() * np.exp(p.logpmf(int(yi))))
    u = np.sort(np.asarray(u)); n = len(u)
    if n == 0:
        return float("nan")
    return float(np.max(np.abs(np.arange(1, n + 1) / n - u)))
