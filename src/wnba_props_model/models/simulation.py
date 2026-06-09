from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from wnba_props_model.constants import DOMAIN_MAX


def normalize_pmf(pmf: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    arr = np.asarray(pmf, dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= eps:
        arr = np.zeros_like(arr, dtype=float)
        arr[0] = 1.0
        return arr
    return arr / s


def pmf_to_json(pmf: np.ndarray) -> str:
    arr = normalize_pmf(pmf)
    return json.dumps({str(i): float(p) for i, p in enumerate(arr) if p > 0})


def json_to_pmf(payload: str | Mapping[str, float], domain_max: int | None = None) -> np.ndarray:
    d = json.loads(payload) if isinstance(payload, str) else payload
    kmax = max([int(k) for k in d.keys()] + [domain_max or 0])
    arr = np.zeros(kmax + 1)
    for k, p in d.items():
        arr[int(k)] = float(p)
    if domain_max is not None:
        if len(arr) < domain_max + 1:
            arr = np.pad(arr, (0, domain_max + 1 - len(arr)))
        else:
            arr = arr[:domain_max + 1]
    return normalize_pmf(arr)


def enforce_monotone_tail(pmf: np.ndarray) -> np.ndarray:
    # Tail probabilities must be non-increasing by construction; this is a defensive no-op
    # after clipping/renormalization but keeps delivery invariant explicit.
    return normalize_pmf(pmf)


def convolve_pmfs(*pmfs: np.ndarray, domain_max: int | None = None) -> np.ndarray:
    out = np.array([1.0])
    for p in pmfs:
        out = np.convolve(out, normalize_pmf(p))
    if domain_max is not None:
        out = out[: domain_max + 1]
    return normalize_pmf(out)


def sample_from_quantile_ladder(q_values: Mapping[float, float] | dict[str, float], n: int, rng: np.random.Generator) -> np.ndarray:
    qs = np.array(sorted(float(k) for k in q_values.keys()), dtype=float)
    vals = np.array([float(q_values[str(q)] if str(q) in q_values else q_values[q]) for q in qs], dtype=float)
    vals = np.maximum.accumulate(vals)
    u = rng.uniform(qs.min(), qs.max(), size=n)
    return np.interp(u, qs, vals)


def simulate_count_pmf(
    stat: str,
    minutes_quantiles: Mapping[float, float] | dict[str, float],
    rate_quantiles: Mapping[float, float] | dict[str, float],
    n_draws: int = 50_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    minutes = np.clip(sample_from_quantile_ladder(minutes_quantiles, n_draws, rng), 0, 40)
    rates = np.clip(sample_from_quantile_ladder(rate_quantiles, n_draws, rng), 0, None)
    raw = np.clip(minutes * rates, 0, DOMAIN_MAX.get(stat, 100))
    draws = rng.poisson(raw).astype(int)
    domain = DOMAIN_MAX[stat]
    draws = np.clip(draws, 0, domain)
    counts = np.bincount(draws, minlength=domain + 1)
    return normalize_pmf(counts.astype(float))


def simulate_joint_count_pmfs(
    minutes_quantiles: Mapping[float, float] | dict[str, float],
    rate_quantiles_by_stat: Mapping[str, Mapping[float, float] | dict[str, float]],
    n_draws: int = 50_000,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    rng = rng or np.random.default_rng()
    minutes = np.clip(sample_from_quantile_ladder(minutes_quantiles, n_draws, rng), 0, 40)
    out: dict[str, np.ndarray] = {}
    for stat, q in rate_quantiles_by_stat.items():
        rates = np.clip(sample_from_quantile_ladder(q, n_draws, rng), 0, None)
        raw = np.clip(minutes * rates, 0, DOMAIN_MAX.get(stat, 100))
        draws = np.clip(rng.poisson(raw).astype(int), 0, DOMAIN_MAX[stat])
        counts = np.bincount(draws, minlength=DOMAIN_MAX[stat] + 1)
        out[stat] = normalize_pmf(counts.astype(float))
    return out


def build_combo_pmfs(component_pmfs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    if "stl" in component_pmfs and "blk" in component_pmfs:
        out["stocks"] = convolve_pmfs(component_pmfs["stl"], component_pmfs["blk"], domain_max=DOMAIN_MAX["stocks"])
    if "pts" in component_pmfs and "ast" in component_pmfs:
        out["pa"] = convolve_pmfs(component_pmfs["pts"], component_pmfs["ast"], domain_max=DOMAIN_MAX["pa"])
    if "pts" in component_pmfs and "reb" in component_pmfs:
        out["pr"] = convolve_pmfs(component_pmfs["pts"], component_pmfs["reb"], domain_max=DOMAIN_MAX["pr"])
    if "reb" in component_pmfs and "ast" in component_pmfs:
        out["ra"] = convolve_pmfs(component_pmfs["reb"], component_pmfs["ast"], domain_max=DOMAIN_MAX["ra"])
    if all(s in component_pmfs for s in ("pts", "reb", "ast")):
        out["pra"] = convolve_pmfs(component_pmfs["pts"], component_pmfs["reb"], component_pmfs["ast"], domain_max=DOMAIN_MAX["pra"])
    return out
