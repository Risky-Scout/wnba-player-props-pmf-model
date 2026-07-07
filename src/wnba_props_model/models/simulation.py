from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.constants import DOMAIN_MAX

# Empirical OOF correlations — updated by weekly calibration via estimate_oof_correlations().
# Conservative defaults (slightly below full empirical to avoid overfitting small samples).
_COMBO_CORRELATIONS: dict[str, float] = {
    "pts_reb":     0.42,
    "pts_ast":     0.28,
    "reb_ast":     0.10,
    "pts_reb_ast": 0.30,
    "stocks":      0.15,
}
_COMBO_CORRELATIONS_PATH = Path(__file__).parents[3] / "config" / "model" / "combo_correlations.json"
if _COMBO_CORRELATIONS_PATH.exists():
    try:
        import json as _json
        _COMBO_CORRELATIONS.update(_json.loads(_COMBO_CORRELATIONS_PATH.read_text()))
    except Exception:
        pass


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
    # Support both list format [p0, p1, ...] and sparse dict format {"0": p0, "3": p3, ...}
    if isinstance(d, list):
        arr = np.array(d, dtype=float)
        if domain_max is not None:
            if len(arr) < domain_max + 1:
                arr = np.pad(arr, (0, domain_max + 1 - len(arr)))
            else:
                arr = arr[:domain_max + 1]
        return normalize_pmf(arr)
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


def convolve_pmfs_correlated(
    pmf_a: np.ndarray,
    pmf_b: np.ndarray,
    correlation: float,
    domain_max: int | None = None,
    n_monte_carlo: int = 100_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convolve two PMFs using a Gaussian copula to account for correlation.

    Instead of assuming independence, this uses a Gaussian copula:
    1. Draw (u, v) from a bivariate normal with correlation r
    2. Convert to quantiles via the normal CDF
    3. Map quantiles to values using each PMF's CDF (inverse transform)
    4. Sum the values → sample of the joint distribution
    5. Build empirical PMF from the sum samples

    Positive correlation increases P(over) for combo props relative to
    independent convolution, correcting the chronic UNDER tilt on combos.
    """
    rng = rng or np.random.default_rng(42)
    corr = float(np.clip(correlation, -0.99, 0.99))
    if abs(corr) < 0.01:
        return convolve_pmfs(pmf_a, pmf_b, domain_max=domain_max)
    a = normalize_pmf(pmf_a)
    b = normalize_pmf(pmf_b)
    cdf_a = np.clip(np.cumsum(a), 1e-10, 1.0 - 1e-10)
    cdf_b = np.clip(np.cumsum(b), 1e-10, 1.0 - 1e-10)
    cov = np.array([[1.0, corr], [corr, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n_monte_carlo)
    from scipy.stats import norm as _scipy_norm
    u = _scipy_norm.cdf(z[:, 0])
    v = _scipy_norm.cdf(z[:, 1])
    x_a = np.clip(np.searchsorted(cdf_a, u, side="left"), 0, len(a) - 1)
    x_b = np.clip(np.searchsorted(cdf_b, v, side="left"), 0, len(b) - 1)
    combo_vals = x_a + x_b
    if domain_max is not None:
        combo_vals = np.clip(combo_vals, 0, domain_max)
    max_val = int(combo_vals.max())
    if domain_max is not None:
        max_val = max(max_val, domain_max)
    counts = np.bincount(combo_vals, minlength=max_val + 1).astype(float)
    if domain_max is not None:
        counts = counts[: domain_max + 1]
    return normalize_pmf(counts)


def estimate_oof_correlations(oof_df: pd.DataFrame) -> dict[str, float]:
    """Estimate pairwise stat correlations from OOF residuals.

    For each player, computes residual = actual_outcome - pmf_mean per stat,
    then computes Pearson correlation across players between stat pairs.
    Only uses players with >= 10 OOF rows for stability.
    """
    combo_pairs = {
        "pts_reb": ("pts", "reb"),
        "pts_ast": ("pts", "ast"),
        "reb_ast": ("reb", "ast"),
        "stocks": ("stl", "blk"),
    }
    df = oof_df.copy()
    actual_col = "actual_outcome" if "actual_outcome" in df.columns else "outcome"
    df["residual"] = df[actual_col] - df["pmf_mean"]
    player_stat_residuals: dict[str, pd.Series] = {}
    for stat in ["pts", "reb", "ast", "stl", "blk"]:
        sub = df[df["stat"] == stat]
        counts = sub.groupby("player_id").size()
        eligible = counts[counts >= 10].index
        player_stat_residuals[stat] = (
            sub[sub["player_id"].isin(eligible)]
            .groupby("player_id")["residual"]
            .mean()
        )
    correlations: dict[str, float] = {}
    for combo_name, (stat_a, stat_b) in combo_pairs.items():
        res_a = player_stat_residuals.get(stat_a)
        res_b = player_stat_residuals.get(stat_b)
        if res_a is None or res_b is None or len(res_a) == 0 or len(res_b) == 0:
            correlations[combo_name] = 0.0
            continue
        common = res_a.index.intersection(res_b.index)
        if len(common) < 20:
            correlations[combo_name] = 0.0
            continue
        r = float(res_a[common].corr(res_b[common]))
        correlations[combo_name] = round(float(np.clip(r, -0.5, 0.7)), 3)
    correlations["pts_reb_ast"] = round(
        (correlations.get("pts_reb", 0.3) + correlations.get("pts_ast", 0.2)) / 2, 3
    )
    return correlations


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
        out["stocks"] = convolve_pmfs_correlated(
            component_pmfs["stl"], component_pmfs["blk"],
            correlation=_COMBO_CORRELATIONS.get("stocks", 0.0),
            domain_max=DOMAIN_MAX["stocks"],
        )
    if "pts" in component_pmfs and "ast" in component_pmfs:
        out["pa"] = convolve_pmfs_correlated(
            component_pmfs["pts"], component_pmfs["ast"],
            correlation=_COMBO_CORRELATIONS.get("pts_ast", 0.0),
            domain_max=DOMAIN_MAX["pa"],
        )
    if "pts" in component_pmfs and "reb" in component_pmfs:
        out["pr"] = convolve_pmfs_correlated(
            component_pmfs["pts"], component_pmfs["reb"],
            correlation=_COMBO_CORRELATIONS.get("pts_reb", 0.0),
            domain_max=DOMAIN_MAX["pr"],
        )
    if "reb" in component_pmfs and "ast" in component_pmfs:
        out["ra"] = convolve_pmfs_correlated(
            component_pmfs["reb"], component_pmfs["ast"],
            correlation=_COMBO_CORRELATIONS.get("reb_ast", 0.0),
            domain_max=DOMAIN_MAX["ra"],
        )
    if all(s in component_pmfs for s in ("pts", "reb", "ast")):
        # PRA: convolve pts+reb first, then convolve result with ast.
        # Use pts_reb_ast correlation for the second step (covers full triple correlation).
        _pr_tmp = convolve_pmfs_correlated(
            component_pmfs["pts"], component_pmfs["reb"],
            correlation=_COMBO_CORRELATIONS.get("pts_reb", 0.0),
        )
        out["pra"] = convolve_pmfs_correlated(
            _pr_tmp, component_pmfs["ast"],
            correlation=_COMBO_CORRELATIONS.get("pts_reb_ast", 0.0),
            domain_max=DOMAIN_MAX["pra"],
        )
    return out
