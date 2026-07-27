"""Full PTS decomposition PMF (owner directive section 7).

Points = 2*(2P makes) + 3*(3P makes) + 1*(FT makes). Each makes-count is a Beta-Binomial marginal
of an attempt-count opportunity distribution and a shrunk Beta conversion:

    PTS PMF = stretch(2PM, 2)  (x)  stretch(3PM, 3)  (x)  FTM

where 2PM = BetaBinomial(2PA, 2P%), 3PM = BetaBinomial(3PA, 3P%), FTM = BetaBinomial(FTA, FT%),
averaged over conditional-active minute samples. This path is only valid on rows whose 2P% / FT%
conversions were fit from the verified tracking-based reconstruction; rows without it fall back to
the diagnostic proxy (handled by the caller).
"""
from __future__ import annotations

import numpy as np

from .pmf_builders import (
    convolve_pmfs,
    marginal_beta_binomial_pmf,
    poisson_or_nbinom_pmf,
    weighted_mix_pmfs,
)


def stretch_pmf(pmf: np.ndarray, mult: int) -> np.ndarray:
    """Map a count PMF to points by placing mass at count*mult (each make = ``mult`` points)."""
    pmf = np.asarray(pmf, float)
    if mult == 1:
        return pmf
    out = np.zeros((pmf.size - 1) * mult + 1)
    out[::mult] = pmf
    return out


def build_pts_pmf_for_minutes(
    minute: float,
    *,
    rate_2pa: float, r_2pa: float, alpha2: float, beta2: float,
    rate_3pa: float, r_3pa: float, alpha3: float, beta3: float,
    rate_fta: float, r_fta: float, alpha_ft: float, beta_ft: float,
    tail_tolerance: float = 1e-8, maximum_cap: int = 120,
) -> np.ndarray:
    """Convolved PTS PMF for a single minutes value."""
    a2 = poisson_or_nbinom_pmf(max(rate_2pa * minute, 1e-6), r_2pa,
                               tail_tolerance=tail_tolerance, maximum_cap=maximum_cap)
    a3 = poisson_or_nbinom_pmf(max(rate_3pa * minute, 1e-6), r_3pa,
                               tail_tolerance=tail_tolerance, maximum_cap=maximum_cap)
    aft = poisson_or_nbinom_pmf(max(rate_fta * minute, 1e-6), r_fta,
                                tail_tolerance=tail_tolerance, maximum_cap=maximum_cap)
    pm2 = marginal_beta_binomial_pmf(a2, alpha2, beta2)     # 2P makes count
    pm3 = marginal_beta_binomial_pmf(a3, alpha3, beta3)     # 3P makes count
    pmft = marginal_beta_binomial_pmf(aft, alpha_ft, beta_ft)  # FT makes count
    pts = convolve_pmfs(convolve_pmfs(stretch_pmf(pm2, 2), stretch_pmf(pm3, 3)), pmft)
    s = pts.sum()
    return pts / s if s > 0 else pts


def build_pts_pmf_over_minutes(
    minute_samples, weights, **components,
) -> np.ndarray:
    """Minutes-averaged convolved PTS PMF (equal-probability minute weights)."""
    per_sample = [build_pts_pmf_for_minutes(float(m), **components) for m in minute_samples]
    return weighted_mix_pmfs(per_sample, list(weights))
