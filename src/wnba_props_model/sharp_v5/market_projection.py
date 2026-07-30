"""Push-aware, overflow-aware, multi-line market-consistent projection (Section 4).

One player-stat-timestamp gets ONE distribution fit to ALL exact same-time no-vig lines using the
settled constraint A/(A+B)=q_over (NOT A=q_over). A single low-dimensional tilt (mean, dispersion,
zero-mass) is fit to all line constraints jointly. Fails closed as MARKET_PROJECTION_INFEASIBLE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from wnba_props_model.sharp_v5.distribution import DiscreteDistribution, TabularDistribution

FEASIBLE_MAX_RESIDUAL = 0.02


def no_vig_settled_over(over_odds: float, under_odds: float) -> float:
    """Two-way no-vig Over settled probability (already conditional on no push)."""
    def imp(a):
        a = float(a)
        if not np.isfinite(a) or (-100 < a < 100):
            return float("nan")
        return (100.0 / (a + 100.0)) if a > 0 else (abs(a) / (abs(a) + 100.0))
    po, pu = imp(over_odds), imp(under_odds)
    if not (np.isfinite(po) and np.isfinite(pu)) or (po + pu) <= 0:
        return float("nan")
    return float(po / (po + pu))


@dataclass
class ProjectionResult:
    distribution: TabularDistribution | None
    theta: dict
    line_residuals: list[dict]
    max_abs_residual: float
    status: str            # PROJECTED | MARKET_PROJECTION_INFEASIBLE
    feasible: bool


def _settled_over(atoms: np.ndarray, line: float) -> float:
    k = np.arange(atoms.size)
    A = float(atoms[k > line].sum())
    B = float(atoms[k < line].sum())
    den = A + B
    return A / den if den > 0 else float("nan")


def project_multiline(base: DiscreteDistribution, constraints: list[dict],
                      required_max: int | None = None) -> ProjectionResult:
    """constraints: list of {line, q_over (settled), weight}. Returns one market-consistent PMF."""
    if not constraints:
        return ProjectionResult(None, {}, [], float("nan"), "NO_CONSTRAINTS", False)
    max_line = max(c["line"] for c in constraints)
    need = max(int(np.ceil(max_line)) + 8, required_max or 0)
    m = base.materialize(required_max=need)
    p0 = m.atoms.copy()
    p0 = p0 / p0.sum()
    k = np.arange(p0.size)
    mu = float(np.dot(k, p0))

    def tilt(theta):
        t_mean, t_disp, t_zero = theta
        logw = t_mean * k + t_disp * (k - mu) ** 2 + t_zero * (k == 0)
        logw -= logw.max()
        w = p0 * np.exp(logw)
        return w / w.sum()

    reg = 1e-3   # prefer the minimal tilt (min-KL) when constraints under-determine theta
    def resid(theta):
        atoms = tilt(theta)
        out = []
        for c in constraints:
            so = _settled_over(atoms, c["line"])
            wgt = np.sqrt(c.get("weight", 1.0))
            out.append(wgt * ((so if np.isfinite(so) else 1.0) - c["q_over"]))
        out.extend((reg * theta[0], reg * theta[1], reg * theta[2]))   # L2 shrink to zero tilt
        return out

    sol = least_squares(resid, x0=[0.0, 0.0, 0.0], method="trf", max_nfev=800)
    atoms = tilt(sol.x)
    line_res = []
    for c in constraints:
        so = _settled_over(atoms, c["line"])
        line_res.append({"line": c["line"], "q_over": c["q_over"], "fitted_over": so,
                         "residual": float(abs((so if np.isfinite(so) else 1.0) - c["q_over"]))})
    max_res = max(r["residual"] for r in line_res)
    feasible = max_res <= FEASIBLE_MAX_RESIDUAL
    dist = TabularDistribution(atoms, overflow=float(m.overflow_probability), tail_method="tilted_analytic") \
        if feasible else None
    return ProjectionResult(dist, {"mean_tilt": float(sol.x[0]), "dispersion_tilt": float(sol.x[1]),
                                   "zero_tilt": float(sol.x[2])}, line_res, float(max_res),
                            "PROJECTED" if feasible else "MARKET_PROJECTION_INFEASIBLE", feasible)
