"""V6 market projection: fit a TiltedDistribution (tail-aware) to push-aware A/(A+B) constraints.

Returns a proper distribution whose stored atoms + overflow = 1 exactly (the V5 bug that attached
unchanged overflow after normalizing stored atoms is gone). Fails closed when infeasible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from wnba_props_model.sharp_v6.distribution import TiltedDistribution

FEASIBLE_MAX_RESIDUAL = 0.02


def no_vig_settled_over(over_odds: float, under_odds: float) -> float:
    def imp(a):
        a = float(a)
        return float("nan") if (-100 < a < 100) else ((100.0 / (a + 100.0)) if a > 0 else (abs(a) / (abs(a) + 100.0)))
    po, pu = imp(over_odds), imp(under_odds)
    return float(po / (po + pu)) if np.isfinite(po) and np.isfinite(pu) and po + pu > 0 else float("nan")


@dataclass
class ProjectionResult:
    distribution: TiltedDistribution | None
    theta: dict
    line_residuals: list[dict]
    max_abs_residual: float
    status: str
    feasible: bool


def project_multiline(base, constraints: list[dict]) -> ProjectionResult:
    if not constraints:
        return ProjectionResult(None, {}, [], float("nan"), "NO_CONSTRAINTS", False)
    reg = 1e-3

    def resid(theta):
        td = TiltedDistribution(base, theta_mean=theta[0], theta_zero=theta[1], theta_disp=theta[2])
        out = []
        for c in constraints:
            s = td.settle_over_under(c["line"])
            so = s.p_over_settled if np.isfinite(s.p_over_settled) else 1.0
            out.append(np.sqrt(c.get("weight", 1.0)) * (so - c["q_over"]))
        out.extend((reg * theta[0], reg * theta[1], reg * theta[2]))
        return out

    sol = least_squares(resid, x0=[0.0, 0.0, 0.0], method="trf", max_nfev=500)
    td = TiltedDistribution(base, theta_mean=sol.x[0], theta_zero=sol.x[1], theta_disp=sol.x[2])
    line_res = []
    for c in constraints:
        s = td.settle_over_under(c["line"])
        so = s.p_over_settled if np.isfinite(s.p_over_settled) else 1.0
        line_res.append({"line": c["line"], "q_over": c["q_over"], "fitted_over": float(so),
                         "residual": float(abs(so - c["q_over"]))})
    max_res = max(r["residual"] for r in line_res)
    feasible = max_res <= FEASIBLE_MAX_RESIDUAL
    try:
        if feasible:
            td.validate()
    except ValueError:
        feasible = False
    return ProjectionResult(td if feasible else None,
                            {"theta_mean": float(sol.x[0]), "theta_zero": float(sol.x[1]),
                             "theta_disp": float(sol.x[2])}, line_res, float(max_res),
                            "PROJECTED" if feasible else "MARKET_PROJECTION_INFEASIBLE", feasible)
