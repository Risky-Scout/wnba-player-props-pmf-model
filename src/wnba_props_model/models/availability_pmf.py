"""Availability-conditioned PMF handling (STEP 4 of the pure-model-supremacy repair).

Two explicit distributions per player-game-stat row:

  * ``active_pmf``               = P(stat | player appears and the wager settles).
  * ``availability_mixture_pmf`` = P(stat including did-not-play uncertainty), i.e. the active
                                   PMF with the DNP mass folded onto outcome 0.

The forward availability mixture is::

    mixture[0]   = p_dnp + (1 - p_dnp) * active[0]
    mixture[k>0] = (1 - p_dnp) * active[k]

For **sportsbook binary scoring** and the Edge Board, a DNP does NOT settle as an Under — it
voids. The correct binary probability therefore conditions on appearance (the ``active_pmf``)
and then removes integer-line push mass, in this order:

    p_over_settled_from_active = settled_probabilities_from_pmf(active_pmf, line).p_over_settled
    model_prob_over_final      = binary_calibrator(p_over_settled_from_active)   # monotone, pure

This module deliberately does NOT provide the invalid post-hoc shortcut
``model_prob_over_final / (1 - p_dnp)``: that shortcut is exact only for a half-line,
pre-calibration, exact DNP-zero mixture, and is WRONG for integer lines with push mass and for
any nonlinear binary calibration already applied. See
``recover_active_pmf`` / ``settle_over_from_active_pmf`` for the correct PMF-level path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from wnba_props_model.models.market import settled_probabilities_from_pmf
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json

_EPS = 1e-12


def _dense(pmf) -> np.ndarray:
    if isinstance(pmf, str) or isinstance(pmf, dict):
        return normalize_pmf(json_to_pmf(pmf))
    return normalize_pmf(np.asarray(pmf, dtype=float))


def build_availability_mixture(active_pmf, p_dnp: float) -> np.ndarray:
    """Fold DNP mass onto outcome 0: mixture[0]=p_dnp+(1-p_dnp)active[0]; mixture[k>0] scaled."""
    a = _dense(active_pmf)
    d = float(np.clip(p_dnp, 0.0, 1.0 - _EPS))
    mix = (1.0 - d) * a.copy()
    mix[0] += d
    return normalize_pmf(mix)


def recover_active_pmf(mixture_pmf, p_dnp: float) -> np.ndarray:
    """Invert the availability mixture to the conditional-on-play (active) PMF.

    active[k>0] = mixture[k] / (1 - p_dnp);  active[0] = (mixture[0] - p_dnp) / (1 - p_dnp).
    The recovered index-0 mass is clipped at 0 (numerical guard) and the PMF renormalized.
    This is a PMF-level operation used only when an availability-mixture PMF must be reduced to
    its active component; production should build the active PMF directly.
    """
    mix = _dense(mixture_pmf)
    d = float(np.clip(p_dnp, 0.0, 1.0 - _EPS))
    denom = 1.0 - d
    active = mix.copy() / denom
    active[0] = max((mix[0] - d) / denom, 0.0)
    return normalize_pmf(active)


def pmf_mean(pmf) -> float:
    a = _dense(pmf)
    return float(np.dot(np.arange(a.size), a))


def settle_over_from_active_pmf(active_pmf, line: float):
    """Push-safe settled probabilities computed from the ACTIVE (conditional-on-play) PMF."""
    return settled_probabilities_from_pmf(_dense(active_pmf), float(line))


@dataclass(frozen=True)
class AvailabilityConditionedRow:
    """Persisted per-row availability decomposition for delivery/lineage."""
    p_dnp: float
    active_pmf_json: str
    active_pmf_mean: float
    availability_mixture_pmf_json: str
    availability_mixture_mean: float
    sportsbook_settlement_basis: str
    model_prob_over_settled_from_active_pmf: float
    model_prob_over_final: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_availability_conditioned_row(
    active_pmf,
    p_dnp: float,
    line: float,
    *,
    binary_calibrator: Callable[[float], float] | None = None,
    settlement_basis: str = "active_pmf_push_safe_void_on_dnp",
) -> AvailabilityConditionedRow:
    """Produce the correct availability-conditioned row for sportsbook binary scoring.

    The binary probability is derived from the ACTIVE PMF (void-on-DNP), push-safe at integer
    lines, then run through an optional monotone pure ``binary_calibrator`` (model-vs-outcome
    only). ``p_dnp`` is retained separately for availability/void suppression; DNP mass is NEVER
    placed into the settled over/under probability.
    """
    a = _dense(active_pmf)
    mix = build_availability_mixture(a, p_dnp)
    settled = settle_over_from_active_pmf(a, line)
    p_settled = float(settled.p_over_settled)
    p_final = float(binary_calibrator(p_settled)) if binary_calibrator is not None else p_settled
    p_final = float(np.clip(p_final, 0.0, 1.0))
    return AvailabilityConditionedRow(
        p_dnp=float(np.clip(p_dnp, 0.0, 1.0)),
        active_pmf_json=pmf_to_json(a),
        active_pmf_mean=pmf_mean(a),
        availability_mixture_pmf_json=pmf_to_json(mix),
        availability_mixture_mean=pmf_mean(mix),
        sportsbook_settlement_basis=settlement_basis,
        model_prob_over_settled_from_active_pmf=p_settled,
        model_prob_over_final=p_final,
    )


def invalid_posthoc_dednp_over(model_prob_over_final_mixture: float, p_dnp: float) -> float:
    """The DEPRECATED/INVALID post-hoc shortcut, kept ONLY so tests can prove it is wrong.

    Divides an already-final (post-calibration, mixture-based) over probability by (1 - p_dnp).
    This is NOT the inverse of the availability mixture at integer lines with push mass, nor of a
    nonlinear binary calibration. Never use in production.
    """
    return float(model_prob_over_final_mixture / max(1.0 - p_dnp, _EPS))
