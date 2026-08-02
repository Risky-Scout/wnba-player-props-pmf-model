"""Four distinct probability tracks for the pick engine."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from wnba_props_model.models.market import (
    UndefinedSettledProbabilityError,
    settled_probabilities_from_pmf,
)
from wnba_props_model.pick_engine.constants import PROB_EPS

EPS = PROB_EPS


def _clip01(p: float) -> float:
    return float(min(1.0 - EPS, max(EPS, p)))


def logit(p: float) -> float:
    p = _clip01(float(p))
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def pick_probability(
    pure_probability: float,
    reference_market_probability: float,
    w_segment: float,
) -> float:
    """Pick probability via reliability-weighted logit shrinkage.

    logit(p_pick) = logit(p_reference) + w_segment * [logit(p_pure) - logit(p_reference)]

    Constrains 0 <= w_segment <= 1. Never equals production_probability by construction.
    """
    w = float(w_segment)
    if not math.isfinite(w):
        raise ValueError(f"w_segment must be finite; got {w_segment!r}")
    w = min(1.0, max(0.0, w))
    lp = logit(pure_probability)
    lr = logit(reference_market_probability)
    return _clip01(inv_logit(lr + w * (lp - lr)))


def pure_settled_from_active_pmf(
    active_pmf: Mapping[int, float] | Sequence[float] | str,
    line: float,
) -> dict[str, Any]:
    """Independently modeled active-conditional settled probabilities from a pure PMF.

    Never accepts sportsbook prices, consensus, or market-consistent tilt.
    """
    from wnba_props_model.models.simulation import json_to_pmf  # local import

    pmf = json_to_pmf(active_pmf) if isinstance(active_pmf, str) else active_pmf
    try:
        settled = settled_probabilities_from_pmf(pmf, float(line))
    except UndefinedSettledProbabilityError:
        return {
            "pure_probability_over": None,
            "pure_probability_under": None,
            "p_over_unconditional": None,
            "p_under_unconditional": None,
            "p_push": None,
            "valid": False,
            "reason": "ABSTAIN_MISSING_PURE_PROBABILITY",
        }
    return {
        "pure_probability_over": settled.p_over_settled,
        "pure_probability_under": settled.p_under_settled,
        "p_over_unconditional": settled.p_over_unconditional,
        "p_under_unconditional": settled.p_under_unconditional,
        "p_push": settled.p_push,
        "valid": (
            settled.p_over_settled is not None
            and settled.p_under_settled is not None
            and math.isfinite(settled.p_over_settled)
            and math.isfinite(settled.p_under_settled)
        ),
        "reason": "",
    }


def side_pure_probability(settled: Mapping[str, Any], side: str) -> float | None:
    s = str(side).strip().lower()
    if not settled.get("valid"):
        return None
    if s == "over":
        return settled.get("pure_probability_over")
    if s == "under":
        return settled.get("pure_probability_under")
    raise ValueError(f"side must be over/under; got {side!r}")


def production_probability_for_side(
    *,
    production_p_over: float | None,
    side: str,
) -> float | None:
    """Conservative pricing probability for market making / fair-price delivery.

    Kept visibly separate from pick_probability. May be market-consistent upstream;
    this function never substitutes it into the pick track.
    """
    if production_p_over is None:
        return None
    try:
        p = float(production_p_over)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p):
        return None
    s = str(side).strip().lower()
    if s == "over":
        return _clip01(p)
    if s == "under":
        return _clip01(1.0 - p)
    raise ValueError(f"side must be over/under; got {side!r}")


def assert_tracks_distinct(
    *,
    pure_probability: float,
    reference_market_probability: float | None,
    production_probability: float | None,
    pick_prob: float,
) -> None:
    """Guard: pick must not silently equal production merely because production is market-consistent."""
    if production_probability is None:
        return
    # If pure differs from production (e.g. zero-residual market pricing), pick must follow
    # the pure/reference blend, not collapse to production.
    if (
        reference_market_probability is not None
        and abs(float(pure_probability) - float(production_probability)) > 1e-9
        and abs(float(pick_prob) - float(production_probability)) < 1e-12
        and abs(float(pick_prob) - float(pure_probability)) > 1e-9
    ):
        raise AssertionError(
            "pick_probability collapsed onto production_probability while pure differs; "
            "zero-residual production must not suppress pure alpha"
        )


def validate_pmf_mass(
    active_pmf: Mapping[int, float] | Sequence[float] | str,
    *,
    mass_tol: float = 1e-6,
) -> tuple[bool, float, str]:
    """Return (ok, total_mass, reason)."""
    from wnba_props_model.models.simulation import json_to_pmf

    try:
        pmf = json_to_pmf(active_pmf) if isinstance(active_pmf, str) else active_pmf
        if isinstance(pmf, Mapping):
            arr = np.asarray(list(pmf.values()), dtype=float)
        else:
            arr = np.asarray(list(pmf), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return False, float("nan"), f"pmf_parse:{exc}"
    if arr.size == 0:
        return False, 0.0, "empty_pmf"
    if not np.all(np.isfinite(arr)):
        return False, float("nan"), "nonfinite"
    if np.any(arr < -1e-15):
        return False, float(arr.sum()), "negative"
    total = float(arr.sum())
    if abs(total - 1.0) > mass_tol:
        return False, total, "mass_tolerance"
    return True, total, ""
