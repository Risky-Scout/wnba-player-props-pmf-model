"""Strict, fail-closed contract for the delivered binary probability (PR 1A).

Decision-grade consumers (fair odds, edge, CLV selection, historical scoring, market
evaluation, reporting, recommendation inputs) must consume the immutable delivered
probability ``model_prob_over_final`` and nothing else. This module centralizes the column
names and the fail-closed validators so no consumer invents its own fallback.

Explicitly forbidden anywhere in decision-grade code:

    row.get("model_prob_over_final", row.get("model_prob_over"))   # silent legacy fallback

There is no silent clipping, no fallback to the legacy column, and no fallback to PMF
reconstruction. Missing/NaN/inf/out-of-range values fail closed with the consumer named.
"""
from __future__ import annotations

import math
from typing import Iterable

FINAL_PROBABILITY_COLUMN = "model_prob_over_final"
LEGACY_PROBABILITY_COLUMN = "model_prob_over"
PROBABILITY_LINEAGE_VERSION = "v1"


class ProbabilityContractError(ValueError):
    """Fail-closed violation of the delivered-probability contract."""


def validate_final_probability(value, *, consumer: str, allow_none: bool = False) -> float:
    """Validate a single delivered ``model_prob_over_final`` value.

    Fails closed (never clips/rounds/falls back). ``allow_none=True`` permits a genuinely
    binary-ineligible row (all-push) to carry None, which the caller must then exclude from
    binary scoring rather than substitute a value.
    """
    if value is None:
        if allow_none:
            return float("nan")
        raise ProbabilityContractError(
            f"[{consumer}] {FINAL_PROBABILITY_COLUMN} is missing (None); no legacy/PMF fallback")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbabilityContractError(
            f"[{consumer}] {FINAL_PROBABILITY_COLUMN} not numeric: {value!r}") from exc
    if math.isnan(v):
        if allow_none:
            return float("nan")
        raise ProbabilityContractError(f"[{consumer}] {FINAL_PROBABILITY_COLUMN} is NaN")
    if not math.isfinite(v):
        raise ProbabilityContractError(f"[{consumer}] {FINAL_PROBABILITY_COLUMN} is not finite: {v}")
    if not (0.0 <= v <= 1.0):
        raise ProbabilityContractError(
            f"[{consumer}] {FINAL_PROBABILITY_COLUMN} outside [0,1]: {v} (no silent clipping)")
    return v


def require_final_probability(df, *, consumer: str, allow_ineligible: bool = False):
    """Return the validated ``model_prob_over_final`` Series from a DataFrame.

    Fails closed if the column is absent or (when ``allow_ineligible`` is False) contains
    any missing/NaN/inf/out-of-range value. Never falls back to the legacy column or PMF.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if FINAL_PROBABILITY_COLUMN not in getattr(df, "columns", []):
        raise ProbabilityContractError(
            f"[{consumer}] required column {FINAL_PROBABILITY_COLUMN!r} is absent "
            f"(present: {list(getattr(df, 'columns', []))[:12]}...)")
    s = df[FINAL_PROBABILITY_COLUMN]
    if not pd.api.types.is_numeric_dtype(s):
        raise ProbabilityContractError(
            f"[{consumer}] {FINAL_PROBABILITY_COLUMN} must be numeric, got dtype {s.dtype}")
    vals = s.to_numpy(dtype="float64", copy=False)
    if allow_ineligible:
        finite = vals[np.isfinite(vals)]
        if finite.size and ((finite < 0).any() or (finite > 1).any()):
            raise ProbabilityContractError(
                f"[{consumer}] {FINAL_PROBABILITY_COLUMN} has values outside [0,1]")
        return s
    if not np.all(np.isfinite(vals)):
        raise ProbabilityContractError(
            f"[{consumer}] {FINAL_PROBABILITY_COLUMN} has missing/NaN/inf values; fail closed")
    if (vals < 0).any() or (vals > 1).any():
        raise ProbabilityContractError(
            f"[{consumer}] {FINAL_PROBABILITY_COLUMN} has values outside [0,1] (no silent clipping)")
    return s


def assert_alias_invariant(df, *, consumer: str, tol: float = 1e-12) -> None:
    """Assert the deprecated output alias equals the final column within ``tol``.

    The legacy column is output-only. Where both are present and final is finite, they must
    match; a mismatch is a fail-closed violation (used to prove no consumer silently reads
    the legacy value)."""
    import numpy as np  # noqa: PLC0415

    cols = getattr(df, "columns", [])
    if FINAL_PROBABILITY_COLUMN not in cols or LEGACY_PROBABILITY_COLUMN not in cols:
        return
    f = df[FINAL_PROBABILITY_COLUMN].to_numpy(dtype="float64", copy=False)
    lg = df[LEGACY_PROBABILITY_COLUMN].to_numpy(dtype="float64", copy=False)
    both = np.isfinite(f) & np.isfinite(lg)
    if both.any() and np.nanmax(np.abs(f[both] - lg[both])) > tol:
        raise ProbabilityContractError(
            f"[{consumer}] alias invariant violated: {LEGACY_PROBABILITY_COLUMN} != "
            f"{FINAL_PROBABILITY_COLUMN} within {tol}")
