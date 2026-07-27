"""Single source of truth for the delivered binary probability (PR 1A B2/B4).

``build_probability_lineage`` is the ONLY function permitted to create
``model_prob_over_final``. Both the live delivery path and the historical replay/scoring
path must invoke this function (or consume its serialized output); there must be no
second implementation.

Lineage stages (PR 1A: pure track, identity binary calibration, no market anchor):

    final PMF
      -> push-safe settled probability (settled_probabilities_from_pmf)
      -> binary probability calibration (identity in 1A; fail-closed when enabled)
      -> optional market-anchored residual correction (null in 1A)
      -> model_prob_over_final
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from wnba_props_model.models.binary_probability_calibration import (
    BinaryCalibrationRegistry,
    CalibrationResult,
)
from wnba_props_model.models.market import (
    UndefinedSettledProbabilityError,
    settled_probabilities_from_pmf,
)

PROBABILITY_LINEAGE_VERSION = "1A.1"


@dataclass(frozen=True)
class ProbabilityLineage:
    model_prob_over_unconditional: float
    model_prob_under_unconditional: float
    model_prob_push: float
    model_prob_over_settled_from_final_pmf: float | None
    model_prob_over_binary_calibrated: float | None
    model_prob_over_market_anchored: float | None
    model_prob_over_final: float | None
    probability_track: str
    probability_lineage_version: str
    calibration_status: str
    calibrator_id: str | None
    calibrator_hash: str | None
    structural_model_id: str | None
    structural_model_hash: str | None
    binary_score_eligible: bool

    def as_row(self) -> dict:
        """Serializable dict for Parquet evidence (float64 preserved; never rounded)."""
        return asdict(self)


def build_probability_lineage(
    *,
    final_pmf: "Mapping[int, float] | Sequence[float]",
    line: float,
    prop: str,
    role: str,
    binary_calibration_registry: BinaryCalibrationRegistry | None = None,
    market_anchor: float | None = None,
    structural_model_id: str | None = None,
    structural_model_hash: str | None = None,
    probability_track: str = "pure_forecast",
) -> ProbabilityLineage:
    """Create the full probability lineage for one (player, game, prop, line) row.

    In PR 1A: pure_forecast track only, identity binary calibration, no market anchor.
    ``market_anchor`` must be None on the pure track.
    """
    if probability_track == "pure_forecast" and market_anchor is not None:
        raise ValueError("pure_forecast track must not receive a market anchor")

    registry = binary_calibration_registry or BinaryCalibrationRegistry(enabled=False)

    # Stage 1: push-safe settled probability from the FINAL pmf.
    try:
        settled = settled_probabilities_from_pmf(final_pmf, float(line))
        p_over_unc = settled.p_over_unconditional
        p_under_unc = settled.p_under_unconditional
        p_push = settled.p_push
        p_settled = settled.p_over_settled           # defined (not None) here
        binary_eligible = True
    except UndefinedSettledProbabilityError:
        # All mass on the push -> binary-ineligible row. Never fabricate 0.5.
        # Recompute unconditional/push for reporting via a direct decomposition.
        from wnba_props_model.models.market import _pmf_to_dense_array  # noqa: PLC0415
        import numpy as _np  # noqa: PLC0415
        arr = _pmf_to_dense_array(final_pmf)
        k = _np.arange(arr.size)
        p_over_unc = float(arr[k > line].sum())
        p_under_unc = float(arr[k < line].sum())
        p_push = float(arr[int(round(line))]) if int(round(line)) < arr.size else 0.0
        return ProbabilityLineage(
            model_prob_over_unconditional=p_over_unc,
            model_prob_under_unconditional=p_under_unc,
            model_prob_push=p_push,
            model_prob_over_settled_from_final_pmf=None,
            model_prob_over_binary_calibrated=None,
            model_prob_over_market_anchored=None,
            model_prob_over_final=None,
            probability_track=probability_track,
            probability_lineage_version=PROBABILITY_LINEAGE_VERSION,
            calibration_status="binary_ineligible_push",
            calibrator_id=None,
            calibrator_hash=None,
            structural_model_id=structural_model_id,
            structural_model_hash=structural_model_hash,
            binary_score_eligible=False,
        )

    # Stage 2: binary probability calibration (identity in 1A; fail-closed when enabled).
    cal: CalibrationResult = registry.apply(prop, role, p_settled)
    p_binary_cal = cal.p_calibrated

    # Stage 3: optional market-anchored residual correction (null on the pure track / 1A).
    p_market_anchored = market_anchor

    # Stage 4: final = market-anchored when present, else binary-calibrated (pure track).
    p_final = p_market_anchored if p_market_anchored is not None else p_binary_cal

    return ProbabilityLineage(
        model_prob_over_unconditional=p_over_unc,
        model_prob_under_unconditional=p_under_unc,
        model_prob_push=p_push,
        model_prob_over_settled_from_final_pmf=p_settled,
        model_prob_over_binary_calibrated=p_binary_cal,
        model_prob_over_market_anchored=p_market_anchored,
        model_prob_over_final=p_final,
        probability_track=probability_track,
        probability_lineage_version=PROBABILITY_LINEAGE_VERSION,
        calibration_status=cal.calibration_status,
        calibrator_id=cal.calibrator_id,
        calibrator_hash=cal.calibrator_hash,
        structural_model_id=structural_model_id,
        structural_model_hash=structural_model_hash,
        binary_score_eligible=binary_eligible,
    )


def build_settled_probability_from_active_pmf(
    *,
    active_pmf: "Mapping[int, float] | Sequence[float]",
    line: float,
    prop: str,
    role: str,
    binary_calibration_registry: BinaryCalibrationRegistry | None = None,
    structural_model_id: str | None = None,
    structural_model_hash: str | None = None,
) -> ProbabilityLineage:
    """SHARED sportsbook settlement for the VOID_DNP rule (owner item 3).

    This is the SINGLE settlement implementation used by BOTH live delivery
    (``deliver.build_market_comparison``) and OOF scoring/replay
    (``scripts/evaluate_pure_oof.py``). It settles the binary from the ACTIVE
    (conditional-on-appearance) PMF — push-safe, void-on-DNP — then applies the
    monotone pure binary calibrator, via ``build_probability_lineage``.

    ``p_dnp`` is intentionally NOT an argument: under VOID_DNP a did-not-play voids
    the wager and must not move the settled Over/Under probability. Availability
    suppression / void-risk status is handled separately by the caller.

    Returns a :class:`ProbabilityLineage` exposing p_over/under_unconditional,
    p_push, p_over_settled (``model_prob_over_settled_from_final_pmf``),
    model_prob_over_binary_calibrated, model_prob_over_final, binary_score_eligible,
    and calibration provenance.
    """
    return build_probability_lineage(
        final_pmf=active_pmf,
        line=float(line),
        prop=str(prop),
        role=str(role),
        binary_calibration_registry=binary_calibration_registry,
        market_anchor=None,
        structural_model_id=structural_model_id,
        structural_model_hash=structural_model_hash,
        probability_track="pure_forecast",
    )


def fail_closed_lineage(
    status: str,
    *,
    probability_track: str = "pure_forecast",
    structural_model_id: str | None = None,
    structural_model_hash: str | None = None,
) -> ProbabilityLineage:
    """A fully fail-closed lineage: NO settled/final probability, binary-ineligible.

    Used when a certified sportsbook probability cannot legitimately be produced —
    a missing active PMF in certified mode, or an unknown book DNP-settlement rule.
    NEVER fabricates 0.5, an identity fallback, a legacy PMF, or a ``/(1-p_dnp)``
    shortcut. ``status`` records the reason (audited downstream).
    """
    import math as _math
    return ProbabilityLineage(
        model_prob_over_unconditional=_math.nan,
        model_prob_under_unconditional=_math.nan,
        model_prob_push=_math.nan,
        model_prob_over_settled_from_final_pmf=None,
        model_prob_over_binary_calibrated=None,
        model_prob_over_market_anchored=None,
        model_prob_over_final=None,
        probability_track=probability_track,
        probability_lineage_version=PROBABILITY_LINEAGE_VERSION,
        calibration_status=status,
        calibrator_id=None,
        calibrator_hash=None,
        structural_model_id=structural_model_id,
        structural_model_hash=structural_model_hash,
        binary_score_eligible=False,
    )
