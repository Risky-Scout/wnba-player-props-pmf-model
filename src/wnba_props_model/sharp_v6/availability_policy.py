"""Fail-closed availability decisions used before PMF materialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AvailabilityStatus(StrEnum):
    OUT = "OUT"
    SUSPENDED = "SUSPENDED"
    DOUBTFUL = "DOUBTFUL"
    QUESTIONABLE = "QUESTIONABLE"
    PROBABLE = "PROBABLE"
    NOT_LISTED = "NOT_LISTED"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"
    UNSPECIFIED = "UNSPECIFIED"


@dataclass(frozen=True)
class AvailabilityDecision:
    status: AvailabilityStatus
    action: str
    p_active: float | None
    dnp_mass: float | None
    reason: str
    requires_injury_model: bool = False
    historically_calibrated: bool = False

    @property
    def should_abstain(self) -> bool:
        return self.action == "ABSTAIN"

    @property
    def overrides_model_p_active(self) -> bool:
        return self.p_active is not None


def normalize_status(value: object, *, snapshot_success: bool) -> AvailabilityStatus:
    """Normalize provider text; a failed snapshot can never imply healthy.

    Missing/blank status on a successful snapshot is UNSPECIFIED (keep model),
    not NOT_LISTED. NOT_LISTED requires an explicit successful empty listing.
    """
    if not snapshot_success:
        return AvailabilityStatus.UNKNOWN_SOURCE
    if value is None:
        return AvailabilityStatus.UNSPECIFIED
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if not text or text in {"NONE", "NAN", "NULL", "UNSPECIFIED"}:
        return AvailabilityStatus.UNSPECIFIED
    aliases = {
        "DOUBT": "DOUBTFUL",
        "QUESTION": "QUESTIONABLE",
        "PROB": "PROBABLE",
        "O": "OUT",
        "ACTIVE": "NOT_LISTED",
        "HEALTHY": "NOT_LISTED",
    }
    text = aliases.get(text, text)
    try:
        return AvailabilityStatus(text)
    except ValueError:
        return AvailabilityStatus.UNKNOWN_SOURCE


def decide_availability(
    status: AvailabilityStatus | str | None,
    *,
    snapshot_success: bool,
    injury_model_in_domain: bool = False,
    explicitly_not_listed: bool = False,
) -> AvailabilityDecision:
    """Return a pricing decision without making network or model calls."""
    if isinstance(status, AvailabilityStatus):
        s = (
            status
            if snapshot_success or status == AvailabilityStatus.UNKNOWN_SOURCE
            else AvailabilityStatus.UNKNOWN_SOURCE
        )
    else:
        s = normalize_status(status, snapshot_success=snapshot_success)
        if explicitly_not_listed and s == AvailabilityStatus.UNSPECIFIED and snapshot_success:
            s = AvailabilityStatus.NOT_LISTED

    if s in (AvailabilityStatus.OUT, AvailabilityStatus.SUSPENDED):
        return AvailabilityDecision(
            s, "ABSTAIN", 0.0, 1.0, "ABSTAIN_PLAYER_OUT", historically_calibrated=False
        )
    if s == AvailabilityStatus.DOUBTFUL:
        if injury_model_in_domain:
            return AvailabilityDecision(
                s, "INJURY_MODEL", None, None, "INJURY_CONDITIONED", True, True
            )
        return AvailabilityDecision(s, "ABSTAIN", None, None, "DOUBTFUL_OUT_OF_DOMAIN", True, False)
    if s in (AvailabilityStatus.QUESTIONABLE, AvailabilityStatus.PROBABLE):
        if injury_model_in_domain:
            return AvailabilityDecision(
                s, "INJURY_MODEL", None, None, "INJURY_CONDITIONED", True, True
            )
        return AvailabilityDecision(
            s, "ABSTAIN", None, None, "INJURY_STATUS_OUT_OF_DOMAIN", True, False
        )
    if s == AvailabilityStatus.NOT_LISTED:
        # Operational gate only — not a historically calibrated unconditional p_active.
        return AvailabilityDecision(
            s,
            "ACTIVE_CONDITIONAL",
            1.0,
            0.0,
            "ACTIVE_ROSTER_NOT_LISTED",
            historically_calibrated=False,
        )
    if s == AvailabilityStatus.UNSPECIFIED:
        return AvailabilityDecision(
            s,
            "MODEL_DEFAULT",
            None,
            None,
            "NO_AVAILABILITY_OVERRIDE",
            historically_calibrated=False,
        )
    return AvailabilityDecision(
        s,
        "ABSTAIN",
        None,
        None,
        "AVAILABILITY_SOURCE_UNAVAILABLE",
        historically_calibrated=False,
    )
