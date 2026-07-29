"""Explicit, fail-closed per-stat feature policy.

Replaces the previous ``stat_feature_subset`` semantics, whose fallbacks were unsafe:

* a **missing** or **empty** feature map silently reverted to the full feature matrix;
* an explicit map with **fewer than eight** available columns *also* reverted to the
  full matrix (the "minimum-eight-column floor").

Both meant a compact, deliberately-chosen one- or two-feature model was never actually
trained on those features in production. This module makes the contract explicit:

``explicit``
    Train on exactly ``required_columns`` (which MUST all be present -> otherwise raise)
    plus whatever ``optional_columns`` happen to be available. **Never** falls back to
    the full matrix. A one-feature policy stays a one-feature policy.

``base_rate``
    An intentionally empty feature set. Means an intercept / hierarchical-prior /
    structured base-rate model. It does **not** mean "use all features".

``legacy_full_diagnostic``
    The full shared matrix, permitted ONLY as a frozen, **non-certifiable** comparison
    candidate (this is the current production P0 behavior). :pyattr:`certifiable` is
    ``False`` for this mode.

A ``pure_forecast`` policy additionally rejects any current-game / lagged market feature
(sourced from :mod:`wnba_props_model.features.feature_provenance`) at construction time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Sequence

FeatureMode = Literal[
    "explicit",
    "base_rate",
    "legacy_full_diagnostic",
]

InformationContract = Literal[
    "pure_forecast",
    "internal_game_context",
    "external_market_anchored",
]


def feature_schema_hash(columns: Sequence[str]) -> str:
    """Deterministic, order-sensitive hash of an ordered feature list."""
    payload = json.dumps(list(columns), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class FeaturePolicyError(ValueError):
    """Raised when a policy is violated (missing required column, market leak, ...)."""


@dataclass(frozen=True)
class PropFeaturePolicy:
    stat: str
    mode: FeatureMode
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    information_contract: InformationContract
    missing_required_policy: Literal["raise"]
    feature_set_id: str

    def __post_init__(self) -> None:
        if self.mode not in ("explicit", "base_rate", "legacy_full_diagnostic"):
            raise FeaturePolicyError(f"unknown feature mode: {self.mode!r}")
        if self.missing_required_policy != "raise":
            raise FeaturePolicyError(
                "missing_required_policy must be 'raise' (fail-closed contract)")
        if self.mode == "base_rate" and (self.required_columns or self.optional_columns):
            raise FeaturePolicyError(
                f"{self.stat}: base_rate policy must declare NO columns (intercept model); "
                f"got required={self.required_columns} optional={self.optional_columns}")
        # A pure_forecast policy must never reference a market-derived feature.
        if self.information_contract == "pure_forecast":
            self._assert_no_market(self.required_columns + self.optional_columns)

    @staticmethod
    def _assert_no_market(columns: Sequence[str]) -> None:
        # Local import to avoid a hard dependency cycle at module import time.
        from wnba_props_model.features.feature_provenance import classify, Provenance
        bad = sorted(
            c for c in columns
            if classify(c) in (
                Provenance.EXTERNAL_MARKET_CURRENT_GAME,
                Provenance.EXTERNAL_MARKET_LAGGED,
            )
        )
        if bad:
            raise FeaturePolicyError(
                f"pure_forecast policy must not contain market-derived features: {bad}")

    @property
    def certifiable(self) -> bool:
        """legacy_full_diagnostic is a frozen comparison candidate only."""
        return self.mode != "legacy_full_diagnostic"

    def resolve_columns(self, available_columns: Sequence[str]) -> list[str]:
        """Return the ordered training columns for this policy given the columns that
        actually exist in the training matrix.

        * ``base_rate`` -> ``[]`` (intercept).
        * ``explicit``  -> required (all must exist, else raise) + available optionals,
          **never** the full matrix.
        * ``legacy_full_diagnostic`` -> every available column (the full matrix).
        """
        available = list(available_columns)
        avail_set = set(available)

        if self.mode == "base_rate":
            return []

        if self.mode == "legacy_full_diagnostic":
            return available

        # explicit
        missing = [c for c in self.required_columns if c not in avail_set]
        if missing:
            raise FeaturePolicyError(
                f"{self.stat}: explicit policy '{self.feature_set_id}' is missing required "
                f"columns {missing}. Refusing to fall back to the full feature matrix.")
        keep = list(self.required_columns) + [c for c in self.optional_columns if c in avail_set]
        # de-duplicate while preserving policy order
        seen: set[str] = set()
        ordered: list[str] = []
        for c in keep:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    def to_dict(self) -> dict:
        return {
            "stat": self.stat,
            "mode": self.mode,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "information_contract": self.information_contract,
            "missing_required_policy": self.missing_required_policy,
            "feature_set_id": self.feature_set_id,
            "certifiable": self.certifiable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PropFeaturePolicy":
        return cls(
            stat=d["stat"],
            mode=d["mode"],
            required_columns=tuple(d.get("required_columns", ())),
            optional_columns=tuple(d.get("optional_columns", ())),
            information_contract=d["information_contract"],
            missing_required_policy=d.get("missing_required_policy", "raise"),
            feature_set_id=d["feature_set_id"],
        )


@dataclass(frozen=True)
class FittedFeatureSpec:
    """Frozen identity of the feature set a stat model was actually fitted on. Stored on
    the artifact and re-checked at inference so a train/inference schema drift fails loudly
    instead of silently NaN-filling or reordering."""

    feature_set_id: str
    ordered_feature_names: tuple[str, ...]
    feature_schema_hash: str
    information_contract: InformationContract
    training_cutoff: str | None = None
    training_row_hash: str | None = None

    @classmethod
    def build(
        cls,
        policy: PropFeaturePolicy,
        ordered_feature_names: Sequence[str],
        training_cutoff: str | None = None,
        training_row_hash: str | None = None,
    ) -> "FittedFeatureSpec":
        cols = tuple(ordered_feature_names)
        return cls(
            feature_set_id=policy.feature_set_id,
            ordered_feature_names=cols,
            feature_schema_hash=feature_schema_hash(cols),
            information_contract=policy.information_contract,
            training_cutoff=training_cutoff,
            training_row_hash=training_row_hash,
        )

    def verify_inference(self, inference_columns: Sequence[str]) -> None:
        """Raise if the inference frame's ordered columns do not match training exactly."""
        got = tuple(inference_columns)
        if got != self.ordered_feature_names:
            raise FeaturePolicyError(
                f"inference feature schema mismatch for feature_set_id="
                f"{self.feature_set_id!r}: expected {list(self.ordered_feature_names)} "
                f"got {list(got)}")
        got_hash = feature_schema_hash(got)
        if got_hash != self.feature_schema_hash:
            raise FeaturePolicyError(
                f"inference feature_schema_hash mismatch for {self.feature_set_id!r}: "
                f"expected {self.feature_schema_hash} got {got_hash}")

    def to_dict(self) -> dict:
        return {
            "feature_set_id": self.feature_set_id,
            "ordered_feature_names": list(self.ordered_feature_names),
            "feature_schema_hash": self.feature_schema_hash,
            "information_contract": self.information_contract,
            "training_cutoff": self.training_cutoff,
            "training_row_hash": self.training_row_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FittedFeatureSpec":
        return cls(
            feature_set_id=d["feature_set_id"],
            ordered_feature_names=tuple(d["ordered_feature_names"]),
            feature_schema_hash=d["feature_schema_hash"],
            information_contract=d["information_contract"],
            training_cutoff=d.get("training_cutoff"),
            training_row_hash=d.get("training_row_hash"),
        )


def resolve_policy_columns(
    policy: PropFeaturePolicy | None,
    available_columns: Sequence[str],
) -> list[str]:
    """Convenience resolver. ``None`` policy means the caller opted out of the explicit
    contract entirely; that is treated as the legacy full-matrix behavior (non-certifiable)
    and callers should label it accordingly."""
    if policy is None:
        return list(available_columns)
    return policy.resolve_columns(available_columns)
