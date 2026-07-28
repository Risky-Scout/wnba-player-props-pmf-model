"""Phase 2 - CERTIFIED feature-parity contract (zero tolerance).

Kept in a SEPARATE module from the foundation-locked ``feature_contract.py`` so the certified guard is
purely additive and does not mutate the locked file. It reuses the locked primitives
(``FORBIDDEN_MODEL_FEATURES``, ``feature_schema_hash``, ``FeatureArtifactParityError``,
``assert_inference_parity``) without changing them.

CERTIFIED mode tolerates NO deviation: no benign absence, no optional dtype check, no silent NaN/zero
insertion, no forbidden feature, no duplicate, no order mutation, no schema-hash divergence. An artifact
that expects a now-forbidden feature is REJECTED (retrain), never silently pruned. Extra frame columns
are ignored ONLY after the full contract passes.
"""
from __future__ import annotations

from wnba_props_model.features.feature_contract import (
    FORBIDDEN_MODEL_FEATURES,
    FeatureArtifactParityError,
    assert_inference_parity,
    feature_schema_hash,
)

FEATURE_MODE_CERTIFIED = "certified"
FEATURE_MODE_DIAGNOSTIC = "diagnostic"


def assert_certified_inference_parity(
    frame,
    model,
    context: str = "",
    *,
    expected_schema_hash: "str | None" = None,
) -> None:
    """Fail-closed CERTIFIED parity. Raises ``FeatureArtifactParityError`` on ANY deviation."""
    import pandas as pd  # local import to avoid module-level cycle
    usable = getattr(model, "_usable_cols", None)
    if not usable:
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: artifact has no recorded feature contract "
            f"(_usable_cols); cannot certify.")
    usable = list(usable)

    # 1) no duplicate feature name in the contract
    dupes = sorted({f for f in usable if usable.count(f) > 1})
    if dupes:
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: duplicate feature name(s) in artifact contract: {dupes}")

    # 2) artifact must not expect a now-forbidden feature (reject -> retrain, do NOT silently drop)
    forbidden = sorted(set(usable) & FORBIDDEN_MODEL_FEATURES)
    if forbidden:
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: artifact expects now-FORBIDDEN feature(s) {forbidden}; "
            f"reject the artifact and retrain a valid pure artifact (silent removal at inference is banned).")

    # 3) exact presence: every contract feature must be in the frame (no benign tolerance)
    cols = list(frame.columns) if isinstance(frame, pd.DataFrame) else list(frame)
    colset = set(cols)
    missing = [f for f in usable if f not in colset]
    if missing:
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: {len(missing)}/{len(usable)} contract features absent "
            f"(certified mode tolerates ZERO absences). First missing: {missing[:12]}")

    # 4) exact schema hash (order-sensitive -> catches order mutation)
    want_hash = getattr(model, "_feature_schema_hash", None)
    got_hash = feature_schema_hash(usable)
    if want_hash is not None and got_hash != want_hash:
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: feature schema hash mismatch (order mutation or "
            f"contract drift): artifact={want_hash[:12]} computed={got_hash[:12]}")
    if expected_schema_hash is not None and got_hash != expected_schema_hash:
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: schema hash != expected: computed={got_hash[:12]} "
            f"expected={expected_schema_hash[:12]}")

    # 5) builder version must be recorded
    if not getattr(model, "_feature_builder_version", None):
        raise FeatureArtifactParityError(
            f"{context or 'certified parity'}: artifact has no recorded feature builder version.")

    if isinstance(frame, pd.DataFrame) and len(frame):
        # 6) nullability contract: no entirely-null required feature (no silent NaN insertion)
        all_null = [f for f in usable if frame[f].isna().all()]
        if all_null:
            raise FeatureArtifactParityError(
                f"{context or 'certified parity'}: {len(all_null)} required feature(s) entirely null "
                f"(silent NaN substitution banned). First: {all_null[:12]}")

        # 7) exact dtype-kind map on EVERY contract feature
        dtmap = getattr(model, "_feature_dtype_kinds", None)
        if not dtmap:
            raise FeatureArtifactParityError(
                f"{context or 'certified parity'}: artifact has no recorded dtype-kind map; cannot certify.")
        bad = []
        for f in usable:
            want = dtmap.get(f)
            if want is None:
                bad.append((f, "MISSING_FROM_DTYPE_MAP", None))
            elif frame[f].dtype.kind != str(want):
                bad.append((f, frame[f].dtype.kind, want))
        if bad:
            raise FeatureArtifactParityError(
                f"{context or 'certified parity'}: dtype-kind mismatch on {len(bad)} feature(s): {bad[:8]}")


def assert_inference_parity_mode(frame, model, context: str, *, mode: str = FEATURE_MODE_DIAGNOSTIC,
                                 strict_dtype: bool = False,
                                 expected_schema_hash: "str | None" = None) -> None:
    """Dispatch to CERTIFIED (zero tolerance) or DIAGNOSTIC (benign-absence tolerant) parity."""
    if mode == FEATURE_MODE_CERTIFIED:
        assert_certified_inference_parity(frame, model, context, expected_schema_hash=expected_schema_hash)
    else:
        assert_inference_parity(frame, model, context, strict_dtype=strict_dtype)
