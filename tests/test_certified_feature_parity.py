"""Phase 2 - certified feature-parity guard: each defect independently prevents prediction.

CERTIFIED mode tolerates ZERO deviations: one missing / one forbidden / one all-null / one
dtype-mismatch / one order-mismatch (schema-hash) / one duplicate each raise
FeatureArtifactParityError, while DIAGNOSTIC mode keeps the benign-absence tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.feature_contract import (
    FEATURE_MODE_CERTIFIED,
    FEATURE_MODE_DIAGNOSTIC,
    FeatureArtifactParityError,
    assert_certified_inference_parity,
    assert_inference_parity_mode,
    capture_feature_dtype_kinds,
    feature_schema_hash,
)


class _Artifact:
    def __init__(self, cols, frame):
        self._usable_cols = list(cols)
        self._feature_schema_hash = feature_schema_hash(list(cols))
        self._feature_dtype_kinds = capture_feature_dtype_kinds(frame, list(cols))
        self._feature_builder_version = "test-v1"


def _good_frame():
    return pd.DataFrame({
        "minutes_lag1": np.array([20.0, 25.0, 30.0]),
        "rest_days": np.array([1, 2, 3]),
        "is_home": np.array([1, 0, 1]),
    })


def _artifact():
    f = _good_frame()
    return _Artifact(["minutes_lag1", "rest_days", "is_home"], f)


def test_certified_passes_on_exact_match():
    assert_certified_inference_parity(_good_frame(), _artifact(), "ok")  # no raise


def test_missing_feature_raises_certified():
    f = _good_frame().drop(columns=["rest_days"])
    with pytest.raises(FeatureArtifactParityError, match="absent"):
        assert_certified_inference_parity(f, _artifact(), "missing")


def test_forbidden_feature_in_artifact_raises_certified():
    f = _good_frame()
    f["pts"] = [1.0, 2.0, 3.0]           # 'pts' is a forbidden box-score leakage feature
    art = _Artifact(["minutes_lag1", "rest_days", "is_home", "pts"], f)
    with pytest.raises(FeatureArtifactParityError, match="FORBIDDEN"):
        assert_certified_inference_parity(f, art, "forbidden")


def test_all_null_feature_raises_certified():
    f = _good_frame()
    f["rest_days"] = np.nan
    with pytest.raises(FeatureArtifactParityError, match="null"):
        assert_certified_inference_parity(f, _artifact(), "allnull")


def test_dtype_mismatch_raises_certified():
    art = _artifact()                     # trained: rest_days is integer-kind
    f = _good_frame()
    f["rest_days"] = f["rest_days"].astype(str)   # arrives as object -> kind mismatch
    with pytest.raises(FeatureArtifactParityError, match="dtype-kind"):
        assert_certified_inference_parity(f, art, "dtype")


def test_order_mutation_schema_hash_raises_certified():
    art = _artifact()
    # mutate the artifact's contract order without recomputing the stored hash
    art._usable_cols = ["rest_days", "minutes_lag1", "is_home"]
    with pytest.raises(FeatureArtifactParityError, match="schema hash"):
        assert_certified_inference_parity(_good_frame(), art, "order")


def test_expected_hash_mismatch_raises_certified():
    with pytest.raises(FeatureArtifactParityError, match="expected"):
        assert_certified_inference_parity(_good_frame(), _artifact(), "hash",
                                          expected_schema_hash="deadbeef" * 8)


def test_duplicate_feature_raises_certified():
    f = _good_frame()
    art = _Artifact(["minutes_lag1", "minutes_lag1", "is_home"], f)
    with pytest.raises(FeatureArtifactParityError, match="duplicate"):
        assert_certified_inference_parity(f, art, "dup")


def test_diagnostic_mode_tolerates_small_absence():
    # one benign absence within tolerance should NOT raise in diagnostic mode
    f = _good_frame().drop(columns=["rest_days"])
    assert_inference_parity_mode(f, _artifact(), "diag", mode=FEATURE_MODE_DIAGNOSTIC)
    # but certified mode raises on the same frame
    with pytest.raises(FeatureArtifactParityError):
        assert_inference_parity_mode(f, _artifact(), "cert", mode=FEATURE_MODE_CERTIFIED)
