"""W0.3 exact feature-artifact parity.

Training/OOF/delivery must require the artifact's exact feature list; a truncated
inference frame (the invalidated 52-of-128 failure) must be a FATAL error, never a
silent drop or all-null substitution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.feature_contract import (
    FeatureArtifactParityError,
    assert_feature_artifact_parity,
    feature_schema_hash,
)


def test_52_of_128_input_is_fatal():
    artifact_features = [f"f{i}" for i in range(128)]         # artifact trained on 128
    frame = pd.DataFrame({f"f{i}": [1.0, 2.0] for i in range(52)})  # inference provides 52
    with pytest.raises(FeatureArtifactParityError) as exc:
        assert_feature_artifact_parity(frame, artifact_features, context="predict")
    assert "76/128" in str(exc.value) or "absent" in str(exc.value)


def test_full_match_passes():
    feats = [f"f{i}" for i in range(10)]
    frame = pd.DataFrame({f: np.arange(3, dtype=float) for f in feats})
    assert_feature_artifact_parity(frame, feats, context="ok")  # must not raise


def test_extra_columns_allowed():
    feats = ["a", "b"]
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "c_extra": [3.0]})
    assert_feature_artifact_parity(frame, feats)  # extras ignored


def test_all_null_expected_feature_is_fatal():
    feats = ["a", "b"]
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    with pytest.raises(FeatureArtifactParityError):
        assert_feature_artifact_parity(frame, feats, check_all_null=True)


def test_dtype_kind_mismatch_is_fatal():
    feats = ["a"]
    frame = pd.DataFrame({"a": ["x", "y"]})  # object where numeric expected
    with pytest.raises(FeatureArtifactParityError):
        assert_feature_artifact_parity(frame, feats, dtype_map={"a": "f"})


def test_feature_schema_hash_is_deterministic_and_order_sensitive():
    a = feature_schema_hash(["x", "y", "z"])
    assert a == feature_schema_hash(["x", "y", "z"])          # deterministic
    assert a != feature_schema_hash(["z", "y", "x"])          # order is part of the contract
    assert len(a) == 64
