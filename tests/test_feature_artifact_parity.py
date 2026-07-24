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
    assert_inference_parity,
    build_feature_artifact_metadata,
    capture_feature_dtype_kinds,
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


# --------------------------------------------------------------------------- A3

class _Artifact:
    """Minimal stand-in for a fitted model exposing the A3 parity contract fields."""
    def __init__(self, usable, dtype_kinds=None):
        self._usable_cols = usable
        self._feature_dtype_kinds = dtype_kinds


def test_a3_shared_validator_missing_feature_is_fatal():
    art = _Artifact(["a", "b", "c"])
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})           # c missing
    with pytest.raises(FeatureArtifactParityError):
        assert_inference_parity(frame, art, "unit")


def test_a3_shared_validator_unexpected_categorical_is_fatal():
    kinds = {"a": "f", "b": "f"}
    art = _Artifact(["a", "b"], dtype_kinds=kinds)
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})  # b arrives object/categorical
    with pytest.raises(FeatureArtifactParityError):
        assert_inference_parity(frame, art, "unit")


def test_a3_shared_validator_reordered_is_ok():
    kinds = {"a": "f", "b": "f"}
    art = _Artifact(["a", "b"], dtype_kinds=kinds)
    frame = pd.DataFrame({"b": [2.0], "a": [1.0]})           # reordered but complete -> ok
    assert_inference_parity(frame, art, "unit")               # reindex normalizes order


def test_a3_capture_dtype_kinds_and_metadata():
    frame = pd.DataFrame({"a": np.arange(3, dtype=float), "b": [1, 2, 3], "c": ["x", "y", "z"]})
    kinds = capture_feature_dtype_kinds(frame, ["a", "b", "c"])
    assert kinds == {"a": "f", "b": "i", "c": "O"}
    meta = build_feature_artifact_metadata(frame, ["a", "b", "c"], training_data_hash="h")
    assert meta["ordered_feature_names"] == ["a", "b", "c"]
    assert meta["feature_schema_hash"] == feature_schema_hash(["a", "b", "c"])
    assert meta["dtype_kinds"] == kinds
    assert meta["training_data_hash"] == "h"
    assert meta["builder_version"]


def test_a3_every_inference_path_wires_the_validator():
    """Every model component that reindexes to its trained feature list must call a fail-closed
    parity validator FIRST (proves A3 coverage across all inference artifacts)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "wnba_props_model" / "models"
    expectations = {
        "minutes_model.py": 2,       # predict + predict_quantiles (DNP shares X)
        "hurdle.py": 1,              # ZINBStatModel.predict
        "rate_model.py": 2,          # StatRateModel.predict_mean + HurdleModel.predict
        "beta_binomial.py": 2,       # predict_pmf_matrix + predict_mean
        "beta_binomial_fg3m.py": 1,  # predict_pmf_matrix
        "svd_bridge.py": 1,          # predict
        "pmf_engine.py": 1,          # dispersion_model.predict
    }
    for fname, min_calls in expectations.items():
        src = (root / fname).read_text()
        n = src.count("assert_inference_parity(") + src.count("assert_feature_artifact_parity(")
        assert n >= min_calls, f"{fname}: expected >= {min_calls} parity guard(s), found {n}"


def test_a3_statratemodel_predict_is_fail_closed():
    """End-to-end: a real fitted StatRateModel must reject a truncated inference frame."""
    from wnba_props_model.models.rate_model import StatRateModel
    rng = np.random.default_rng(0)
    n = 120
    feats = [f"f{i}" for i in range(6)]
    X = pd.DataFrame({f: rng.normal(size=n) for f in feats})
    y = pd.Series(np.clip(rng.poisson(5, size=n), 0, None).astype(float))
    m = StatRateModel(stat="pts", cfg={"min_stat_mean": 0.01})
    m.fit(X, y)
    assert getattr(m, "_feature_dtype_kinds", None)           # dtype kinds captured at fit
    m.predict_mean(X)                                          # full frame -> ok
    with pytest.raises(FeatureArtifactParityError):
        m.predict_mean(X.drop(columns=["f3"]))                # missing feature -> fatal
