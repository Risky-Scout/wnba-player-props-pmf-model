"""W0.6: one required-mode binary-calibration policy resolver.

The same resolver + policy must be usable by delivery, market comparison, proof, scoring,
and OOF, applying identical calibration. Required mode is fail-closed.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from wnba_props_model.models.binary_calibrators import IsotonicCalibrator
from wnba_props_model.models.binary_probability_calibration import (
    CalibrationError,
    load_binary_calibration_registry,
)


def _write_policy(tmp: Path, with_calibrator=True):
    props = {p: {"method": "identity"} for p in
             ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]}
    if with_calibrator:
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 500)
        y = (rng.uniform(0, 1, 500) < p ** 2).astype(int)
        model = IsotonicCalibrator().fit(p, y)
        art = tmp / "binary_isotonic_reb.pkl"
        joblib.dump(model, art)
        import hashlib
        sha = hashlib.sha256(art.read_bytes()).hexdigest()
        props["reb"] = {"method": "isotonic", "path": str(art), "sha256": sha}
    policy = tmp / "policy.json"
    policy.write_text(json.dumps({"version": "binary-cal-v1", "enabled": True, "props": props}))
    return policy


def test_missing_policy_required_is_fatal(tmp_path):
    with pytest.raises(CalibrationError):
        load_binary_calibration_registry(tmp_path / "nope.json", mode="required")


def test_missing_policy_optional_is_identity(tmp_path):
    reg = load_binary_calibration_registry(tmp_path / "nope.json", mode="optional")
    r = reg.apply("reb", "all", 0.7)
    assert r.p_calibrated == 0.7 and r.calibration_status.startswith("identity")


def test_disabled_mode_always_identity(tmp_path):
    reg = load_binary_calibration_registry(_write_policy(tmp_path), mode="disabled")
    assert reg.apply("reb", "all", 0.42).p_calibrated == 0.42


def test_identity_prop_passthrough_and_calibrated_prop_changes(tmp_path):
    reg = load_binary_calibration_registry(_write_policy(tmp_path), mode="required")
    assert reg.apply("pts", "all", 0.6).p_calibrated == 0.6           # identity method
    out = reg.apply("reb", "all", 0.8)                                # isotonic method
    assert out.calibration_status == "calibrated"
    assert 0.0 < out.p_calibrated < 1.0
    assert out.calibrator_hash is not None


def test_required_missing_prop_entry_is_fatal(tmp_path):
    pol = _write_policy(tmp_path)
    cfg = json.loads(pol.read_text()); del cfg["props"]["ast"]; pol.write_text(json.dumps(cfg))
    reg = load_binary_calibration_registry(pol, mode="required")
    with pytest.raises(CalibrationError):
        reg.apply("ast", "all", 0.5)


def test_hash_mismatch_is_fatal(tmp_path):
    pol = _write_policy(tmp_path)
    cfg = json.loads(pol.read_text()); cfg["props"]["reb"]["sha256"] = "0" * 64
    pol.write_text(json.dumps(cfg))
    reg = load_binary_calibration_registry(pol, mode="required")
    with pytest.raises(CalibrationError):
        reg.apply("reb", "all", 0.8)


def test_delivery_proof_parity_same_policy_same_result(tmp_path):
    """Delivery and proof both load through the resolver -> identical calibrated value."""
    pol = _write_policy(tmp_path)
    reg_delivery = load_binary_calibration_registry(pol, mode="required")
    reg_proof = load_binary_calibration_registry(pol, mode="required")
    for p in (0.2, 0.55, 0.83):
        assert reg_delivery.apply("reb", "core", p).p_calibrated == \
               reg_proof.apply("reb", "starter", p).p_calibrated   # role-agnostic, identical
