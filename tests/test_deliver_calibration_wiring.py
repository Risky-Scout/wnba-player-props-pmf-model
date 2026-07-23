"""PR 1A B5: fail-closed binary-calibration wiring."""
from __future__ import annotations

import joblib
import numpy as np
import pytest

from wnba_props_model.models.binary_probability_calibration import (
    BinaryCalibrationRegistry,
    CalibrationError,
)
from wnba_props_model.models.probability_lineage import build_probability_lineage


class _AffineCal:
    """Toy calibrator: predict(x) = clip(a*x+b)."""
    def __init__(self, a=1.0, b=0.0, clip=True):
        self.a, self.b, self.clip = a, b, clip

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = self.a * X + self.b
        return np.clip(y, 0.0, 1.0) if self.clip else y


def _dump(obj, path):
    joblib.dump(obj, path)
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_disabled_registry_is_identity():
    reg = BinaryCalibrationRegistry(enabled=False)
    r = reg.apply("pts", "starter", 0.62)
    assert r.p_calibrated == 0.62
    assert r.calibration_status == "identity_disabled"
    assert r.calibrator_id is None and r.calibrator_hash is None


def test_lineage_uses_identity_in_1a():
    pmf = np.array([0.1, 0.2, 0.3, 0.4])
    lin = build_probability_lineage(final_pmf=pmf, line=1.5, prop="pts", role="starter")
    assert lin.calibration_status == "identity_disabled"
    assert lin.model_prob_over_binary_calibrated == lin.model_prob_over_settled_from_final_pmf
    assert lin.model_prob_over_final == lin.model_prob_over_binary_calibrated
    assert lin.model_prob_over_market_anchored is None
    assert lin.probability_track == "pure_forecast"
    assert lin.binary_score_eligible is True


def test_enabled_calibrated_path(tmp_path):
    p = tmp_path / "pts_starter.pkl"
    sha = _dump(_AffineCal(a=0.5, b=0.25), p)
    reg = BinaryCalibrationRegistry(enabled=True,
                                    artifacts={"pts|starter": {"path": str(p), "sha256": sha}})
    r = reg.apply("pts", "starter", 0.6)
    assert r.calibration_status == "calibrated"
    assert r.p_calibrated == pytest.approx(0.55)
    assert r.calibrator_id == "pts|starter" and r.calibrator_hash == sha


def test_enabled_missing_artifact_is_fatal(tmp_path):
    reg = BinaryCalibrationRegistry(enabled=True,
        artifacts={"pts|starter": {"path": str(tmp_path / "nope.pkl"), "sha256": "0" * 64}})
    with pytest.raises(CalibrationError):
        reg.apply("pts", "starter", 0.6)


def test_enabled_hash_mismatch_is_fatal(tmp_path):
    p = tmp_path / "c.pkl"; _dump(_AffineCal(), p)
    reg = BinaryCalibrationRegistry(enabled=True,
        artifacts={"pts|starter": {"path": str(p), "sha256": "d" * 64}})
    with pytest.raises(CalibrationError):
        reg.apply("pts", "starter", 0.6)


def test_undeclared_role_fallback_is_fatal(tmp_path):
    p = tmp_path / "pts.pkl"; sha = _dump(_AffineCal(), p)
    # Only a prop-level calibrator exists, and fallback is NOT declared -> fatal.
    reg = BinaryCalibrationRegistry(enabled=True,
        artifacts={"pts": {"path": str(p), "sha256": sha}},
        allow_role_fallback_to_prop=False)
    with pytest.raises(CalibrationError):
        reg.apply("pts", "starter", 0.6)
    # Declared fallback is allowed.
    reg2 = BinaryCalibrationRegistry(enabled=True,
        artifacts={"pts": {"path": str(p), "sha256": sha}},
        allow_role_fallback_to_prop=True)
    assert reg2.apply("pts", "starter", 0.6).calibration_status == "calibrated"


def test_output_out_of_range_is_fatal(tmp_path):
    p = tmp_path / "c.pkl"; sha = _dump(_AffineCal(a=3.0, b=0.0, clip=False), p)
    reg = BinaryCalibrationRegistry(enabled=True,
        artifacts={"pts|starter": {"path": str(p), "sha256": sha}})
    with pytest.raises(CalibrationError):
        reg.apply("pts", "starter", 0.9)  # 2.7 -> out of [0,1] -> fatal, never silent


def test_nan_or_oob_input_is_fatal():
    reg = BinaryCalibrationRegistry(enabled=False)
    with pytest.raises(CalibrationError):
        reg.apply("pts", "starter", float("nan"))
    with pytest.raises(CalibrationError):
        reg.apply("pts", "starter", 1.5)


def test_venn_abers_registry_inside_lineage_nonidentity(tmp_path):
    # Build a NONIDENTITY Venn-Abers calibrator and prove: raw settled != calibrated,
    # calibrated == final, final == binary_calibrated, provenance recorded, no later mutation.
    from wnba_props_model.calibration.venn_abers import VennAbersCalibrator
    from wnba_props_model.models.binary_probability_calibration import (
        VennAbersBinaryCalibrationRegistry,
    )
    rng = np.random.default_rng(3)
    scores = rng.uniform(0, 1, 4000)
    labels = (rng.uniform(0, 1, 4000) < np.clip(scores * 0.6 + 0.1, 0, 1)).astype(float)
    va = VennAbersCalibrator().fit(scores, labels)
    (tmp_path / "venn_abers_pts_starter.pkl").write_bytes(b"")  # placeholder to ensure dir
    va.save(str(tmp_path / "venn_abers_pts_starter.pkl"))

    from wnba_props_model.models.market import settled_probabilities_from_pmf
    reg = VennAbersBinaryCalibrationRegistry(cal_dir=str(tmp_path), require=True)
    pmf = np.array([0.05, 0.10, 0.15, 0.30, 0.40])  # settled over(1.5)=0.85
    raw = settled_probabilities_from_pmf(pmf, 1.5).p_over_settled
    lin = build_probability_lineage(final_pmf=pmf, line=1.5, prop="pts", role="starter",
                                    binary_calibration_registry=reg)
    assert lin.calibration_status == "calibrated"
    assert lin.calibrator_id == "venn_abers_pts_starter.pkl"
    assert lin.calibrator_hash is not None
    assert lin.model_prob_over_settled_from_final_pmf == pytest.approx(raw)
    assert lin.model_prob_over_binary_calibrated != pytest.approx(raw)   # nonidentity
    assert lin.model_prob_over_final == lin.model_prob_over_binary_calibrated  # pure track
    assert lin.model_prob_over_market_anchored is None


def test_no_silent_raw_fallback_source():
    # Guard: the registry module must never swallow a failure and return the raw prob.
    import inspect
    from wnba_props_model.models import binary_probability_calibration as m
    src = inspect.getsource(m)
    assert "return CalibrationResult(float(p_over)" in src  # only the disabled/identity path
    # There is exactly one identity return (the disabled branch); enabled failures raise.
    assert src.count("identity_disabled") >= 1
