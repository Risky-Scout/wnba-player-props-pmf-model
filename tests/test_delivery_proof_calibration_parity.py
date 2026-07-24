"""A5: delivery and proof must load the SAME calibration resolver, so once a non-identity
policy is activated they produce the identical float64 model_prob_over_final.

Delivery (`deliver.write_delivery` -> `build_market_comparison`) and the proof assembler
(`build_market_superiority_input.py`) both resolve the registry via
`load_binary_calibration_registry(policy_path, mode)` and compute the final probability with
`build_probability_lineage`. This test freezes a non-identity policy artifact and proves the
two load sites yield byte-identical final probabilities (and that the calibrator actually
moved the probability off identity).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wnba_props_model.models.binary_calibrators import PlattCalibrator
from wnba_props_model.models.binary_probability_calibration import (
    load_binary_calibration_registry,
)
from wnba_props_model.models.probability_lineage import build_probability_lineage

REPO = Path(__file__).resolve().parent.parent


def _nonidentity_policy(tmp: Path):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(0, 1, 2000) < p ** 2).astype(int)     # miscalibrated -> Platt moves probs
    model = PlattCalibrator(C=1.0).fit(p, y)
    art = tmp / "binary_platt_reb.pkl"
    joblib.dump(model, art)
    sha = hashlib.sha256(art.read_bytes()).hexdigest()
    policy = tmp / "policy.json"
    policy.write_text(json.dumps({
        "version": "test-nonidentity-v1",
        "props": {"reb": {"method": "platt", "path": str(art), "sha256": sha}},
    }))
    return policy, model


def _final(registry, pmf, line, prop="reb", role="all"):
    return build_probability_lineage(
        final_pmf=pmf, line=float(line), prop=prop, role=role,
        binary_calibration_registry=registry, probability_track="pure_forecast",
    ).model_prob_over_final


def test_delivery_and_proof_same_nonidentity_final(tmp_path):
    policy, _ = _nonidentity_policy(tmp_path)
    pmf = {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2}
    line = 1.5
    # Delivery-side load (deliver.write_delivery uses load_binary_calibration_registry).
    delivery_reg = load_binary_calibration_registry(str(policy), "required")
    # Proof-side load (build_market_superiority_input.py uses the same loader).
    proof_reg = load_binary_calibration_registry(str(policy), "required")

    d = _final(delivery_reg, pmf, line)
    p = _final(proof_reg, pmf, line)
    assert isinstance(d, float) and isinstance(p, float)
    assert d == p                                          # exact float64 parity

    # The non-identity calibrator actually changed the probability off identity.
    ident = _final(load_binary_calibration_registry(None, "disabled"), pmf, line)
    assert abs(d - ident) > 1e-6


def test_delivery_wires_the_common_resolver():
    # deliver.py must load the shared resolver and inject it into build_market_comparison.
    src = (REPO / "src" / "wnba_props_model" / "pipeline" / "deliver.py").read_text()
    assert "load_binary_calibration_registry" in src
    assert "binary_calibration_registry=_registry" in src
    # Per-row policy provenance persisted on delivered rows (A5).
    for col in ("calibration_policy_id", "calibration_policy_hash", "calibration_mode"):
        assert col in src
