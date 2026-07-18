"""P3 Task 3/7 — turnover promotion: registry, immutable calibration package, and
evaluated-vs-deployed parity for the certified forecast calibration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wnba_props_model.evaluation.pmf_recalibration import (
    apply_certified_forecast_calibration, recalibrate_pmf,
)

REPO = Path(__file__).resolve().parent.parent
REGISTRY = json.loads((REPO / "config/stat_registry.json").read_text())
MANIFEST = json.loads((REPO / "config/champion_manifest.json").read_text())
CALIB = json.loads((REPO / "config/certified_forecast_calibration.json").read_text())


def test_registry_turnover_certified_betting_off():
    assert REGISTRY["turnover"]["forecast_allowed"] is True
    assert REGISTRY["turnover"]["betting_recommendation_allowed"] is False
    # the six non-passing stats are not forecast_allowed
    for s in ("pts", "reb", "ast", "fg3m", "blk", "stl"):
        assert REGISTRY[s]["forecast_allowed"] is False


def test_manifest_hash_matches_calibration_artifact():
    h = hashlib.sha256(json.dumps(CALIB, sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert MANIFEST["calibration_hash"] == h
    assert REGISTRY["turnover"]["calibration_hash"] == h
    assert MANIFEST["status"] == "LIVE_VALIDATED_FORECAST_ONLY"
    assert MANIFEST["feature_schema"] == "schema_v2"
    # manifest carries the full lineage set incl. prequential ledger hash (no stale commit)
    for k in ("github_sha", "policy_hash", "ledger_hash", "registry_hash", "feature_hash", "split_dates"):
        assert MANIFEST.get(k)
    assert MANIFEST["ledger_hash"] == "ef22ea7799fd9a0a"


def test_prequential_calibration_is_location_only():
    # the prequential winner was LOCATION (bias-only) -> scale factor must be 1.0
    assert CALIB["method"] == "location_recalibration"
    assert CALIB["stats"]["turnover"]["_pooled"][1] == 1.0
    for _role, (_d, _s) in CALIB["stats"]["turnover"]["by_role"].items():
        assert _s == 1.0


def test_release_manifest_verification_passes():
    import subprocess, sys
    r = subprocess.run([sys.executable, str(REPO / "scripts/verify_release_manifest.py")],
                       capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, r.stdout + r.stderr


def test_evaluated_vs_deployed_parity():
    # the production applier must equal the SAME recalibrate_pmf transform used in
    # validation, with the persisted per-role factors -> byte-identical PMF.
    pmf = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
    entry = CALIB["stats"]["turnover"]
    role = "starter"
    delta, scale = entry["by_role"][role]
    expected = recalibrate_pmf(pmf, float(delta), float(scale))
    got = apply_certified_forecast_calibration(pmf, "turnover", role, CALIB)
    assert np.allclose(expected, got)


def test_non_certified_stat_unchanged():
    pmf = np.array([0.5, 0.3, 0.2])
    got = apply_certified_forecast_calibration(pmf, "pts", "starter", CALIB)
    assert np.allclose(got, pmf)   # pts not certified -> untouched
