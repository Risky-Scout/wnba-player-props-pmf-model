"""Foundation Lock tests for the feature-ablation-maps regeneration check.

Guards: committed map matches deterministic regeneration; drift is detected; unknown/
forbidden features are rejected; G0 must equal the full feature contract.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "scripts" / "build_feature_ablation_maps.py"
PLAN = REPO / "config" / "feature_ablation_plan_v1.json"
CAND = REPO / "config" / "prop_feature_map_candidate_v1.json"
COMMITTED = REPO / "config" / "feature_ablation_maps_v1.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("bfam", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_check(*extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILDER), "check", *extra],
        capture_output=True, text=True, cwd=str(REPO))


def test_check_passes_on_committed_map():
    r = _run_check()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CHECK PASS" in r.stdout


def test_resolution_is_deterministic():
    mod = _load_builder()
    a = mod.build_resolved(str(PLAN), str(CAND))
    b = mod.build_resolved(str(PLAN), str(CAND))
    assert mod._canonical(a) == mod._canonical(b)
    # G0 equals full feature contract for every direct prop.
    from wnba_props_model.features.feature_contract import MODEL_FEATURES
    for prop in mod.DIRECT_PROPS:
        assert a["candidates"]["G0"][prop] == list(MODEL_FEATURES)


def test_check_detects_drift(tmp_path):
    tampered = tmp_path / "maps.json"
    obj = json.loads(COMMITTED.read_text())
    # Drop one feature from a candidate -> must be detected as drift.
    some_prop = next(iter(obj["candidates"]["G0"]))
    obj["candidates"]["G0"][some_prop] = obj["candidates"]["G0"][some_prop][:-1]
    tampered.write_text(json.dumps(obj, indent=2))
    r = _run_check("--out", str(tampered))
    assert r.returncode == 1
    assert "CHECK FAIL" in (r.stdout + r.stderr)


def test_unknown_feature_rejected(tmp_path):
    mod = _load_builder()
    bad_cand = tmp_path / "cand.json"
    prop_map = json.loads(CAND.read_text())
    # Inject a non-contract feature into one prop's candidate list.
    first = next(iter(prop_map))
    prop_map[first] = list(prop_map[first]) + ["__definitely_not_a_real_feature__"]
    bad_cand.write_text(json.dumps(prop_map))
    try:
        mod.build_resolved(str(PLAN), str(bad_cand))
        raised = False
    except ValueError:
        raised = True
    assert raised, "unknown feature must raise ValueError"


def test_g0_truncation_detected(tmp_path):
    tampered = tmp_path / "maps.json"
    obj = json.loads(COMMITTED.read_text())
    for prop in list(obj["candidates"]["G0"]):
        obj["candidates"]["G0"][prop] = obj["candidates"]["G0"][prop][:3]
    tampered.write_text(json.dumps(obj, indent=2))
    r = _run_check("--out", str(tampered))
    assert r.returncode == 1
    assert "G0" in (r.stdout + r.stderr)
