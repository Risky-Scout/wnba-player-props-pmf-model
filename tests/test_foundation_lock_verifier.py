"""Foundation Lock tests for the manifest verifier itself.

Asserts the committed manifest verifies clean, and that verification FAILS on each
prohibited condition: missing path, hash mismatch, missing required test, schema/manifest
mismatch, and a component labeled promotion-eligible while its limitations prohibit it.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "verify_foundation_lock.py"
MANIFEST = REPO / "config" / "foundation_lock_v1.json"


def _mod():
    spec = importlib.util.spec_from_file_location("vfl", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


VFL = _mod()
BASE = json.loads(MANIFEST.read_text())


def test_committed_manifest_verifies_clean():
    failures, _ = VFL.verify(BASE)
    assert failures == [], failures


def test_all_items_are_not_promotion_eligible():
    # Foundation components must never be promotion-eligible.
    for item in BASE["items"]:
        assert item["promotion_eligible"] is False, item["id"]


def test_fail_on_missing_path():
    m = copy.deepcopy(BASE)
    m["items"][0]["paths"].append(
        {"path": "scripts/__does_not_exist__.py", "availability": "in_repo", "sha256": "0" * 64})
    failures, _ = VFL.verify(m)
    assert any("missing in-repo path" in f for f in failures)


def test_fail_on_hash_mismatch():
    m = copy.deepcopy(BASE)
    m["items"][0]["paths"][0]["sha256"] = "d" * 64  # wrong hash for a real file
    failures, _ = VFL.verify(m)
    assert any("hash mismatch" in f for f in failures)


def test_fail_on_missing_required_test():
    m = copy.deepcopy(BASE)
    m["items"][0]["required_tests"] = ["tests/__no_such_test__.py"]
    failures, _ = VFL.verify(m)
    assert any("missing required test" in f for f in failures)


def test_fail_on_schema_mismatch_bad_version():
    m = copy.deepcopy(BASE)
    m["schema_version"] = 2
    failures, _ = VFL.verify(m)
    assert any("schema_version" in f for f in failures)


def test_fail_on_missing_item_key():
    m = copy.deepcopy(BASE)
    del m["items"][0]["invariants"]
    failures, _ = VFL.verify(m)
    assert any("missing keys" in f for f in failures)


def test_fail_on_bad_status():
    m = copy.deepcopy(BASE)
    m["items"][0]["status"] = "totally_locked"
    failures, _ = VFL.verify(m)
    assert any("bad status" in f for f in failures)


def test_fail_on_promotion_eligible_when_prohibited():
    m = copy.deepcopy(BASE)
    # Find a prohibits_promotion item and flip promotion_eligible true.
    target = next(it for it in m["items"] if it["prohibits_promotion"])
    target["promotion_eligible"] = True
    failures, _ = VFL.verify(m)
    assert any("promotion_eligible while limitations prohibit" in f for f in failures)


def test_data_artifact_absence_is_deferred_not_failure(tmp_path):
    m = copy.deepcopy(BASE)
    # Point a data artifact at a guaranteed-absent path; must DEFER, not FAIL.
    for item in m["items"]:
        for entry in item["paths"]:
            if entry.get("availability") == "data_artifact_untracked":
                entry["path"] = "definitely/absent/artifact.parquet"
    failures, deferrals = VFL.verify(m)
    assert failures == [], failures
    assert any("DEFERRED" in d for d in deferrals)
