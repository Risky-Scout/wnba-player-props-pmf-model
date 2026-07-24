"""Contract tests for the data-durability registry and its scripts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "data_registry.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_schema_valid():
    reg = json.loads(REGISTRY.read_text())
    assert reg.get("repo"), "registry must name the repo"
    assert reg.get("datasets"), "registry must have datasets"
    for name, d in reg["datasets"].items():
        for k in ("path", "release_tag", "asset"):
            assert d.get(k), f"{name} missing required field {k}"
        assert d["path"].endswith(".parquet"), f"{name} path should be a parquet"
        if d.get("sha256") is not None:
            assert len(d["sha256"]) == 64, f"{name} sha256 must be a full hex digest"


def test_seeded_hashes_present_for_tracking():
    reg = json.loads(REGISTRY.read_text())
    for name in ("wnba_tracking", "wnba_hustle"):
        assert reg["datasets"][name]["sha256"], f"{name} should ship with a real sha256"


def test_scripts_import_cleanly():
    for name in ("data_registry_lib", "fetch_data", "publish_data", "verify_data_registry"):
        _load(name)


def test_verify_runs_without_crash():
    mod = _load("verify_data_registry")
    rc = mod.main()
    assert rc in (0, 1)
