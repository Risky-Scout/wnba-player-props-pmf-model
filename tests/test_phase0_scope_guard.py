"""Foundation Lock tests for the Phase-0 scope-protection guard."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_phase0_scope.py"


def _mod():
    spec = importlib.util.spec_from_file_location("cps", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CPS = _mod()


def test_protected_files_detected():
    assert CPS.is_protected("src/wnba_props_model/models/market.py")
    assert CPS.is_protected("src/wnba_props_model/pipeline/deliver.py")
    assert CPS.is_protected("src/wnba_props_model/pipeline/calibrate.py")
    assert CPS.is_protected("scripts/build_scored_candidates.py")
    assert CPS.is_protected("config/recommendation_policy.yaml")
    assert CPS.is_protected("config/stat_registry.json")


def test_live_workflows_detected():
    assert CPS.is_protected(".github/workflows/daily_pipeline.yml")
    assert CPS.is_protected(".github/workflows/live_inplay.yml")
    assert CPS.is_protected(".github/workflows/pregame_final.yml")
    assert CPS.is_protected(".github/workflows/deploy_forecast_shell.yml")


def test_non_protected_paths():
    assert not CPS.is_protected(".github/workflows/foundation_lock.yml")
    assert not CPS.is_protected(".github/workflows/ci.yml")
    assert not CPS.is_protected("scripts/run_prop_ablation.py")
    assert not CPS.is_protected("config/foundation_lock_v1.json")
    assert not CPS.is_protected("tests/test_phase0_scope_guard.py")


def test_violations_respects_approved_exceptions():
    changed = ["src/wnba_props_model/models/market.py", "scripts/run_prop_ablation.py"]
    assert CPS.violations(changed, approved_paths=[]) == [
        "src/wnba_props_model/models/market.py"]
    assert CPS.violations(changed, approved_paths=[
        "src/wnba_props_model/models/market.py"]) == []


def test_cli_passes_on_this_branch():
    # The Foundation Lock branch must not touch protected paths.
    r = subprocess.run([sys.executable, str(SCRIPT), "--base", "origin/main"],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SCOPE PASS" in r.stdout
