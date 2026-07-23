"""Foundation Lock / PR 1A tests for the single-use Phase-0 scope-protection guard."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_phase0_scope.py"
EXC = REPO / "config" / "phase0_scope_exception.json"

PR1A_BRANCH = "cursor/probability-delivery-correctness-v1"
PR1A_BASE = "d8c9681d6a6334ad8b0897b53009fc28d28dd342"
MARKET = "src/wnba_props_model/models/market.py"
DELIVER = "src/wnba_props_model/pipeline/deliver.py"


def _mod():
    spec = importlib.util.spec_from_file_location("cps", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CPS = _mod()
COMMITTED_EXC = json.loads(EXC.read_text())


def test_protected_files_detected():
    for p in (MARKET, DELIVER, "src/wnba_props_model/pipeline/calibrate.py",
              "scripts/build_scored_candidates.py", "config/recommendation_policy.yaml",
              "config/stat_registry.json"):
        assert CPS.is_protected(p)


def test_live_workflows_detected():
    for p in (".github/workflows/daily_pipeline.yml", ".github/workflows/live_inplay.yml",
              ".github/workflows/pregame_final.yml", ".github/workflows/deploy_forecast_shell.yml"):
        assert CPS.is_protected(p)


def test_non_protected_paths():
    for p in (".github/workflows/foundation_lock.yml", ".github/workflows/ci.yml",
              "scripts/run_prop_ablation.py", "config/foundation_lock_v1.json",
              "tests/test_phase0_scope_guard.py"):
        assert not CPS.is_protected(p)


def test_no_protected_change_always_passes_even_with_exception():
    ok, _ = CPS.evaluate(["scripts/run_prop_ablation.py"], COMMITTED_EXC,
                         current_branch="some/other-branch", base_sha="deadbeef",
                         today="2027-01-01")
    assert ok is True  # persisted exception cannot block unrelated PRs


def test_valid_exception_authorizes_pr1a():
    ok, msgs = CPS.evaluate([MARKET, DELIVER], COMMITTED_EXC,
                            current_branch=PR1A_BRANCH, base_sha=PR1A_BASE, today="2026-07-23")
    assert ok is True, msgs


def test_protected_change_without_exception_fails():
    ok, _ = CPS.evaluate([MARKET], None, current_branch=PR1A_BRANCH,
                         base_sha=PR1A_BASE, today="2026-07-23")
    assert ok is False


def test_exception_cannot_be_reused_by_pr1b_branch():
    # cursor/exact-quotes-settlement-v1 (PR 1B) must NOT reuse PR 1A's exception.
    ok, msgs = CPS.evaluate([MARKET], COMMITTED_EXC,
                            current_branch="cursor/exact-quotes-settlement-v1",
                            base_sha=PR1A_BASE, today="2026-07-23")
    assert ok is False
    assert any("cannot be reused" in m or "approved_branch" in m for m in msgs)


def test_exception_base_sha_mismatch_fails():
    ok, _ = CPS.evaluate([MARKET], COMMITTED_EXC, current_branch=PR1A_BRANCH,
                         base_sha="0000000000000000000000000000000000000000", today="2026-07-23")
    assert ok is False


def test_exception_expired_fails():
    ok, msgs = CPS.evaluate([MARKET], COMMITTED_EXC, current_branch=PR1A_BRANCH,
                            base_sha=PR1A_BASE, today="2099-01-01")
    assert ok is False
    assert any("expired" in m for m in msgs)


def test_wildcard_path_rejected():
    exc = dict(COMMITTED_EXC); exc["approved_paths"] = ["src/**"]
    ok, msgs = CPS.evaluate([MARKET], exc, current_branch=PR1A_BRANCH,
                            base_sha=PR1A_BASE, today="2026-07-23")
    assert ok is False
    assert any("wildcard" in m for m in msgs)


def test_unlisted_protected_path_fails():
    ok, msgs = CPS.evaluate([MARKET, "config/stat_registry.json"], COMMITTED_EXC,
                            current_branch=PR1A_BRANCH, base_sha=PR1A_BASE, today="2026-07-23")
    assert ok is False
    assert any("not in the exception allowlist" in m for m in msgs)


def test_cli_passes_on_this_branch():
    import pytest
    base = subprocess.run(["git", "rev-parse", "--verify", "origin/main"],
                          capture_output=True, text=True, cwd=str(REPO))
    if base.returncode != 0:
        pytest.skip("origin/main not available in this checkout")
    r = subprocess.run([sys.executable, str(SCRIPT), "--base", "origin/main",
                        "--branch", PR1A_BRANCH],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SCOPE PASS" in r.stdout
