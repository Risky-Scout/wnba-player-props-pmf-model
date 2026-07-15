"""Regression tests for PMF manifest contract — updated for duplicate-only validation.

The PMF manifest no longer checks coverage (expected vs actual player universe)
because predict_today.py may score more players than the BDL tonight-slate.
Validation only catches duplicate (game_id, player_id, stat) identities.
"""
from __future__ import annotations

import yaml
from pathlib import Path

WF_PATH = Path(".github/workflows/pregame_initial.yml")


def _load_wf() -> dict:
    raw = yaml.safe_load(WF_PATH.read_text())
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return raw


def _pmf_step_run() -> str:
    wf = _load_wf()
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "Build expected PMF and edge manifests" in str(step.get("name", "")):
                return step.get("run", "")
    return ""


def test_pmf_manifest_validates_duplicates():
    """PMF manifest must check for duplicate (game_id, player_id, stat) identities."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    assert "duplicate" in run.lower() or "duplicated" in run, (
        "PMF manifest step must validate duplicate identities"
    )


def test_pmf_manifest_expected_equals_actual():
    """expected_pmf == actual_pmf so validate_pmf_manifest only checks duplicates."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    assert "expected_pmf = actual_pmf" in run, (
        "expected_pmf must equal actual_pmf — coverage check removed "
        "because predictor universe > tonight-slate"
    )


def test_pmf_manifest_fatal_when_empty_with_games():
    """Empty PMF file when games are scheduled must remain fatal."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    assert "actual_pmf.empty" in run or "empty" in run.lower(), (
        "Must detect empty PMF file when games are scheduled"
    )


def test_market_actionable_does_not_filter_pmf_manifest():
    """is_market_actionable must NOT filter the PMF manifest."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    # Find PMF section (before edge section)
    edge_idx = run.find("edge manifest")
    pmf_section = run[:edge_idx] if edge_idx != -1 else run
    # is_market_actionable should not be used as a filter for eligible players
    for line in pmf_section.split("\n"):
        stripped = line.strip()
        if "is_market_actionable" in stripped and "eligible" in stripped and "==" in stripped:
            raise AssertionError(
                f"is_market_actionable must not filter PMF eligibility: {stripped!r}"
            )
