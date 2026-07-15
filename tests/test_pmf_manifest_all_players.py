"""PMF manifest regression tests — updated for slate-based independent contract."""
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
    run = _pmf_step_run()
    assert run, "Build expected PMF step not found"
    assert "validate_pmf_manifest" in run or "duplicate" in run.lower()


def test_pmf_manifest_uses_slate_not_actual_pmf():
    """expected_pmf must come from the slate, NOT from full_pmfs_wide.parquet."""
    run = _pmf_step_run()
    assert run, "Build expected PMF step not found"
    assert "expected_pmf = actual_pmf" not in run, (
        "expected_pmf must not be derived from actual_pmf — "
        "the slate is the authoritative source"
    )


def test_pmf_manifest_fatal_when_empty_with_games():
    run = _pmf_step_run()
    assert run, "Build expected PMF step not found"
    assert "actual_pmf.empty" in run or "empty" in run.lower()


def test_market_actionable_does_not_filter_pmf_manifest():
    run = _pmf_step_run()
    assert run, "Build expected PMF step not found"
    edge_idx = run.find("edge manifest")
    pmf_section = run[:edge_idx] if edge_idx != -1 else run
    for line in pmf_section.split("\n"):
        stripped = line.strip()
        if "is_market_actionable" in stripped and "eligible" in stripped and "==" in stripped:
            raise AssertionError(
                f"is_market_actionable must not filter PMF eligibility: {stripped!r}"
            )
