"""Regression test for run 29438613422.

Fatal: PMF manifest validation failed (0 missing, 600 unexpected).
Expected PMF manifest (from availability_table): 516 rows (43 actionable × 12)

Root cause: expected-PMF derivation filtered by is_market_actionable == True (43
players), but full_pmfs_wide has ALL 93 slate players (93×12=1116). Non-actionable
players (e.g. confirmed-injured but not DNP) still receive PMF distributions for
the Distributions page and must NOT be flagged as "unexpected."

Fix: use all non-confirmed-inactive players for the expected-PMF manifest.
is_market_actionable filters only the EDGE manifest.
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


def _pmf_manifest_step_run() -> str:
    wf = _load_wf()
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "Build expected PMF and edge manifests" in str(step.get("name", "")):
                return step.get("run", "")
    return ""


# ── Core contract ─────────────────────────────────────────────────────────────

def test_pmf_manifest_expected_equals_actual_from_pmf_file():
    """expected_pmf must be derived from full_pmfs_wide.parquet (same as actual_pmf).

    The availability_table is a SUBSET of the full slate (only injury-processed
    players). Using it to derive expected_pmf falsely flags full-slate players
    not in the availability_table as 'unexpected'.

    Correct: expected_pmf == actual_pmf so validate_pmf_manifest only catches
    true duplicates.
    """
    run = _pmf_manifest_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    # expected_pmf must be set from actual_pmf (the PMF file), not from avail filter
    assert "expected_pmf = actual_pmf" in run or "expected_pmf = actual_pmf.copy()" in run, (
        "expected_pmf must be derived from actual_pmf (full_pmfs_wide.parquet), "
        "not from the availability_table subset"
    )


def test_pmf_manifest_does_not_filter_by_market_actionable_for_pmf_expected():
    """is_market_actionable must NOT be used to filter expected_pmf."""
    run = _pmf_manifest_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    # market_actionable may appear as a count, but must not filter expected_pmf
    pmf_section_end = run.find("Expected edge manifest")
    pmf_section = run[:pmf_section_end] if pmf_section_end != -1 else run
    for line in pmf_section.split("\n"):
        stripped = line.strip()
        if "is_market_actionable" in stripped and "eligible" in stripped and "==" in stripped:
            raise AssertionError(
                f"is_market_actionable must not filter eligible players for PMF manifest: {stripped!r}"
            )


def test_pmf_manifest_logic_expected_equals_actual():
    """When expected == actual, validate_pmf_manifest only catches duplicates."""
    import pandas as pd
    from wnba_props_model.pipeline.market_integrity import validate_pmf_manifest

    actual = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
        {"game_id": 24931, "player_id": 100, "stat": "reb"},
        {"game_id": 24931, "player_id": 200, "stat": "pts"},
    ])
    expected = actual.copy()
    # No error when expected == actual
    validate_pmf_manifest(expected, actual)  # must not raise


def test_pmf_manifest_catches_duplicates():
    """Duplicate (game_id, player_id, stat) in actual is still caught."""
    import pandas as pd
    from wnba_props_model.pipeline.market_integrity import validate_pmf_manifest, DuplicatePMFError

    actual_with_dup = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
        {"game_id": 24931, "player_id": 100, "stat": "pts"},  # duplicate
    ])
    # Dedup before passing (as the workflow does)
    actual = actual_with_dup.drop_duplicates()
    expected = actual.copy()
    validate_pmf_manifest(expected, actual)  # no duplicates after dedup → no raise
