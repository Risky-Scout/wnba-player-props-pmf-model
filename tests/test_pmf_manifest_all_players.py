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

def test_pmf_manifest_uses_confirmed_inactive_not_market_actionable():
    """PMF manifest must filter by is_confirmed_inactive, not is_market_actionable.

    PMFs are generated for ALL non-confirmed-inactive players (including those
    not market-actionable). Using is_market_actionable would exclude ~50 players
    and falsely flag them as unexpected when comparing to full_pmfs_wide.parquet.
    """
    run = _pmf_manifest_step_run()
    assert run, "Build expected PMF and edge manifests step not found"

    # Must use confirmed_inactive to derive expected PMF manifest
    assert "is_confirmed_inactive" in run, (
        "PMF manifest derivation must filter by is_confirmed_inactive, "
        "not is_market_actionable"
    )


def test_pmf_manifest_does_not_filter_by_market_actionable_for_pmfs():
    """is_market_actionable must NOT be used to filter the expected PMF manifest."""
    run = _pmf_manifest_step_run()
    assert run, "Build expected PMF and edge manifests step not found"

    # Find the PMF manifest section (before the edge manifest section)
    pmf_section_end = run.find("Expected edge manifest")
    pmf_section = run[:pmf_section_end] if pmf_section_end != -1 else run

    # is_market_actionable should NOT appear as a filter for expected PMF derivation
    # (it may appear for edge manifest or as a count, but not as the eligibility filter)
    lines = pmf_section.split("\n")
    for line in lines:
        stripped = line.strip()
        if "is_market_actionable" in stripped and "==" in stripped:
            # It's ok if it's used for counting, not for the eligible filter
            assert "eligible" not in stripped, (
                f"Line uses is_market_actionable to filter 'eligible' for PMFs: {stripped!r}\n"
                "PMF manifest must include all non-confirmed-inactive players."
            )


def test_pmf_manifest_logic_produces_correct_counts():
    """Verify expected-PMF logic: 93 total - 4 inactive = 89 eligible, 89×12=1068 expected."""
    import pandas as pd

    SUPPORTED_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                       "pts_reb", "pts_ast", "reb_ast", "pts_reb_ast", "stocks"]

    # Simulate availability_table with 93 players: 43 market-actionable, 89 non-inactive
    avail_rows = []
    for i in range(93):
        avail_rows.append({
            "player_id": i,
            "game_id": 24931 if i < 50 else 24932,
            "is_confirmed_inactive": i < 4,       # 4 confirmed inactive
            "is_market_actionable": i >= 50,       # 43 market-actionable
        })
    avail = pd.DataFrame(avail_rows)

    # Apply the CORRECTED filter (non-confirmed-inactive)
    eligible = avail[avail["is_confirmed_inactive"] != True]
    assert len(eligible) == 89, f"Expected 89 non-inactive, got {len(eligible)}"

    rows = []
    key_cols = ["game_id", "player_id"]
    for _, row in eligible[key_cols].drop_duplicates().iterrows():
        for stat in SUPPORTED_STATS:
            rows.append({**row.to_dict(), "stat": stat})
    expected_pmf = pd.DataFrame(rows)
    assert len(expected_pmf) == 89 * 12, f"Expected {89*12}, got {len(expected_pmf)}"

    # OLD (wrong) filter using is_market_actionable would give 43 players → 516 rows
    old_eligible = avail[avail["is_market_actionable"] == True]
    assert len(old_eligible) == 43
    old_expected = len(old_eligible) * 12
    assert old_expected == 516, "Old (wrong) filter gives 516 rows — matches the failure"
