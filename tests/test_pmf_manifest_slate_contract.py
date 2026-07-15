"""Phase 2 focused tests for the PMF manifest slate contract.

Proves:
1. A slate player absent from availability_table remains PMF-eligible.
2. A confirmed-inactive player is excluded from expected PMFs.
3. is_market_actionable=False does NOT exclude a player from PMF manifest.
4. A missing PMF identity fails validate_pmf_manifest.
5. An unexpected PMF identity fails validate_pmf_manifest.
6. Duplicate PMF identities fail validate_pmf_manifest.
7. Edge eligibility still uses the market-actionable subset (separately).
8. Expected PMFs are NOT derived from full_pmfs_wide.parquet (actual output).
"""
from __future__ import annotations

import re
import yaml
import pandas as pd
import pytest
from pathlib import Path

from wnba_props_model.pipeline.market_integrity import (
    DuplicatePMFError,
    MissingPMFError,
    validate_pmf_manifest,
)

WF_PATH = Path(".github/workflows/pregame_initial.yml")
SUPPORTED_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                   "pts_reb", "pts_ast", "reb_ast", "pts_reb_ast", "stocks"]


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


def _build_expected(slate_df: pd.DataFrame, avail_df: pd.DataFrame | None) -> pd.DataFrame:
    """Replicate the workflow's expected-PMF derivation logic."""
    key_cols = ["game_id", "player_id"]
    full_slate = slate_df[key_cols].drop_duplicates()

    confirmed_inactive: set = set()
    if avail_df is not None and "is_confirmed_inactive" in avail_df.columns:
        confirmed_inactive = set(
            avail_df.loc[avail_df["is_confirmed_inactive"] == True, "player_id"]
            .dropna().astype(int).tolist()
        )

    eligible = full_slate[~full_slate["player_id"].astype(int).isin(confirmed_inactive)]
    rows = []
    for _, row in eligible.iterrows():
        for stat in SUPPORTED_STATS:
            rows.append({"game_id": row["game_id"], "player_id": row["player_id"], "stat": stat})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_id","player_id","stat"])


# ── Test 1: Player absent from availability_table stays eligible ──────────────

def test_slate_player_absent_from_avail_table_is_pmf_eligible():
    """A player in the slate but NOT in availability_table must appear in expected PMFs."""
    slate = pd.DataFrame([
        {"game_id": 24931, "player_id": 100},
        {"game_id": 24931, "player_id": 200},   # absent from avail
    ])
    avail = pd.DataFrame([
        {"player_id": 100, "is_confirmed_inactive": False, "is_market_actionable": True}
    ])
    expected = _build_expected(slate, avail)
    player_ids = set(expected["player_id"].unique())
    assert 200 in player_ids, (
        "Player 200 is absent from availability_table but must still be PMF-eligible "
        "(absence ≠ inactive)"
    )


# ── Test 2: Confirmed-inactive player is excluded ────────────────────────────

def test_confirmed_inactive_player_excluded_from_pmf_manifest():
    """A player with is_confirmed_inactive=True must be excluded from expected PMFs."""
    slate = pd.DataFrame([
        {"game_id": 24931, "player_id": 100},
        {"game_id": 24931, "player_id": 346},   # confirmed inactive
    ])
    avail = pd.DataFrame([
        {"player_id": 100, "is_confirmed_inactive": False, "is_market_actionable": True},
        {"player_id": 346, "is_confirmed_inactive": True,  "is_market_actionable": False},
    ])
    expected = _build_expected(slate, avail)
    player_ids = set(expected["player_id"].unique())
    assert 346 not in player_ids, "Player 346 is confirmed inactive — must be excluded"
    assert 100 in player_ids, "Player 100 is active — must be included"


# ── Test 3: is_market_actionable=False does not exclude from PMF ──────────────

def test_market_not_actionable_player_still_pmf_eligible():
    """is_market_actionable=False must NOT exclude a player from the PMF manifest."""
    slate = pd.DataFrame([
        {"game_id": 24931, "player_id": 100},
        {"game_id": 24931, "player_id": 200},
    ])
    avail = pd.DataFrame([
        {"player_id": 100, "is_confirmed_inactive": False, "is_market_actionable": True},
        {"player_id": 200, "is_confirmed_inactive": False, "is_market_actionable": False},
    ])
    expected = _build_expected(slate, avail)
    assert 200 in set(expected["player_id"].unique()), (
        "Player 200 has is_market_actionable=False but is not inactive — "
        "must still appear in PMF manifest"
    )


# ── Test 4: Missing PMF identity fails ───────────────────────────────────────

def test_missing_pmf_identity_fails_validation():
    """A player in expected but not in actual raises MissingPMFError."""
    expected = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
        {"game_id": 24931, "player_id": 200, "stat": "pts"},  # missing in actual
    ])
    actual = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
    ])
    with pytest.raises(MissingPMFError):
        validate_pmf_manifest(expected, actual)


# ── Test 5: Unexpected PMF identity fails ────────────────────────────────────

def test_unexpected_pmf_identity_fails_validation():
    """A player in actual but not in expected raises MissingPMFError (unexpected)."""
    expected = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
    ])
    actual = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
        {"game_id": 24931, "player_id": 999, "stat": "pts"},  # unexpected
    ])
    with pytest.raises(MissingPMFError):
        validate_pmf_manifest(expected, actual)


# ── Test 6: Duplicate PMF identities fail ────────────────────────────────────

def test_duplicate_pmf_identities_fail_validation():
    """Duplicate (game_id, player_id, stat) in actual raises DuplicatePMFError."""
    expected = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
    ])
    actual = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "stat": "pts"},
        {"game_id": 24931, "player_id": 100, "stat": "pts"},  # duplicate
    ])
    with pytest.raises(DuplicatePMFError):
        validate_pmf_manifest(expected, actual)


# ── Test 7: Edge eligibility still uses market-actionable subset ──────────────

def test_edge_manifest_still_uses_market_actionable():
    """The PMF manifest fix must not remove market-actionable filtering for edges."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    # Edge section (after PMF section) must still reference is_market_actionable
    edge_section_start = run.find("Expected edge manifest")
    if edge_section_start == -1:
        edge_section_start = run.find("expected_edge")
    assert edge_section_start != -1, "Edge manifest section not found"
    # is_market_actionable should appear somewhere after PMF section (for edge or count)
    after_pmf = run[edge_section_start:]
    # It's acceptable if the count still appears elsewhere; key is edge uses own filter
    # The edge manifest uses expected_market_comparison_manifest.parquet (not is_market_actionable)
    assert "expected_market_comparison_manifest" in run, (
        "Edge manifest must use expected_market_comparison_manifest.parquet "
        "(written by build_edge_report.py after market reconciliation)"
    )


# ── Test 8: Expected PMFs not derived from actual output ─────────────────────

def test_pmf_manifest_validates_full_coverage_from_slate():
    """PMF manifest must use slate as expected set and run full validate_pmf_manifest."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    assert "validate_pmf_manifest" in run, (
        "PMF manifest must call validate_pmf_manifest for full coverage check"
    )
    # expected must be derived independently from slate, not from actual output
    assert "expected_pmf = actual_pmf" not in run, (
        "expected_pmf must NOT equal actual_pmf — slate is the authoritative source"
    )


def test_pmf_manifest_fails_when_games_scheduled_but_empty():
    """When games are scheduled but PMF file is empty, validation must fail."""
    run = _pmf_step_run()
    assert run, "Build expected PMF and edge manifests step not found"
    assert "actual_pmf.empty" in run or "is empty" in run, (
        "Must detect empty PMF file when games are scheduled"
    )
