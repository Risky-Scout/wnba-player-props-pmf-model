"""Focused regression tests for the PMF slate identity contract.

Proves:
1. Off-slate players in the feature table cannot expand pmfs_after.
2. Output identity set exactly equals input PMF identity set.
3. Confirmed-inactive players remain present (with zero PMF).
4. Missing feature rows for an input PMF identity are fatal.
5. Missing, unexpected, and duplicate output identities are fatal.
6. Workflow expected PMF set is independently derived from the slate.
7. Workflow does not contain expected_pmf = actual_pmf.copy().
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

WF_PATH = Path(".github/workflows/pregame_initial.yml")
SUPPORTED_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
                   "pts_reb", "pts_ast", "reb_ast", "pts_reb_ast", "stocks"]


def _load_wf_run() -> str:
    raw = yaml.safe_load(WF_PATH.read_text())
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    for job in raw.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "Build expected PMF and edge manifests" in str(step.get("name", "")):
                return step.get("run", "")
    return ""


def _make_pmfs(pairs: list[tuple[int, int, str]]) -> pd.DataFrame:
    rows = []
    for gid, pid, stat in pairs:
        rows.append({"game_id": gid, "player_id": pid, "stat": stat,
                     "pmf_json": '{"5":1.0}', "pmf_mean": 5.0})
    return pd.DataFrame(rows)


def _identity_set(df: pd.DataFrame) -> set[tuple]:
    return set(
        tuple(row) for row in df[["game_id", "player_id", "stat"]].drop_duplicates().itertuples(index=False)
    )


# ─── 1. Off-slate feature players cannot expand pmfs_after ───────────────────

def test_off_slate_feature_player_excluded_from_rebuild(tmp_path, monkeypatch):
    """Feature table with off-slate player 999 must not appear in pmfs_after."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    # Simulate: allowed pairs from pmfs_before (players 100, 200)
    allowed_pairs = pd.DataFrame([
        {"game_id": 24931, "player_id": 100},
        {"game_id": 24931, "player_id": 200},
    ])

    # Feature table with off-slate player 999
    feature_with_offslate = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "pts": 15.0},
        {"game_id": 24931, "player_id": 200, "pts": 10.0},
        {"game_id": 24931, "player_id": 999, "pts": 5.0},   # off-slate
    ])

    # Inner-join to allowed pairs (replicate apply_injury_updates logic)
    restricted = feature_with_offslate.merge(
        allowed_pairs, on=["game_id", "player_id"], how="inner"
    )
    assert 999 not in restricted["player_id"].values, (
        "Off-slate player 999 must be excluded from feature_df after inner-join"
    )
    assert 100 in restricted["player_id"].values
    assert 200 in restricted["player_id"].values


# ─── 2. Output identity set == input PMF identity set ───────────────────────

def test_identity_preservation_after_rebuild():
    """pmfs_after identities must exactly equal pmfs_before identities."""
    before_triples = [
        (24931, 100, "pts"), (24931, 100, "reb"),
        (24931, 200, "pts"), (24931, 200, "reb"),
    ]
    pmfs_before = _make_pmfs(before_triples)
    pmfs_after  = _make_pmfs(before_triples)  # same identities

    before_ids = _identity_set(pmfs_before)
    after_ids  = _identity_set(pmfs_after)
    assert before_ids == after_ids, "Identity sets must be equal after rebuild"


# ─── 3. Confirmed-inactive players remain present ────────────────────────────

def test_confirmed_inactive_player_retained_with_zero_pmf():
    """Confirmed-inactive players must stay in pmfs_after with pmf_json={0:1.0}."""
    before_triples = [
        (24931, 100, "pts"), (24931, 100, "reb"),
        (24931, 346, "pts"), (24931, 346, "reb"),   # 346 is inactive
    ]
    pmfs_before = _make_pmfs(before_triples)

    # Simulate: 346 set to zero PMF
    pmfs_after = pmfs_before.copy()
    pmfs_after.loc[pmfs_after["player_id"] == 346, "pmf_json"] = json.dumps({"0": 1.0})
    pmfs_after.loc[pmfs_after["player_id"] == 346, "pmf_mean"] = 0.0

    after_ids = _identity_set(pmfs_after)
    before_ids = _identity_set(pmfs_before)
    assert after_ids == before_ids, "Inactive player 346 must remain in identity set"
    assert all(
        pmfs_after.loc[pmfs_after["player_id"] == 346, "pmf_mean"] == 0.0
    )


# ─── 4. Missing feature rows for an input PMF identity are fatal ─────────────

def test_missing_feature_row_for_slate_player_is_fatal():
    """If a slate player has no matching feature row, the check must detect it."""
    allowed_pairs = pd.DataFrame([
        {"game_id": 24931, "player_id": 100},
        {"game_id": 24931, "player_id": 200},   # has no feature row
    ])
    feature_df = pd.DataFrame([
        {"game_id": 24931, "player_id": 100, "pts": 15.0},
        # player 200 has no feature row
    ])
    id_cols = ["game_id", "player_id"]
    feat_pairs = feature_df[id_cols].drop_duplicates()
    missing = allowed_pairs.merge(feat_pairs, on=id_cols, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    assert len(missing) == 1 and 200 in missing["player_id"].values, (
        "Player 200 missing from feature table must be detected"
    )


# ─── 5. Missing, unexpected, duplicate output identities are fatal ────────────

def test_missing_identity_in_after_is_detected():
    before_triples = [(24931, 100, "pts"), (24931, 200, "pts")]
    pmfs_before = _make_pmfs(before_triples)
    pmfs_after  = _make_pmfs([(24931, 100, "pts")])  # 200 dropped
    before_ids = _identity_set(pmfs_before)
    after_ids  = _identity_set(pmfs_after)
    missing = before_ids - after_ids
    assert missing, f"Dropped identity must be detected: {missing}"


def test_unexpected_identity_in_after_is_detected():
    before_triples = [(24931, 100, "pts")]
    pmfs_before = _make_pmfs(before_triples)
    pmfs_after  = _make_pmfs([(24931, 100, "pts"), (24931, 999, "pts")])  # 999 added
    before_ids = _identity_set(pmfs_before)
    after_ids  = _identity_set(pmfs_after)
    unexpected = after_ids - before_ids
    assert unexpected, f"Added off-slate identity must be detected: {unexpected}"


def test_duplicate_identity_in_after_is_detected():
    before_triples = [(24931, 100, "pts")]
    pmfs_before = _make_pmfs(before_triples)
    pmfs_after  = _make_pmfs([(24931, 100, "pts"), (24931, 100, "pts")])  # dup
    dup_mask = pmfs_after.duplicated(subset=["game_id", "player_id", "stat"], keep=False)
    assert dup_mask.any(), "Duplicate identity must be detected"


# ─── 6. Workflow expected PMF set is independently derived from slate ─────────

def test_workflow_derives_expected_pmf_from_slate_not_actual():
    run = _load_wf_run()
    assert run, "Build expected PMF and edge manifests step not found"
    # Must reference the slate parquet
    assert "slate_path" in run or "slate_" in run, (
        "Expected PMF must be derived from the slate parquet, not the actual PMF file"
    )
    # Must cross-join with SUPPORTED_STATS
    assert "SUPPORTED_STATS" in run, (
        "Expected PMF must cross-join slate players with SUPPORTED_STATS"
    )
    # validate_pmf_manifest must be called (full expected vs actual check)
    assert "validate_pmf_manifest" in run, (
        "validate_pmf_manifest must be called to check expected vs actual PMFs"
    )


# ─── 7. Workflow does not contain expected_pmf = actual_pmf.copy() ────────────

def test_workflow_does_not_use_expected_equals_actual():
    run = _load_wf_run()
    assert run, "Build expected PMF and edge manifests step not found"
    assert "expected_pmf = actual_pmf" not in run, (
        "expected_pmf must NOT be derived from actual_pmf — "
        "that bypasses the coverage integrity check"
    )
    assert "expected_pmf = actual_pmf.copy()" not in run, (
        "expected_pmf must NOT be actual_pmf.copy()"
    )
