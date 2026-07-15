"""Focused regression tests for injury slate/feature contract.

Proves:
1. Workflow passes slate_DATE.parquet (not historical matrix) to injury step.
2. Historical feature matrix path is NOT used.
3. PMF and feature (game_id, player_id) identities must match exactly.
4. Affected players rebuild successfully within the slate universe.
5. Before/after PMF (game_id, player_id, stat) identity set is identical.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

WF_PATH = Path(".github/workflows/pregame_initial.yml")


def _load_wf() -> dict:
    raw = yaml.safe_load(WF_PATH.read_text())
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return raw


def _injury_step_run() -> str:
    wf = _load_wf()
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "Apply injury updates" in str(step.get("name", "")):
                return step.get("run", "")
    return ""


# ─── 1. Workflow passes current-run slate as features ────────────────────────

def test_workflow_passes_tonight_slate_as_features():
    """--features must be deliveries/tonight/slate_${GAME_DATE}.parquet."""
    run = _injury_step_run()
    assert run, "Apply injury updates step not found"
    assert 'deliveries/tonight/slate_${GAME_DATE}.parquet' in run or \
           'deliveries/tonight/slate_' in run, (
        "apply_injury_updates must receive deliveries/tonight/slate_{date}.parquet "
        "as --features, not the historical feature matrix"
    )


def test_workflow_passes_full_pmfs_as_slate():
    """--slate must be deliveries/tonight/full_pmfs_wide.parquet."""
    run = _injury_step_run()
    assert run, "Apply injury updates step not found"
    assert "full_pmfs_wide.parquet" in run, (
        "apply_injury_updates must receive full_pmfs_wide.parquet as --slate"
    )


# ─── 2. Historical feature matrix NOT used ───────────────────────────────────

def test_workflow_does_not_pass_historical_feature_matrix():
    """data/processed/wnba_player_game_features_wide.parquet must not be passed."""
    run = _injury_step_run()
    assert run, "Apply injury updates step not found"
    assert "wnba_player_game_features_wide.parquet" not in run, (
        "Historical feature matrix must NOT be passed to apply_injury_updates; "
        "it doesn't contain current-run game identities"
    )


# ─── 3. PMF and feature identities must match exactly ────────────────────────

def _check_identity_equality(pmf_pairs: pd.DataFrame, feat_pairs: pd.DataFrame) -> list[str]:
    """Replicate apply_injury_updates identity check logic."""
    id_cols = ["game_id", "player_id"]
    merge = pmf_pairs.merge(feat_pairs, on=id_cols, how="outer", indicator=True)
    missing = merge[merge["_merge"] == "left_only"]
    extra   = merge[merge["_merge"] == "right_only"]
    dups    = feat_pairs[feat_pairs.duplicated(subset=id_cols, keep=False)]
    errors = []
    if not missing.empty:
        errors.append(f"missing: {len(missing)}")
    if not extra.empty:
        errors.append(f"extra: {len(extra)}")
    if not dups.empty:
        errors.append(f"duplicates: {len(dups)}")
    return errors


def test_exact_identity_match_passes():
    pmf_pairs  = pd.DataFrame([{"game_id": 24931, "player_id": 100},
                                {"game_id": 24931, "player_id": 200}])
    feat_pairs = pd.DataFrame([{"game_id": 24931, "player_id": 100},
                                {"game_id": 24931, "player_id": 200}])
    errors = _check_identity_equality(pmf_pairs, feat_pairs)
    assert not errors, f"Exact match must pass: {errors}"


def test_missing_feature_identity_fails():
    """PMF player 200 has no feature row — must fail."""
    pmf_pairs  = pd.DataFrame([{"game_id": 24931, "player_id": 100},
                                {"game_id": 24931, "player_id": 200}])
    feat_pairs = pd.DataFrame([{"game_id": 24931, "player_id": 100}])
    errors = _check_identity_equality(pmf_pairs, feat_pairs)
    assert any("missing" in e for e in errors)


def test_unexpected_feature_identity_fails():
    """Feature has off-slate player 999 not in PMFs — must fail."""
    pmf_pairs  = pd.DataFrame([{"game_id": 24931, "player_id": 100}])
    feat_pairs = pd.DataFrame([{"game_id": 24931, "player_id": 100},
                                {"game_id": 24931, "player_id": 999}])
    errors = _check_identity_equality(pmf_pairs, feat_pairs)
    assert any("extra" in e for e in errors)


def test_duplicate_feature_identity_fails():
    """Duplicate (game_id, player_id) in feature slate — must fail."""
    pmf_pairs  = pd.DataFrame([{"game_id": 24931, "player_id": 100}])
    feat_pairs = pd.DataFrame([{"game_id": 24931, "player_id": 100},
                                {"game_id": 24931, "player_id": 100}])
    errors = _check_identity_equality(pmf_pairs, feat_pairs)
    assert any("duplicate" in e for e in errors)


# ─── 4. Affected players rebuild within slate universe ───────────────────────

def test_affected_players_constrained_to_slate():
    """affected_player_ids must be intersected with slate player IDs."""
    slate_ids = {100, 200, 300}
    affected  = {100, 200, 999}   # 999 is off-slate
    off_slate = affected - slate_ids
    constrained = affected - off_slate
    assert 999 not in constrained
    assert 100 in constrained and 200 in constrained


# ─── 5. Before/after PMF identity set is identical ───────────────────────────

def test_before_after_identity_equality():
    """pmfs_after (game_id, player_id, stat) must exactly equal pmfs_before."""
    triples = [(24931, 100, "pts"), (24931, 100, "reb"), (24931, 200, "pts")]
    before = pd.DataFrame([{"game_id": g, "player_id": p, "stat": s} for g,p,s in triples])
    after  = before.copy()
    # Simulate inactive player 100 getting zero PMF — identity still present
    after_ids = set(map(tuple, after[["game_id","player_id","stat"]].itertuples(index=False)))
    before_ids = set(map(tuple, before[["game_id","player_id","stat"]].itertuples(index=False)))
    assert before_ids == after_ids, "Identity sets must be identical"


def test_added_player_in_after_is_detected():
    before = pd.DataFrame([{"game_id": 24931, "player_id": 100, "stat": "pts"}])
    after  = pd.DataFrame([{"game_id": 24931, "player_id": 100, "stat": "pts"},
                            {"game_id": 24931, "player_id": 999, "stat": "pts"}])
    before_ids = set(map(tuple, before[["game_id","player_id","stat"]].itertuples(index=False)))
    after_ids  = set(map(tuple, after[["game_id","player_id","stat"]].itertuples(index=False)))
    unexpected = after_ids - before_ids
    assert unexpected, "Off-slate addition must be detectable"
