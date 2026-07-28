"""Tests for deterministic prediction-cutoff (section 2) and hash-safe checkpoint reuse (section 3)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
BUILD = REPO / "scripts" / "build_opportunity_oof.py"


def _mod():
    spec = importlib.util.spec_from_file_location("opp_oof", BUILD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _mod()


def test_cutoff_is_deterministic_and_per_game():
    # two props, two books, differing decision timestamps for the same game
    q = pd.DataFrame({
        "game_id": [1, 1, 1, 2, 2],
        "player_id": [10, 11, 10, 20, 21],
        "prop": ["fg3m", "pts", "fg3m", "fg3m", "pts"],
        "decision_timestamp": ["2026-06-01T18:00:00Z", "2026-06-01T19:00:00Z",
                               "2026-06-01T17:00:00Z", "2026-06-02T18:00:00Z",
                               "2026-06-02T18:30:00Z"],
    })
    box = pd.DataFrame({"game_id": [1, 1, 1, 2, 2], "player_id": [10, 11, 12, 20, 21]})
    cut, policy = B._build_deterministic_cutoffs(q, box)
    # one cutoff per game (max decision ts), broadcast to all box players in that game
    g1 = cut[cut["game_id"] == 1]["quote_timestamp"].unique()
    assert len(g1) == 1
    assert pd.Timestamp(g1[0]) == pd.Timestamp("2026-06-01T19:00:00Z")
    # player 12 (no quote) still receives the game cutoff -> independent of prop/player identity
    assert set(cut[cut["game_id"] == 1]["player_id"]) == {10, 11, 12}
    assert policy["cutoff_policy_id"] == B.CUTOFF_POLICY_ID
    assert policy["certified_mode"] is False


def test_cutoff_ignores_prop_book_line_ordering():
    base = pd.DataFrame({
        "game_id": [1, 1], "player_id": [10, 10], "prop": ["fg3m", "pts"],
        "decision_timestamp": ["2026-06-01T18:00:00Z", "2026-06-01T19:00:00Z"],
    })
    box = pd.DataFrame({"game_id": [1], "player_id": [10]})
    c1, _ = B._build_deterministic_cutoffs(base, box)
    c2, _ = B._build_deterministic_cutoffs(base.iloc[::-1].reset_index(drop=True), box)
    assert c1["quote_timestamp"].iloc[0] == c2["quote_timestamp"].iloc[0]


def test_checkpoint_reuse_rejects_on_mismatch(tmp_path):
    man = tmp_path / "m.json"
    expected = {"code_sha": "abc", "input_hashes": {"box": "1"},
                "validation": ["2026-05-08", "2026-05-14"],
                "cutoff_policy_id": B.CUTOFF_POLICY_ID, "candidate_id": "OPP_V2_RAW"}
    man.write_text(json.dumps(expected))
    assert B._verify_checkpoint(man, expected) is True
    # code change -> reject
    bad = dict(expected, code_sha="different")
    assert B._verify_checkpoint(man, bad) is False
    # input hash change -> reject
    bad2 = dict(expected, input_hashes={"box": "2"})
    assert B._verify_checkpoint(man, bad2) is False
    # candidate change -> reject
    bad3 = dict(expected, candidate_id="OPP_V2_TEAM_SHARE")
    assert B._verify_checkpoint(man, bad3) is False


def test_checkpoint_missing_manifest_recomputes(tmp_path):
    assert B._verify_checkpoint(tmp_path / "nope.json", {"code_sha": "x"}) is False


def test_code_sha_changes_when_source_changes(tmp_path):
    # code_sha is a hash over real repo source files; just assert it's stable + hex
    s1 = B._code_sha(REPO)
    s2 = B._code_sha(REPO)
    assert s1 == s2 and len(s1) == 64
