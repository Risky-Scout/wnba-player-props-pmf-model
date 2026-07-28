"""build_candidate_comparison must FAIL CLOSED on duplicates (owner directive item 3).

Proves there is no silent drop_duplicates: the comparison's uniqueness guard raises with the key,
source, duplicate row count, and conflicting fields; and the candidate join routes through the
canonical evaluator's fail-closed build_canonical_scored_rows (which itself raises on duplicate OOF
predictions).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "bcc", REPO / "scripts" / "build_candidate_comparison.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_assert_unique_raises_with_details():
    bcc = _load_module()
    df = pd.DataFrame({
        "game_id": [1, 1, 2], "player_id": [10, 10, 20], "prop": ["pts", "pts", "pts"],
        "line": [10.5, 12.5, 8.5], "market": [0.5, 0.6, 0.4],
    })
    with pytest.raises(ValueError) as ei:
        bcc._assert_unique(df, bcc.KEY, "quotes:test")
    msg = str(ei.value)
    assert "quotes:test" in msg
    assert "duplicate rows" in msg
    assert "conflicting_fields" in msg
    # line and market both differ across the duplicate key -> reported as conflicting
    assert "line" in msg and "market" in msg


def test_assert_unique_passes_when_unique():
    bcc = _load_module()
    df = pd.DataFrame({"game_id": [1, 2], "player_id": [10, 20], "prop": ["pts", "pts"]})
    bcc._assert_unique(df, bcc.KEY, "ok")  # must not raise


def test_no_silent_drop_duplicates_in_source():
    """Guard against reintroducing a silent dedup in the comparison script."""
    src = (REPO / "scripts" / "build_candidate_comparison.py").read_text()
    assert ".drop_duplicates(" not in src, "silent drop_duplicates() call reintroduced in build_candidate_comparison.py"


def test_candidate_join_routes_through_canonical_evaluator_and_fails_on_dup_oof():
    bcc = _load_module()
    EV = bcc.EV
    # minimal quotes (unique) and a DUPLICATED oof for the same key -> evaluator must raise
    q = pd.DataFrame({
        "game_id": [1, 2], "player_id": [10, 20], "prop": ["fg3m", "fg3m"],
        "line": [1.5, 1.5], "outcome_over": [1, 0], "binary_score_eligible": [True, True],
        "quote_pair_id": ["a", "b"], "market_prob_over_no_vig": [0.5, 0.5],
        "game_date": ["2026-05-10", "2026-05-11"],
    })
    dup_oof = pd.DataFrame({
        "game_id": [1, 1], "player_id": [10, 10], "prop": ["fg3m", "fg3m"],
        "active_pmf_json": ["[0.4,0.6]", "[0.3,0.7]"],
    })
    with pytest.raises(EV.EvaluatorContractError, match="duplicate OOF predictions"):
        EV.build_canonical_scored_rows(dup_oof, q)
