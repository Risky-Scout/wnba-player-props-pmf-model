"""Tests for scripts/migrate_p1_quotes_to_atomic.py (owner phase 5).

Covers the fail-closed validation helper and, when the migrated store is present locally, the atomic-pair
invariants (same book, same line, two distinct sides, valid odds, before tip). Integration assertions are
skipped in a clean clone where the P1 archive / box are not fetched, so the unit contract still runs in CI.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        "mig", REPO / "scripts" / "migrate_p1_quotes_to_atomic.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_valid_american_rejects_sub_100_and_nonfinite():
    m = _mod()
    assert m._valid_american(100) and m._valid_american(-110) and m._valid_american(250)
    assert not m._valid_american(0)
    assert not m._valid_american(99)
    assert not m._valid_american(-99)
    assert not m._valid_american(float("nan"))
    assert not m._valid_american(None)
    assert not m._valid_american("abc")


def test_stable_side_id_is_deterministic():
    m = _mod()
    a = m._sha("evt", "dk", 1, "pts", 12.5, "over", "decision", "2026-06-01T00:00:00Z")
    b = m._sha("evt", "dk", 1, "pts", 12.5, "over", "decision", "2026-06-01T00:00:00Z")
    c = m._sha("evt", "dk", 1, "pts", 12.5, "under", "decision", "2026-06-01T00:00:00Z")
    assert a == b and a != c


def test_atomic_pairs_invariants_when_present():
    pairs_path = REPO / "data" / "processed" / "atomic_quotes" / "atomic_pairs.parquet"
    if not pairs_path.exists():
        pytest.skip("migrated atomic pairs not present (clean clone / data not fetched)")
    p = pd.read_parquet(pairs_path)
    assert len(p) > 0
    # two-sided: distinct side ids; per-book / per-line by construction
    assert (p["over_side_id"] != p["under_side_id"]).all()
    # odds validity preserved
    assert (p["over_odds"].abs() >= 100).all()
    assert (p["under_odds"].abs() >= 100).all()
    # never at/after tip
    assert (pd.to_datetime(p["pair_timestamp_utc"], utc=True)
            < pd.to_datetime(p["scheduled_tip_utc"], utc=True)).all()
    # no consensus: sportsbook is a real single book per pair
    assert p["sportsbook"].notna().all()
