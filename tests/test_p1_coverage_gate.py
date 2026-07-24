"""Tests for the P1 backfill fail-closed identity/coverage gate.

Regression for the silent-empty-output failure: a stale games/roster table used to
produce zero quotes and exit 0. classify_coverage now flags that catastrophically.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "p1_historical_backfill", ROOT / "scripts" / "p1_historical_backfill.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
classify = p1.classify_coverage


def test_no_events_is_fatal():
    assert classify(0, 0, True, 0.0, 0.5, False) == ("fatal_no_events", None)


def test_stale_games_zero_quotes_is_fatal():
    # events exist, none resolved to a game -> empty quotes -> fatal (games)
    assert classify(7, 0, True, 0.0, 0.5, False) == ("fatal_stale", "games")


def test_stale_roster_zero_quotes_is_fatal():
    # events matched games but no player resolved -> empty quotes -> fatal (roster)
    assert classify(7, 3, True, 0.4286, 0.5, False) == ("fatal_stale", "roster")


def test_allow_empty_downgrades_to_warn():
    assert classify(7, 0, True, 0.0, 0.5, True) == ("warn_empty_allowed", None)


def test_low_coverage_but_nonempty_is_warn_not_fatal():
    # real quotes produced but low match rate (e.g. cross-date window artifact)
    assert classify(7, 3, False, 0.4286, 0.5, False) == ("warn_low", None)


def test_healthy_coverage_is_ok():
    assert classify(10, 10, False, 1.0, 0.5, False) == ("ok", None)


def test_cli_has_gate_options():
    import inspect
    src = inspect.getsource(p1.main)
    assert "--min-event-match-rate" in src
    assert "--allow-empty" in src
