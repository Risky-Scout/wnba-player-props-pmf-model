"""Path B MARKET_DISLOCATION acceptance-gate + hardened-scan tests.

One or more focused tests per acceptance requirement (1-9), plus the mandatory gate
behaviour: it PASSES a clean fixture audit and FAILS CLOSED (non-passing report / non-zero
CLI exit) on every seeded violation. Deterministic fixtures only — no live odds.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from wnba_props_model.edge.path_b_audit import build_audit, row_provenance
from wnba_props_model.edge.path_b_collect import extract_side_rows
from wnba_props_model.edge.path_b_fixtures import make_events, make_roster
from wnba_props_model.edge.path_b_gate import (
    FORBIDDEN_STAKE_FIELDS,
    PROVENANCE_FIELDS,
    SOURCE_TYPE_MARKET_DISLOCATION,
    load_and_validate,
    validate_audit,
)
from wnba_props_model.edge.prop_identity import (
    STATUS_AMBIGUOUS,
    STATUS_RESOLVED,
    STATUS_UNMATCHED,
    build_name_index,
    resolve_player_id,
)
from wnba_props_model.edge.soft_book_scan import (
    REFERENCE_BOOKS,
    SHARP_BOOKS,
    scan_soft_book_edges,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_MARKETS = [
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_steals", "player_blocks", "player_turnovers", "player_points_alternate",
]


# --------------------------------------------------------------------------- #
# Shared builders
# --------------------------------------------------------------------------- #
def _now():
    return datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc)


def _fixture_quotes(now=None):
    now = now or _now()
    sides = []
    for ev in make_events(now):
        sides.extend(extract_side_rows(ev, FIXTURE_MARKETS))
    return pd.DataFrame(sides)


def _run_scan(now=None, **kw):
    now = now or _now()
    quotes = _fixture_quotes(now)
    index = build_name_index(make_roster())
    defaults = dict(
        ev_threshold=0.025, min_consensus_books=3, sharp_books=SHARP_BOOKS,
        reference_books=REFERENCE_BOOKS, max_quote_age_seconds=3600.0,
        identity_index=index, require_identity=True, now=now, return_rejections=True,
    )
    defaults.update(kw)
    return scan_soft_book_edges(quotes, **defaults)


def _fixture_audit(now=None):
    board, rej = _run_scan(now)
    return build_audit(
        board, rej, game_date="2026-07-28",
        config={"ev_threshold_pct": 2.5, "min_consensus_books": 3,
                "max_quote_age_seconds": 3600.0,
                "reference_books": sorted(str(b) for b in REFERENCE_BOOKS),
                "no_vig_fail_closed": True},
        discovery={"games_discovered": 2}, credit_usage={"mode": "fixture", "consumed": 0},
        diagnostic_ev_threshold=0.025,
    )


# --------------------------------------------------------------------------- #
# Requirement 1 — exact identity match (canonical player_id, no name-only)
# --------------------------------------------------------------------------- #
def test_req1_identity_resolution_exact_unmatched_ambiguous():
    index = build_name_index([
        {"player_name": "Alpha Guard", "player_id": 1001},
        {"player_name": "Dup Name", "player_id": 1},
        {"player_name": "Dup Name", "player_id": 2},
    ])
    assert resolve_player_id("Alpha Guard", index) == (1001, STATUS_RESOLVED)
    assert resolve_player_id("Nobody Here", index) == (None, STATUS_UNMATCHED)
    assert resolve_player_id("Dup Name", index)[1] == STATUS_AMBIGUOUS


def test_req1_unresolved_identity_is_rejected_not_scored():
    board, rej = _run_scan()
    # Bravo Center is absent from the roster -> rejected, never scored.
    assert "Bravo Center" not in set(board["player_name"])
    reasons = set(rej["reason"])
    assert "unresolved_identity" in reasons
    # Every displayed row carries a resolved canonical player_id.
    assert board["player_id_resolved"].all()
    assert board["player_id"].notna().all()


# --------------------------------------------------------------------------- #
# Requirement 2 — leave-one-book-out consensus (self excluded, recorded)
# --------------------------------------------------------------------------- #
def test_req2_self_excluded_recorded_and_book_not_in_own_consensus():
    board, _ = _run_scan()
    assert board["self_excluded"].all()
    for _, r in board.iterrows():
        assert r["book"] not in list(r["consensus_books"])


# --------------------------------------------------------------------------- #
# Requirement 3 — atomic line matching (standard vs alternate segregated)
# --------------------------------------------------------------------------- #
def test_req3_alternate_market_not_mixed_with_standard():
    board, _ = _run_scan()
    ag = board[board["player_name"] == "Alpha Guard"]
    std = ag[ag["market_key"] == "player_points"]
    alt = ag[ag["market_key"] == "player_points_alternate"]
    assert len(std) and len(alt)
    assert (alt["is_alternate_market"]).all()
    assert not (std["is_alternate_market"]).any()
    # Same line (18.5) but different atomic groups => the standard group has 4 books
    # (3 others when scored) and the alternate group has 3 (2 others) — never merged.
    assert std["consensus_n_books"].max() == 3
    assert alt["consensus_n_books"].max() == 2


def test_req3_different_lines_never_compared():
    # Two books at DIFFERENT lines never form a consensus with each other.
    rows = []
    for bk in ("pinnacle", "betonlineag", "draftkings"):
        rows += [_q(bk, "over", -110, line=18.5), _q(bk, "under", -110, line=18.5)]
    rows += [_q("softbook", "over", 120, line=19.5), _q("softbook", "under", -140, line=19.5)]
    board = scan_soft_book_edges(pd.DataFrame(rows), min_consensus_books=1)
    # softbook @19.5 has no other book at 19.5 -> per_book < 2 -> not scored.
    assert "softbook" not in set(board["book"])


# --------------------------------------------------------------------------- #
# Requirement 4 — timestamp integrity
# --------------------------------------------------------------------------- #
def test_req4_post_tip_event_rejected():
    board, rej = _run_scan()
    assert "post_tip_or_stale_event" in set(rej["reason"])
    # evt-post-tip produced no scored rows.
    assert "evt-post-tip" not in set(board["event_id"])


def test_req4_stale_quote_rejected_and_reasoned():
    board, rej = _run_scan()
    assert "stale_quote_age" in set(rej["reason"])
    stale_rows = board[board["rejection_reason"] == "stale_quote_age"]
    assert len(stale_rows) >= 1
    assert (stale_rows["validation_status"] == "REJECTED").all()
    assert not stale_rows["actionable"].any()


def test_req4_malformed_timestamp_warned_and_age_null():
    board, _ = _run_scan()
    mal = board[board["warning_reason"] == "malformed_timestamp"]
    assert len(mal) >= 1
    assert mal["quote_age_seconds"].isna().all()


def test_req4_timestamps_present_and_age_reported():
    board, _ = _run_scan()
    for col in ("provider_timestamp", "ingestion_timestamp", "scan_timestamp", "scheduled_tip"):
        assert col in board.columns
    fresh = board[board["warning_reason"].isna() & board["rejection_reason"].isna()]
    assert (fresh["quote_age_seconds"] >= 0).all()


def test_req4_configurable_age_gate_disabled_keeps_rows():
    # With no age gate, the previously-stale draftkings quote is not rejected for age.
    board, rej = _run_scan(max_quote_age_seconds=None)
    assert "stale_quote_age" not in set(rej["reason"])


# --------------------------------------------------------------------------- #
# Requirement 5 — consensus quality (min books, dispersion, sharp reference)
# --------------------------------------------------------------------------- #
def test_req5_min_consensus_books_guard_flags_not_qualified():
    board, _ = _run_scan(min_consensus_books=5)
    assert not board["qualified"].any()


def test_req5_dispersion_and_sharp_reference_recorded():
    board, _ = _run_scan()
    assert "consensus_dispersion_stdev" in board.columns
    assert "consensus_dispersion_iqr" in board.columns
    # Alpha standard consensus includes pinnacle/betonlineag (reference books).
    ag = board[(board["player_name"] == "Alpha Guard")
               & (board["market_key"] == "player_points")]
    assert ag["consensus_includes_sharp"].any()


def test_req5_soft_only_consensus_flagged_not_sharp():
    # Consensus of ONLY soft books must record consensus_includes_sharp=False.
    rows = []
    for bk in ("softa", "softb", "softc"):
        rows += [_q(bk, "over", -110), _q(bk, "under", -110)]
    rows += [_q("softd", "over", 120), _q("softd", "under", -140)]
    board = scan_soft_book_edges(pd.DataFrame(rows), min_consensus_books=1,
                                 reference_books=REFERENCE_BOOKS)
    assert not board["consensus_includes_sharp"].any()


# --------------------------------------------------------------------------- #
# Requirement 6 — no-vig correctness / fail closed on missing opposite side
# --------------------------------------------------------------------------- #
def test_req6_missing_opposite_side_fails_closed_with_reason():
    board, rej = _run_scan()
    assert "missing_opposite_side_no_vig_fail_closed" in set(rej["reason"])
    # The one-sided softbook Charlie-assists quote was never scored (no fabrication).
    charlie = board[board["player_name"] == "Charlie Wing"]
    assert "softbook" not in set(charlie["book"])


# --------------------------------------------------------------------------- #
# Requirement 7 — execution realism (theoretical vs executable EV)
# --------------------------------------------------------------------------- #
def test_req7_theoretical_and_executable_ev_distinct():
    board, _ = _run_scan()
    assert (board["theoretical_ev_pct"].notna()).all()
    # EXECUTABLE_EV is unknown until a forward recheck runs.
    assert board["executable_ev_pct"].isna().all()
    assert board["price_survived_30s"].isna().all()
    assert board["price_survived_60s"].isna().all()


def test_req7_high_ev_row_not_actionable():
    board, _ = _run_scan()
    qual = board[board["qualified"]]
    assert len(qual) >= 1
    assert (qual["theoretical_ev_pct"] >= 2.5).all()
    assert not qual["actionable"].any()          # never actionable on theoretical EV alone


# --------------------------------------------------------------------------- #
# Requirement 8 — initial safety status
# --------------------------------------------------------------------------- #
def test_req8_actionable_false_and_no_stake_or_kelly_emitted():
    board, _ = _run_scan()
    assert not board["actionable"].any()
    for f in FORBIDDEN_STAKE_FIELDS:
        assert f not in board.columns


# --------------------------------------------------------------------------- #
# Requirement 9 — full provenance on every displayed row
# --------------------------------------------------------------------------- #
def test_req9_every_row_carries_full_provenance():
    board, _ = _run_scan()
    for _, r in board.iterrows():
        prov = row_provenance(r)
        for f in PROVENANCE_FIELDS:
            assert f in prov
        assert prov["source_type"] == SOURCE_TYPE_MARKET_DISLOCATION


# --------------------------------------------------------------------------- #
# The MANDATORY gate
# --------------------------------------------------------------------------- #
def test_gate_passes_clean_fixture_audit():
    report = validate_audit(_fixture_audit())
    assert report.passed, [str(v) for v in report.violations]
    assert report.n_rows_checked > 0


@pytest.mark.parametrize("mutate,expected_code", [
    (lambda a: a["board_rows"][0].__setitem__("actionable", True), "PREMATURE_ACTIONABLE"),
    (lambda a: a["board_rows"][0].__setitem__("source_type", "MODEL_EDGE"), "WRONG_SOURCE_TYPE"),
    (lambda a: a["board_rows"][0].__setitem__("kelly_fraction", 0.1), "STAKE_EMITTED"),
    (lambda a: (a["board_rows"][0].__setitem__("player_id", None),
                a["board_rows"][0].__setitem__("player_id_resolved", False)), "IDENTITY_UNRESOLVED"),
    (lambda a: a["board_rows"][0].__setitem__("self_excluded", False), "SELF_EXCLUSION_NOT_RECORDED"),
    (lambda a: a["board_rows"][0].pop("quote_age_seconds"), "MISSING_PROVENANCE_FIELD"),
    (lambda a: a["config"].__setitem__("no_vig_fail_closed", False), "NO_VIG_NOT_FAIL_CLOSED"),
    (lambda a: a.__setitem__("source_type", "MODEL_EDGE"), "TOPLEVEL_SOURCE_TYPE"),
    (lambda a: a.__setitem__("profitable", True), "OVERSTATED_CLAIM"),
    (lambda a: a["latency"].pop("median_quote_age_seconds"), "LATENCY_NOT_DISCLOSED"),
])
def test_gate_fails_closed_on_seeded_violation(mutate, expected_code):
    audit = _fixture_audit()
    mutate(audit)
    report = validate_audit(audit)
    assert not report.passed
    assert expected_code in {v.code for v in report.violations}


def test_gate_missing_file_fails():
    report = load_and_validate("/nonexistent/audit.json")
    assert not report.passed


def test_gate_cli_exits_nonzero_on_seeded_violation(tmp_path):
    audit = _fixture_audit()
    audit["board_rows"][0]["source_type"] = "MODEL_EDGE"   # seed a violation
    bad = tmp_path / "seeded_audit.json"
    bad.write_text(json.dumps(audit, default=str))
    env = {"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "path_b_acceptance_gate.py"), str(bad)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_gate_cli_exits_zero_on_clean_audit(tmp_path):
    good = tmp_path / "clean_audit.json"
    good.write_text(json.dumps(_fixture_audit(), default=str))
    env = {"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "path_b_acceptance_gate.py"), str(good)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# Minimal single-side row builder for atomic/consensus micro-tests
# --------------------------------------------------------------------------- #
_FUTURE = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()


def _q(book, side, odds, *, player="A. Player", stat="pts", line=18.5,
       market_key="player_points", event="evt1", commence=_FUTURE):
    return {
        "collected_utc": "2026-07-28T18:00:00+00:00",
        "event_id": event, "commence_time": commence,
        "home_team": "HOME", "away_team": "AWAY",
        "book": book, "book_last_update": "2026-07-28T17:59:00+00:00",
        "market_key": market_key, "stat": stat, "player_name": player,
        "side": side, "line": line, "american_odds": odds,
    }
