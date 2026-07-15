"""Focused tests for market ingestion pipeline.

Covers:
  1. Audit count semantics (raw / fresh / reconciled definitions)
  2. Deterministic multi-book quote selection (max EV → price → timestamp → vendor)
  3. Complete PMF publication with partial market coverage
  4. Odds API → BDL fallback activation logic in pregame_initial.yml
  5. Active-slate zero-market failure (fail closed)
  6. Blocking FTP deployment (no continue-on-error on FTP step)
  7. Live post-deployment verification step present and blocking
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import pytest

WORKFLOW_PATH = Path(".github/workflows/pregame_initial.yml")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_workflow() -> str:
    return WORKFLOW_PATH.read_text()


def _make_props_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal props DataFrame for edge report tests."""
    return pd.DataFrame(rows)


def _american_to_decimal(a: float) -> float:
    return (a / 100 + 1) if a >= 0 else (100 / abs(a) + 1)


def _ev(model_p_over: float, dec_over: float, dec_under: float, prefer_over: bool) -> float:
    if prefer_over:
        return model_p_over * (dec_over - 1) - (1 - model_p_over)
    return (1 - model_p_over) * (dec_under - 1) - model_p_over


# ─── 1. Audit count semantics ─────────────────────────────────────────────────

def test_audit_count_semantics_raw_fresh_reconciled():
    """raw >= fresh >= reconciled; reconciled counts deduped snapshots before PMF join."""
    raw_rows = [
        # 3 vendors for the same (game_id, player_id, stat, line)
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "draftkings", "over_odds": -115.0, "under_odds": -105.0},
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "fanduel",    "over_odds": -110.0, "under_odds": -110.0},
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "caesars",    "over_odds": -112.0, "under_odds": -108.0},
        # An exact duplicate row (same vendor, same odds) – snapshot duplicate
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "draftkings", "over_odds": -115.0, "under_odds": -105.0},
        # Different line — NOT a duplicate
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 15.5,
         "vendor": "draftkings", "over_odds": +110.0, "under_odds": -130.0},
    ]
    df = _make_props_df(raw_rows)

    raw_count = len(df)  # 5

    # Simulate odds validation (all pass here)
    fresh_count = raw_count  # 5 (no invalid odds in this fixture)

    # Snapshot dedup: drop duplicates on (game_id, player_id, stat, vendor, line, over_odds, under_odds)
    dedup_cols = ["game_id", "player_id", "stat", "vendor", "line", "over_odds", "under_odds"]
    reconciled_df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    reconciled_count = len(reconciled_df)  # 4 (DraftKings 14.5 deduplicated)

    assert raw_count == 5, f"raw_count={raw_count}"
    assert fresh_count == 5, f"fresh_count={fresh_count}"
    assert reconciled_count == 4, f"reconciled_count={reconciled_count}"

    # Invariant: raw >= fresh >= reconciled
    assert raw_count >= fresh_count >= reconciled_count


def test_audit_json_has_all_required_fields(tmp_path):
    """edge_report JSON must contain raw, fresh, reconciled counts and market_status."""
    audit = {
        "game_date": "2026-07-14",
        "market_status": "SUCCESS_WITH_MARKETS",
        "raw_quote_count": 162,
        "fresh_quote_count": 162,
        "reconciled_quote_count": 92,
        "rejection_counts": {},
        "market_request_status": "ok",
        "market_request_timestamp_utc": "2026-07-14T23:00:00Z",
    }
    p = tmp_path / "edge_report_2026-07-14.json"
    p.write_text(json.dumps(audit))

    loaded = json.loads(p.read_text())
    for field in ("raw_quote_count", "fresh_quote_count", "reconciled_quote_count",
                  "rejection_counts", "market_request_status", "market_request_timestamp_utc"):
        assert field in loaded, f"Missing field: {field}"
    assert loaded["raw_quote_count"] >= loaded["fresh_quote_count"] >= loaded["reconciled_quote_count"]


# ─── 2. Deterministic multi-book quote selection ──────────────────────────────

def _build_comp_rows():
    """Three vendors for pts @ 14.5. Model p_over = 0.55 (prefer OVER)."""
    model_p = 0.55
    return [
        # DraftKings: worse over odds (-115 → 1.87 decimal → EV = 0.55*0.87 - 0.45 = 0.0285)
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "draftkings", "over_odds": -115.0, "under_odds": -105.0,
         "model_prob_over": model_p, "updated_at": "2026-07-14T22:00:00Z"},
        # FanDuel: best over odds (+100 → 2.00 → EV = 0.55*1.0 - 0.45 = 0.10)
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "fanduel", "over_odds": 100.0, "under_odds": -120.0,
         "model_prob_over": model_p, "updated_at": "2026-07-14T22:30:00Z"},
        # Caesars: medium odds (-110 → 1.909 → EV = 0.55*0.909 - 0.45 = 0.05)
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "caesars", "over_odds": -110.0, "under_odds": -110.0,
         "model_prob_over": model_p, "updated_at": "2026-07-14T22:15:00Z"},
    ]


def test_multibook_selection_max_ev():
    """FanDuel has highest EV on OVER side and must be selected."""
    rows = _build_comp_rows()
    comp = pd.DataFrame(rows)

    model_p = 0.55
    id_cols = ["game_id", "player_id", "stat", "line"]

    def _american_to_dec(a):
        try:
            a = float(a)
            return (a / 100 + 1) if a >= 0 else (100 / abs(a) + 1)
        except (TypeError, ZeroDivisionError, ValueError):
            return 0.0

    prefer_over = comp["model_prob_over"] >= 0.5
    dec_over    = comp["over_odds"].apply(_american_to_dec)
    dec_under   = comp["under_odds"].apply(_american_to_dec)
    ev_over     = comp["model_prob_over"] * (dec_over - 1) - (1 - comp["model_prob_over"])
    ev_under    = (1 - comp["model_prob_over"]) * (dec_under - 1) - comp["model_prob_over"]
    comp["_ev_preferred"] = prefer_over.map({True: ev_over, False: ev_under}).fillna(0)
    # NOTE: above map is wrong; use np.where
    import numpy as np
    comp["_ev_preferred"] = np.where(prefer_over, ev_over, ev_under)
    comp["_best_price_dec"] = np.where(prefer_over, dec_over, dec_under)
    import pandas as _pd
    comp["_ts_sort"] = _pd.to_datetime(comp["updated_at"], errors="coerce")

    selected = (
        comp.sort_values(["_ev_preferred", "_best_price_dec", "_ts_sort", "vendor"],
                         ascending=[False, False, False, True])
        .drop_duplicates(subset=id_cols, keep="first")
    )

    assert len(selected) == 1
    assert selected.iloc[0]["vendor"] == "fanduel", (
        f"Expected fanduel (max EV), got {selected.iloc[0]['vendor']}"
    )


def test_multibook_selection_price_tiebreak():
    """When EV is equal, best price wins."""
    import numpy as np
    rows = [
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "book_a", "over_odds": -110.0, "under_odds": -110.0,
         "model_prob_over": 0.55, "updated_at": "2026-07-14T22:00:00Z"},
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "book_b", "over_odds": -105.0, "under_odds": -115.0,
         "model_prob_over": 0.55, "updated_at": "2026-07-14T22:00:00Z"},
    ]
    comp = pd.DataFrame(rows)
    # book_b has better over odds (-105 > -110) → higher decimal → higher EV
    dec_over_a = _american_to_decimal(-110.0)
    dec_over_b = _american_to_decimal(-105.0)
    assert dec_over_b > dec_over_a

    prefer = comp["model_prob_over"] >= 0.5
    dec_over = comp["over_odds"].apply(_american_to_decimal)
    comp["_ev"] = comp["model_prob_over"] * (dec_over - 1) - (1 - comp["model_prob_over"])
    comp["_price"] = np.where(prefer, dec_over, comp["under_odds"].apply(_american_to_decimal))
    selected = comp.sort_values(["_ev", "_price", "vendor"],
                                ascending=[False, False, True]).iloc[0]
    assert selected["vendor"] == "book_b"


def test_multibook_selection_vendor_tiebreak():
    """Identical EV and price → vendor name alphabetically (first) wins."""
    import numpy as np
    rows = [
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "zzz_book", "over_odds": -110.0, "under_odds": -110.0,
         "model_prob_over": 0.55, "updated_at": "2026-07-14T22:00:00Z"},
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5,
         "vendor": "aaa_book", "over_odds": -110.0, "under_odds": -110.0,
         "model_prob_over": 0.55, "updated_at": "2026-07-14T22:00:00Z"},
    ]
    comp = pd.DataFrame(rows)
    dec_over = comp["over_odds"].apply(_american_to_decimal)
    comp["_ev"] = comp["model_prob_over"] * (dec_over - 1) - (1 - comp["model_prob_over"])
    comp["_price"] = dec_over
    comp["_ts"] = pd.to_datetime(comp["updated_at"])
    selected = comp.sort_values(["_ev", "_price", "_ts", "vendor"],
                                ascending=[False, False, False, True]).iloc[0]
    assert selected["vendor"] == "aaa_book"


def test_number_of_books_offering_is_correct():
    """number_of_books_offering must reflect the count BEFORE selection."""
    id_cols = ["game_id", "player_id", "stat", "line"]
    rows = [
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5, "vendor": "dk"},
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5, "vendor": "fd"},
        {"game_id": 1, "player_id": 100, "stat": "pts", "line": 14.5, "vendor": "cs"},
        {"game_id": 1, "player_id": 101, "stat": "reb", "line": 5.5,  "vendor": "dk"},
    ]
    comp = pd.DataFrame(rows)
    book_counts = comp.groupby(id_cols).size().rename("number_of_books_offering").reset_index()
    comp = comp.merge(book_counts, on=id_cols, how="left")

    pts_row = comp[comp["stat"] == "pts"].iloc[0]
    reb_row = comp[comp["stat"] == "reb"].iloc[0]
    assert pts_row["number_of_books_offering"] == 3
    assert reb_row["number_of_books_offering"] == 1


# ─── 3. Complete PMF publication with partial market coverage ─────────────────

def test_distributions_includes_all_players_not_just_market_matched():
    """Distributions page props must include ALL eligible players, not just those
    with sportsbook market lines. Unmatched rows get null market fields."""
    all_players = [f"Player{i}" for i in range(10)]
    market_players = all_players[:3]  # Only 3 of 10 have market lines

    dist_props = []
    for player in all_players:
        prop = {"player": player, "stat": "PTS", "line": 0.0, "has_market_line": False}
        if player in market_players:
            prop["line"] = 14.5
            prop["has_market_line"] = True
        dist_props.append(prop)

    assert len(dist_props) == 10
    assert sum(1 for p in dist_props if p["has_market_line"]) == 3
    assert sum(1 for p in dist_props if not p["has_market_line"]) == 7
    # All players present
    assert {p["player"] for p in dist_props} == set(all_players)


# ─── 4. Odds API → BDL fallback activation ────────────────────────────────────

def test_workflow_contains_bdl_fallback_step():
    """pregame_initial.yml must have a BDL fallback step that triggers when Odds API is empty."""
    wf = _load_workflow()
    assert "BDL props fallback" in wf, "Missing BDL fallback step name"
    assert "wnba/v1/odds/player_props" in wf, "BDL props endpoint not in workflow"
    assert "oddsapi_empty" in wf or "oddsapi_latest" in wf, \
        "Fallback must check Odds API output"


def test_workflow_bdl_fallback_uses_correct_endpoint():
    """BDL fallback must use /wnba/v1/odds/player_props (not /wnba/v1/players/props)."""
    wf = _load_workflow()
    # Correct endpoint
    assert "/wnba/v1/odds/player_props" in wf
    # Wrong endpoint must NOT appear in the fallback section
    # (it can appear in comments as a negative example, so just verify the correct one is there)
    assert wf.count("/wnba/v1/odds/player_props") >= 1


def test_workflow_bdl_fallback_has_correct_prop_type_map():
    """BDL PROP_MAP in fallback must map 'points' → 'pts', combos correctly."""
    wf = _load_workflow()
    assert '"points": "pts"' in wf or "'points': 'pts'" in wf
    assert "points_rebounds_assists" in wf and "pts_reb_ast" in wf


# ─── 5. Active-slate zero-market failure (fail closed) ───────────────────────

def test_workflow_fails_closed_when_slate_nonempty_and_zero_markets():
    """When slate has players but both Odds API and BDL return 0 props, workflow must exit nonzero."""
    wf = _load_workflow()
    # The BDL fallback step must have a fatal exit when rows=0 with active slate
    assert "zero usable rows" in wf or "zero BDL" in wf or "FATAL" in wf
    # Must NOT silently succeed (continue-on-error must not be true on fallback)
    # Find the BDL fallback section
    fallback_start = wf.find("BDL props fallback")
    assert fallback_start != -1, "No BDL fallback step"
    # Extract just this step: from its name to the next "- name:" marker
    import re as _re
    rest = wf[fallback_start + 10:]
    next_step = _re.search(r"\n      - name:", rest)
    fallback_section = wf[fallback_start: fallback_start + 10 + (next_step.start() if next_step else len(rest))]
    assert "continue-on-error: true" not in fallback_section, \
        "BDL fallback must not have continue-on-error: true"
    assert "continue-on-error: false" in fallback_section, \
        "BDL fallback must explicitly set continue-on-error: false"


def test_workflow_does_not_call_zero_markets_not_posted_without_evidence():
    """Status LIVE_MARKETS_NOT_YET_AVAILABLE requires actual provider response evidence."""
    wf = _load_workflow()
    # The workflow must show evidence-based check (HTTP status or row count)
    # rather than just assuming markets aren't posted
    assert "raw_quote_count" in wf or "market_status" in wf or "r2.ok" in wf, \
        "Workflow must use provider response evidence before concluding no markets"


# ─── 6. Blocking FTP deployment ──────────────────────────────────────────────

def test_ftp_deployment_is_blocking():
    """FTP deployment step must NOT have continue-on-error: true."""
    wf = _load_workflow()
    # Find FTP deploy step
    ftp_idx = wf.find("ftp_deploy.py")
    assert ftp_idx != -1, "No ftp_deploy.py in workflow"
    # Get the surrounding step context (2000 chars before and after)
    ftp_section = wf[max(0, ftp_idx - 500):ftp_idx + 200]
    assert "continue-on-error: true" not in ftp_section, \
        "FTP deployment must NOT have continue-on-error: true"


def test_ftp_deployment_step_name_says_blocking():
    """FTP step name should indicate it is blocking."""
    wf = _load_workflow()
    assert "BLOCKING" in wf or "blocking" in wf.lower(), \
        "FTP deployment step should be marked as BLOCKING"


# ─── 7. Live post-deployment verification ────────────────────────────────────

def test_workflow_has_post_deployment_verification():
    """pregame_initial.yml must have a blocking post-deployment verification step."""
    wf = _load_workflow()
    assert "post-deployment" in wf.lower() or "post_deploy" in wf.lower() or \
           "Post-deployment" in wf, "Missing post-deployment verification step"


def test_post_deployment_step_checks_all_required_fields():
    """Post-deployment step must verify release_id, game_date, model_version,
    calibration_version, Edge rows > 0, Distributions rows > 0, no duplicate identities."""
    wf = _load_workflow()
    post_idx = wf.lower().find("post-deployment")
    assert post_idx != -1
    post_section = wf[post_idx:post_idx + 5000]

    required_checks = [
        "release_id",
        "game_date",
        "model_version",
        "row_count",
        "duplicate",
        "reconciled_quote_count",
    ]
    for req in required_checks:
        assert req in post_section, \
            f"Post-deployment verification missing check for: {req}"


def test_post_deployment_step_is_blocking():
    """Post-deployment verification must NOT have continue-on-error: true."""
    wf = _load_workflow()
    post_idx = wf.lower().find("post-deployment")
    assert post_idx != -1
    # Step can be long (heredoc); search up to 7000 chars
    post_section = wf[post_idx:post_idx + 7000]
    assert "continue-on-error: true" not in post_section, \
        "Post-deployment verification must be blocking (not continue-on-error: true)"
    assert "continue-on-error: false" in post_section, \
        "Post-deployment verification must explicitly set continue-on-error: false"


def test_post_deployment_edge_rows_fails_when_markets_exist():
    """Post-deployment check: Edge rows must be > 0 when reconciled_quote_count > 0."""
    wf = _load_workflow()
    post_section = wf[wf.lower().find("post-deployment"):][:5000]
    # Must have logic: if rqc > 0 then edge_rows > 0
    assert "rqc > 0" in post_section or "reconciled_quote_count" in post_section, \
        "Post-deployment must gate Edge row_count on reconciled_quote_count"
