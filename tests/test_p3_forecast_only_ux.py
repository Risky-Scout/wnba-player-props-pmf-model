"""P3 / PR#49 completion — Edge forecast-only abstention UX safety tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gwp", REPO / "scripts" / "generate_web_pages.py")
gwp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gwp)

BIDI = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069,
        0x200E, 0x200F, 0x061C}
ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}


def test_no_bidi_or_zero_width_in_edge_template():
    html = gwp._EDGE_HTML
    assert not [c for c in html if ord(c) in BIDI], "bidirectional control chars present"
    assert not [c for c in html if ord(c) in ZERO_WIDTH], "zero-width/BOM chars present"


def test_edge_payload_abstain_fields():
    edges = pd.DataFrame(columns=["player_id", "stat", "direction", "abs_edge", "edge_pp"])
    proj = pd.DataFrame(columns=["player_id", "stat", "pmf_mean", "median"])
    p = gwp._build_edge_json(
        edges, proj, "2026-07-18", abstain=True,
        abstain_reason="No validated betting edges currently qualify",
        validation_status="LAUNCH_READY_FORECAST_ONLY",
        enabled_stats=["reb", "turnover"], suppressed_stats=["pts", "ast", "fg3m", "blk", "stl"],
    )
    assert p["abstain"] is True
    assert p["total_props"] == 0
    assert p["publication_mode"] == "forecast_only"
    assert p["validation_status"] == "LAUNCH_READY_FORECAST_ONLY"
    assert p["enabled_stats"] == ["reb", "turnover"]
    assert "pts" in p["suppressed_stats"]
    assert "no profitability" in p["disclaimer"].lower()
    assert "no validated betting edges" in p["abstain_reason"].lower()


def test_edge_shell_abstain_hides_betting_controls():
    html = gwp._EDGE_HTML
    # dispatches to abstain rendering when the payload abstains
    assert "if (data.abstain) { renderAbstain(data); return; }" in html
    assert "function renderAbstain(data)" in html
    # hides counters, filters (BET/SMALL BET/LEAN, Over/Under) and the recommendation table
    for sel in ("'.kpis'", "'.filters'", "'.tbl-wrap'"):
        assert sel in html, f"abstain mode must hide {sel}"
    # neutralizes betting language in the footer
    assert "Kelly|BET|Edge =|Confidence" in html
    # explicit forecast-only labelling
    assert "Forecast-only release" in html
    assert "Betting edges abstained" in html


def test_edge_shell_exposes_release_metadata():
    html = gwp._EDGE_HTML
    for token in ("model_version", "calibration_version", "release_id", "validation_status",
                  "enabled_stats", "suppressed_stats"):
        assert token in html, f"abstain banner must expose {token}"


def test_non_abstain_payload_unaffected():
    edges = pd.DataFrame(columns=["player_id", "stat", "direction", "abs_edge", "edge_pp"])
    proj = pd.DataFrame(columns=["player_id", "stat", "pmf_mean", "median"])
    p = gwp._build_edge_json(edges, proj, "2026-07-18", abstain=False)
    assert p["abstain"] is False
    assert "abstain_reason" not in p
