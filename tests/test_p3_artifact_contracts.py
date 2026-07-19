"""P3 — immediate safety (VALIDATION_PENDING), candidate/publishable artifact
separation (Defect 1), and fail-closed release-critical workflows (Defect 2)."""
from __future__ import annotations

from pathlib import Path

from wnba_props_model.pipeline.policy import load_policy

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "config" / "recommendation_policy.yaml"
BUILD_EDGE = (REPO / "scripts/build_edge_report.py").read_text()
GEN_PAGES = (REPO / "scripts/generate_web_pages.py").read_text()


# ---- Immediate safety correction ----
def test_forecast_state_seven_certified_rest_suppressed():
    p = load_policy(POLICY)
    assert p.forecast_status == "LIVE_VALIDATED_FORECAST_ONLY"
    assert set(p.forecast_certified_stats) == {"turnover", "pts", "ast", "stl", "stocks", "pts_ast", "reb"}
    # the non-passing markets remain suppressed (reb now passes via Candidate D)
    for s in ("fg3m", "blk", "pts_reb", "pts_reb_ast"):
        assert s in p.forecast_suppress_stats
    assert "turnover" not in p.forecast_suppress_stats and "reb" not in p.forecast_suppress_stats


def test_pages_expose_pending_banner_and_uncertified():
    # generate_web_pages must carry the pending banner and mark forecast uncertified
    assert "VALIDATION_PENDING" in GEN_PAGES
    assert "pending_banner" in GEN_PAGES
    assert "forecast_certified" in GEN_PAGES
    # must NOT restrict to (empty) publish stats in a way that implies certification
    assert "distributions shown as UNCERTIFIED" in GEN_PAGES


# ---- Defect 1: candidate vs publishable separation ----
def test_candidate_publishable_contract_in_builder():
    assert "candidate_edges.parquet" in BUILD_EDGE
    # abstain: candidates written, publishable emptied to zero rows (schema preserved)
    assert "publishable = candidates.iloc[0:0].copy()" in BUILD_EDGE
    # audit reports both counts separately
    assert "candidate_edge_rows" in BUILD_EDGE and "public_recommendation_rows" in BUILD_EDGE


def test_page_generator_consumes_publishable_not_candidate():
    # the page generator's edges input is publishable_edges (per workflows), never candidate_edges
    for wf in ["pregame_initial.yml", "pregame_odds_refresh.yml"]:
        txt = (REPO / ".github/workflows" / wf).read_text()
        assert "publishable_edges.parquet" in txt
        assert "candidate_edges.parquet" not in txt  # candidates never fed to the public page


# ---- Defect 2: release-critical steps fail closed ----
def test_release_critical_steps_fail_closed():
    for wf in ["pregame_odds_refresh.yml", "pregame_injury_update.yml"]:
        txt = (REPO / ".github/workflows" / wf).read_text()
        # the edge build step must NOT be continue-on-error: true
        idx = txt.index("build_edge_report.py")
        # scan the ~15 lines after the run block for the step's continue-on-error
        seg = txt[idx: idx + 900]
        assert "continue-on-error: true" not in seg, f"{wf}: edge build still continue-on-error true"


def test_odds_refresh_ftp_deploy_blocking():
    txt = (REPO / ".github/workflows/pregame_odds_refresh.yml").read_text()
    idx = txt.index("ftp_deploy.py")
    seg = txt[idx: idx + 120]
    assert "continue-on-error: true" not in seg
