"""P2 Phase 1 — blocking tests for the 10 hardened release-contract conditions
and the canonical policy loader."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wnba_props_model.pipeline.policy import load_policy, PolicyError

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "config" / "recommendation_policy.yaml"
PREGAME_INIT = (REPO / ".github/workflows/pregame_initial.yml").read_text()
PREGAME_FINAL = (REPO / ".github/workflows/pregame_final.yml").read_text()
DAILY = (REPO / ".github/workflows/daily_pipeline.yml").read_text()


# ---- canonical policy ----
def test_policy_loads_and_is_shared():
    p = load_policy(POLICY)
    assert p.version >= 1
    # single source of truth referenced by BOTH the builder and the web page generator
    assert "--policy" in (REPO / "scripts/build_edge_report.py").read_text()
    assert "load_policy" in (REPO / "scripts/build_edge_report.py").read_text()
    assert "load_policy" in (REPO / "scripts/generate_web_pages.py").read_text()


# Condition 4: nonzero threshold selected on dev folds, frozen before holdout.
def test_condition4_nonzero_threshold():
    p = load_policy(POLICY)
    assert p.edge_threshold > 0.0

def test_policy_rejects_zero_threshold(tmp_path):
    doc = yaml.safe_load(POLICY.read_text())
    doc["edge"]["edge_threshold"] = 0.0
    bad = tmp_path / "bad.yaml"; bad.write_text(yaml.safe_dump(doc))
    with pytest.raises(PolicyError):
        load_policy(bad)


# Condition 1: production passes the current-run slate manifest explicitly.
def test_condition1_slate_manifest_passed():
    assert "--slate-manifest deliveries/tonight/slate_manifest.json" in PREGAME_INIT
    assert "--slate-manifest deliveries/tonight/slate_manifest.json" in PREGAME_FINAL


# Condition 2: production requires the calibrated probability layer explicitly.
def test_condition2_calibration_required():
    p = load_policy(POLICY)
    assert p.required_calibration  # non-empty
    # the builder wires the policy (which declares required_calibration)
    assert "--policy config/recommendation_policy.yaml" in PREGAME_INIT


# Condition 3: fail if calibration missing / zero eligible (require-venn-abers path exits nonzero).
def test_condition3_calibration_failclosed_exists():
    src = (REPO / "scripts/build_edge_report.py").read_text()
    assert "require_venn_abers" in src and "require-venn-abers" in src


# Condition 5 + 6: the AUTHORITATIVE edge build is blocking (no continue-on-error true);
# publication is prohibited if it fails.
def test_condition5_edge_build_blocking():
    # pregame_initial: the "Build edge report (BLOCKING)" step must be continue-on-error: false
    idx = PREGAME_INIT.index("Build edge report (BLOCKING)")
    # the step body ends at the next step's continue-on-error; check up to the next step
    seg = PREGAME_INIT[idx: idx + 1600]
    assert "continue-on-error: false" in seg
    # daily preview edge build must NOT be marked continue-on-error: true after hardening
    didx = DAILY.index("Build edge report")
    assert "continue-on-error: true" not in DAILY[didx: didx + 500]


# Condition 7 + 10: no reuse of stale edge artifacts; never publish a stale board.
def test_condition7_no_stale_edge_fallback():
    # the archive fallback that copied a stale publishable_edges as a PMF archive is removed
    assert "Archived publishable edges as PMF archive" not in DAILY


# Condition 8: run id / commit / date / manifest verified in the publishing workflow.
def test_condition8_lineage_verified():
    assert "workflow_run_artifact" in PREGAME_INIT
    assert "slate_manifest" in PREGAME_INIT


# Condition 9: abstaining board is published when nothing qualifies.
def test_condition9_abstain_board(tmp_path):
    p = load_policy(POLICY)
    assert p.abstain  # current status is forecast-only -> abstain
    # build_edge_report writes an explicit empty board + abstain audit under abstain mode
    src = (REPO / "scripts/build_edge_report.py").read_text()
    assert "_policy_abstain" in src and "abstain_reason" in src
    # generate_web_pages surfaces the abstention (no unvalidated picks, no profit claim)
    gsrc = (REPO / "scripts/generate_web_pages.py").read_text()
    assert "No validated betting edges currently qualify" in gsrc
    assert "no profitability" in gsrc.lower() or "no_profit" in gsrc.lower()


# Policy suppression semantics
def test_side_and_stat_suppression_applied():
    p = load_policy(POLICY)
    assert "over" in [s.lower() for s in p.suppress_sides]  # Overs diagnosed harmful (Phase 5)
    src = (REPO / "scripts/build_edge_report.py").read_text()
    assert "_policy_suppress_sides" in src and "_policy_suppress_stats" in src


# P3 Phase 1: production must fail closed without the canonical policy.
def test_production_requires_policy_flag():
    src = (REPO / "scripts/build_edge_report.py").read_text()
    assert "--require-policy" in src and "require_policy" in src
    assert "policy_required_but_missing" in src
    # every production edge build passes --require-policy
    for wf in ["pregame_initial.yml", "pregame_final.yml", "pregame_odds_refresh.yml",
               "pregame_injury_update.yml", "daily_pipeline.yml"]:
        assert "--require-policy" in (REPO / ".github/workflows" / wf).read_text(), \
            f"{wf}: production edge build must pass --require-policy"


def test_invalid_policy_fails_closed(tmp_path):
    from wnba_props_model.pipeline.policy import load_policy, PolicyError
    bad = tmp_path / "missing.yaml"
    with pytest.raises((PolicyError, FileNotFoundError, OSError)):
        load_policy(bad)


def test_preregistration_doc_exists():
    doc = (REPO / "docs/p3_forecasting_gates_preregistration.md")
    assert doc.exists()
    txt = doc.read_text()
    # forbids the invalid gate and mandates two-sided coverage + line-level calibration
    assert "midpoint-PIT" in txt and "two-sided coverage" in txt.lower()
    assert "randomized PIT" in txt


# Every workflow that PUBLISHES the Edge board to the custom domain must apply the
# canonical policy on EVERY run, or a mid-day refresh could overwrite the abstaining
# board with an unvalidated threshold-0 board.
def test_all_publishers_apply_policy():
    publishers = [
        "pregame_initial.yml", "pregame_final.yml",
        "pregame_odds_refresh.yml", "pregame_injury_update.yml",
    ]
    for wf in publishers:
        txt = (REPO / ".github/workflows" / wf).read_text()
        assert "build_edge_report.py" in txt, f"{wf}: no edge build"
        assert "--policy config/recommendation_policy.yaml" in txt, (
            f"{wf} builds edges without --policy — would bypass forecast-only abstention")


def test_publishers_run_on_daily_schedule():
    # Each custom-domain publisher must be scheduled (cron) so the board refreshes daily.
    for wf in ["pregame_initial.yml", "pregame_final.yml", "pregame_odds_refresh.yml",
               "pregame_injury_update.yml"]:
        txt = (REPO / ".github/workflows" / wf).read_text()
        assert "cron:" in txt, f"{wf} has no schedule — board would not refresh daily"


# Forecast gate outcome is encoded and total stats reconcile.
def test_forecast_suppression_encoded():
    # Under VALIDATION_PENDING nothing is certified/published; all seven stats are
    # suppressed from certified publication until the corrected gate runs.
    p = load_policy(POLICY)
    # seven markets certified (LIVE_VALIDATED_FORECAST_ONLY); reb now passes via Candidate D.
    # fg3m/blk + reb-independent combos (pts_reb, pts_reb_ast) remain suppressed.
    assert p.forecast_status == "LIVE_VALIDATED_FORECAST_ONLY"
    assert set(p.forecast_publish_stats) == {"turnover", "pts", "ast", "stl", "stocks", "pts_ast", "reb"}
    assert {"fg3m", "blk", "pts_reb", "pts_reb_ast"} <= set(p.forecast_suppress_stats)
