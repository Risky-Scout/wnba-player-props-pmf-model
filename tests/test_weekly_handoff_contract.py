"""Contract tests for the weekly -> daily -> publish hand-off.

The weekly calibration job commits a config/model/stage4_baseline.yaml change
(dispersion auto-update) that invalidates the existing model artifact's
config_hash. If the weekly job published directly, pregame_initial's BLOCKING
manifest validation would fail until the next daily retrain — leaving the live
pages stale for up to a day.

Fix: weekly hands off to daily_pipeline (which retrains + re-syncs the
config_hash), and daily then triggers the publish. These tests lock that wiring
so the direct weekly->publish path can never be reintroduced accidentally.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

WF_DIR = Path(".github/workflows")
WEEKLY = WF_DIR / "weekly_calibration.yml"
PREGAME = WF_DIR / "pregame_initial.yml"
DAILY = WF_DIR / "daily_pipeline.yml"


def _on_block(path: Path) -> dict:
    wf = yaml.safe_load(path.read_text())
    on = wf.get("on", wf.get(True))
    assert isinstance(on, dict), f"{path} 'on' block must be a mapping"
    return on


# ── weekly hands off to daily, not to the publisher ───────────────────────────

def test_weekly_triggers_daily_not_pregame():
    content = WEEKLY.read_text()
    assert "gh workflow run daily_pipeline.yml" in content, (
        "weekly_calibration.yml must hand off to daily_pipeline.yml so the model is "
        "retrained (config_hash re-synced) before publishing"
    )
    assert "gh workflow run pregame_initial.yml" not in content, (
        "weekly_calibration.yml must NOT trigger pregame_initial.yml directly — that "
        "publishes before the retrain and blocks/stales the pages"
    )


# ── publisher no longer auto-fires on weekly completion ───────────────────────

def test_pregame_workflow_run_excludes_weekly():
    on = _on_block(PREGAME)
    wr = on.get("workflow_run", {})
    workflows = wr.get("workflows", [])
    assert "Daily WNBA PMF Pipeline" in workflows, (
        "pregame_initial.yml must still auto-trigger on the daily pipeline"
    )
    assert "Weekly OOF Refresh & Calibration" not in workflows, (
        "pregame_initial.yml must NOT auto-trigger on weekly calibration completion; "
        "the weekly job hands off via daily_pipeline instead"
    )


# ── daily still triggers the publish (so the chain completes) ─────────────────

def test_daily_triggers_pregame():
    content = DAILY.read_text()
    assert "gh workflow run pregame_initial.yml" in content, (
        "daily_pipeline.yml must trigger pregame_initial.yml so the publish still "
        "happens after the retrain (weekly -> daily -> publish)"
    )


# ── no trigger loop: neither daily nor pregame triggers weekly ────────────────

def test_no_trigger_loop_back_to_weekly():
    for wf in (DAILY, PREGAME):
        content = wf.read_text()
        assert "gh workflow run weekly_calibration.yml" not in content, (
            f"{wf} must not trigger weekly_calibration.yml (would create a loop)"
        )
    # pregame must not listen for weekly completion either (covered above), and
    # daily's workflow_run (if any) must not include the weekly workflow name.
    daily_on = _on_block(DAILY)
    daily_wr = daily_on.get("workflow_run", {}) or {}
    assert "Weekly OOF Refresh & Calibration" not in (daily_wr.get("workflows", []) or []), (
        "daily_pipeline.yml must not be workflow_run-triggered by weekly calibration"
    )
