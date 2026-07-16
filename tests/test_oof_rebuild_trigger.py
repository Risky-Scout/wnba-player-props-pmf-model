"""Focused contract test for the weekly_calibration.yml sentinel push trigger.

Option-2 execution requires firing a full OOF rebuild in environments where
workflow_dispatch is unavailable (app token lacks actions:write). A dedicated
sentinel path push trigger provides that, without letting routine pushes or the
workflow's own [skip ci] calibrator commits re-fire the 6h job.

Proves:
1. weekly_calibration.yml has a push trigger scoped ONLY to the sentinel path
   on main.
2. The sentinel file exists.
3. The push trigger does NOT broaden to source/config/artifact paths (which
   would recursively re-fire on the workflow's own calibrator commits).
"""
from __future__ import annotations

from pathlib import Path

import yaml

WF_PATH = Path(".github/workflows/weekly_calibration.yml")
SENTINEL = Path(".github/oof_rebuild.trigger")


def _load_wf() -> dict:
    # PyYAML parses the bare `on:` key as boolean True.
    return yaml.safe_load(WF_PATH.read_text())


def _on_block(wf: dict) -> dict:
    on = wf.get("on", wf.get(True))
    assert isinstance(on, dict), "weekly_calibration.yml 'on' block must be a mapping"
    return on


def test_sentinel_file_exists():
    assert SENTINEL.exists(), f"{SENTINEL} sentinel must exist to fire the rebuild"
    assert SENTINEL.read_text().strip(), "sentinel file must not be empty"


def test_push_trigger_scoped_to_sentinel_only():
    on = _on_block(_load_wf())
    assert "push" in on, "weekly_calibration.yml must define a push trigger"
    push = on["push"]
    assert push.get("branches") == ["main"], "push trigger must be limited to main"
    paths = push.get("paths")
    assert paths == [".github/oof_rebuild.trigger"], (
        "push trigger must be scoped to ONLY the sentinel path; broadening it would "
        "let the workflow's own calibrator commits recursively re-fire the rebuild"
    )


def test_dispatch_and_schedule_preserved():
    on = _on_block(_load_wf())
    assert "workflow_dispatch" in on, "workflow_dispatch must remain available"
    assert "schedule" in on, "weekly schedule must remain available"


def test_oof_rebuild_runs_on_push_event():
    """On a push event github.event.inputs.skip_oof_build is empty, so the
    rebuild gate `skip_oof_build != 'true'` is truthy and the OOF rebuild runs.
    Assert the rebuild step keeps that condition rather than requiring dispatch.
    """
    wf = _load_wf()
    steps = wf["jobs"]["weekly-calibration"]["steps"]
    build = next(s for s in steps if s.get("name") == "Build OOF PMFs (Stage 5)")
    cond = build.get("if", "")
    assert "skip_oof_build != 'true'" in cond, (
        "Build OOF PMFs must run when skip_oof_build is not 'true' (true on push)"
    )
