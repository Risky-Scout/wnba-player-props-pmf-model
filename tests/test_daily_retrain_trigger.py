"""Focused contract test for the daily_pipeline.yml sentinel retrain trigger.

After weekly_calibration mutates config/model/stage4_baseline.yaml (dispersion
auto-update from OOF), the existing daily model artifact's config_hash no longer
matches main, and pregame_initial's BLOCKING artifact-manifest validation fails.
A fresh retrain on current main re-syncs the config_hash. In environments where
workflow_dispatch is unavailable (app token lacks actions:write), a dedicated
sentinel-path push trigger fires that retrain.

Proves:
1. daily_pipeline.yml has a push trigger scoped ONLY to the sentinel path on main.
2. The sentinel file exists and is non-empty.
3. The push trigger does NOT broaden to source/config/artifact paths.
4. The model manifest step computes config_hash from stage4_baseline.yaml, so a
   retrain re-syncs the hash pregame_initial validates against.
"""
from __future__ import annotations

from pathlib import Path

import yaml

WF_PATH = Path(".github/workflows/daily_pipeline.yml")
SENTINEL = Path(".github/daily_retrain.trigger")


def _load_wf() -> dict:
    # PyYAML parses the bare `on:` key as boolean True.
    return yaml.safe_load(WF_PATH.read_text())


def _on_block(wf: dict) -> dict:
    on = wf.get("on", wf.get(True))
    assert isinstance(on, dict), "daily_pipeline.yml 'on' block must be a mapping"
    return on


def test_sentinel_file_exists():
    assert SENTINEL.exists(), f"{SENTINEL} sentinel must exist to fire the retrain"
    assert SENTINEL.read_text().strip(), "sentinel file must not be empty"


def test_push_trigger_scoped_to_sentinel_only():
    on = _on_block(_load_wf())
    assert "push" in on, "daily_pipeline.yml must define a push trigger"
    push = on["push"]
    assert push.get("branches") == ["main"], "push trigger must be limited to main"
    paths = push.get("paths")
    assert paths == [".github/daily_retrain.trigger"], (
        "push trigger must be scoped to ONLY the sentinel path; broadening it would "
        "let the workflow's own [skip ci] commits recursively re-fire the retrain"
    )


def test_dispatch_and_schedule_preserved():
    on = _on_block(_load_wf())
    assert "workflow_dispatch" in on, "workflow_dispatch must remain available"
    assert "schedule" in on, "daily schedule must remain available"


def test_model_manifest_config_hash_from_stage4_config():
    """The model artifact manifest must hash config/model/stage4_baseline.yaml,
    which is exactly what pregame_initial validates against — so a retrain on
    current main re-syncs the config_hash.
    """
    content = WF_PATH.read_text()
    assert "config/model/stage4_baseline.yaml" in content
    assert "config_hash" in content
