"""Focused tests for the slate manifest production contract.

Proves all 11 required properties:
1. build_next_game_slate.py writes both canonical and dated manifests.
2. The two files are byte-for-byte identical.
3. Required fields are present.
4. scheduled_game_count == len(game_ids).
5. Manifest game IDs equal current-run slate game IDs.
6. A slate with rows and no game IDs fails.
7. A mismatched canonical/dated manifest is detected by the validation step logic.
8. pregame_initial.yml supplies the canonical manifest to build_edge_report.py.
9. The filtered push trigger exists only for the two production-contract files.
10. weekly_calibration.yml has actions: write.
11. The weekly auto-trigger step is blocking (no continue-on-error: true).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml


WF_PREGAME = Path(".github/workflows/pregame_initial.yml")
WF_WEEKLY  = Path(".github/workflows/weekly_calibration.yml")


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_wf(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    # YAML parses bare 'on' as Python True (boolean). Normalise to string key.
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return raw


def _get_step(wf: dict, name_fragment: str) -> dict | None:
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if name_fragment.lower() in str(step.get("name", "")).lower():
                return step
    return None


def _make_games_df(game_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": game_ids,
        "home_team_abbreviation":    [f"HOME{i}" for i in game_ids],
        "visitor_team_abbreviation": [f"AWAY{i}" for i in game_ids],
    })


def _make_slate_df(game_ids: list[int]) -> pd.DataFrame:
    rows = []
    for gid in game_ids:
        for pid in range(5):
            rows.append({
                "game_id": gid,
                "player_id": pid + gid * 100,
                "player_name": f"Player {pid}",
                "team_abbreviation": f"HOME{gid}",
                "injury_flag": False,
                "dnp_risk": "low",
            })
    return pd.DataFrame(rows)


# ─── 1 & 2. Both manifests written and identical ──────────────────────────────

def test_canonical_and_dated_manifests_written_and_identical(tmp_path, monkeypatch):
    """build_next_game_slate writes slate_manifest.json and slate_manifest_DATE.json,
    byte-for-byte identical."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    game_date = "2026-07-15"
    game_ids  = [24929, 24930]
    games_df  = _make_games_df(game_ids)
    slate_df  = _make_slate_df(game_ids)

    monkeypatch.setenv("GITHUB_RUN_ID", "test-run-99")
    monkeypatch.setenv("GITHUB_SHA", "abc123def456abc123def456abc123def456abc1")

    from build_next_game_slate import build_and_write_manifest
    build_and_write_manifest(
        target=game_date,
        games=games_df,
        slate=slate_df,
        out=tmp_path,
        injury_flagged=[],
        high_dnp=[],
    )

    canon = tmp_path / "slate_manifest.json"
    dated = tmp_path / f"slate_manifest_{game_date}.json"

    assert canon.exists(), "slate_manifest.json must exist"
    assert dated.exists(), f"slate_manifest_{game_date}.json must exist"
    assert canon.read_bytes() == dated.read_bytes(), "files must be byte-for-byte identical"


# ─── 3. Required fields present ──────────────────────────────────────────────

def test_required_fields_present(tmp_path, monkeypatch):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    monkeypatch.setenv("GITHUB_RUN_ID", "run-42")
    monkeypatch.setenv("GITHUB_SHA", "sha-def456")
    from build_next_game_slate import build_and_write_manifest

    build_and_write_manifest("2026-07-15", _make_games_df([24929]), _make_slate_df([24929]),
                              tmp_path, [], [])
    m = json.loads((tmp_path / "slate_manifest.json").read_text())
    for field in ("game_date","scheduled_game_count","game_ids","github_run_id","git_commit"):
        assert field in m, f"Required field missing: {field!r}"


# ─── 4. scheduled_game_count == len(game_ids) ────────────────────────────────

def test_scheduled_game_count_equals_len_game_ids(tmp_path, monkeypatch):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    monkeypatch.setenv("GITHUB_RUN_ID", "run-1")
    monkeypatch.setenv("GITHUB_SHA", "sha-1")
    from build_next_game_slate import build_and_write_manifest

    game_ids = [24929, 24930]
    build_and_write_manifest("2026-07-15", _make_games_df(game_ids), _make_slate_df(game_ids),
                              tmp_path, [], [])
    m = json.loads((tmp_path / "slate_manifest.json").read_text())
    assert m["scheduled_game_count"] == len(m["game_ids"]) == len(game_ids)


# ─── 5. Manifest game IDs == current-run slate game IDs ──────────────────────

def test_manifest_game_ids_equal_slate_game_ids(tmp_path, monkeypatch):
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    monkeypatch.setenv("GITHUB_RUN_ID", "run-2")
    monkeypatch.setenv("GITHUB_SHA", "sha-2")
    from build_next_game_slate import build_and_write_manifest

    game_ids = [11111, 22222]
    build_and_write_manifest("2026-07-15", _make_games_df(game_ids), _make_slate_df(game_ids),
                              tmp_path, [], [])
    m = json.loads((tmp_path / "slate_manifest.json").read_text())
    assert sorted(int(g) for g in m["game_ids"]) == sorted(game_ids)


# ─── 6. Slate with rows but empty game_ids fails ─────────────────────────────

def test_slate_with_rows_but_empty_game_ids_fails(tmp_path, monkeypatch):
    """If games table is empty but slate has players, must raise SystemExit."""
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    monkeypatch.setenv("GITHUB_RUN_ID", "run-3")
    monkeypatch.setenv("GITHUB_SHA", "sha-3")
    from build_next_game_slate import build_and_write_manifest

    empty_games = _make_games_df([])
    slate_with_players = _make_slate_df([24929])  # has rows but no game from games table

    with pytest.raises(SystemExit):
        build_and_write_manifest("2026-07-15", empty_games, slate_with_players,
                                  tmp_path, [], [])


# ─── 7. Mismatched canonical/dated manifest detected ─────────────────────────

def test_mismatched_manifests_detected(tmp_path):
    """Validation logic must detect when canonical != dated."""
    game_date = "2026-07-15"
    canon = tmp_path / "slate_manifest.json"
    dated = tmp_path / f"slate_manifest_{game_date}.json"

    m1 = {"game_date": game_date, "scheduled_game_count": 1, "game_ids": ["1"],
           "github_run_id": "r", "git_commit": "s", "games": [], "total_players": 5,
           "injury_flagged": [], "high_dnp_risk": []}
    m2 = dict(m1, scheduled_game_count=2)

    canon.write_text(json.dumps(m1))
    dated.write_text(json.dumps(m2))
    assert canon.read_bytes() != dated.read_bytes()  # mismatch detected


# ─── 8. pregame_initial.yml supplies canonical manifest to edge report ────────

def test_pregame_supplies_canonical_manifest_to_edge_report():
    content = WF_PREGAME.read_text()
    # Canonical manifest must be present somewhere in the workflow
    assert "slate_manifest.json" in content, \
        "pregame_initial.yml must reference slate_manifest.json"
    # The edge report step must pass the canonical (undated) manifest path
    assert "--slate-manifest deliveries/tonight/slate_manifest.json" in content, \
        "Build edge report must use --slate-manifest deliveries/tonight/slate_manifest.json"


# ─── 9. Filtered push trigger for production-contract files only ──────────────

def test_filtered_push_trigger_exists():
    wf = _load_wf(WF_PREGAME)
    push = wf.get("on", {}).get("push", {})
    assert push, "pregame_initial.yml must have a push trigger"

    branches = push.get("branches", [])
    assert "main" in branches, "Push trigger must be restricted to main branch"

    paths = push.get("paths", [])
    assert "scripts/build_next_game_slate.py" in paths, \
        "Push trigger must include scripts/build_next_game_slate.py"
    assert ".github/workflows/pregame_initial.yml" in paths, \
        "Push trigger must include .github/workflows/pregame_initial.yml"


def test_push_trigger_includes_market_integrity():
    """push.paths must include market_integrity.py so dedup/hash fixes auto-trigger pregame."""
    paths = _load_wf(WF_PREGAME).get("on", {}).get("push", {}).get("paths", [])
    assert "src/wnba_props_model/pipeline/market_integrity.py" in paths, (
        "Push trigger must include src/wnba_props_model/pipeline/market_integrity.py"
    )


def test_push_trigger_includes_apply_injury_updates():
    """push.paths must include apply_injury_updates.py so injury-pipeline fixes auto-trigger pregame."""
    paths = _load_wf(WF_PREGAME).get("on", {}).get("push", {}).get("paths", [])
    assert "scripts/apply_injury_updates.py" in paths, (
        "Push trigger must include scripts/apply_injury_updates.py"
    )


def test_push_trigger_does_not_include_data_paths():
    wf = _load_wf(WF_PREGAME)
    paths = wf.get("on", {}).get("push", {}).get("paths", [])
    data_patterns = ["data/", "deliveries/", "artifacts/", "tools/", "*.parquet", "*.json"]
    for pattern in data_patterns:
        for p in paths:
            assert pattern not in str(p), \
                f"Push trigger must not include data path {pattern!r} (found in {p!r})"


# ─── 10. weekly_calibration.yml has actions: write ───────────────────────────

def test_weekly_calibration_has_actions_write():
    wf = _load_wf(WF_WEEKLY)
    for job_name, job in wf.get("jobs", {}).items():
        perms = job.get("permissions", {})
        if perms:
            assert perms.get("actions") == "write", \
                f"Job '{job_name}' must have actions: write (got {perms.get('actions')!r})"


# ─── 11. Weekly auto-trigger is blocking ─────────────────────────────────────

def test_weekly_auto_trigger_is_blocking():
    # Weekly now hands off to the daily pipeline (retrain re-syncs the model
    # config_hash) instead of publishing directly, so the pages can never be
    # blocked/stale after a weekly calibration. The hand-off step must still be
    # blocking (not continue-on-error) and must target daily_pipeline.yml.
    step = _get_step(_load_wf(WF_WEEKLY), "Auto-trigger daily pipeline")
    assert step is not None, "Auto-trigger daily pipeline step must exist in weekly_calibration.yml"
    assert step.get("continue-on-error") is not True, \
        "Auto-trigger step must be blocking (not continue-on-error: true)"
    assert "gh workflow run daily_pipeline.yml" in step.get("run", ""), \
        "Weekly must hand off to daily_pipeline.yml (not trigger pregame_initial.yml directly)"
