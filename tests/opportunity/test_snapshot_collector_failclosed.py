"""Forward snapshot collector must FAIL CLOSED (owner directive item 2).

Script-level tests: the collector exits non-zero on missing injuries / zero rows when games are
scheduled, always writes a coverage manifest, and exits 0 with a real snapshot when injuries exist.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "build_opportunity_snapshots.py"


def _run(args, env=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO / "src")
    if env:
        e.update(env)
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=e)


def _manifest(p):
    return json.loads(Path(p).read_text())


def test_missing_injuries_with_games_fails_closed(tmp_path):
    man = tmp_path / "cov.json"
    r = _run(["--injuries", str(tmp_path / "nope.parquet"),
              "--availability-root", str(tmp_path / "avail"),
              "--coverage-manifest", str(man), "--phase", "final",
              "--game-date", "2026-07-28", "--scheduled-games", "1"])
    assert r.returncode == 1, r.stdout + r.stderr
    assert _manifest(man)["failure_status"] == "FAIL_missing_injuries_response"


def test_missing_injuries_no_games_is_ok(tmp_path):
    man = tmp_path / "cov.json"
    r = _run(["--injuries", str(tmp_path / "nope.parquet"),
              "--availability-root", str(tmp_path / "avail"),
              "--coverage-manifest", str(man), "--scheduled-games", "0"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert _manifest(man)["failure_status"] == "ok_no_games_missing_injuries"


def test_zero_injury_rows_with_games_fails_closed(tmp_path):
    empty = tmp_path / "inj.parquet"
    pd.DataFrame({"player_id": [], "team_id": [], "injury_status": []}).to_parquet(empty)
    man = tmp_path / "cov.json"
    r = _run(["--injuries", str(empty), "--availability-root", str(tmp_path / "avail"),
              "--coverage-manifest", str(man), "--scheduled-games", "5"])
    assert r.returncode == 1, r.stdout + r.stderr
    assert _manifest(man)["failure_status"] == "FAIL_zero_injury_rows"


def test_valid_injuries_produces_snapshot_and_manifest(tmp_path):
    inj = tmp_path / "inj.parquet"
    pd.DataFrame({
        "player_id": [101, 202, 303], "team_id": [1, 1, 2],
        "injury_status": ["out", "questionable", "available"],
        "injury_description": ["knee", "ankle", ""],
    }).to_parquet(inj)
    man = tmp_path / "cov.json"
    r = _run(["--injuries", str(inj), "--availability-root", str(tmp_path / "avail"),
              "--coverage-manifest", str(man), "--phase", "initial",
              "--game-date", "2026-07-28", "--scheduled-games", "3"])
    assert r.returncode == 0, r.stdout + r.stderr
    m = _manifest(man)
    assert m["failure_status"] == "ok"
    assert m["injury_rows"] == 3 and m["availability_rows"] == 3 and m["players"] == 3
    assert m["payload_hash_count"] == 3 and m["prediction_cutoff"]
    # all required manifest keys present
    for k in ("phase", "game_date", "games", "players", "injury_rows", "availability_rows",
              "prediction_cutoff", "source_timestamp", "payload_hashes", "failure_status"):
        assert k in m, f"coverage manifest missing {k}"
