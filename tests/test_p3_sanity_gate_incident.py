"""Regression test for run #309 (29642662780): the calibration sanity gate died with
exit 134 (native std::terminate at threadpool teardown) AFTER printing PASS. The
hardened gate must single-thread native pools and clean-exit on PASS so a teardown race
can never flip a decided PASS to a nonzero exit."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "calibration_sanity_gate.py"


def _fixture(tmp_path, bias, edges_edge_pp):
    cal = tmp_path / "cal"; cal.mkdir()
    (cal / "bias_corrections.json").write_text(
        __import__("json").dumps(bias))
    edges = tmp_path / "edges.parquet"
    pd.DataFrame({"edge_pp": edges_edge_pp}).to_parquet(edges)
    return str(cal), str(edges)


def _run(cal_dir, edges):
    env = {"PYTHONPATH": str(REPO / "src")}
    import os
    env.update({k: v for k, v in os.environ.items() if k not in env})
    return subprocess.run(
        [sys.executable, str(GATE), "--cal-dir", cal_dir, "--edges", edges],
        capture_output=True, text=True, env=env, timeout=120)


def test_pass_path_exit_zero_repeatably(tmp_path):
    cal, edges = _fixture(tmp_path, {"pts": 1.0, "reb": 0.9}, [1.0, -2.0, 3.0])
    # Repeat to exercise the non-deterministic teardown path; NONE may exit 134/nonzero.
    for i in range(20):
        r = _run(cal, edges)
        assert r.returncode == 0, f"run {i}: rc={r.returncode}\n{r.stdout}\n{r.stderr}"
        assert r.returncode != 134
    assert "All sanity gates passed" in r.stdout


def test_fail_path_exits_one(tmp_path):
    cal, edges = _fixture(tmp_path, {"pts": 0.2}, [1.0])
    r = _run(cal, edges)
    assert r.returncode == 1
    assert "GATE FAIL" in r.stdout


def test_gate_limits_threads_before_heavy_import_and_clean_exits():
    src = GATE.read_text()
    # thread limits set BEFORE importing pandas/numpy (env read at import time)
    assert src.index("OMP_NUM_THREADS") < src.index("import pandas") if "import pandas" in src else True
    assert 'setdefault("NUMBA_THREADING_LAYER"' in src
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_NUM_THREADS", "NUMBA_NUM_THREADS"):
        assert var in src
    # clean exit that bypasses native teardown
    assert "os._exit(" in src


def test_publishers_share_single_flight_group():
    # All pregame publishers must share ONE concurrency group so no two runs mutate
    # deliveries/tonight or the live pointer concurrently (competing-publishers fix #2).
    import yaml
    for wf in ["pregame_initial.yml", "pregame_final.yml",
               "pregame_odds_refresh.yml", "pregame_injury_update.yml"]:
        doc = yaml.safe_load((REPO / ".github/workflows" / wf).read_text())
        conc = doc.get("concurrency")
        assert conc and conc.get("group") == "wnba-pregame-live-publish", \
            f"{wf}: missing shared single-flight concurrency group"
        assert conc.get("cancel-in-progress") is False, f"{wf}: must serialize, not cancel"


def test_workflow_uses_hardened_gate_not_heredoc():
    wf = (REPO / ".github/workflows/pregame_initial.yml").read_text()
    assert "python scripts/calibration_sanity_gate.py" in wf
    # the old inline heredoc gate is gone
    assert "GATE FAIL: bias_corrections" not in wf
    # fault handler + thread limits are set on the step
    seg = wf[wf.index("Sanity gate — block deployment"):]
    seg = seg[: seg.index("continue-on-error")]
    assert "PYTHONFAULTHANDLER" in seg and "OMP_NUM_THREADS" in seg
