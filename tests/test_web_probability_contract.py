"""B1: the public pages' bettor-facing selected-line probability must be the decision-grade
model_prob_over_final (push-safe + binary-calibrated), NEVER recomputed from PMF mass.

Regression uses an INTEGER line with nonzero push mass and a model_prob_over_final that differs
from the unconditional PMF P(X>line) (as non-identity binary calibration + push conditioning
would produce). The page must surface `final` as the bettor-facing prob and expose the raw PMF
masses only under the clearly-labeled *unconditional* fields.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

LANE = Path(__file__).resolve().parent.parent


def _load(script):
    spec = importlib.util.spec_from_file_location(script, LANE / "scripts" / f"{script}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_pmf_payload_uses_final_not_pmf_mass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                     # avoid picking up repo feature parquet
    gwp = _load("generate_web_pages")
    # PMF over an integer line (2): P(X>2)=0.30, P(X=2)=0.30 (push), P(X<2)=0.40 unconditional.
    pmf = json.dumps({"0": 0.25, "1": 0.15, "2": 0.30, "3": 0.20, "4": 0.10})
    proj = pd.DataFrame([{"player_id": "p1", "player_name": "A. Player", "stat": "reb",
                          "pmf_json": pmf, "pmf_mean": 1.85}])
    # Non-identity calibrated, push-safe final that is NOT equal to unconditional P(X>2)=0.30.
    edges = pd.DataFrame([{"player_id": "p1", "stat": "reb", "line": 2.0,
                           "model_prob_over_final": 0.62, "model_prob_push": 0.30,
                           "edge_over": 0.0, "kelly_fraction": 0.0,
                           "market_prob_over_no_vig": 0.5}])
    payload = gwp._build_pmf_json(edges, proj, "2026-07-20")
    row = next(p for p in payload["props"] if p["stat_raw"] == "reb")

    # Bettor-facing prob == final (NOT the unconditional PMF mass).
    assert row["model_p_over"] == 0.62
    assert row["model_prob_over_final"] == 0.62
    assert abs(row["model_p_under"] - 0.38) < 1e-9          # settled Under = 1 - final
    # Unconditional PMF masses exposed SEPARATELY and correctly.
    assert abs(row["pmf_p_over_unconditional"] - 0.30) < 1e-9
    assert abs(row["pmf_p_push"] - 0.30) < 1e-9              # nonzero push mass on the integer line
    assert abs(row["pmf_p_under_unconditional"] - 0.40) < 1e-9
    # The two concepts differ - final must NOT have been overwritten by PMF mass.
    assert row["model_p_over"] != row["pmf_p_over_unconditional"]


def test_distributions_page_consumes_final(tmp_path):
    gdp = _load("generate_distributions_page")
    src = {
        "release_id": "rel1", "game_date": "2026-07-20",
        "props": [{
            "player": "A. Player", "stat": "REB", "stat_raw": "reb", "line": 2.0,
            "pmf": [[0, 0.25], [1, 0.15], [2, 0.30], [3, 0.20], [4, 0.10]],
            "model_prob_over_final": 0.62, "model_p_over": 0.30, "model_p_push": 0.30,
        }],
    }
    pmf_path = tmp_path / "pmf.json"; pmf_path.write_text(json.dumps(src))
    out = gdp._build_json(pmf_path, "2026-07-20", release_id="rel1")
    row = out["props"][0]
    assert row["model_p_over"] == 0.62                      # consumes final, not PMF P(X>line)=0.30
    assert row["model_prob_over_final"] == 0.62
    assert abs(row["pmf_p_over_unconditional"] - 0.30) < 1e-9
    assert abs(row["pmf_p_push"] - 0.30) < 1e-9
