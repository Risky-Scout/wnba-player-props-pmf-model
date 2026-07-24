"""Foundation Lock regression tests for the market-superiority evaluator.

Locks the evaluator mechanics that every future proof depends on:
selection/proof isolation, deterministic clustered bootstrap, correct sign and metric
directions, push exclusion, minimum-observation and minimum-cluster requirements, Holm
monotonicity, and fail-closed behavior on missing columns.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "scripts" / "evaluate_market_superiority.py"


def _mod():
    spec = importlib.util.spec_from_file_location("ems", EVAL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EMS = _mod()

_COLS = dict(prop_col="prop", candidate_col="candidate", split_col="split",
             date_col="game_date", actual_col="actual", line_col="line",
             model_prob_col="model_prob_over_final", market_prob_col="market_prob_over_no_vig")


def _prep(df):
    return EMS._prepare(df, **_COLS)


def test_prepare_excludes_push_rows():
    df = pd.DataFrame({
        "prop": ["pts"] * 3, "candidate": ["c"] * 3, "split": ["test"] * 3,
        "game_date": ["2025-07-01"] * 3, "actual": [10, 12, 8], "line": [10, 9.5, 9.5],
        "model_prob_over_final": [0.5, 0.6, 0.4], "market_prob_over_no_vig": [0.5, 0.55, 0.45],
    })
    out = _prep(df)
    # The actual==line push row (10 vs 10) is dropped; 2 remain.
    assert len(out) == 2
    assert not (out["actual"] == out["line"]).any()


def test_prepare_fail_closed_on_missing_column():
    df = pd.DataFrame({"prop": ["pts"], "game_date": ["2025-07-01"], "actual": [10],
                       "line": [9.5], "model_prob_over_final": [0.6]})  # market col missing
    with pytest.raises(ValueError):
        _prep(df)


def test_metrics_sign_and_direction():
    # Model closer to truth than market on every row.
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.2, 4000)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    # Market ranking is degraded by independent noise, so the model both calibrates
    # better (log loss/Brier) AND discriminates better (AUC is monotone-invariant, so a
    # sign change in AUC requires a genuinely worse market ranking).
    p_model = 1 / (1 + np.exp(-(0.98 * z)))
    p_market = 1 / (1 + np.exp(-(0.45 * z + rng.normal(0, 0.9, z.size))))
    m = EMS._metrics(y, p_model, p_market)
    assert m["model_logloss"] < m["market_logloss"]
    assert m["model_brier"] < m["market_brier"]
    assert m["model_auc"] > 0.5
    # Delta convention: challenger minus market.
    assert m["logloss_delta"] == pytest.approx(m["model_logloss"] - m["market_logloss"])
    assert m["brier_delta"] == pytest.approx(m["model_brier"] - m["market_brier"])
    assert m["auc_delta"] == pytest.approx(m["model_auc"] - m["market_auc"])
    assert m["logloss_delta"] < 0 and m["brier_delta"] < 0 and m["auc_delta"] > 0


def test_select_candidates_picks_lower_logloss():
    rng = np.random.default_rng(1)
    rows = []
    for day in range(12):
        z = rng.normal(0, 1.2, 60)
        y = rng.binomial(1, 1 / (1 + np.exp(-z)))
        pm = 1 / (1 + np.exp(-(0.45 * z)))
        good = 1 / (1 + np.exp(-(0.95 * z)))
        bad = 1 / (1 + np.exp(-(0.30 * z + rng.normal(0, 0.8, 60))))
        for cand, pred in [("good", good), ("bad", bad)]:
            for j in range(60):
                rows.append({"prop": "pts", "candidate": cand, "split": "selection",
                             "game_date": f"2025-05-{day + 1:02d}", "actual": int(y[j]),
                             "line": 0.5, "model_prob_over_final": float(pred[j]),
                             "market_prob_over_no_vig": float(pm[j])})
    df = _prep(pd.DataFrame(rows))
    point = EMS._point_table(df, prop_col="prop", candidate_col="candidate",
                             date_col="game_date", model_prob_col="model_prob_over_final",
                             market_prob_col="market_prob_over_no_vig")
    _, selected = EMS._select_candidates(point)
    assert selected["pts"] == "good"


def test_bootstrap_requires_two_clusters():
    df = pd.DataFrame({"prop": ["pts"] * 5, "candidate": ["c"] * 5,
                       "game_date": ["2025-07-01"] * 5, "_outcome_over": [1, 0, 1, 0, 1],
                       "model_prob_over_final": [0.6] * 5, "market_prob_over_no_vig": [0.5] * 5})
    with pytest.raises(ValueError):
        EMS._bootstrap_deltas(df, date_col="game_date", model_prob_col="model_prob_over_final",
                              market_prob_col="market_prob_over_no_vig", n_boot=10, seed=0)


def test_bootstrap_is_deterministic_under_seed():
    rng = np.random.default_rng(2)
    n_days, per = 8, 50
    frames = []
    for d in range(n_days):
        z = rng.normal(0, 1, per)
        y = rng.binomial(1, 1 / (1 + np.exp(-z)))
        frames.append(pd.DataFrame({
            "game_date": [f"2025-07-{d + 1:02d}"] * per, "_outcome_over": y,
            "model_prob_over_final": 1 / (1 + np.exp(-(0.9 * z))),
            "market_prob_over_no_vig": 1 / (1 + np.exp(-(0.5 * z)))}))
    g = pd.concat(frames, ignore_index=True)
    kw = dict(date_col="game_date", model_prob_col="model_prob_over_final",
              market_prob_col="market_prob_over_no_vig", n_boot=500, seed=123)
    a = EMS._bootstrap_deltas(g, **kw)
    b = EMS._bootstrap_deltas(g, **kw)
    for k in a:
        assert np.allclose(a[k], b[k], equal_nan=True)


def test_holm_adjust_monotone_nondecreasing():
    p = pd.Series([0.001, 0.02, 0.03, 0.5])
    adj = EMS._holm_adjust(p)
    vals = adj.sort_index().tolist()
    ordered = adj.loc[p.sort_values().index].tolist()
    assert all(ordered[i] <= ordered[i + 1] + 1e-12 for i in range(len(ordered) - 1))
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_prove_insufficient_below_min_rows():
    rng = np.random.default_rng(3)
    rows = []
    for d in range(4):
        z = rng.normal(0, 1, 20)
        y = rng.binomial(1, 1 / (1 + np.exp(-z)))
        for j in range(20):
            rows.append({"prop": "pts", "candidate": "c", "split": "test",
                         "game_date": f"2025-07-{d + 1:02d}", "actual": int(y[j]), "line": 0.5,
                         "model_prob_over_final": float(1 / (1 + np.exp(-(0.9 * z[j])))),
                         "market_prob_over_no_vig": float(1 / (1 + np.exp(-(0.5 * z[j]))))})
    df = _prep(pd.DataFrame(rows))
    res = EMS._prove(df, {"pts": "c"}, prop_col="prop", candidate_col="candidate",
                     date_col="game_date", model_prob_col="model_prob_over_final",
                     market_prob_col="market_prob_over_no_vig", n_boot=200, seed=0,
                     min_rows=300, min_clusters=30, alpha=0.05, min_logloss_delta=0.0,
                     min_brier_delta=0.0, min_auc_delta=0.0)
    assert (res["market_superiority_gate"] == "INSUFFICIENT").all()
    assert (res["proper_score_market_superiority_gate"] == "INSUFFICIENT").all()
    assert (res["strict_market_superiority_gate"] == "INSUFFICIENT").all()


def test_select_precedes_proof_and_no_leakage(tmp_path):
    """End-to-end: selection uses only the selection split; proof scores only test rows,
    and selection dates strictly precede proof dates."""
    rng = np.random.default_rng(4)
    rows = []
    def block(split, start_day, n_days, cands):
        for d in range(n_days):
            z = rng.normal(0, 1.2, 45)
            y = rng.binomial(1, 1 / (1 + np.exp(-z)))
            pm = 1 / (1 + np.exp(-(0.45 * z)))
            preds = {"good": 1 / (1 + np.exp(-(0.97 * z))),
                     "bad": 1 / (1 + np.exp(-(0.30 * z + rng.normal(0, 0.8, 45))))}
            for cand in cands:
                for j in range(45):
                    rows.append({"prop": "pts", "candidate": cand, "split": split,
                                 "game_date": f"2025-{start_day}-{d + 1:02d}", "actual": int(y[j]),
                                 "line": 0.5, "model_prob_over_final": float(preds[cand][j]),
                                 "market_prob_over_no_vig": float(pm[j])})
    block("selection", "05", 10, ["good", "bad"])   # May
    block("test", "07", 12, ["good"])               # July (strictly later)
    src = tmp_path / "scored.csv"
    pd.DataFrame(rows).to_csv(src, index=False)

    out = tmp_path / "out"
    r1 = subprocess.run([sys.executable, str(EVAL), "--mode", "select",
                         "--input", str(src), "--output-dir", str(out)],
                        capture_output=True, text=True, cwd=str(REPO))
    assert r1.returncode == 0, r1.stdout + r1.stderr
    sel = json.loads((out / "selected_candidates.json").read_text())["selected_candidates"]
    assert sel["pts"] == "good"

    # W0.1: prove mode requires a FROZEN split manifest (no automatic splitting).
    split_manifest = out / "split_manifest.json"
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.write_text(json.dumps(
        {"proof_date_min": "2025-07-01", "proof_date_max": "2025-07-31"}))
    r2 = subprocess.run([sys.executable, str(EVAL), "--mode", "prove", "--input", str(src),
                         "--selected-candidates", str(out / "selected_candidates.json"),
                         "--split-manifest", str(split_manifest),
                         "--output-dir", str(out), "--min-rows", "300", "--bootstrap", "600"],
                        capture_output=True, text=True, cwd=str(REPO))
    assert r2.returncode == 0, r2.stdout + r2.stderr
    proof = json.loads((out / "market_superiority_proof.json").read_text())
    res = {r["prop"]: r for r in proof["results"]}["pts"]
    # Proof scored only the 12 July frozen-window dates (never the 10 May selection dates).
    assert res["date_min"].startswith("2025-07")
    assert res["n_clusters"] == 12
    # Selection dates (May) strictly precede proof dates (July).
    assert "2025-05" < res["date_min"]


def test_real_proof_rejects_legacy_model_prob_over(tmp_path):
    # PR 1A: real proof mode must reject a CLI override selecting the legacy column.
    src = tmp_path / "s.csv"
    pd.DataFrame({"prop": ["pts"], "candidate": ["c"], "split": ["test"],
                  "game_date": ["2025-07-01"], "actual": [1], "line": [0.5],
                  "model_prob_over_final": [0.6], "market_prob_over_no_vig": [0.5]}).to_csv(src, index=False)
    r = subprocess.run([sys.executable, str(EVAL), "--mode", "prove", "--input", str(src),
                        "--model-prob-col", "model_prob_over", "--output-dir", str(tmp_path / "o")],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0
    assert "model_prob_over" in (r.stdout + r.stderr) and "forbidden" in (r.stdout + r.stderr)


def test_real_proof_default_column_is_final():
    src = EVAL.read_text()
    assert '--model-prob-col", default="model_prob_over_final"' in src


def test_prove_requires_frozen_split_manifest(tmp_path):
    # W0.1: prove mode must refuse to run without a frozen split manifest (no auto-splitting).
    src = tmp_path / "s.csv"
    pd.DataFrame({"prop": ["pts"], "candidate": ["c"], "split": ["test"],
                  "game_date": ["2025-07-01"], "actual": [1], "line": [0.5],
                  "model_prob_over_final": [0.6], "market_prob_over_no_vig": [0.5]}).to_csv(src, index=False)
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"selected_candidates": {"pts": "c"}}))
    r = subprocess.run([sys.executable, str(EVAL), "--mode", "prove", "--input", str(src),
                        "--selected-candidates", str(sel), "--output-dir", str(tmp_path / "o")],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode != 0
    assert "split-manifest" in (r.stdout + r.stderr)


def test_prove_emits_both_gates_and_cluster_floor():
    # Two separate gates exist; min_clusters hard floor is 30 in prove mode.
    src = EVAL.read_text()
    assert "proper_score_market_superiority_gate" in src
    assert "strict_market_superiority_gate" in src
    assert "max(30, int(args.min_clusters))" in src
