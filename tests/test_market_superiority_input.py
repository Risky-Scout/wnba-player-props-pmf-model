"""Tests for the market-superiority input assembler (archive + OOF -> evaluator input)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_market_superiority_input", ROOT / "scripts" / "build_market_superiority_input.py")
msi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msi)

CONTRACT_COLS = ["game_date", "prop", "candidate", "split", "actual", "line",
                 "model_prob_over_final", "market_prob_over_no_vig"]


def test_pmf_array_decodes_list_dict_and_pairs():
    assert np.allclose(msi._pmf_array([0.2, 0.8]), [0.2, 0.8])
    assert np.allclose(msi._pmf_array(json.dumps([0.5, 0.5])), [0.5, 0.5])
    assert np.allclose(msi._pmf_array({"2": 0.3, "0": 0.7}), [0.7, 0.0, 0.3])


def test_chronological_split_respects_cutoff():
    dates = pd.Series(["2024-05-01", "2024-05-10", "2024-05-20"])
    labels = msi._chronological_split(dates, "2024-05-15", 0.4)
    assert list(labels) == ["selection", "selection", "test"]


def _make_inputs(tmp: Path, n_games=8, players=6):
    from scipy.stats import poisson
    rng = np.random.default_rng(0)
    cc_rows, oof_rows = [], []
    base = pd.Timestamp("2024-05-01")
    for g in range(n_games):
        gd = (base + pd.Timedelta(days=g)).strftime("%Y-%m-%d")
        for p in range(players):
            line = float(rng.integers(8, 20)) + 0.5
            cc_rows.append({"game_id": str(1000 + g), "player_id": str(p), "stat": "pts",
                            "line": line, "market_prob_over_no_vig": float(rng.uniform(0.35, 0.65)),
                            "commence_time": f"{gd}T23:00:00Z"})
            lam = line
            xs = np.arange(0, int(2 * line + 6))
            pmf = poisson.pmf(xs, lam); pmf = pmf / pmf.sum()
            oof_rows.append({"game_id": str(1000 + g), "player_id": str(p), "stat": "pts",
                             "actual_outcome": float(poisson.rvs(lam, random_state=rng)),
                             "pmf_json": json.dumps(pmf.round(6).tolist()), "game_date": gd})
    cc_p, oof_p = tmp / "cc.parquet", tmp / "oof.parquet"
    pd.DataFrame(cc_rows).to_parquet(cc_p)
    pd.DataFrame(oof_rows).to_parquet(oof_p)
    return cc_p, oof_p


def test_assembler_emits_evaluator_contract(tmp_path):
    cc_p, oof_p = _make_inputs(tmp_path)
    out_p = tmp_path / "msi.parquet"
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_market_superiority_input.py"),
         "--closing", str(cc_p), "--oof", str(oof_p), "--out", str(out_p),
         "--split-date", "2024-05-06"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    df = pd.read_parquet(out_p)
    assert list(df.columns) == CONTRACT_COLS
    assert set(df["split"].unique()) <= {"selection", "test"}
    assert {"selection", "test"} == set(df["split"].unique())
    assert df["model_prob_over_final"].between(0, 1).all()
    assert df["market_prob_over_no_vig"].between(0, 1).all()
    assert (df["candidate"] == "production").all()


def test_assembler_fails_on_no_overlap(tmp_path):
    cc_p, oof_p = _make_inputs(tmp_path)
    # Break overlap: shift OOF game_ids so no key matches (keeping keys unique).
    oof = pd.read_parquet(oof_p)
    oof["game_id"] = "9" + oof["game_id"].astype(str)
    oof.to_parquet(oof_p)
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_market_superiority_input.py"),
         "--closing", str(cc_p), "--oof", str(oof_p), "--out", str(tmp_path / "x.parquet")],
        capture_output=True, text=True)
    assert res.returncode == 1
    assert "no (game,player,stat) overlap" in res.stderr
