"""Pure OOF wiring coverage (STEP 3/4/6-9): the walk-forward OOF is regenerated on the
pure_forecast contract and routed through the repaired active-PMF lineage, and the CI evaluation
settles the sportsbook probability from the ACTIVE PMF (never the invalid ÷(1-p_dnp) shortcut).

Covers:
  * config/model/stage5_oof.yaml passes the fail-closed pure guard;
  * generate_fold_pmfs persists active_pmf_json / availability_mixture_pmf_json / p_dnp and the
    mixture mean never exceeds the active mean (DNP mass folds onto 0);
  * build_oof_pmfs re-enforces the pure guard and refuses a market-tainted config;
  * evaluate_pure_oof settles P(over) from the active PMF, its pure-provenance check is correct
    at zero weight, and the nested-CV metrics run;
  * a full synthetic build_oof_pmfs -> evaluate_pure_oof dry-run produces the active-PMF lineage
    columns and the authoritative metrics artifact.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from wnba_props_model.models.pure_model_contract import (  # noqa: E402
    MarketLeakageError,
    assert_pure_model_config,
    is_pure_model,
)
from wnba_props_model.models.training import FoldModel, generate_fold_pmfs  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# A. stage5_oof.yaml is pure
# ---------------------------------------------------------------------------

def test_stage5_oof_config_is_pure():
    cfg = yaml.safe_load((REPO / "config/model/stage5_oof.yaml").read_text())
    assert is_pure_model(cfg)
    assert cfg["information_contract"] == "pure_forecast"
    assert float(cfg["market_prior_lambda"]) == 0.0
    assert float(cfg["market_probability_weight"]) == 0.0
    assert cfg["market_anchor"] is None
    assert cfg["use_clv_head"] is False
    assert cfg.get("train_minutes_on_appearances_only") is True
    # Fail-closed guard must accept it.
    assert_pure_model_config(cfg, context="test_stage5_oof")


def test_stage5_oof_config_would_reject_market_weight():
    cfg = yaml.safe_load((REPO / "config/model/stage5_oof.yaml").read_text())
    cfg["market_prior_lambda"] = 0.1  # re-introduce leakage
    with pytest.raises(MarketLeakageError):
        assert_pure_model_config(cfg, context="test_stage5_oof_leak")


# ---------------------------------------------------------------------------
# B. generate_fold_pmfs active-PMF lineage
# ---------------------------------------------------------------------------

def _oof_cfg():
    return {
        "random_seed": 42, "stats": ["pts"], "sparse_stats": [],
        "pmf_support_caps": {"pts": 40}, "min_train_stat_rows": 5, "min_train_long_rows": 10,
        "dnp_minutes_threshold": 1.0, "low_minutes_zero_inflation_enabled": True,
        "use_minutes_marginalization": True,
        "minutes_marginalization_weights": [0.1, 0.15, 0.5, 0.15, 0.1],
        "pmf_source": "stage5_walk_forward_oof_uncalibrated_model_only",
        "calibration_eligible_prediction_types": ["model_oof"],
        "league_priors": {"pts": {"mean": 8.0, "var": 40.0}},
        "hgb_regressor": {"max_iter": 30, "max_leaf_nodes": 10, "min_samples_leaf": 3},
        "hgb_classifier": {"max_iter": 30, "max_leaf_nodes": 10, "min_samples_leaf": 3},
        "minutes_clip_min": 0.0, "minutes_clip_max": 48.0, "min_minutes_sigma": 3.0,
        "uncertain_sigma_multiplier": 1.5, "min_stat_mean": 0.01,
        "train_minutes_on_appearances_only": True,
    }


def _tiny(n=60):
    rng = np.random.default_rng(0)
    mm = rng.uniform(6, 34, n)
    played = rng.random(n) > 0.15
    return pd.DataFrame({
        "player_id": np.arange(n), "game_id": np.arange(n),
        "game_date": pd.date_range("2026-05-01", periods=n, freq="D"), "season": 2026,
        "team_id": rng.integers(1, 5, n), "player_name": [f"P{i}" for i in range(n)],
        "team_abbreviation": "TST", "opponent_team_id": rng.integers(5, 10, n),
        "position": rng.choice(["G", "F", "C"], n), "is_home": rng.choice([True, False], n),
        "actual_minutes": np.where(played, mm, 0.0), "did_play": played,
        "actual_pts": np.where(played, rng.poisson(np.clip(mm * 0.4, 0.1, None)), 0).astype(float),
        "player_minutes_mean_l5": mm + rng.normal(0, 2, n),
        "player_pts_mean_l5": rng.uniform(4, 18, n),
    })


def _fold_pmfs():
    from wnba_props_model.models.minutes_model import MinutesModel
    from wnba_props_model.models.rate_model import StatRateModel
    from wnba_props_model.models.training import encode_features
    cfg = _oof_cfg()
    wide = _tiny()
    long = wide[["player_id", "game_id", "game_date", "season", "player_name", "team_id",
                 "team_abbreviation", "opponent_team_id", "actual_minutes", "did_play"]].copy()
    long["stat"] = "pts"
    long["actual_outcome"] = wide["actual_pts"].values
    feats = ["player_minutes_mean_l5", "player_pts_mean_l5", "is_home", "position"]
    X, enc = encode_features(wide, feats, fit_encoder=True)
    mm = MinutesModel(cfg)
    mm.fit(X, wide["actual_minutes"], wide)
    mm._pos_encoder = enc
    played = wide["did_play"].to_numpy()
    sm = StatRateModel("pts", cfg)
    sm.fit(X[played].reset_index(drop=True), wide.loc[played, "actual_pts"].reset_index(drop=True))
    fm = FoldModel(mm, {"pts": sm}, {}, enc, list(X.columns), {}, len(wide), len(long),
                   {"pts": int(played.sum())})
    meta = {"fold_id": 0, "oof_prediction_type": "model_oof", "train_start_date": None,
            "train_end_date": date(2026, 4, 30), "val_start_date": date(2026, 5, 1),
            "val_end_date": date(2026, 7, 1), "train_wide_rows": len(wide),
            "train_stat_rows": {"pts": int(played.sum())}, "train_games": 5}
    return generate_fold_pmfs(fm, wide, long, meta, cfg)


def test_generate_fold_pmfs_persists_active_lineage():
    df = _fold_pmfs()
    for col in ("active_pmf_json", "active_pmf_mean", "availability_mixture_pmf_json", "p_dnp"):
        assert col in df.columns, f"missing repaired-lineage column {col}"
    # availability mixture alias equals the historical pmf_json
    assert (df["availability_mixture_pmf_json"] == df["pmf_json"]).all()


def test_mixture_mean_never_exceeds_active_mean():
    df = _fold_pmfs()
    active = df["active_pmf_mean"].to_numpy(float)
    mixture = df["pmf_mean"].to_numpy(float)
    assert np.all(mixture <= active + 1e-6), "folding DNP mass onto 0 must not raise the mean"


def test_active_pmf_json_is_valid_distribution():
    df = _fold_pmfs()
    for js in df["active_pmf_json"]:
        vals = list(json.loads(js).values())
        assert abs(sum(vals) - 1.0) < 1e-6
        assert all(v >= -1e-9 for v in vals)


# ---------------------------------------------------------------------------
# C. evaluate_pure_oof helpers
# ---------------------------------------------------------------------------

ev = _load_script("evaluate_pure_oof")


def test_settle_over_from_active_half_line():
    # Point-ish mass around 5; half line 4.5 -> P(over)=P(>=5).
    pmf = {"3": 0.1, "4": 0.2, "5": 0.4, "6": 0.2, "7": 0.1}
    p = ev._settled_over_from_active(json.dumps(pmf), 4.5)
    assert abs(p - (0.4 + 0.2 + 0.1)) < 1e-9


def test_settle_over_integer_line_is_push_safe():
    # Integer line 5 with push mass at 5; settled conditions out the push.
    pmf = {"3": 0.1, "4": 0.2, "5": 0.4, "6": 0.2, "7": 0.1}
    p = ev._settled_over_from_active(json.dumps(pmf), 5.0)
    assert abs(p - (0.3 / 0.6)) < 1e-9


def test_crps_point_mass_is_zero():
    assert ev._crps_discrete(json.dumps({"0": 0.0, "3": 1.0}), 3) < 1e-12


def test_pure_provenance_zero_weight_is_ok():
    prov = {"information_contract": "pure_forecast", "market_probability_weight": 0.0,
            "market_prior_lambda": 0.0, "clv_head_enabled": False,
            "forbidden_market_columns_present": []}
    # The 0.0-weight case must be recognized as pure (regression: `0.0 or 1.0` bug).
    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "PURE_OOF_RUN_MANIFEST.json"
        mp.write_text(json.dumps(prov))
        loaded = json.loads(mp.read_text())
    def _zero(v):
        return v is not None and float(v) == 0.0
    ok = (loaded["information_contract"] == "pure_forecast"
          and _zero(loaded["market_probability_weight"])
          and _zero(loaded["market_prior_lambda"])
          and not loaded["clv_head_enabled"]
          and not loaded["forbidden_market_columns_present"])
    assert ok is True


def test_nested_eval_runs_on_synthetic_join():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2026-05-01", periods=20, freq="D").astype(str)
    rows = []
    for dt in dates:
        for _ in range(8):
            active = np.zeros(11)
            mu = rng.integers(3, 8)
            active[mu] = 0.6
            active[mu - 1] = 0.2
            active[min(mu + 1, 10)] += 0.2
            line = float(mu) - 0.5
            actual = int(rng.poisson(mu))
            rows.append({
                "prop": "pts", "game_date": dt, "line": line, "actual": float(actual),
                "outcome_over": int(actual > line), "market_prob_over_no_vig": 0.5,
                "p_over_settled_active": ev._settled_over_from_active(
                    json.dumps({str(k): float(v) for k, v in enumerate(active)}), line),
                "crps_active": 1.0,
            })
    pdf = pd.DataFrame(rows)
    res = ev.nested_eval(pdf, "P1_active_settled_platt", min_train_dates=6, val_block_dates=2)
    assert res is not None
    assert res["n"] > 0
    assert "model_logloss" in res and "model_crps_active_pmf" in res


# ---------------------------------------------------------------------------
# D. Full synthetic build -> evaluate dry-run (pure guard + active lineage + metrics)
# ---------------------------------------------------------------------------

def test_build_oof_pmfs_dry_run_routes_through_active_pmf(tmp_path):
    rng = np.random.default_rng(7)
    stats = ["pts", "reb"]
    feats = ["player_minutes_mean_l5", "player_pts_mean_l5", "player_reb_mean_l5",
             "is_home", "position"]
    dates = pd.date_range("2026-05-01", periods=20, freq="D")
    rows = []
    gid = 1000
    for d in dates:
        for pl in range(30):
            gid += 1
            mm = float(rng.uniform(8, 34))
            played = rng.random() > 0.12
            rows.append({
                "player_id": pl, "game_id": gid, "game_date": d, "season": 2026,
                "team_id": pl % 6, "player_name": f"P{pl}", "team_abbreviation": "TST",
                "opponent_team_id": (pl + 1) % 6, "opponent_team_abbreviation": "OPP",
                "position": rng.choice(["G", "F", "C"]), "is_home": bool(rng.random() > .5),
                "actual_minutes": mm if played else 0.0, "did_play": bool(played),
                "player_minutes_mean_l5": mm + rng.normal(0, 2),
                "player_pts_mean_l5": float(rng.uniform(4, 20)),
                "player_reb_mean_l5": float(rng.uniform(1, 9)),
                "actual_pts": float(rng.poisson(max(0.1, mm * 0.4)) if played else 0),
                "actual_reb": float(rng.poisson(max(0.1, mm * 0.15)) if played else 0),
            })
    wide = pd.DataFrame(rows)
    long_rows = []
    for st in stats:
        sub = wide[["player_id", "game_id", "game_date", "season", "player_name", "team_id",
                    "team_abbreviation", "opponent_team_id", "opponent_team_abbreviation",
                    "is_home", "actual_minutes", "did_play"]].copy()
        sub["stat"] = st
        sub["actual_outcome"] = wide[f"actual_{st}"].values
        long_rows.append(sub)
    long = pd.concat(long_rows, ignore_index=True)

    wide_p = tmp_path / "wide.parquet"; wide.to_parquet(wide_p, index=False)
    long_p = tmp_path / "long.parquet"; long.to_parquet(long_p, index=False)
    man_p = tmp_path / "manifest.json"
    man_p.write_text(json.dumps({"model_feature_columns": feats,
                                 "target_columns": [f"actual_{s}" for s in stats]}))
    cfg = yaml.safe_load((REPO / "config/model/stage5_oof.yaml").read_text())
    cfg.update({"stats": stats, "sparse_stats": [], "min_train_long_rows": 150,
                "min_train_stat_rows": 30, "oof_first_val_date": "2026-05-08",
                "validation_window_days": 5, "use_tuned_hyperparams": False,
                "use_model_ensemble": False, "use_role_stratified_training": False})
    cfg["pmf_support_caps"] = {"pts": 45, "reb": 25}
    cfg_p = tmp_path / "cfg.yaml"; cfg_p.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "oof"; audit = tmp_path / "audit.json"
    r = subprocess.run(
        [sys.executable, "scripts/build_oof_pmfs.py",
         "--features-wide", str(wide_p), "--features-long", str(long_p),
         "--manifest", str(man_p), "--config", str(cfg_p),
         "--out-dir", str(out_dir), "--audit-out", str(audit), "--max-folds", "2"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "pure_forecast guard: PASS" in r.stdout
    assert "active-PMF lineage: PASS" in r.stdout

    oof = pd.read_parquet(out_dir / "oof_player_stat_pmfs.parquet")
    assert {"active_pmf_json", "availability_mixture_pmf_json", "p_dnp"} <= set(oof.columns)
    manifest = json.loads((audit.parent / "PURE_OOF_RUN_MANIFEST.json").read_text())
    assert manifest["information_contract"] == "pure_forecast"
    assert manifest["forbidden_market_columns_present"] == []

    # Evaluate against a synthetic exact-quote table joined on the pure OOF keys.
    q = oof[["game_id", "player_id", "stat", "actual_outcome", "active_pmf_json"]].rename(
        columns={"stat": "prop"})
    q = q.merge(oof[["game_id", "player_id", "stat", "game_date"]].rename(columns={"stat": "prop"}),
                on=["game_id", "player_id", "prop"])
    lines, oo = [], []
    for _, rr in q.iterrows():
        pmf = np.array(list(json.loads(rr["active_pmf_json"]).values()))
        line = round(float(np.dot(np.arange(pmf.size), pmf / pmf.sum()))) + 0.5
        lines.append(line)
        oo.append(int(rr["actual_outcome"] > line))
    scored = pd.DataFrame({
        "game_id": q["game_id"].astype(str), "player_id": q["player_id"].astype(str),
        "prop": q["prop"], "game_date": q["game_date"].astype(str),
        "actual": q["actual_outcome"].astype(float), "line": lines,
        "market_prob_over_no_vig": 0.5, "outcome_over": oo})
    scored_p = tmp_path / "scored.parquet"; scored.to_parquet(scored_p, index=False)

    ev_out = tmp_path / "metrics"
    r2 = subprocess.run(
        [sys.executable, "scripts/evaluate_pure_oof.py",
         "--oof", str(out_dir / "oof_player_stat_pmfs.parquet"), "--scored", str(scored_p),
         "--oof-manifest", str(audit.parent / "PURE_OOF_RUN_MANIFEST.json"),
         "--out-dir", str(ev_out), "--min-train-dates", "5", "--val-block-dates", "2"],
        cwd=REPO, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-3000:]
    metrics = json.loads((ev_out / "PRODUCTION_PURE_OOF_METRICS.json").read_text())
    assert metrics["pure_input_provenance_ok"] is True
    assert (ev_out / "PRODUCTION_PURE_OOF_METRICS.csv").exists()
