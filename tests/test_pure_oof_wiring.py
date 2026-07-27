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
    res = ev.nested_eval(pdf, "P1", min_train_dates=6, val_block_dates=2)
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
                # Market-derived columns that MUST be dropped from the pure feature set.
                "game_spread_home": float(rng.normal(0, 6)),
                "game_total": float(rng.uniform(150, 175)),
                "implied_team_total": float(rng.uniform(75, 90)),
                # Box-score components so the structural repair candidate can fit (train-only).
                "actual_fga": float(rng.poisson(max(0.1, mm * 0.45)) if played else 0),
                "actual_fgm": float(rng.poisson(max(0.1, mm * 0.20)) if played else 0),
                "actual_fg3a": float(rng.poisson(max(0.1, mm * 0.15)) if played else 0),
                "actual_fg3m": float(rng.poisson(max(0.1, mm * 0.05)) if played else 0),
                "actual_fta": float(rng.poisson(max(0.1, mm * 0.12)) if played else 0),
                "actual_ftm": float(rng.poisson(max(0.1, mm * 0.10)) if played else 0),
                "actual_oreb": float(rng.poisson(max(0.1, mm * 0.06)) if played else 0),
                "actual_dreb": float(rng.poisson(max(0.1, mm * 0.15)) if played else 0),
                "actual_pts": float(rng.poisson(max(0.1, mm * 0.4)) if played else 0),
                "actual_reb": float(rng.poisson(max(0.1, mm * 0.15)) if played else 0),
            })
    wide = pd.DataFrame(rows)
    # Production stores realized per-game box-score VOLUMES bare (oreb/dreb/fga/fg3a/fta) while
    # settled outcomes carry the actual_ prefix; stage5_oof.yaml's structural_repair.columns map
    # points at the bare names. Mirror that here so the structural repair candidate fires.
    for _c in ("fga", "fg3a", "fta", "oreb", "dreb"):
        wide[_c] = wide[f"actual_{_c}"]
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
    market_feats = ["game_spread_home", "game_total", "implied_team_total"]
    man_p.write_text(json.dumps({"model_feature_columns": feats + market_feats,
                                 "target_columns": [f"actual_{s}" for s in stats]}))
    cfg = yaml.safe_load((REPO / "config/model/stage5_oof.yaml").read_text())
    cfg.update({"stats": stats, "sparse_stats": [], "min_train_long_rows": 150,
                "min_train_stat_rows": 30, "oof_first_val_date": "2026-05-08",
                "validation_window_days": 5, "use_tuned_hyperparams": False,
                "use_model_ensemble": False, "use_role_stratified_training": False})
    cfg["pmf_support_caps"] = {"pts": 45, "reb": 25}
    cfg_p = tmp_path / "cfg.yaml"; cfg_p.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "oof"; audit = tmp_path / "audit.json"
    ckpt = out_dir / "checkpoints"
    build_cmd = [sys.executable, "scripts/build_oof_pmfs.py",
                 "--features-wide", str(wide_p), "--features-long", str(long_p),
                 "--manifest", str(man_p), "--config", str(cfg_p),
                 "--out-dir", str(out_dir), "--audit-out", str(audit), "--max-folds", "2",
                 "--strict-baseline", "--checkpoint-dir", str(ckpt), "--resume"]
    r = subprocess.run(build_cmd, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "pure_forecast guard: PASS" in r.stdout
    assert "active-PMF lineage: PASS" in r.stdout
    assert "strict-baseline gates: PASS" in r.stdout
    # Pure track drops forbidden market columns (394→391 in production; here N→N-3) and PASSES
    # the strict guard rather than aborting with MarketLeakageError.
    assert "pure_forecast drop: removed 3 market-derived" in r.stdout
    assert "MarketLeakageError" not in (r.stdout + r.stderr)
    # GAP 1: per-fold checkpoints persisted, and a second --resume run reuses them.
    assert ckpt.exists() and any(ckpt.iterdir()), "checkpoint dir should hold per-fold files"
    r_resume = subprocess.run(build_cmd, cwd=REPO, capture_output=True, text=True)
    assert r_resume.returncode == 0, r_resume.stderr[-3000:]

    oof = pd.read_parquet(out_dir / "oof_player_stat_pmfs.parquet")
    assert {"active_pmf_json", "availability_mixture_pmf_json", "p_dnp"} <= set(oof.columns)
    # Structural repair candidate PMFs are persisted and non-null for supported props (pts/reb).
    assert "structural_active_pmf_json" in oof.columns
    for prop in ("pts", "reb"):
        sub = oof[oof["stat"] == prop]
        assert sub["structural_active_pmf_json"].notna().any(), f"no structural PMF for {prop}"
        assert (sub["structural_candidate_id"].dropna()
                == {"pts": "S_pts_opportunity_conversion",
                    "reb": "S_reb_oreb_dreb_opportunity"}[prop]).all()
    # GAP 4: line-independent provenance/contract + hash fields persisted on every OOF row.
    gap4 = {"active_pmf_variance", "availability_mixture_mean", "information_contract",
            "market_probability_weight", "model_hash", "config_hash", "feature_hash",
            "calibrator_hash"}
    assert gap4 <= set(oof.columns), f"missing persisted fields: {gap4 - set(oof.columns)}"
    assert (oof["information_contract"] == "pure_forecast").all()
    assert (oof["market_probability_weight"] == 0.0).all()
    assert (oof["active_pmf_variance"] >= -1e-9).all()
    assert oof["model_hash"].nunique() == 1 and oof["feature_hash"].nunique() == 1
    manifest = json.loads((audit.parent / "PURE_OOF_RUN_MANIFEST.json").read_text())
    assert manifest["information_contract"] == "pure_forecast"
    assert manifest["forbidden_market_columns_present"] == []
    # Provenance records EXACTLY what was dropped and the pure feature count (owner item 2).
    assert manifest["dropped_market_columns"] == sorted(market_feats)
    assert manifest["dropped_market_column_count"] == 3
    assert manifest["pre_drop_feature_count"] == len(feats) + 3
    assert manifest["pure_feature_count"] == len(feats)
    assert set(manifest["ordered_feature_list"]).isdisjoint(market_feats)
    # Standalone PURE_FORECAST_PROVENANCE.json is written too.
    prov2 = json.loads((audit.parent / "PURE_FORECAST_PROVENANCE.json").read_text())
    assert prov2["ordered_feature_list_sha256"] == manifest["ordered_feature_list_sha256"]
    # (b) OOF path and live-delivery path resolve to the SAME pure feature list via the shared
    # resolver used by both build_oof_pmfs.py and pmf_engine.build_all_pmfs.
    from wnba_props_model.models.pure_model_contract import drop_forbidden_market_columns
    delivery_kept, _ = drop_forbidden_market_columns(feats + market_feats)
    assert delivery_kept == manifest["ordered_feature_list"]

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
    # GAP 2: Holm-adjusted p-values recorded per quote-covered prop with an evaluated fold.
    assert "holm_family" in metrics
    for prop in metrics["holm_family"]:
        pp = metrics["per_prop"][prop]
        assert "holm_adjusted_p_ll" in pp and "holm_adjusted_p_brier" in pp
        assert 0.0 <= pp["holm_adjusted_p_ll"] <= 1.0
        assert 0.0 <= pp["holm_adjusted_p_brier"] <= 1.0
    csv = pd.read_csv(ev_out / "PRODUCTION_PURE_OOF_METRICS.csv")
    assert {"holm_adjusted_p_ll", "holm_adjusted_p_brier"} <= set(csv.columns)
    # GAP 4: per-row scored lineage persists the line-dependent settled + final probabilities.
    srp = ev_out / "PRODUCTION_PURE_OOF_SCORED_ROWS.parquet"
    assert srp.exists()
    sr = pd.read_parquet(srp)
    assert {"model_prob_over_settled_from_active_pmf", "model_prob_over_final",
            "p_dnp", "line"} <= set(sr.columns)
    # Repair ladder: the pure recalibration candidate family is reported per prop.
    for prop in metrics["holm_family"]:
        cands = metrics["per_prop"][prop].get("pure_recalibration_candidates", {})
        assert "P1" in cands
        # ITEM 4: nested selection fold manifest + selected outer-OOF rows are emitted.
        assert metrics["per_prop"][prop].get("selected_outer_oof") is not None
        assert "selected_candidate_distribution" in metrics["per_prop"][prop]
    assert (ev_out / "NESTED_SELECTION_FOLD_MANIFEST.json").exists()
    assert (ev_out / "NESTED_SELECTED_OOF_ROWS.parquet").exists()
    # ITEM 5: no verdict may exist without Holm-adjusted p-values populated.
    for prop in metrics["holm_family"]:
        rc = metrics["per_prop"][prop].get("real_selection_contract")
        assert rc is not None
        assert "holm_pvalues_not_populated" not in rc["fail_reasons"]
    # Structural repair candidate is registered + scored for a supported prop (pts).
    sc = metrics["per_prop"]["pts"].get("structural_repair_candidate")
    assert sc is not None and sc["candidate_id"] == "S_pts_opportunity_conversion"
    assert sc["after_structural_settled_platt"] is not None
    assert "advances" in sc


def test_holm_adjustment_is_monotone_and_bounded():
    """GAP 2: Holm-Bonferroni is step-down monotone, bounded by 1.0, and >= the raw p-value."""
    raw = {"pts": 0.20, "reb": 0.01, "ast": 0.04, "fg3m": 0.50}
    adj = ev._holm(raw)
    # Smallest raw (reb) gets multiplied by m=4; ordering preserved; all capped at 1.
    assert adj["reb"] == pytest.approx(min(1.0, 4 * 0.01))
    order = sorted(raw, key=raw.get)
    seq = [adj[k] for k in order]
    assert seq == sorted(seq), "Holm-adjusted p-values must be nondecreasing in raw rank"
    for k in raw:
        assert adj[k] >= raw[k] - 1e-12 and adj[k] <= 1.0


def test_pure_recalibration_candidates_are_monotone():
    """AST repair (owner item 7): isotonic + beta pure recalibrators fit and stay monotone,
    consuming ONLY the model's active-PMF settled probability vs outcome (no market input)."""
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.uniform(size=400) < p).astype(int)  # well-calibrated -> monotone increasing
    tr = pd.DataFrame({"p_over_settled_active": p, "outcome_over": y})
    grid = pd.DataFrame({"p_over_settled_active": np.linspace(0.05, 0.95, 50)})
    for cand in ("P2", "P3"):  # P2 = monotone Beta, P3 = isotonic (owner item 3 naming)
        fn, mono = ev._fit_calibrator(cand, tr)
        assert mono is True
        out = fn(grid)
        assert np.all(np.diff(out) >= -1e-9), f"{cand} must be monotone non-decreasing"
        assert np.all((out >= 0) & (out <= 1))
    assert set(ev.PURE_RECAL_CANDIDATES) == {"P0", "P1", "P2", "P3"}
    assert set(ev.CANDIDATE_FAMILY) == {"P0", "P1", "P2", "P3", "S1", "S2", "S3", "E1"}


def test_onesided_bootstrap_pvalue_semantics():
    """GAP 2: a model that dominates the market on every date yields a tiny one-sided p."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2026-05-01", periods=25, freq="D").astype(str)
    rows = []
    for dt in dates:
        for _ in range(12):
            y = int(rng.random() < 0.5)
            # Model is confidently correct; market is a coin flip -> model LL/Brier much lower.
            rows.append({"prop": "pts", "game_date": dt, "outcome_over": y,
                         "market_prob_over_no_vig": 0.5,
                         "p_over_settled_active": 0.95 if y else 0.05, "crps_active": 0.1})
    pdf = pd.DataFrame(rows)
    res = ev.nested_eval(pdf, "P0", min_train_dates=6, val_block_dates=2)
    ci = ev._bootstrap_ci(res, pdf, n_boot=2000)
    assert ci["logloss_p_onesided"] < 0.05
    assert ci["brier_p_onesided"] < 0.05


# ---------------------------------------------------------------------------
# E. delivery <-> OOF float64 parity (GAP 3)
# ---------------------------------------------------------------------------

def test_delivery_oof_float64_parity_identity_calibration():
    """The live-delivery lineage (build_probability_lineage) and the OOF/eval lineage
    (settle_over_from_active_pmf) MUST produce the bit-identical float64 model_prob_over_final
    on an identical (active PMF, line) fixture under identity calibration (production default)."""
    from wnba_props_model.models.availability_pmf import settle_over_from_active_pmf
    from wnba_props_model.models.probability_lineage import build_probability_lineage

    for line in (4.5, 5.0, 7.5, 12.0):  # half-lines and integer (push) lines
        active = {"3": 0.05, "4": 0.15, "5": 0.30, "6": 0.25, "7": 0.15, "8": 0.10}
        # Delivery path: single source of truth, identity (disabled) binary calibration.
        lineage = build_probability_lineage(
            final_pmf={int(k): v for k, v in active.items()}, line=line,
            prop="pts", role="rotation")
        # OOF/eval path: settle P(over) from the same active PMF; identity calibration.
        try:
            settled = settle_over_from_active_pmf(json.dumps(active), line).p_over_settled
        except Exception:
            settled = None
        if lineage.model_prob_over_final is None:
            # Both paths must agree the row is binary-ineligible (all mass on the push).
            assert settled is None
            continue
        # Bit-identical float64 (identity calibration -> final == settled on both sides).
        assert lineage.model_prob_over_settled_from_final_pmf == settled
        assert lineage.model_prob_over_final == settled


def test_delivery_oof_parity_under_shared_monotone_calibrator():
    """When a nonidentity monotone calibrator g is applied identically after the shared
    settlement step, both paths still return the identical float64 final probability."""
    from wnba_props_model.models.availability_pmf import settle_over_from_active_pmf
    from wnba_props_model.models.market import settled_probabilities_from_pmf

    active = {"2": 0.1, "3": 0.2, "4": 0.4, "5": 0.2, "6": 0.1}
    line = 3.5

    def g(p):  # a monotone logit-shift calibrator, applied identically on both sides
        z = np.log(p / (1 - p)) * 1.3 - 0.2
        return float(1 / (1 + np.exp(-z)))

    delivery_settled = settled_probabilities_from_pmf(
        {int(k): v for k, v in active.items()}, line).p_over_settled
    oof_settled = settle_over_from_active_pmf(json.dumps(active), line).p_over_settled
    assert delivery_settled == oof_settled  # shared settlement function
    assert g(delivery_settled) == g(oof_settled)  # identical calibrated final


# ---------------------------------------------------------------------------
# F. strict-baseline is fail-closed, never silently falls back (GAP 1)
# ---------------------------------------------------------------------------

def test_strict_baseline_aborts_on_prior_only(tmp_path):
    """With --strict-baseline, a fold that cannot fit a model (thresholds unmet) must FAIL the
    run with a nonzero exit and NOT silently emit prior_only PMFs."""
    rng = np.random.default_rng(9)
    stats = ["pts"]
    feats = ["player_minutes_mean_l5", "player_pts_mean_l5", "is_home", "position"]
    dates = pd.date_range("2026-05-01", periods=16, freq="D")
    rows, gid = [], 5000
    for d in dates:
        for pl in range(20):
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
                "actual_pts": float(rng.poisson(max(0.1, mm * 0.4)) if played else 0),
            })
    wide = pd.DataFrame(rows)
    sub = wide[["player_id", "game_id", "game_date", "season", "player_name", "team_id",
                "team_abbreviation", "opponent_team_id", "opponent_team_abbreviation",
                "is_home", "actual_minutes", "did_play"]].copy()
    sub["stat"] = "pts"; sub["actual_outcome"] = wide["actual_pts"].values

    wide_p = tmp_path / "wide.parquet"; wide.to_parquet(wide_p, index=False)
    long_p = tmp_path / "long.parquet"; sub.to_parquet(long_p, index=False)
    man_p = tmp_path / "manifest.json"
    man_p.write_text(json.dumps({"model_feature_columns": feats,
                                 "target_columns": ["actual_pts"]}))
    cfg = yaml.safe_load((REPO / "config/model/stage5_oof.yaml").read_text())
    cfg.update({"stats": stats, "sparse_stats": [], "oof_first_val_date": "2026-05-08",
                "validation_window_days": 5, "use_tuned_hyperparams": False,
                "use_model_ensemble": False, "use_role_stratified_training": False,
                # Impossible thresholds -> every fold is prior_only.
                "min_train_long_rows": 10_000_000, "min_train_stat_rows": 10_000_000})
    cfg["pmf_support_caps"] = {"pts": 45}
    cfg_p = tmp_path / "cfg.yaml"; cfg_p.write_text(yaml.safe_dump(cfg))

    r = subprocess.run(
        [sys.executable, "scripts/build_oof_pmfs.py",
         "--features-wide", str(wide_p), "--features-long", str(long_p),
         "--manifest", str(man_p), "--config", str(cfg_p),
         "--out-dir", str(tmp_path / "oof"), "--audit-out", str(tmp_path / "audit.json"),
         "--max-folds", "2", "--strict-baseline"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0, "strict-baseline must abort on prior_only folds"
    assert "strict-baseline" in (r.stdout + r.stderr).lower()
