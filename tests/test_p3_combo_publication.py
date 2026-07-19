"""P3 - blocking tests for pts_reb / pts_reb_ast combo validation, packaging, and parity.

Covers: nonempty output, exact combo actual-outcome invariant, pre-block-only dependence
estimation, correlations actually affecting the PMF, mean reconciliation, production-vs-
validation parity, and no modification of the seven existing markets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.pipeline.forecast_publication import (
    apply_multistat_forecast, apply_market, _cells_from_json, _minbucket, _pmf_mean)
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf
from wnba_props_model.evaluation.distribution_calibration import hierarchical_empirical_pmf
from wnba_props_model.models.simulation import build_combo_pmfs, estimate_oof_correlations

REPO = Path(__file__).resolve().parent.parent
CALIB = json.loads((REPO / "config/certified_forecast_calibration.json").read_text())
REGISTRY = json.loads((REPO / "config/stat_registry.json").read_text())
MANIFEST = json.loads((REPO / "config/champion_manifest.json").read_text())
CERTIFIED = MANIFEST["certified_markets"]
COMBOS = ("pts_reb", "pts_reb_ast")
SEVEN = ["turnover", "pts", "ast", "stl", "stocks", "pts_ast", "reb"]


def _pmf_json(mean, n=45):
    a = np.array([np.exp(-((k - mean) ** 2) / (2 * max(mean, 1))) for k in range(n)])
    a /= a.sum()
    return json.dumps({str(i): float(v) for i, v in enumerate(a) if v > 1e-9})


def _proj():
    rows = []
    for pid in (1, 2, 3):
        for stat, mean in [("pts", 12 + pid), ("reb", 4 + pid), ("ast", 2 + pid),
                           ("stl", 1), ("turnover", 2)]:
            rows.append({"player_id": pid, "player": f"P{pid}", "stat": stat,
                         "pmf_json": _pmf_json(mean), "pmf_mean": float(mean),
                         "role_bucket": "starter", "minutes_mean": 30.0})
    return pd.DataFrame(rows)


def test_registry_and_policy_certify_both_combos_betting_off():
    for c in COMBOS:
        assert REGISTRY[c]["forecast_allowed"] is True
        assert REGISTRY[c]["forecast_method"] == "combo_residual"
        assert REGISTRY[c]["betting_recommendation_allowed"] is False
        assert c in CERTIFIED
        assert CALIB["markets"][c]["method"] == "combo_residual"
        assert CALIB["markets"][c]["certified"] is True
        assert CALIB["markets"][c]["scale"] in (0.97, 0.95)


def test_nonempty_output_for_both_combos():
    out = apply_multistat_forecast(_proj(), CALIB, CERTIFIED)
    stats = set(out["stat"])
    for c in COMBOS:
        assert c in stats, f"{c} missing from publisher output"
        assert (out["stat"] == c).sum() == 3   # one row per eligible player


def test_exact_combo_actual_outcomes():
    # pts_reb = pts + reb; pts_reb_ast = pts + reb + ast, evaluated on real component actuals.
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.p3_combo_repair import _load_components, COMBO_PARTS
    _df, rows = _load_components(
        str(REPO / "artifacts/models/calibration/oof_predictions.parquet"),
        str(REPO / "config/certified_forecast_calibration.json"))
    checked = 0
    for d in list(rows.values())[:500]:
        if all(p in d["actual"] for p in ("pts", "reb", "ast")):
            assert round(d["actual"]["pts"] + d["actual"]["reb"], 6) == round(
                sum(d["actual"][p] for p in COMBO_PARTS["pts_reb"]), 6)
            assert round(d["actual"]["pts"] + d["actual"]["reb"] + d["actual"]["ast"], 6) == round(
                sum(d["actual"][p] for p in COMBO_PARTS["pts_reb_ast"]), 6)
            checked += 1
    assert checked > 50


def _corr_oof(n_players, n_games, factor, seed):
    """OOF rows with a per-player latent factor so cross-player residual means correlate."""
    r = np.random.default_rng(seed)
    recs = []
    for pid in range(n_players):
        latent = r.normal(0, 1)
        for _g in range(n_games):
            for stat, f in (("pts", factor), ("reb", factor), ("ast", factor)):
                recs.append({"player_id": pid, "stat": stat, "pmf_mean": 5.0,
                             "actual_outcome": 5.0 + latent * f + r.normal(0, 0.5)})
    return pd.DataFrame(recs)


def test_pre_block_only_dependence_estimation():
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.p3_combo_repair import _combo_ledger_correlated

    pre = _corr_oof(60, 12, 1.5, 1)                     # strongly correlated pre-block data
    expected = {k: float(v) for k, v in estimate_oof_correlations(pre).items()}
    assert expected["pts_reb"] > 0.2                     # estimation is data-driven, not constant

    # one block, two players; component PMFs + a game_date inside the block
    import pandas as _pd
    gd = _pd.Timestamp("2026-07-10")
    pmf = np.array([np.exp(-((k - 6) ** 2) / 6) for k in range(40)]); pmf /= pmf.sum()
    rows = {(f"g{p}", p): {"game_date": gd, "role_bucket": "starter", "minutes_mean": 28.0,
                           "player_name": f"P{p}", "pmf": {"pts": pmf, "reb": pmf},
                           "actual": {"pts": 6.0, "reb": 5.0}, "mean": {"pts": 6.0, "reb": 5.0}}
            for p in range(2)}
    blocks = [[gd]]
    led1, bc1 = _combo_ledger_correlated("pts_reb", rows, [pre], blocks, pre)
    # frozen correlations equal estimate on PRE-block only
    assert bc1[0]["correlations"]["pts_reb"] == expected["pts_reb"]
    # mutating the scored-block outcomes must NOT change the frozen dependence
    for d in rows.values():
        d["actual"]["pts"] = 30.0
    _led2, bc2 = _combo_ledger_correlated("pts_reb", rows, [pre], blocks, pre)
    assert bc2[0]["correlations"] == bc1[0]["correlations"]
    assert len(led1) == 2


def test_correlation_parameters_actually_affect_pmf():
    pts = np.array([np.exp(-((k - 12) ** 2) / 8) for k in range(45)]); pts /= pts.sum()
    reb = np.array([np.exp(-((k - 5) ** 2) / 4) for k in range(45)]); reb /= reb.sum()
    ast = np.array([np.exp(-((k - 3) ** 2) / 3) for k in range(45)]); ast /= ast.sum()
    lo = build_combo_pmfs({"pts": pts, "reb": reb, "ast": ast}, correlations={
        "pts_reb": 0.0, "pts_ast": 0.0, "reb_ast": 0.0})
    hi = build_combo_pmfs({"pts": pts, "reb": reb, "ast": ast}, correlations={
        "pts_reb": 0.6, "pts_ast": 0.6, "reb_ast": 0.6})
    for key in ("pr", "pra"):
        a, b = np.asarray(lo[key]), np.asarray(hi[key])
        n = min(len(a), len(b))
        assert not np.allclose(a[:n], b[:n], atol=1e-3), f"{key} PMF ignores correlation"


def test_pmf_mean_reconciliation_correlated():
    # correlated combo preserves marginal means: E[combo] ~= sum of component means.
    pts = np.array([np.exp(-((k - 12) ** 2) / 8) for k in range(60)]); pts /= pts.sum()
    reb = np.array([np.exp(-((k - 5) ** 2) / 4) for k in range(60)]); reb /= reb.sum()
    ast = np.array([np.exp(-((k - 3) ** 2) / 3) for k in range(60)]); ast /= ast.sum()
    means = {"pts": _pmf_mean(pts), "reb": _pmf_mean(reb), "ast": _pmf_mean(ast)}
    built = build_combo_pmfs({"pts": pts, "reb": reb, "ast": ast},
                             correlations={"pts_reb": 0.3, "pts_ast": 0.3, "reb_ast": 0.15})
    assert abs(_pmf_mean(np.asarray(built["pr"])) - (means["pts"] + means["reb"])) < 0.6
    assert abs(_pmf_mean(np.asarray(built["pra"])) - sum(means.values())) < 0.8


def test_production_vs_validation_parity_combo_residual():
    proj = _proj()
    out = apply_multistat_forecast(proj, CALIB, CERTIFIED)
    # recompute calibrated component means exactly as the publisher does
    for combo in COMBOS:
        spec = CALIB["markets"][combo]; parts = spec["parts"]
        cells = _cells_from_json(spec["cells"]); scale = float(spec["scale"])
        from wnba_props_model.evaluation.forecasting import pmf_to_array
        pid = 2
        comp_means = []
        for p in parts:
            row = proj[(proj.player_id == pid) & (proj.stat == p)].iloc[0]
            arr = apply_market(pmf_to_array(row["pmf_json"]), float(row["pmf_mean"]),
                               "starter", 30.0, CALIB["markets"][p])
            comp_means.append(_pmf_mean(arr))
        point = float(sum(comp_means))
        max_sup = max(int(point) + 40, 80)
        exp = hierarchical_empirical_pmf(point, f"starter|{_minbucket(30.0)}", cells, max_sup)
        if scale != 1.0:
            exp = recalibrate_pmf(exp, 0.0, scale)
        got_row = out[(out.stat == combo) & (out.player_id == pid)].iloc[0]
        got = np.array([json.loads(got_row["pmf_json"]).get(str(i), 0.0)
                        for i in range(len(exp))])
        assert np.allclose(got, exp[:len(got)], atol=1e-8)
        assert abs(got.sum() - 1.0) < 1e-6


def test_no_modification_of_seven_existing_markets():
    # publishing with the two combos present must not change the seven existing markets' rows.
    proj = _proj()
    with_combos = apply_multistat_forecast(proj, CALIB, CERTIFIED)
    without = apply_multistat_forecast(proj, CALIB, [m for m in CERTIFIED if m not in COMBOS])
    for m in SEVEN:
        a = with_combos[with_combos.stat == m].sort_values("player_id")["pmf_json"].tolist()
        b = without[without.stat == m].sort_values("player_id")["pmf_json"].tolist()
        assert a == b, f"market {m} changed when combos were added"
    # component calibration specs are unchanged and only the two combos are combo_residual
    cr = [k for k, v in CALIB["markets"].items() if v.get("method") == "combo_residual"]
    assert set(cr) == set(COMBOS)
