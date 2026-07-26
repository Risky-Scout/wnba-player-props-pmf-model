"""Tests for scripts/audit_structural_run_integrity.py (owner ITEM 1).

Builds a small synthetic OOF-shaped frame that mirrors the real schema and proves the audit PASSES
on a clean frame and FAILS (per-check) on each injected defect. No market inputs, no BDL.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "audit_structural_run_integrity",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_structural_run_integrity.py",
)
audit_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_mod)  # type: ignore[union-attr]

REQUIRED_PROPS = audit_mod.REQUIRED_PROPS


def _pmf_json(mean: float, cap: int = 20) -> str:
    """A small Poisson-like PMF summing to exactly 1 (renormalized)."""
    k = np.arange(cap + 1)
    w = np.exp(-mean) * mean**k / np.array([math.factorial(i) for i in k])
    w = w / w.sum()
    return json.dumps({str(i): float(p) for i, p in enumerate(w) if p > 0})


def _parse(js):
    d = json.loads(js)
    kmax = max(int(x) for x in d)
    arr = np.zeros(kmax + 1)
    for kk, p in d.items():
        arr[int(kk)] = p
    return arr


def _row(stat: str, mean: float = 5.0, cap: int = 20) -> dict:
    js = _pmf_json(mean, cap)
    arr = _parse(js)
    pm = float(np.dot(np.arange(arr.size), arr))
    pv = float(np.dot((np.arange(arr.size) - pm) ** 2, arr))
    struct_js = _pmf_json(mean, cap) if stat in audit_mod.SUPPORTED_STRUCTURAL_PROPS else None
    struct_id = f"S_{stat}" if stat in audit_mod.SUPPORTED_STRUCTURAL_PROPS else None
    return {
        "game_id": "g1", "player_id": "p1", "stat": stat,
        "oof_prediction_type": "model_oof",
        "pmf_json": js, "active_pmf_json": js, "availability_mixture_pmf_json": js,
        "pmf_mean": pm, "pmf_variance": pv,
        "stat_mean": pm,
        "structural_active_pmf_json": struct_js, "structural_candidate_id": struct_id,
        "support_tail_warning": False,
        "information_contract": "pure_forecast",
        "market_probability_weight": 0.0,
        "fold_train_end_date": "2025-06-01", "fold_validation_start_date": "2025-06-08",
    }


def _clean_frame() -> pd.DataFrame:
    rows = [_row(p) for p in REQUIRED_PROPS]
    return pd.DataFrame(rows)


def test_clean_frame_passes():
    rep = audit_mod.audit_oof(_clean_frame())
    assert rep["overall_passed"] is True, rep
    assert rep["all_seven_props_present"]["passed"]
    assert not rep["all_seven_props_present"]["missing_props"]
    for name, chk in rep["row_checks"].items():
        assert chk["passed"], (name, chk)


def test_missing_prop_fails():
    df = _clean_frame()
    df = df[df["stat"] != "blk"]
    rep = audit_mod.audit_oof(df)
    assert not rep["overall_passed"]
    assert rep["all_seven_props_present"]["passed"] is False
    assert "blk" in rep["all_seven_props_present"]["missing_props"]


def test_pmf_not_summing_to_one_fails():
    df = _clean_frame()
    bad = json.loads(df.loc[df["stat"] == "pts", "pmf_json"].iloc[0])
    bad["0"] = float(bad.get("0", 0.0)) + 0.5  # break the sum
    df.loc[df["stat"] == "pts", "pmf_json"] = json.dumps(bad)
    rep = audit_mod.audit_oof(df)
    assert rep["row_checks"]["pmf_sums_to_one"]["passed"] is False
    assert not rep["overall_passed"]


def test_mean_mismatch_fails():
    df = _clean_frame()
    df.loc[df["stat"] == "reb", "pmf_mean"] = 999.0
    rep = audit_mod.audit_oof(df)
    assert rep["row_checks"]["pmf_mean_matches"]["passed"] is False


def test_prior_only_row_fails():
    df = _clean_frame()
    df.loc[df["stat"] == "ast", "oof_prediction_type"] = "prior_only"
    rep = audit_mod.audit_oof(df)
    assert rep["row_checks"]["no_prior_only_or_failed"]["passed"] is False


def test_non_pure_contract_fails():
    df = _clean_frame()
    df.loc[df["stat"] == "pts", "information_contract"] = "market_blend"
    df2 = _clean_frame()
    df2.loc[df2["stat"] == "pts", "market_probability_weight"] = 0.1
    assert audit_mod.audit_oof(df)["row_checks"]["information_contract_pure"]["passed"] is False
    assert audit_mod.audit_oof(df2)["row_checks"]["market_weight_zero"]["passed"] is False


def test_temporal_leakage_fails():
    df = _clean_frame()
    df.loc[df["stat"] == "stl", "fold_train_end_date"] = "2025-06-10"  # >= val start
    rep = audit_mod.audit_oof(df)
    assert rep["row_checks"]["no_temporal_leakage"]["passed"] is False


def test_forbidden_market_column_in_schema_fails():
    df = _clean_frame()
    df["game_spread_home"] = -3.5
    rep = audit_mod.audit_oof(df)
    assert rep["no_forbidden_market_feature"]["passed"] is False
    assert "game_spread_home" in rep["no_forbidden_market_feature"]["forbidden_columns_present"]


def test_support_tail_warning_must_be_bool_no_missing():
    df = _clean_frame()
    df["support_tail_warning"] = df["support_tail_warning"].astype(object)
    df.loc[df["stat"] == "pts", "support_tail_warning"] = None
    rep = audit_mod.audit_oof(df)
    assert rep["support_tail_warning"]["passed"] is False
    assert rep["support_tail_warning"]["n_missing"] >= 1


def test_ast_tov_integrity_fails_on_mean_drift():
    df = _clean_frame()
    df.loc[df["stat"] == "turnover", "stat_mean"] = 123.0
    rep = audit_mod.audit_oof(df)
    assert rep["row_checks"]["ast_tov_integrity"]["passed"] is False


def test_structural_field_invalid_fails():
    df = _clean_frame()
    df.loc[df["stat"] == "fg3m", "structural_active_pmf_json"] = "not-json"
    rep = audit_mod.audit_oof(df)
    assert rep["row_checks"]["structural_fields_valid"]["passed"] is False


def test_cli_writes_report_and_exit_code(tmp_path):
    import subprocess
    import sys

    pq = tmp_path / "oof.parquet"
    _clean_frame().to_parquet(pq, index=False)
    out = tmp_path / "AUDIT.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_structural_run_integrity.py"
    res = subprocess.run([sys.executable, str(script), str(pq), "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    rep = json.loads(out.read_text())
    assert rep["overall_passed"] is True
    assert rep["n_rows"] == len(REQUIRED_PROPS)
