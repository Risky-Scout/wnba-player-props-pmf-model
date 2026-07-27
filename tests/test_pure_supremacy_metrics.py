"""PHASE 1/2 guards: root-cause decomposition + pure P1 recalibration metrics.

Ensures the pure-supremacy analytics stay honest: the pure path never reads a market column,
de-DNP inverts the blend exactly, and on the canonical artifacts P1 (de-DNP + pure Platt) turns
ast into a genuine pure-model win vs the exact no-vig market. Skips if artifacts are absent.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
_SCORED = REPO / "artifacts" / "market_feature_proof" / "G0_v2" / "PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet"
_OOF = REPO / "artifacts" / "models" / "calibration" / "oof_predictions.parquet"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sup = _load("build_pure_supremacy_metrics")


def test_de_dnp_inverts_blend():
    p_cond = np.array([0.1, 0.4, 0.7])
    p_dnp = np.array([0.05, 0.25, 0.4])
    np.testing.assert_allclose(sup.de_dnp((1 - p_dnp) * p_cond, p_dnp), p_cond, atol=1e-6)


def test_pure_eval_rejects_market_column():
    from wnba_props_model.models.pure_model_contract import MarketLeakageError
    df = pd.DataFrame({
        sup.FINAL_PROBABILITY_COLUMN: [0.3, 0.4],
        "p_dnp": [0.1, 0.1], "outcome_over": [0, 1], "game_date": ["2026-05-01", "2026-05-02"],
    })
    df["market_prob_over_no_vig"] = 0.5
    # Inject the market column into the "pure" allowed set to prove the guard fires.
    orig = sup.PURE_ALLOWED
    try:
        sup.PURE_ALLOWED = orig | {"market_prob_over_no_vig"}
        with pytest.raises(MarketLeakageError):
            sup.nested_eval(df, "P0_identity")
    finally:
        sup.PURE_ALLOWED = orig


def test_genuine_pure_win_label_is_removed():
    # The invalid point-estimate-only label must not exist anywhere in the diagnostic.
    assert not hasattr(sup, "genuine_pure_win")
    src = (REPO / "scripts" / "build_pure_supremacy_metrics.py").read_text()
    assert "genuine_pure_win" not in src


@pytest.mark.skipif(not (_SCORED.exists() and _OOF.exists()), reason="artifacts absent")
def test_ast_diagnostic_FAILS_real_selection_contract():
    # AST has a negative point estimate but FAILS the real contract: AUC<market and the
    # date-cluster bootstrap CIs cross zero. It must NOT be reported as a win.
    m = sup.load_joined(str(_SCORED), str(_OOF))
    pdf = m[m["prop"] == "ast"].sort_values("game_date").reset_index(drop=True)
    after = sup.nested_eval(pdf, "P1_deDNP_platt")
    ci = sup._bootstrap_ci(after, pdf, n_boot=2000)
    contract = sup.real_selection_contract(after, ci)
    assert after["model_auc"] < after["market_auc"]  # AUC below market
    assert not ci["logloss_upper95_below_zero"]        # LL CI crosses zero
    assert contract["selection_contract_pass"] is False
    assert any("AUC<market" in r for r in contract["fail_reasons"])


@pytest.mark.skipif(not (_SCORED.exists() and _OOF.exists()), reason="artifacts absent")
def test_decomposition_runs_and_attributes_availability():
    dec = _load("decompose_projection_bias")
    oof = pd.read_parquet(_OOF).rename(columns={"stat": "prop"})
    g = oof[oof["prop"] == "pts"]
    d = dec._decompose(g)
    # components approximately sum to the total mean bias (identity closure).
    recon = (d["availability_bias"] + d["minutes_bias"] + d["rate_bias"]
             + d["pmf_shape_residual_bias"])
    assert abs(recon - d["total_mean_bias"]) < 1e-6
    # availability (DNP over-suppression) is a genuine downward driver for pts.
    assert d["availability_bias"] < 0
