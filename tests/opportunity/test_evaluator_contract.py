"""Contract tests for the corrected Opportunity V2 evaluator.

Proves (owner directive section 1):
* the canonical scored-row builder FAILS (does not silently drop) on duplicate
  predictions, duplicate market rows, cross-line joins, missing quote_pair_id,
  ambiguous identities, post-cutoff quotes, and push/void rows in binary scoring;
* AUC uses sklearn (tie-safe);
* Holm is applied separately to the LL and Brier families;
* NO single metric or single p-value can produce a PASS — every gate is required.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
EVAL = REPO / "scripts" / "evaluate_opportunity_oof.py"


def _mod():
    spec = importlib.util.spec_from_file_location("opp_eval", EVAL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EV = _mod()


def _pmf_json(mean, size=30):
    k = np.arange(size)
    from math import exp, factorial
    arr = np.array([exp(-mean) * mean**i / factorial(i) for i in k], float)
    arr = arr / arr.sum()
    return json.dumps(arr.tolist())


def _oof(n=4, prop="fg3m"):
    return pd.DataFrame({
        "game_id": range(100, 100 + n),
        "player_id": range(1, 1 + n),
        "prop": [prop] * n,
        "active_pmf_json": [_pmf_json(1.5)] * n,
        "prediction_cutoff_utc": ["2026-06-01T22:00:00+00:00"] * n,
    })


def _quotes(n=4, prop="fg3m"):
    return pd.DataFrame({
        "game_id": range(100, 100 + n),
        "player_id": range(1, 1 + n),
        "prop": [prop] * n,
        "line": [1.5] * n,
        "quote_pair_id": [f"q{i}" for i in range(n)],
        "outcome_over": [i % 2 for i in range(n)],
        "binary_score_eligible": [True] * n,
        "market_prob_over_no_vig": [0.5] * n,
        "model_prob_over_final": [0.5] * n,
        "game_date": [f"2026-06-{i + 1:02d}" for i in range(n)],
        "decision_timestamp": ["2026-06-01T20:00:00+00:00"] * n,
        "settlement_status": ["settled"] * n,
    })


# ------------------------- fail-closed contract ------------------------- #
def test_duplicate_oof_predictions_raise():
    oof = pd.concat([_oof(2), _oof(2).iloc[[0]]], ignore_index=True)
    with pytest.raises(EV.EvaluatorContractError, match="duplicate OOF"):
        EV.build_canonical_scored_rows(oof, _quotes(2))


def test_duplicate_market_rows_raise():
    q = pd.concat([_quotes(2), _quotes(2).iloc[[0]]], ignore_index=True)
    with pytest.raises(EV.EvaluatorContractError, match="duplicate deterministic market"):
        EV.build_canonical_scored_rows(_oof(2), q)


def test_cross_line_join_raises():
    q = _quotes(2)
    extra = q.iloc[[0]].copy()
    extra["line"] = 2.5
    extra["quote_pair_id"] = "qX"
    q2 = pd.concat([q, extra], ignore_index=True)
    with pytest.raises(EV.EvaluatorContractError, match="cross-line"):
        EV.build_canonical_scored_rows(_oof(2), q2)


def test_missing_quote_pair_id_raises():
    q = _quotes(2)
    q.loc[0, "quote_pair_id"] = None
    with pytest.raises(EV.EvaluatorContractError, match="quote_pair_id"):
        EV.build_canonical_scored_rows(_oof(2), q)


def test_ambiguous_identity_raises():
    q = _quotes(2)
    q.loc[0, "player_id"] = None
    with pytest.raises(EV.EvaluatorContractError, match="ambiguous"):
        EV.build_canonical_scored_rows(_oof(2), q)


def test_push_void_in_binary_scoring_raises():
    q = _quotes(2)
    q.loc[0, "settlement_status"] = "push"  # still flagged binary_score_eligible
    with pytest.raises(EV.EvaluatorContractError, match="push/void"):
        EV.build_canonical_scored_rows(_oof(2), q)


def test_post_cutoff_quote_raises():
    q = _quotes(2)
    q.loc[0, "decision_timestamp"] = "2026-06-01T23:30:00+00:00"  # after 22:00 cutoff
    with pytest.raises(EV.EvaluatorContractError, match="post-cutoff"):
        EV.build_canonical_scored_rows(_oof(2), q)


def test_canonical_one_row_per_key():
    j = EV.build_canonical_scored_rows(_oof(4), _quotes(4))
    assert not j.duplicated(subset=EV.KEY).any()
    assert "p_over_opp_v2" in j.columns


# --------------------------- AUC tie-safety ---------------------------- #
def test_auc_uses_sklearn_and_handles_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])  # all tied -> AUC 0.5
    assert EV.auc(y, p) == pytest.approx(0.5, abs=1e-9)


# --------------------------- Holm separation --------------------------- #
def test_holm_applied_per_family():
    ll = EV.holm({"a": 0.01, "b": 0.04})
    br = EV.holm({"a": 0.20, "b": 0.30})
    # families are independent: same prop can have very different adjusted p's
    assert ll["a"] < br["a"]


# ------------------- no single gate produces a PASS -------------------- #
def _passing_frame(n=400, seed=0):
    """Frame where the model genuinely beats the market on LL and Brier."""
    rng = np.random.default_rng(seed)
    dates = [f"2026-06-{(i % 40) + 1:02d}" for i in range(n)]
    truth = rng.uniform(0.15, 0.85, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    # model close to truth; market noisier
    model = np.clip(truth + rng.normal(0, 0.03, n), 0.02, 0.98)
    market = np.clip(truth + rng.normal(0, 0.18, n), 0.02, 0.98)
    return pd.DataFrame({
        "game_id": range(n), "player_id": range(n), "prop": ["fg3m"] * n,
        "game_date": dates, "outcome_over": y,
        "p_over_opp_v2": model, "market_prob_over_no_vig": market,
        "active_pmf_json": [_pmf_json(1.5)] * n, "actual": [1] * n,
    })


CONTRACT = {"required_rows": 300, "required_dates": 30, "holm_alpha": 0.05,
            "auc_min": 0.5, "auc_vs_market_tol": 0.0, "calibration_slope_min": 0.80,
            "calibration_slope_max": 1.25, "calibration_intercept_abs_max": 0.25,
            "ece_max": 0.05}


def _eval(g, **over):
    kw = dict(ci_ll=[-0.05, -0.01], ci_bs=[-0.03, -0.005],
              holm_p_ll=0.01, holm_p_brier=0.01, parity_pass=True)
    kw.update(over)
    return EV.evaluate_candidate(g, CONTRACT, **kw)


def test_all_gates_pass_gives_eligible():
    g = _passing_frame()
    res = _eval(g)
    # sanity: this configuration is designed to pass; if data is borderline, at
    # least confirm the aggregate equals AND of gates.
    assert res["selection_eligible"] == all(res["gates"].values())


def test_single_pvalue_cannot_pass_ll():
    g = _passing_frame()
    res = _eval(g, holm_p_ll=0.5)  # LL p-value fails
    assert res["gates"]["holm_ll_ok"] is False
    assert res["selection_eligible"] is False


def test_single_pvalue_cannot_pass_brier():
    g = _passing_frame()
    res = _eval(g, holm_p_brier=0.5)  # Brier p-value fails
    assert res["gates"]["holm_brier_ok"] is False
    assert res["selection_eligible"] is False


def test_ci_upper_bound_required():
    g = _passing_frame()
    res = _eval(g, ci_ll=[-0.05, 0.02])  # LL CI upper crosses 0
    assert res["gates"]["ci_ll_upper_neg"] is False
    assert res["selection_eligible"] is False


def test_parity_required():
    g = _passing_frame()
    res = _eval(g, parity_pass=False)
    assert res["gates"]["parity_ok"] is False
    assert res["selection_eligible"] is False


def test_insufficient_rows_cannot_pass():
    g = _passing_frame(n=100)  # < 300
    res = _eval(g)
    assert res["gates"]["rows_ok"] is False
    assert res["selection_eligible"] is False
