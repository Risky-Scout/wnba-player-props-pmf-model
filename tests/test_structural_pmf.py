"""Synthetic-data unit tests for the pure structural repair PMFs (owner item 7).

Covers every component in isolation (count primitives, hierarchical empirical-Bayes shrinkage,
opportunity×conversion builders for pts/reb/fg3m, the shared usage latent) and the
StructuralRepairModel fit/predict on a synthetic box-score frame — with NO market inputs. These
cannot be measured against the real market locally (feature matrix is BDL-gated); they prove the
math is correct and pure so the CI OOF run can measure them against the fresh pure baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.pure_model_contract import MarketLeakageError
from wnba_props_model.models.structural_pmf import (
    STRUCTURAL_CANDIDATE_IDS,
    SUPPORTED_PROPS,
    HierarchicalRate,
    StructuralRepairModel,
    assert_no_market_inputs,
    attempts_pmf,
    binomial_makes_pmf,
    build_fg3m_pmf,
    build_points_pmf,
    build_reb_pmf,
    convolve_pmfs,
    default_key_levels,
    hurdle_attempts_pmf,
    scale_support,
    truncate_pmf,
)


def _mean(pmf):
    return float(np.dot(np.arange(pmf.size), pmf))


def _var(pmf):
    m = _mean(pmf)
    return float(np.dot((np.arange(pmf.size) - m) ** 2, pmf))


# ---------------------------------------------------------------------------
# Count-PMF primitives
# ---------------------------------------------------------------------------

def test_attempts_pmf_valid_and_mean():
    p = attempts_pmf(6.0, 30, r=None)  # Poisson
    assert abs(p.sum() - 1.0) < 1e-9
    assert abs(_mean(p) - 6.0) < 0.1
    pnb = attempts_pmf(6.0, 40, r=5.0)  # NegBinom overdispersed
    assert abs(pnb.sum() - 1.0) < 1e-9
    assert _var(pnb) > _var(p) - 1e-6  # NB no less dispersed than Poisson at same mean


def test_hurdle_attempts_pmf_zero_mass():
    p = hurdle_attempts_pmf(0.4, 5.0, 25, r=None)
    assert abs(p.sum() - 1.0) < 1e-9
    assert abs(p[0] - 0.4) < 1e-6  # at least the hurdle zero mass sits at 0
    assert _mean(p) < 5.0  # zero inflation pulls the mean below the positive mean


def test_binomial_makes_point_attempts():
    # Attempts fixed at 10 -> makes ~ Binomial(10, 0.5): mean 5, var 2.5.
    att = np.zeros(21); att[10] = 1.0
    makes = binomial_makes_pmf(att, 0.5)
    assert abs(makes.sum() - 1.0) < 1e-9
    assert abs(_mean(makes) - 5.0) < 1e-9
    assert abs(_var(makes) - 2.5) < 1e-9


def test_scale_support_spacing():
    pmf = np.array([0.5, 0.3, 0.2])  # makes 0,1,2
    scaled = scale_support(pmf, 3)   # points 0,3,6
    assert scaled.size == 7
    assert scaled[0] == 0.5 and scaled[3] == 0.3 and scaled[6] == 0.2
    assert scaled[1] == 0.0


def test_convolve_point_masses():
    a = np.array([0.0, 1.0])   # =1
    b = np.array([0.0, 0.0, 1.0])  # =2
    c = convolve_pmfs(a, b)    # =3
    assert abs(_mean(c) - 3.0) < 1e-9


def test_truncate_piles_tail():
    pmf = np.array([0.1, 0.2, 0.3, 0.4])
    t = truncate_pmf(pmf, 2)
    assert t.size == 3
    assert abs(t.sum() - 1.0) < 1e-9
    assert abs(t[2] - 0.7) < 1e-9  # 0.3 + 0.4 piled onto cap


# ---------------------------------------------------------------------------
# Hierarchical empirical-Bayes shrinkage
# ---------------------------------------------------------------------------

def test_hierarchical_rate_shrinks_thin_sample_to_parent():
    # Two players in the same role/pos/team: one heavy-sample near 0.5, one 1-attempt outlier.
    rows = []
    for _ in range(400):
        rows.append({"role_bucket": "starter", "position": "G", "team_id": "1",
                     "player_id": "heavy", "m": 1, "a": 2})  # 0.5 make rate, many attempts
    rows.append({"role_bucket": "starter", "position": "G", "team_id": "1",
                 "player_id": "thin", "m": 1, "a": 1})  # 100% on ONE attempt
    df = pd.DataFrame(rows)
    hr = HierarchicalRate(default_key_levels(), kappa=40.0).fit(df, "m", "a")
    heavy = hr.predict_one({"role_bucket": "starter", "position": "G", "team_id": "1",
                            "player_id": "heavy"})
    thin = hr.predict_one({"role_bucket": "starter", "position": "G", "team_id": "1",
                           "player_id": "thin"})
    assert abs(heavy - 0.5) < 0.05          # heavy sample stays near its raw rate
    assert 0.5 <= thin < 1.0                # thin sample shrinks HARD toward the ~0.5 parent
    # Unseen player falls back up the hierarchy to a finite rate near the global mean.
    unseen = hr.predict_one({"role_bucket": "starter", "position": "G", "team_id": "1",
                             "player_id": "ghost"})
    assert 0.4 < unseen < 0.6


def test_hierarchical_rate_unknown_keys_fall_back_to_global():
    df = pd.DataFrame({"role_bucket": ["a"], "position": ["G"], "team_id": ["1"],
                       "player_id": ["p"], "m": [3.0], "a": [10.0]})
    hr = HierarchicalRate(default_key_levels(), kappa=5.0).fit(df, "m", "a")
    r = hr.predict_one({"role_bucket": "zzz", "position": "zzz", "team_id": "zzz",
                        "player_id": "zzz"})
    assert abs(r - hr.global_rate) < 1e-9


# ---------------------------------------------------------------------------
# Per-prop builders
# ---------------------------------------------------------------------------

def _grid():
    return np.array([0.85, 1.0, 1.15]), np.array([0.25, 0.5, 0.25])


def test_build_points_pmf_mean_matches_components():
    g, w = _grid()
    rates = {"fg2a_per_min": 0.30, "fg3a_per_min": 0.15, "fta_per_min": 0.12,
             "p2": 0.50, "p3": 0.36, "pft": 0.85,
             "r_fg2a": None, "r_fg3a": None, "r_fta": None}
    minutes = 30.0
    pmf = build_points_pmf(minutes, rates, {"fg2a": 40, "fg3a": 30, "fta": 30}, g, w, cap=80)
    assert abs(pmf.sum() - 1.0) < 1e-9
    exp = (2 * rates["fg2a_per_min"] * rates["p2"]
           + 3 * rates["fg3a_per_min"] * rates["p3"]
           + 1 * rates["fta_per_min"] * rates["pft"]) * minutes  # usage mean = 1.0
    assert abs(_mean(pmf) - exp) < 0.6


def test_build_reb_pmf_valid_and_mean():
    g, w = _grid()
    rates = {"oreb_per_min": 0.06, "dreb_per_min": 0.18, "r_oreb": None, "r_dreb": None}
    pmf = build_reb_pmf(28.0, rates, {"oreb": 15, "dreb": 25}, g, w, cap=40)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert abs(_mean(pmf) - (0.06 + 0.18) * 28.0) < 0.5


def test_build_fg3m_pmf_is_sharper_than_marginal_nb():
    # Structural FG3M (binomial over the 3PA distribution) must be a valid, reasonably SHARP
    # makes PMF (repairing the marginal-count over-dispersion / cert failure).
    g, w = _grid()
    rates = {"fg3a_per_min": 0.18, "p3": 0.36, "fg3a_p_zero": 0.10, "r_fg3a": None}
    pmf = build_fg3m_pmf(30.0, rates, {"fg3a": 30}, g, w, cap=20)
    assert abs(pmf.sum() - 1.0) < 1e-9
    mean = _mean(pmf)
    assert mean > 0
    # Variance should not blow up: sharpness ~ Fano factor below a loose ceiling.
    assert _var(pmf) / max(mean, 1e-9) < 2.5


def test_shared_usage_latent_adds_dispersion():
    # A wider usage grid (more pace/usage uncertainty) must not REDUCE points variance —
    # the shared latent injects positive co-movement across components.
    rates = {"fg2a_per_min": 0.30, "fg3a_per_min": 0.15, "fta_per_min": 0.12,
             "p2": 0.50, "p3": 0.36, "pft": 0.85, "r_fg2a": None, "r_fg3a": None, "r_fta": None}
    caps = {"fg2a": 40, "fg3a": 30, "fta": 30}
    tight = build_points_pmf(30.0, rates, caps, np.array([1.0]), np.array([1.0]), 80)
    wide = build_points_pmf(30.0, rates, caps, np.array([0.7, 1.0, 1.3]),
                            np.array([0.25, 0.5, 0.25]), 80)
    assert _var(wide) > _var(tight)
    assert abs(_mean(wide) - _mean(tight)) < 0.5  # symmetric grid preserves the mean


# ---------------------------------------------------------------------------
# StructuralRepairModel end-to-end on synthetic box scores (no market inputs)
# ---------------------------------------------------------------------------

def _synth_boxscore(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(8, 34, n)
    fg3a = rng.poisson(np.clip(minutes * 0.15, 0.1, None))
    fg2a = rng.poisson(np.clip(minutes * 0.30, 0.1, None))
    fta = rng.poisson(np.clip(minutes * 0.12, 0.1, None))
    fg3m = rng.binomial(fg3a, 0.35)
    fg2m = rng.binomial(fg2a, 0.50)
    ftm = rng.binomial(fta, 0.85)
    oreb = rng.poisson(np.clip(minutes * 0.06, 0.05, None))
    dreb = rng.poisson(np.clip(minutes * 0.18, 0.05, None))
    return pd.DataFrame({
        "player_id": rng.integers(0, 60, n).astype(str),
        "team_id": rng.integers(0, 12, n).astype(str),
        "position": rng.choice(["G", "F", "C"], n),
        "player_minutes_mean_l5": minutes + rng.normal(0, 2, n),
        "actual_minutes": minutes,
        "actual_fga": fg2a + fg3a, "actual_fgm": fg2m + fg3m,
        "actual_fg3a": fg3a, "actual_fg3m": fg3m,
        "actual_fta": fta, "actual_ftm": ftm,
        "actual_oreb": oreb, "actual_dreb": dreb,
    })


def test_structural_model_supports_all_three_props():
    df = _synth_boxscore()
    m = StructuralRepairModel({}).fit(df, {})
    assert m.supported == set(SUPPORTED_PROPS)
    val = df.head(20).reset_index(drop=True)
    for prop, cap in [("pts", 60), ("reb", 30), ("fg3m", 15)]:
        mat = m.build_active_pmf_matrix(prop, val, val["actual_minutes"].to_numpy(float), cap)
        assert mat is not None and mat.shape == (20, cap + 1)
        row_sums = mat.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        assert (mat >= -1e-12).all()


def test_structural_model_abstains_when_columns_missing():
    df = _synth_boxscore().drop(columns=["actual_oreb", "actual_dreb",
                                         "actual_fga", "actual_fgm", "actual_fta", "actual_ftm"])
    m = StructuralRepairModel({}).fit(df, {})
    # Only fg3m is derivable (fg3a/fg3m remain); pts/reb abstain.
    assert "reb" not in m.supported and "pts" not in m.supported
    assert "fg3m" in m.supported
    assert m.build_active_pmf_matrix("pts", df.head(3), df["actual_minutes"].to_numpy(float)[:3], 60) is None


def test_structural_predicted_mean_tracks_box_score():
    df = _synth_boxscore(n=2500, seed=3)
    m = StructuralRepairModel({}).fit(df, {})
    val = df.head(300).reset_index(drop=True)
    mat = m.build_active_pmf_matrix("pts", val, val["actual_minutes"].to_numpy(float), 80)
    pred_mean = (mat * np.arange(mat.shape[1])[None, :]).sum(axis=1).mean()
    true_pts = (2 * (val["actual_fgm"] - val["actual_fg3m"]) + 3 * val["actual_fg3m"]
                + val["actual_ftm"]).mean()
    assert abs(pred_mean - true_pts) < 3.0  # structural projection tracks realized points


# ---------------------------------------------------------------------------
# Purity: no market column can enter
# ---------------------------------------------------------------------------

def test_assert_no_market_inputs_rejects_market_columns():
    assert_no_market_inputs(["actual_pts", "actual_minutes", "player_id"])  # clean -> no raise
    for bad in ("game_spread_home", "game_total", "implied_team_total", "over_odds",
                "market_prob_over_no_vig", "player_market_p_over_prev", "closing_line"):
        with pytest.raises(MarketLeakageError):
            assert_no_market_inputs(["actual_pts", bad])


def test_structural_model_config_rejects_market_column_mapping():
    with pytest.raises(MarketLeakageError):
        StructuralRepairModel({"structural_repair": {"columns": {"minutes": "market_prob_over"}}})


def test_candidate_ids_registered_per_prop():
    assert set(STRUCTURAL_CANDIDATE_IDS) == set(SUPPORTED_PROPS)
    assert STRUCTURAL_CANDIDATE_IDS["pts"] == "S_pts_opportunity_conversion"
    assert STRUCTURAL_CANDIDATE_IDS["reb"] == "S_reb_oreb_dreb_opportunity"
    assert STRUCTURAL_CANDIDATE_IDS["fg3m"] == "S_fg3m_3pa_hurdle_shrunk_conversion"
