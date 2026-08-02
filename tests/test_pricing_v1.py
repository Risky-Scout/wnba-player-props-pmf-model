"""WNBA Pricing PMF v1 — engine, joint generator, registry, calibration, first-basket tests."""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.data.bdl_client import WNBA_ENDPOINTS
from wnba_props_model.data.odds_api_client import SPORT_KEY
from wnba_props_model.pricing import calibration as CAL
from wnba_props_model.pricing import engine as E
from wnba_props_model.pricing import first_basket as FB
from wnba_props_model.pricing import market_registry as MR
from wnba_props_model.pricing.joint_generator import PlayerGameParams, simulate_player


# ---- data-contract fixes ----
def test_bdl_uses_prop_type_not_type(monkeypatch):
    from wnba_props_model.data.bdl_client import BDLClient
    c = BDLClient(api_key="x")
    captured = {}
    monkeypatch.setattr(c, "iter_endpoint", lambda name, params=None, **k: captured.update(params=params) or iter(()))
    c.list_player_props_for_game(1, prop_type="points")
    assert captured["params"].get("prop_type") == "points" and "type" not in captured["params"]


def test_plays_not_assumed_paginated():
    assert WNBA_ENDPOINTS["plays"].paginated is False


def test_odds_sport_key_is_basketball_wnba():
    assert SPORT_KEY == "basketball_wnba"


# ---- registry ----
def test_registry_covers_required_markets_and_fantasy_needs_config():
    for k in ["player_points", "player_rebounds", "player_threes", "player_field_goals",
              "player_frees_made", "player_points_q1", "player_blocks_steals",
              "player_points_rebounds_assists", "player_double_double", "player_first_basket",
              "player_points_alternate"]:
        assert k in MR.MARKET_REGISTRY
    assert MR.get("player_fantasy_points").required_scoring_config.startswith("REQUIRES")
    # alternates settle from the base distribution (same internal outcome key)
    assert MR.get("player_points_alternate").internal_outcome_key == MR.get("player_points").internal_outcome_key


# ---- joint generator: identities + validity ----
@pytest.fixture(scope="module")
def joint():
    return simulate_player(PlayerGameParams(player_id="p1"), n_samples=30000, seed=7)


def test_active_pmfs_sum_to_one_and_nonneg(joint):
    for key, pmf in joint.pmfs.items():
        assert abs(pmf.sum() - 1.0) < 1e-9, key
        assert np.all(pmf >= 0) and np.all(np.isfinite(pmf)), key
        assert pmf[0] >= 0  # support starts at zero


def test_points_identity_and_fgm_and_reb_identity_hold(joint):
    assert joint.identities_hold is True


def test_dnp_is_separate_from_zero_atom(joint):
    assert 0.0 <= joint.p_dnp <= 1.0
    # p_dnp is NOT folded into pts atom 0
    assert joint.p_dnp == pytest.approx(1 - PlayerGameParams(player_id="p1").p_active)


def test_combination_market_uses_joint_dependence(joint):
    # pts+ast joint PMF must differ from the convolution of independent marginals (shared minutes
    # induces positive correlation).
    indep = np.convolve(joint.pmfs["pts"], joint.pmfs["ast"])   # independent marginals
    joint_combo = joint.pmfs["pts_ast"]
    n = max(indep.size, joint_combo.size)
    a = np.zeros(n); a[:indep.size] = indep
    b = np.zeros(n); b[:joint_combo.size] = joint_combo
    assert np.abs(a - b).sum() > 0.02          # materially different -> dependence, not sum


def test_q1_layer_not_flat_quarter(joint):
    # Q1 pts mean should not be exactly 0.25 * full pts mean (separate Q1 layer)
    full_mean = float(np.dot(np.arange(joint.pmfs["pts"].size), joint.pmfs["pts"]))
    q1_mean = float(np.dot(np.arange(joint.q1_pmfs["pts_q1"].size), joint.q1_pmfs["pts_q1"]))
    assert abs(q1_mean - 0.25 * full_mean) > 1e-6


# ---- pricing engine ----
def test_over_under_push_and_settled():
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])   # atoms 0..4
    pl = E.price_over_under(pmf, 2.0, "player_points")     # integer line -> push possible
    assert pl.p_push == pytest.approx(0.4)
    assert pl.p_over_win == pytest.approx(0.3) and pl.p_under_win == pytest.approx(0.3)
    assert pl.p_over_settled == pytest.approx(0.5) and pl.p_under_settled == pytest.approx(0.5)


def test_half_point_line_cannot_push():
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    pl = E.price_over_under(pmf, 2.5, "player_points")
    assert pl.p_push == 0.0
    assert pl.p_over_win + pl.p_under_win == pytest.approx(1.0)


def test_alternate_ladder_is_monotone():
    pmf = simulate_player(PlayerGameParams(player_id="pz"), n_samples=20000, seed=3).pmfs["pts"]
    ladder = E.price_alternate_ladder(pmf, [9.5, 14.5, 19.5, 24.5], "player_points_alternate")
    p_over = [x.p_over_win for x in ladder]
    assert all(p_over[i] >= p_over[i + 1] - 1e-12 for i in range(len(p_over) - 1))  # non-increasing


def test_fair_odds_conversions_correct():
    assert E.prob_to_decimal(0.5) == pytest.approx(2.0)
    assert E.decimal_to_american(2.0) == pytest.approx(100)
    assert E.decimal_to_american(1.5) == pytest.approx(-200)
    assert E.american_to_prob(-110) == pytest.approx(110 / 210, rel=1e-6)


def test_margin_does_not_change_pmf():
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    before = pmf.copy()
    pl = E.price_over_under(pmf, 2.5, "player_points", margin_method="proportional", overround=0.05)
    assert np.array_equal(pmf, before)                       # PMF untouched
    # quoted implied over+under > 1 (margin added); fair settled sum == 1
    assert pl.p_over_settled + pl.p_under_settled == pytest.approx(1.0)
    q_over = 1 / pl.quoted_decimal_over; q_under = 1 / pl.quoted_decimal_under
    assert q_over + q_under > 1.0


# ---- event markets ----
def test_double_triple_double_settlement(joint):
    assert 0 <= joint.event_probs["double_double"] <= 1
    assert joint.event_probs["triple_double"] <= joint.event_probs["double_double"]


def test_categorical_normalized():
    out = E.price_categorical({"a": 2.0, "b": 1.0, "c": 1.0}, "player_method_of_first_basket")
    assert out["normalized_sum"] == pytest.approx(1.0)
    assert out["categories"]["a"]["probability"] == pytest.approx(0.5)


def test_first_basket_sums_to_one_and_per_team():
    hz = [FB.FirstBasketHazard("p1", "T1", 3.0), FB.FirstBasketHazard("p2", "T1", 1.0),
          FB.FirstBasketHazard("p3", "T2", 2.0), FB.FirstBasketHazard("p4", "T2", 2.0)]
    fb = FB.price_first_basket(hz, residual_hazard=0.5)
    assert fb["normalized_sum"] == pytest.approx(1.0)
    ftb = FB.price_first_team_basket(hz)
    for s in ftb["per_team_normalized_sums"].values():
        assert s == pytest.approx(1.0)


def test_method_is_categorical_normalized():
    hz = FB.FirstBasketHazard("p1", "T1", 1.0, method_mix={"two_point_make": 3, "three_point_make": 1, "free_throw": 1})
    m = FB.price_method_of_first_basket(hz)
    assert m["normalized_sum"] == pytest.approx(1.0)


# ---- calibration ----
def test_calibration_preserves_monotone_cdf_and_sum():
    pmf = np.array([0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
    out = CAL.monotone_cdf_recalibrate(pmf, cdf_link=lambda c: c ** 1.2)
    CAL.assert_calibrated_pmf_valid(out)
    assert out.sum() == pytest.approx(1.0)


def test_fantasy_requires_config():
    assert MR.get("player_fantasy_points").release_status == "CONFIG_REQUIRED"
