"""Owner items 1-6/9 — active-PMF sportsbook settlement (void-on-DNP).

These tests fail on the pre-patch behaviour (delivery settling ``model_prob_over_final`` from
the DNP availability-mixture PMF) and pass after the patch (settling from the ACTIVE PMF via the
single shared function). Synthetic examples with analytically calculable expectations.
"""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.models.availability_pmf import (
    build_availability_mixture,
    invalid_posthoc_dednp_over,
    recover_active_pmf,
)
from wnba_props_model.models.pmf_engine import _blend_with_dnp
from wnba_props_model.models.probability_lineage import (
    build_probability_lineage,
    build_settled_probability_from_active_pmf,
    fail_closed_lineage,
)
from wnba_props_model.models.settlement_rules import (
    UNKNOWN,
    VOID_DNP,
    resolve_dnp_settlement_rule,
    settlement_basis_for_rule,
)
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json


def _active():
    # Nonzero zero-mass, spread so an integer line carries real push mass.
    return normalize_pmf(np.array([0.15, 0.20, 0.20, 0.18, 0.12, 0.10, 0.05]))


def _mean(pmf):
    a = normalize_pmf(np.asarray(pmf, dtype=float))
    return float(np.dot(np.arange(a.size), a))


# ---------------------------------------------------------------------------
# Availability mixture folds DNP exactly ONCE; active is recoverable
# ---------------------------------------------------------------------------

def test_availability_mixture_contains_dnp_once():
    active = _active()
    p_dnp = 0.20
    mix = build_availability_mixture(active, p_dnp)
    # mixture[0] = p_dnp + (1-p_dnp)*active[0]; mixture[k>0] = (1-p_dnp)*active[k]
    assert mix[0] == pytest.approx(p_dnp + (1 - p_dnp) * active[0], abs=1e-12)
    for k in range(1, active.size):
        assert mix[k] == pytest.approx((1 - p_dnp) * active[k], abs=1e-12)


def test_active_pmf_persisted_before_dnp_mix():
    # _blend_with_dnp on the ACTIVE matrix reproduces the mixture; the active (pre-blend) PMF is
    # recoverable exactly -> confirms active is a distinct first-class distribution, folded once.
    active = _active()
    p_dnp = 0.20
    mix = _blend_with_dnp(active[None, :].copy(), np.array([p_dnp]))[0]
    np.testing.assert_allclose(mix, build_availability_mixture(active, p_dnp), atol=1e-12)
    np.testing.assert_allclose(recover_active_pmf(mix, p_dnp), active, atol=1e-12)


def test_active_and_mixture_pmfs_each_sum_to_one():
    active = _active()
    mix = build_availability_mixture(active, 0.2)
    assert active.sum() == pytest.approx(1.0, abs=1e-12)
    assert mix.sum() == pytest.approx(1.0, abs=1e-12)


def test_active_and_mixture_mean_consistency():
    active = _active()
    p_dnp = 0.2
    mix = build_availability_mixture(active, p_dnp)
    # Folding DNP mass onto 0 can only lower the mean by exactly p_dnp * E[active].
    assert _mean(mix) == pytest.approx((1 - p_dnp) * _mean(active), abs=1e-12)
    assert _mean(mix) < _mean(active)


# ---------------------------------------------------------------------------
# Settled probability: active (void-on-DNP) != mixture (DNP-as-Under)
# ---------------------------------------------------------------------------

def test_integer_push_active_pmf_parity():
    active = _active()
    p_dnp = 0.20
    line = 3.0  # integer line -> real push mass
    mix = build_availability_mixture(active, p_dnp)
    la = build_settled_probability_from_active_pmf(
        active_pmf=json_to_pmf(pmf_to_json(active)), line=line, prop="pts", role="all")
    lm = build_probability_lineage(
        final_pmf=json_to_pmf(pmf_to_json(mix)), line=line, prop="pts", role="all")
    # Active settlement (void-on-DNP) is higher than mixture settlement (DNP-as-Under).
    assert la.model_prob_over_final > lm.model_prob_over_final
    assert la.model_prob_over_final - lm.model_prob_over_final > 0.05
    # Active push mass is the raw active mass at the line; mixture scales it by (1-p_dnp).
    assert la.model_prob_push == pytest.approx(active[3], abs=1e-12)
    assert lm.model_prob_push == pytest.approx((1 - p_dnp) * active[3], abs=1e-12)


def test_posthoc_dednp_division_is_wrong():
    active = _active()
    p_dnp = 0.20
    line = 3.0
    mix = build_availability_mixture(active, p_dnp)
    correct = build_settled_probability_from_active_pmf(
        active_pmf=json_to_pmf(pmf_to_json(active)), line=line, prop="pts", role="all"
    ).model_prob_over_final
    mixture_final = build_probability_lineage(
        final_pmf=json_to_pmf(pmf_to_json(mix)), line=line, prop="pts", role="all"
    ).model_prob_over_final
    posthoc = invalid_posthoc_dednp_over(mixture_final, p_dnp)
    # The /(1 - p_dnp) shortcut does NOT reproduce the correct integer-line active result.
    assert abs(posthoc - correct) > 1e-3


def test_half_line_active_pmf_parity():
    active = _active()
    p_dnp = 0.20
    line = 2.5  # half line -> zero push mass
    mix = build_availability_mixture(active, p_dnp)
    la = build_settled_probability_from_active_pmf(
        active_pmf=json_to_pmf(pmf_to_json(active)), line=line, prop="pts", role="all")
    lm = build_probability_lineage(
        final_pmf=json_to_pmf(pmf_to_json(mix)), line=line, prop="pts", role="all")
    assert la.model_prob_push == pytest.approx(0.0, abs=1e-12)
    # At a half line the active settled over is exactly P(active > 2.5) = sum active[3:].
    assert la.model_prob_over_final == pytest.approx(active[3:].sum(), abs=1e-12)
    # DNP-as-under (mixture) still depresses the over at a half line.
    assert la.model_prob_over_final > lm.model_prob_over_final


# ---------------------------------------------------------------------------
# Book DNP rules + fail-closed
# ---------------------------------------------------------------------------

def test_unknown_book_dnp_rule_fails_closed():
    assert resolve_dnp_settlement_rule("draftkings")[1] == VOID_DNP
    rid, rule = resolve_dnp_settlement_rule("some_random_local_book")
    assert rule == UNKNOWN
    assert settlement_basis_for_rule(rule) == "unknown_book_dnp_rule_fail_closed"
    fc = fail_closed_lineage("unknown_book_dnp_rule_fail_closed")
    assert fc.model_prob_over_final is None
    assert fc.binary_score_eligible is False
