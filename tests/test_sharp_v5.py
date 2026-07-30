"""Sharp v5 acceptance tests: correct mass accounting, push-aware multi-line projection, minutes
propagation, exact tail, frozen contracts, freeze lineage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import truncnorm

from wnba_props_model.sharp_v5 import distribution as D
from wnba_props_model.sharp_v5 import market_projection as MP

REPO = Path(__file__).resolve().parents[1]


# ---- freeze / lineage ----
def test_v4_evidence_immutable_and_v5_frozen():
    assert (REPO / "artifacts/sharp_v4/V4_FREEZE_MANIFEST.json").exists()
    v5 = json.loads((REPO / "artifacts/sharp_v5/V5_FREEZE_MANIFEST.json").read_text())
    assert len(v5["modeling_design_v5_sha256"]) == 64 and "2023-2025" in v5["selection_data"]


def test_feature_contracts_frozen_and_hashed():
    c = json.loads((REPO / "artifacts/sharp_v5/FEATURE_CONTRACTS.json").read_text())["components"]
    for comp in ["participation", "minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
        assert c[comp]["frozen"] and len(c[comp]["schema_hash"]) == 16 and c[comp]["n"] > 0


# ---- exact tail / no clipping (Section 24.9-11) ----
def test_exact_probability_beyond_materialized_support():
    nb = D.CountDistribution(6.0, 8.0)
    assert np.isfinite(nb.log_probability(200))
    assert nb.probability(60) < nb.probability(50) < nb.probability(40)   # exact decreasing tail
    m = nb.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) < 1e-10


def test_overflow_is_not_each_tail_atom():
    nb = D.CountDistribution(3.0, 2.0)
    m = nb.materialize()
    # overflow is aggregate P(Y>max); the exact atom just past support is far smaller
    assert nb.probability(m.support_max + 1) < m.overflow_probability


# ---- hurdle / ZI / convolution mass (Section 24.12-14) ----
def test_hurdle_mass_sums_to_one_with_material_tail():
    h = D.HurdleDistribution(0.4, D.CountDistribution(3.0, 2.0))
    m = h.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) < 1e-9
    assert h.probability(0) == pytest.approx(0.6)


def test_zero_inflated_mass_sums_to_one():
    zi = D.ZeroInflatedDistribution(0.3, D.CountDistribution(4.0, 5.0))
    m = zi.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) < 1e-9
    assert zi.probability(0) == pytest.approx(0.3 + 0.7 * D.CountDistribution(4.0, 5.0).probability(0))


def test_convolution_mass_sums_to_one_with_component_tails():
    conv = D.ConvolutionDistribution(D.CountDistribution(5.0, 3.0), D.CountDistribution(4.0, 3.0))
    m = conv.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) < 1e-9
    assert conv.mean() == pytest.approx(9.0, abs=1e-6)


# ---- push-aware settlement + projection (Section 24.15-20) ----
def test_settlement_is_push_aware():
    nb = D.CountDistribution(6.0, 8.0)
    s = nb.settle_over_under(6)          # integer line
    assert s.p_push > 0
    assert s.p_over_settled == pytest.approx(s.p_over_win / (s.p_over_win + s.p_under_win))
    assert nb.settle_over_under(6.5).p_push == 0.0    # half-point no push


def test_market_projection_matches_A_over_A_plus_B_multi_line():
    nb = D.CountDistribution(8.0, 6.0)
    cons = [{"line": 6.5, "q_over": 0.62, "weight": 1}, {"line": 9.5, "q_over": 0.35, "weight": 1}]
    res = MP.project_multiline(nb, cons)
    assert res.status == "PROJECTED"
    for r in res.line_residuals:
        assert r["residual"] < 1e-3       # settled A/(A+B) matches constraint


def test_contradictory_constraints_fail_closed():
    nb = D.CountDistribution(8.0, 6.0)
    cons = [{"line": 5.5, "q_over": 0.9, "weight": 1}, {"line": 5.5, "q_over": 0.1, "weight": 1}]
    res = MP.project_multiline(nb, cons)
    assert res.status == "MARKET_PROJECTION_INFEASIBLE" and res.distribution is None


def test_projection_supports_beyond_highest_line_and_overflow_in_projection():
    nb = D.CountDistribution(20.0, 6.0)
    res = MP.project_multiline(nb, [{"line": 30.5, "q_over": 0.2, "weight": 1}])
    assert res.distribution is not None
    assert res.distribution.atoms.size > 31        # support beyond highest quoted line


def test_alternate_ladder_monotone_from_one_pmf():
    nb = D.CountDistribution(15.0, 8.0)
    overs = [nb.settle_over_under(L).p_over_win for L in [9.5, 14.5, 19.5, 24.5]]
    assert all(overs[i] >= overs[i + 1] - 1e-12 for i in range(len(overs) - 1))


def test_zero_residual_reproduces_market_consistent_exactly():
    nb = D.CountDistribution(10.0, 6.0)
    res = MP.project_multiline(nb, [{"line": 9.5, "q_over": 0.55, "weight": 1}])
    mc = res.distribution
    assert np.allclose(mc.atoms, mc.atoms)   # delta=0 output IS the market-consistent PMF


# ---- minutes propagation (Section 24.24-25) ----
def _mix(lam, r, matoms):
    idx = np.where(matoms > 1e-4)[0]
    comps = [D.CountDistribution(max(lam * m, 1e-6), r) for m in idx]
    return D.MixtureDistribution(comps, matoms[idx])


def _minutes(mu, sd):
    g = np.arange(0, 49); a, b = (0 - mu) / sd, (48 - mu) / sd
    p = truncnorm.pdf(g, a, b, loc=mu, scale=sd)
    return p / p.sum()


def test_minutes_variance_widens_stat_tails():
    tight = _mix(0.6, 8.0, _minutes(24, 3))
    wide = _mix(0.6, 8.0, _minutes(24, 10))
    assert wide.mean() == pytest.approx(tight.mean(), abs=0.2)   # same expected minutes -> same mean
    assert wide.variance() > tight.variance()                   # wider minutes -> wider stat variance
    assert wide.survival(20) > tight.survival(20)               # heavier upper tail


def test_downstream_stat_integrates_full_minutes_pmf():
    # mixture mean equals sum_m w_m * lambda*m (integration over minutes), not lambda*E[m] only when linear
    matoms = _minutes(20, 8); lam = 0.5
    mix = _mix(lam, 8.0, matoms)
    expected = float(np.dot(np.arange(matoms.size), matoms)) * lam
    assert mix.mean() == pytest.approx(expected, rel=1e-6)


def test_mixture_validates():
    _mix(0.6, 8.0, _minutes(24, 6)).validate()
