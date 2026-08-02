"""Sharp v6 acceptance tests: BDL endpoint correction, FGM/FTM labels, tail-aware tilted mass,
hurdle analytic variance, push-aware multi-line projection, freeze lineage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wnba_props_model.sharp_v6 import market_projection as MP
from wnba_props_model.sharp_v6.distribution import (
    CountDistribution,
    HurdleDistribution,
    TiltedDistribution,
    analytic_hurdle_variance,
)

REPO = Path(__file__).resolve().parents[1]


# ---- endpoint + labels + freeze ----
def test_correct_bdl_endpoint_recorded():
    a = json.loads((REPO / "artifacts/sharp_v6/BDL_PLAYER_STATS_ENDPOINT_AUDIT.json").read_text())
    assert a["correct_endpoint"] == "/wnba/v1/player_stats" and a["status"] == 200 and a["fgm_ftm_present"]


def test_fgm_ftm_labels_recovered_and_identity():
    p = REPO / "data/recovered_v2/wnba_player_shooting_labels.parquet"
    if not p.exists():
        pytest.skip("private shooting labels absent")
    import pandas as pd
    df = pd.read_parquet(p)
    assert "fgm" in df.columns and "ftm" in df.columns and len(df) > 0
    fg2m = df["fgm"] - df["fg3m"]
    assert int((2 * fg2m + 3 * df["fg3m"] + df["ftm"] != df["pts"]).sum()) == 0   # pts identity holds


def test_v5_immutable_and_v6_frozen():
    assert (REPO / "artifacts/sharp_v5/V5_FREEZE_MANIFEST.json").exists()
    v6 = json.loads((REPO / "artifacts/sharp_v6/V6_FREEZE_MANIFEST.json").read_text())
    assert len(v6["modeling_design_v6_sha256"]) == 64 and "2023-2025" in v6["selection_data"]


# ---- tail-aware tilted mass (Section 4) ----
def test_tilted_stored_plus_overflow_equals_one_with_material_base_tail():
    base = CountDistribution(6.0, 2.0)                 # overdispersed -> material tail
    td = TiltedDistribution(base, theta_mean=0.12, theta_disp=0.3, theta_zero=0.2)
    m = td.materialize()
    assert abs(m.stored_mass + m.overflow_probability - 1.0) <= 1e-10


def test_tilt_transforms_the_tail_not_just_stored_atoms():
    base = CountDistribution(6.0, 2.0)
    td = TiltedDistribution(base, theta_mean=0.15)
    # tail atom is reweighted relative to base (ratio != ratio at mode) -> tail transformed
    r_tail = td.probability(30) / max(base.probability(30), 1e-300)
    r_mode = td.probability(6) / max(base.probability(6), 1e-300)
    assert abs(r_tail - r_mode) > 1e-6


def test_projected_mean_variance_include_tail():
    base = CountDistribution(10.0, 3.0)
    td = TiltedDistribution(base, theta_mean=0.05)
    # mean via full materialization equals sum k*p over certified support (tail < tol)
    m = td.materialize()
    assert td.mean() == pytest.approx(float(np.dot(np.arange(m.atoms.size), m.atoms)), rel=1e-9)
    assert td.variance() > 0


# ---- hurdle analytic variance (Section 5) ----
def test_hurdle_analytic_variance_matches_materialized():
    h = HurdleDistribution(0.5, CountDistribution(4.0, 3.0))
    ana = analytic_hurdle_variance(h)
    mat = h.materialize()
    k = np.arange(mat.atoms.size)
    mean = float(np.dot(k, mat.atoms)); num = float(np.dot((k - mean) ** 2, mat.atoms))
    assert ana == pytest.approx(num, rel=0.05)     # analytic ~ materialized (tail small)


# ---- push-aware multi-line projection ----
def test_multiline_projection_one_pmf_matches_settled_constraints():
    base = CountDistribution(8.0, 5.0)
    res = MP.project_multiline(base, [{"line": 6.5, "q_over": 0.6}, {"line": 10.5, "q_over": 0.32}])
    assert res.status == "PROJECTED"
    for r in res.line_residuals:
        assert r["residual"] < 5e-3
    mm = res.distribution.materialize()
    assert abs(mm.stored_mass + mm.overflow_probability - 1.0) <= 1e-10


def test_contradictory_projection_fails_closed():
    base = CountDistribution(8.0, 5.0)
    res = MP.project_multiline(base, [{"line": 5.5, "q_over": 0.9}, {"line": 5.5, "q_over": 0.1}])
    assert res.status == "MARKET_PROJECTION_INFEASIBLE" and res.distribution is None


def test_no_clipping_tilted_beyond_support():
    td = TiltedDistribution(CountDistribution(5.0, 4.0), theta_mean=0.05)
    assert np.isfinite(td.log_probability(150))


def test_integer_push_and_half_point():
    td = TiltedDistribution(CountDistribution(6.0, 6.0))
    assert td.settle_over_under(6).p_push > 0
    assert td.settle_over_under(6.5).p_push == 0.0
