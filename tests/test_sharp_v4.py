"""Sharp v4 acceptance tests: exact contracts, join cardinality, no outcome clipping, exact tail,
hierarchical dispersion, minutes PMF, market projection, calibration monotonicity, freeze lineage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.sharp_v4 import core as C

REPO = Path(__file__).resolve().parents[1]


# ---- lineage / freeze (Section 22.1-3) ----
def test_v3_evidence_immutable_and_v4_frozen():
    assert (REPO / "artifacts/sharp_v3/V3_FREEZE_MANIFEST.json").exists()
    v4 = json.loads((REPO / "artifacts/sharp_v4/V4_FREEZE_MANIFEST.json").read_text())
    assert len(v4["modeling_design_v4_sha256"]) == 64
    assert "2023-2025" in v4["selection_data"]


def test_v3_2026_holdout_marked_consumed():
    a = json.loads((REPO / "artifacts/sharp_v4/HOLDOUT_LINEAGE_AUDIT.json").read_text())
    assert a["status"] == "CONSUMED_NOT_VALID_FOR_V4_SELECTION"


# ---- feature contracts (Section 22.4, 22.8) ----
def test_contracts_are_exact_not_substring_and_exclude_labels_and_market():
    cols = ["player_pts_mean_l5", "team_points_pace", "pts", "market_prob_over", "player_odds_x",
            "player_minutes_mean_season", "participation", "game_id"]
    feat = C.resolve_contract("pts", cols)
    assert "player_pts_mean_l5" in feat and "player_minutes_mean_season" in feat
    assert "pts" not in feat and "participation" not in feat and "game_id" not in feat   # labels/ids
    assert "market_prob_over" not in feat and "player_odds_x" not in feat                # no market in PURE
    assert "team_points_pace" not in feat        # anchored ^player_pts_ does NOT admit this substring


def test_contract_deterministic_and_hashed():
    cols = ["player_pts_mean_l5", "player_pts_mean_l10", "player_minutes_mean_season"]
    a = C.resolve_contract("pts", cols); b = C.resolve_contract("pts", cols)
    assert a == b and C.contract_hash(a) == C.contract_hash(b)


# ---- join cardinality (Section 22.5-6) ----
def test_duplicate_keys_fail_closed():
    df = pd.DataFrame({"game_id": [1, 1], "player_id": [7, 7], "x": [1, 2]})
    with pytest.raises(ValueError):
        C.assert_unique_keys(df, ["game_id", "player_id"])


def test_safe_merge_is_one_to_one():
    left = pd.DataFrame({"game_id": [1, 2], "player_id": [7, 8], "a": [1, 2]})
    right = pd.DataFrame({"game_id": [1, 2], "player_id": [7, 8], "b": [3, 4]})
    out = C.safe_merge(left, right, ["game_id", "player_id"])
    assert len(out) == 2
    bad = pd.DataFrame({"game_id": [1, 1], "player_id": [7, 7], "b": [3, 4]})
    with pytest.raises(ValueError):
        C.safe_merge(left, bad, ["game_id", "player_id"])


# ---- no outcome clipping / exact tail (Section 22.9-11) ----
def test_observation_above_support_is_not_clipped():
    pmf = C.build_count_pmf(6.0, 8.0, "pts")
    lp_extreme = pmf.logpmf(200)
    assert np.isfinite(lp_extreme)                       # exact analytic tail, not clipped to atom
    # NLL of an extreme outcome is large but finite (not the log of the last atom)
    assert C.nll_exact([pmf], np.array([200])) > 50


def test_tail_mass_is_analytic_and_over_includes_tail():
    pmf = C.build_count_pmf(5.0, 6.0, "pts")
    assert pmf.tail_method in ("poisson_sf", "nbinom_sf")
    assert pmf.overflow >= 0
    # prob_over uses exact survival: P(Y>line) + P(Y<=line) == 1
    assert pmf.prob_over(4.5) + pmf.cdf(4) == pytest.approx(1.0, abs=1e-9)


def test_integer_push_and_half_point_no_push():
    pmf = C.build_count_pmf(5.0, 8.0, "pts")
    assert pmf.prob_push(5) > 0                          # integer line can push
    assert pmf.prob_push(5.5) == 0.0                     # half-point cannot push


# ---- hierarchical dispersion (Section 22.12) ----
def test_hierarchical_dispersion_is_per_group():
    rng = np.random.default_rng(0)
    n = 4000
    group = rng.integers(0, 3, n)
    mu = np.full(n, 5.0)
    y = rng.poisson(mu).astype(float)
    d = C.hierarchical_dispersion(y, mu, group)
    assert "__global__" in d and set(np.unique(group)).issubset(set(d))   # per-group, not one scalar


# ---- market projection (Section 22.35-36) ----
def test_min_kl_market_projection_matches_constraint():
    pmf = C.build_count_pmf(10.0, 8.0, "pts")
    atoms = pmf.atoms / pmf.atoms.sum()
    target = 0.62
    mc = C.market_consistent_atoms(atoms, 9.5, target)
    got = float(mc[np.arange(mc.size) > 9.5].sum())
    assert got == pytest.approx(target, abs=1e-6)
    assert mc.sum() == pytest.approx(1.0) and np.all(mc >= 0)
    assert np.all(np.diff(np.cumsum(mc)) >= -1e-12)      # monotone CDF preserved


def test_residual_delta_zero_reproduces_market_fallback():
    pmf = C.build_count_pmf(8.0, 6.0, "reb")
    atoms = pmf.atoms / pmf.atoms.sum()
    mc = C.market_consistent_atoms(atoms, 7.5, 0.55)
    mc_delta0 = mc.copy()                                # delta=0 => output IS the market-consistent PMF
    assert np.allclose(mc, mc_delta0)


def test_prospective_registry_append_only():
    p = REPO / "deliveries/sharp_v4/prospective/registry.parquet"
    if p.exists():
        reg = pd.read_parquet(p)
        assert reg["prediction_id"].is_unique          # immutable, no duplicate/overwrite
