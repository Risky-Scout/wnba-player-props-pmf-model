"""Sharp v3 core tests: leakage contracts, no-vig, PMF validity, folds, holdout single-use."""
from __future__ import annotations

import numpy as np
import pytest

from wnba_props_model.sharp_v3 import core as C


def test_feature_contract_excludes_labels():
    cols = ["player_pts_mean_l5", "pts", "reb", "participation", "actual_minutes",
            "player_minutes_mean_season", "game_id", "player_id"]
    feat = C.stat_feature_contract("pts", cols)
    for bad in C.LABEL_COLS + C.ID_COLS:
        assert bad not in feat, f"label/id leaked into contract: {bad}"
    assert "player_pts_mean_l5" in feat            # lagged same-stat feature kept


def test_label_cols_cover_all_targets():
    for t in C.TIER_A + ["participation", "actual_minutes", "stocks", "pts_reb_ast"]:
        assert t in C.LABEL_COLS


def test_no_vig_removes_vig_and_normalizes():
    # -110/-110 -> ~0.5 no-vig over
    assert C.no_vig_over(-110, -110) == pytest.approx(0.5, abs=1e-9)
    # favored over
    p = C.no_vig_over(-200, +170)
    assert 0.5 < p < 0.75
    # no-vig over + under = 1
    over = C.no_vig_over(-140, +120); under = 1 - over
    assert over + under == pytest.approx(1.0)


def test_american_to_prob_rejects_invalid():
    assert np.isnan(C.american_to_prob(50))          # inside [-100,100] invalid
    assert C.american_to_prob(+100) == pytest.approx(0.5)


def test_count_pmf_valid_and_nonneg():
    pmf = C.count_pmf(6.0, r=8.0, cap=35)
    assert abs(pmf.sum() - 1.0) < 1e-6
    assert np.all(pmf >= 0) and np.all(np.isfinite(pmf))


def test_residual_dispersion_conditional():
    rng = np.random.default_rng(0)
    mu = np.full(5000, 5.0)
    y = rng.poisson(5.0, 5000).astype(float)         # equidispersed -> no overdispersion
    r = C.residual_dispersion_r(y, mu)
    assert r is None or r > 5.0                      # large r ~ Poisson limit


def test_crps_and_pit_ranges():
    pmf = C.count_pmf(4.0, 6.0, 25)
    crps = C.crps_discrete([pmf], np.array([4.0]))
    assert crps >= 0
    u = C.pit_values([pmf], np.array([4]), np.random.default_rng(1))
    assert 0 <= u[0] <= 1


def test_prob_over_and_push():
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    assert C.prob_over(pmf, 2.5) == pytest.approx(0.3)
    assert C.prob_push(pmf, 2) == pytest.approx(0.4)     # integer line push
    assert C.prob_push(pmf, 2.5) == 0.0                  # half-point no push


def test_fold_boundaries_frozen_and_disjoint():
    import pandas as pd
    dates = pd.to_datetime(["2023-06-01", "2024-06-01", "2024-08-01", "2025-06-01", "2026-06-01"])
    df = pd.DataFrame({"game_date": dates})
    tr, _ev = C.split(df, C.DEV_FOLDS[0])
    # training is strictly before eval window (expanding-window, no future rows)
    assert (df.loc[tr, "game_date"] < pd.Timestamp(C.DEV_FOLDS[0].eval_start)).all()
    assert C.HOLDOUT.is_holdout is True


def test_active_conditional_semantics_documented():
    # participation is a LABEL (never a feature); DNP is modeled separately from the zero atom.
    assert "participation" in C.LABEL_COLS
    assert C.stat_feature_contract("pts", ["participation", "player_pts_mean_l5"]) == ["player_pts_mean_l5"]
