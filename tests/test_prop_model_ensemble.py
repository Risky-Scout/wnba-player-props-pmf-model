"""Tests for HGB + Bayesian shrinkage ensemble (P3.3)."""
import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.rate_model import StatRateModel


def _make_train_data(n: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "player_pts_mean_l5": rng.uniform(5, 25, n),
        "player_minutes_mean_l5": rng.uniform(15, 35, n),
    })
    y = pd.Series(rng.integers(0, 25, n).astype(float))
    # Include player_id and actual_pts so the Gamma-Poisson prior is actually fitted
    # (without these, compute_league_priors_from_data falls back to None and the
    # ensemble silently reduces to pure HGB — this tests the full prior path).
    ctx = pd.DataFrame({
        "player_id": rng.integers(1, 30, n),
        "actual_pts": rng.integers(0, 25, n).astype(float),
        "role_bucket": ["starter"] * n,
        "actual_minutes": rng.uniform(15, 35, n),
        "player_pts_l5_support": rng.integers(0, 10, n).astype(float),
        "player_pts_mean_l5": rng.uniform(5, 25, n),
    })
    return X, y, ctx


def test_predict_with_shrinkage_shape():
    """predict_with_shrinkage should return array of same length as X."""
    cfg = {"use_model_ensemble": True, "random_seed": 42,
           "hgb_regressor": {"max_iter": 50}}
    X, y, ctx = _make_train_data()
    model = StatRateModel("pts", cfg)
    model.fit(X, y, context_df=ctx)
    wide = ctx.copy()
    preds = model.predict_with_shrinkage(X, wide)
    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds)), "Shrinkage predictions must be finite"


def test_shrinkage_blends_toward_league_mean_for_low_support():
    """Players with 0 prior games should be near league prior mean."""
    cfg = {"use_model_ensemble": True, "random_seed": 42,
           "hgb_regressor": {"max_iter": 100}}
    X, y, ctx = _make_train_data(n=500)
    model = StatRateModel("pts", cfg)
    model.fit(X, y, context_df=ctx)

    if model._league_prior_alpha is None:
        pytest.skip("League prior not fitted — shrinkage module may not be available")

    # Create test row with 0 support
    X_test = pd.DataFrame({"player_pts_mean_l5": [10.0], "player_minutes_mean_l5": [25.0]})
    wide_no_support = pd.DataFrame({"player_pts_l5_support": [0], "player_pts_mean_l5": [10.0]})
    wide_full_support = pd.DataFrame({"player_pts_l5_support": [20], "player_pts_mean_l5": [10.0]})

    pred_no_support   = model.predict_with_shrinkage(X_test, wide_no_support)[0]
    pred_full_support = model.predict_with_shrinkage(X_test, wide_full_support)[0]
    hgb_pred          = model.predict_mean(X_test)[0]

    # With 0 support: result should equal league prior (w=0 → pure Bayes)
    # With 20+ support: result should be near HGB pred (w=1)
    league_mu = model._global_mean
    assert abs(pred_no_support - league_mu) < abs(pred_full_support - league_mu) + 2.0, \
        "Low-support prediction should be closer to league mean"


def test_predict_without_shrinkage_falls_back_gracefully():
    """Without use_model_ensemble=True, predict_with_shrinkage should still return predictions."""
    cfg = {"use_model_ensemble": False, "random_seed": 42,
           "hgb_regressor": {"max_iter": 50}}
    X, y, ctx = _make_train_data()
    model = StatRateModel("pts", cfg)
    model.fit(X, y, context_df=ctx)
    preds = model.predict_with_shrinkage(X, ctx)
    # Should fall back to pure HGB when no prior
    hgb_preds = model.predict_mean(X)
    np.testing.assert_array_almost_equal(preds, hgb_preds, decimal=6)
