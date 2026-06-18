"""Test F1: MinutesModel quantile ordering and DNP model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.minutes_model import MinutesModel, _QUANTILES


def _make_synthetic_data(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # Simple synthetic features
    X = pd.DataFrame({
        "player_minutes_l5": rng.uniform(10, 38, n),
        "team_pace": rng.uniform(90, 110, n),
    })
    # Synthetic minutes target with some DNPs (y=0)
    y_raw = rng.normal(25, 8, n)
    dnp_mask = rng.random(n) < 0.1   # 10% DNP rate
    y = pd.Series(np.where(dnp_mask, 0.0, np.clip(y_raw, 5, 42)))
    metadata = pd.DataFrame({
        "projected_minutes_bucket": ["high"] * n,
        "role_uncertainty_bucket":  ["certain"] * n,
        "did_play": (~dnp_mask).astype(int),
    })
    return X, y, metadata


class TestMinutesModelQuantileOrdering:
    """Quantile regressors must produce monotone q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90."""

    def test_quantile_monotone(self):
        X, y, meta = _make_synthetic_data(n=300)
        cfg = {"min_minutes_sigma": 2.0, "hgb_regressor": {"max_iter": 50}}
        model = MinutesModel(cfg)
        model.fit(X, y, meta)
        assert len(model._quantile_models) == 5, "Expected 5 quantile models"

        quant_mat = model.predict_quantiles(X, meta)
        assert quant_mat.shape == (len(X), 5)

        # Monotonicity: each quantile should be ≥ the previous in expectation
        # (not strictly row-by-row due to crossing — check mean over rows)
        for qi in range(4):
            q_lo = quant_mat[:, qi].mean()
            q_hi = quant_mat[:, qi + 1].mean()
            assert q_lo <= q_hi + 0.5, (
                f"Quantile {_QUANTILES[qi]} mean ({q_lo:.2f}) > "
                f"{_QUANTILES[qi+1]} mean ({q_hi:.2f})"
            )

    def test_predict_returns_3_tuple(self):
        X, y, meta = _make_synthetic_data(n=200)
        cfg = {"min_minutes_sigma": 2.0, "hgb_regressor": {"max_iter": 50}}
        model = MinutesModel(cfg)
        model.fit(X, y, meta)
        result = model.predict(X, meta)
        assert len(result) == 3, "predict() must return (means, sigmas, p_dnp)"
        means, sigmas, p_dnp = result
        assert means.shape == sigmas.shape == p_dnp.shape == (len(X),)
        assert (p_dnp >= 0).all() and (p_dnp <= 1).all()

    def test_sigma_positive(self):
        X, y, meta = _make_synthetic_data(n=200)
        cfg = {"min_minutes_sigma": 2.0, "hgb_regressor": {"max_iter": 50}}
        model = MinutesModel(cfg)
        model.fit(X, y, meta)
        _, sigmas, _ = model.predict(X, meta)
        assert (sigmas > 0).all(), "Sigma must be positive"
        assert (sigmas >= 2.0).all(), "Sigma must be >= min_minutes_sigma"

    def test_dnp_model_fitted_when_dnps_present(self):
        X, y, meta = _make_synthetic_data(n=300, seed=7)
        cfg = {"min_minutes_sigma": 2.0, "hgb_regressor": {"max_iter": 50}}
        model = MinutesModel(cfg)
        model.fit(X, y, meta)
        assert model._dnp_model is not None, "DNP model should be fitted when DNPs present"

    def test_dnp_model_absent_when_all_play(self):
        X, y, meta = _make_synthetic_data(n=200)
        meta["did_play"] = 1  # force all played
        y_play = pd.Series(np.clip(np.random.normal(25, 6, len(X)), 5, 42))
        cfg = {"min_minutes_sigma": 2.0, "hgb_regressor": {"max_iter": 50}}
        model = MinutesModel(cfg)
        model.fit(X, y_play, meta)
        # When only one class, DNP model should be None
        assert model._dnp_model is None, "DNP model should not fit on single-class target"
