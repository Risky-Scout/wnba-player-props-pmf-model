"""Test F2: BetaBinomialStatModel and beta_binomial_pmf_batch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.beta_binomial import (
    BetaBinomialStatModel,
    beta_binomial_pmf_batch,
)
from wnba_props_model.models.pmf_utils import beta_binomial_pmf_batch as pmf_utils_bb


class TestBetaBinomialPMFBatch:
    """beta_binomial_pmf_batch() must produce valid PMFs."""

    def test_rows_sum_to_one(self):
        n = np.array([3.0, 5.0, 7.0, 0.0, 2.5])
        pmf = beta_binomial_pmf_batch(n, alpha=2.0, beta_param=3.0, cap=12)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)

    def test_non_negative(self):
        n = np.array([3.0, 5.0, 7.0])
        pmf = beta_binomial_pmf_batch(n, alpha=2.0, beta_param=3.0, cap=12)
        assert (pmf >= 0).all()

    def test_zero_attempts_is_degenerate(self):
        n = np.array([0.0, 0.0])
        pmf = beta_binomial_pmf_batch(n, alpha=2.0, beta_param=3.0, cap=12)
        assert pmf[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert pmf[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_shape(self):
        n = np.ones(10) * 4.0
        cap = 12
        pmf = beta_binomial_pmf_batch(n, alpha=2.0, beta_param=3.0, cap=cap)
        assert pmf.shape == (10, cap + 1)

    def test_mass_beyond_cap_is_zero(self):
        """All mass must be within support."""
        n = np.array([2.0, 3.0, 4.0])
        pmf = beta_binomial_pmf_batch(n, alpha=2.0, beta_param=3.0, cap=12)
        # For n=2, P(k>2) should be zero since n=2 means max 2 made
        # (beta_binomial_pmf_batch rounds n to int)
        for row_idx, ni in enumerate([2, 3, 4]):
            assert pmf[row_idx, ni + 1:].sum() < 1e-6

    def test_pmf_utils_reexport(self):
        """beta_binomial_pmf_batch in pmf_utils.py should work identically."""
        n = np.array([3.0, 5.0])
        r1 = beta_binomial_pmf_batch(n, 2.0, 3.0, 12)
        r2 = pmf_utils_bb(n, 2.0, 3.0, 12)
        np.testing.assert_allclose(r1, r2)


class TestBetaBinomialStatModel:
    """BetaBinomialStatModel fit/predict interface."""

    def _make_data(self, n: int = 300, seed: int = 42):
        rng = np.random.default_rng(seed)
        fg3a = rng.integers(0, 7, n).astype(float)
        pct  = rng.uniform(0.2, 0.45, n)
        fg3m = np.array([rng.binomial(int(a), p) for a, p in zip(fg3a, pct)]).astype(float)
        X = pd.DataFrame({
            "player_fg3a_mean_l5": fg3a + rng.normal(0, 0.5, n),
            "player_minutes_l5": rng.uniform(15, 35, n),
        })
        return X, pd.Series(fg3m), pd.Series(fg3a)

    def test_fit_predict_pmf_matrix(self):
        X, y_made, y_att = self._make_data()
        cfg = {"hgb_regressor": {"max_iter": 50}}
        model = BetaBinomialStatModel(cfg)
        model.fit(X, y_made, y_att)
        pmf = model.predict_pmf_matrix(X, cap=12)
        assert pmf.shape == (len(X), 13)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)
        assert (pmf >= 0).all()

    def test_beta_params_positive(self):
        X, y_made, y_att = self._make_data()
        cfg = {"hgb_regressor": {"max_iter": 50}}
        model = BetaBinomialStatModel(cfg)
        model.fit(X, y_made, y_att)
        assert model.alpha_ > 0
        assert model.beta_ > 0

    def test_fallback_without_attempts(self):
        """Should not crash when fg3a is None — falls back to NegBinom."""
        X, y_made, _ = self._make_data()
        cfg = {"hgb_regressor": {"max_iter": 50}}
        model = BetaBinomialStatModel(cfg)
        model.fit(X, y_made, None)
        pmf = model.predict_pmf_matrix(X, cap=12)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_mean_reasonable(self):
        X, y_made, y_att = self._make_data()
        cfg = {"hgb_regressor": {"max_iter": 50}}
        model = BetaBinomialStatModel(cfg)
        model.fit(X, y_made, y_att)
        means = model.predict_mean(X)
        assert means.shape == (len(X),)
        assert (means >= 0).all()
        assert (means < 10).all()   # fg3m rarely > 10
