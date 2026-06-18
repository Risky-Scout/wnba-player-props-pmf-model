"""Test F3: ZINBStatModel and zinb_pmf_batch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.hurdle import ZINBStatModel
from wnba_props_model.models.pmf_utils import negbinom_pmf_batch, zinb_pmf_batch


class TestZINBPMFBatch:
    """zinb_pmf_batch must produce valid PMFs with inflated zero mass."""

    def test_rows_sum_to_one(self):
        pi  = np.array([0.3, 0.5, 0.1, 0.0, 0.99])
        mu  = np.array([0.5, 0.3, 1.0, 0.8, 0.2])
        pmf = zinb_pmf_batch(pi, mu, r=2.0, cap=10)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)

    def test_non_negative(self):
        pi  = np.full(5, 0.3)
        mu  = np.full(5, 0.5)
        pmf = zinb_pmf_batch(pi, mu, r=2.0, cap=10)
        assert (pmf >= 0).all()

    def test_zero_mass_greater_than_negbinom(self):
        """ZINB P(k=0) must exceed NegBinom P(k=0) for same mu when pi > 0."""
        pi  = np.array([0.3, 0.4, 0.2])
        mu  = np.array([0.5, 0.3, 1.0])
        r   = 2.0
        zinb_pmf = zinb_pmf_batch(pi, mu, r, cap=10)
        nb_pmf   = negbinom_pmf_batch(mu, r, cap=10)

        for i in range(len(pi)):
            assert zinb_pmf[i, 0] > nb_pmf[i, 0], (
                f"Row {i}: ZINB P(0)={zinb_pmf[i,0]:.4f} should exceed NB P(0)={nb_pmf[i,0]:.4f}"
            )

    def test_zero_pi_equals_negbinom(self):
        """When π=0, ZINB should equal NegBinom."""
        pi  = np.zeros(3)
        mu  = np.array([0.5, 1.0, 2.0])
        r   = 3.0
        zinb_pmf = zinb_pmf_batch(pi, mu, r, cap=10)
        nb_pmf   = negbinom_pmf_batch(mu, r, cap=10)
        np.testing.assert_allclose(zinb_pmf, nb_pmf, atol=1e-5)

    def test_shape(self):
        n = 7
        pmf = zinb_pmf_batch(np.full(n, 0.2), np.full(n, 0.5), r=2.0, cap=10)
        assert pmf.shape == (n, 11)


class TestZINBStatModel:
    """ZINBStatModel must fit and predict correctly."""

    def _make_data(self, n: int = 400, seed: int = 99):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({
            "player_stl_per_min_l5": rng.uniform(0, 0.1, n),
            "opp_pace_proxy_l5": rng.uniform(90, 110, n),
            "player_minutes_mean_l5": rng.uniform(15, 35, n),
        })
        # Simulate ZINB steals: ~40% structural zeros
        pi = 0.4
        y_raw = rng.poisson(0.7, n).astype(float)
        zero_mask = rng.random(n) < pi
        y = pd.Series(np.where(zero_mask, 0.0, y_raw))
        return X, y

    def test_fit_and_predict(self):
        X, y = self._make_data()
        model = ZINBStatModel("stl", {"hgb_regressor": {"max_iter": 50}})
        model.fit(X, y)
        p_nz, mu = model.predict(X)
        assert p_nz.shape == (len(X),)
        assert mu.shape == (len(X),)
        assert (p_nz >= 0).all() and (p_nz <= 1).all()
        assert (mu >= 0).all()

    def test_dispersion_r_positive(self):
        X, y = self._make_data()
        model = ZINBStatModel("blk", {"hgb_regressor": {"max_iter": 50}})
        model.fit(X, y)
        assert model._r > 0
        assert model.pos_dispersion_r > 0

    def test_pmf_from_predict_sums_to_one(self):
        """Building a PMF using predict() output should sum to 1."""
        X, y = self._make_data()
        model = ZINBStatModel("stl", {"hgb_regressor": {"max_iter": 50}})
        model.fit(X, y)
        p_nz, mu = model.predict(X)
        pi = 1.0 - p_nz
        pmf = zinb_pmf_batch(pi, mu, model._r, cap=10)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)

    def test_only_stl_blk_allowed(self):
        with pytest.raises(ValueError, match="stl/blk"):
            ZINBStatModel("pts", {})

    def test_training_summary(self):
        X, y = self._make_data()
        model = ZINBStatModel("stl", {"hgb_regressor": {"max_iter": 50}})
        model.fit(X, y)
        summary = model.get_training_summary()
        assert summary["model_type"] == "ZINBStatModel"
        assert "dispersion_r" in summary
