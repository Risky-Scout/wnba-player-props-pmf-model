"""Component opportunity-rate model tests for Opportunity V2 (section 21)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.component_models import OpportunityRateModel


def _data(n=800, seed=3, sparse=False):
    rng = np.random.default_rng(seed)
    skill = rng.uniform(0.2, 1.2, n)          # per-minute opportunity rate driver
    minutes = rng.uniform(6, 34, n)
    rate = skill * (0.15 if sparse else 0.6)
    count = rng.poisson(rate * minutes)
    X = pd.DataFrame({"skill_ewma": skill, "role": rng.integers(0, 2, n).astype(float)})
    return X, pd.Series(count), pd.Series(minutes), pd.Series(np.ones(n, dtype=bool))


def test_mean_increases_with_minutes():
    X, cnt, mins, played = _data()
    m = OpportunityRateModel().fit(X, cnt, mins, played)
    Xi = X.head(1)
    lo = m.predict_for_minutes(Xi, np.array([10.0]))
    hi = m.predict_for_minutes(Xi, np.array([30.0]))
    assert hi.mean[0] > lo.mean[0]


def test_zero_probability_is_minutes_sensitive_for_sparse():
    X, cnt, mins, played = _data(sparse=True)
    m = OpportunityRateModel(zero_heavy_threshold=0.05).fit(X, cnt, mins, played)
    Xi = X.head(1)
    p_lo = m.predict_for_minutes(Xi, np.array([8.0])).p_zero
    p_hi = m.predict_for_minutes(Xi, np.array([30.0])).p_zero
    assert p_lo is not None and p_hi is not None
    assert p_hi[0] < p_lo[0]  # more minutes -> lower zero probability


def test_dispersion_estimated_when_overdispersed():
    rng = np.random.default_rng(9)
    n = 600
    minutes = rng.uniform(10, 30, n)
    # negative-binomial-ish overdispersed counts
    lam = rng.gamma(2.0, 0.3, n) * minutes
    count = rng.poisson(lam)
    X = pd.DataFrame({"f": rng.normal(size=n)})
    m = OpportunityRateModel().fit(X, pd.Series(count), pd.Series(minutes),
                                   pd.Series(np.ones(n, dtype=bool)))
    pred = m.predict_for_minutes(X.head(3), np.array([20.0, 20.0, 20.0]))
    assert pred.dispersion_r is not None and pred.dispersion_r[0] > 0


def test_source_tier_reported():
    X, cnt, mins, played = _data()
    m = OpportunityRateModel(source_tier=0).fit(X, cnt, mins, played)
    pred = m.predict_for_minutes(X.head(2), np.array([20.0, 25.0]))
    assert list(pred.source_tier) == [0, 0]
