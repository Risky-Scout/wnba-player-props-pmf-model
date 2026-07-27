"""Conditional minutes distribution tests for Opportunity V2 (section 17)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.minutes_distribution import ConditionalMinutesDistributionV2


def _data(n=800, seed=2):
    rng = np.random.default_rng(seed)
    base = rng.uniform(10, 34, n)
    start = (base > 24).astype(float)
    minutes = np.clip(base + rng.normal(0, 3, n), 1, 40)
    did_play = np.ones(n, dtype=bool)
    # inject some DNP zero rows that must be excluded
    dnp_idx = rng.choice(n, size=80, replace=False)
    minutes[dnp_idx] = 0.0
    did_play[dnp_idx] = False
    X = pd.DataFrame({
        "player_minutes_ewma": base,
        "player_minutes_std_l10": rng.uniform(2, 6, n),
        "player_start_rate_ewma": start,
    })
    return X, pd.Series(minutes), pd.Series(did_play)


def test_quantiles_are_ordered_and_bounded():
    X, mins, played = _data()
    m = ConditionalMinutesDistributionV2().fit(X, mins, played)
    q = m.predict_quantiles(X.head(50))
    assert np.all(np.diff(q, axis=1) >= -1e-9)          # non-decreasing
    assert q.min() >= 0.5 - 1e-9 and q.max() <= 60.0 + 1e-9


def test_deterministic_samples_weights_sum_to_one():
    X, mins, played = _data()
    m = ConditionalMinutesDistributionV2().fit(X, mins, played)
    samples, weights = m.deterministic_samples(X.head(10), n_samples=21)
    assert samples.shape == (10, 21)
    assert weights.shape == (21,)
    assert abs(weights.sum() - 1.0) < 1e-12
    assert np.allclose(weights, 1.0 / 21)               # no arbitrary 50% median weight
    assert samples.min() >= 0.5 - 1e-9 and samples.max() <= 60.0 + 1e-9


def test_reported_cap_enforced():
    X, mins, played = _data()
    m = ConditionalMinutesDistributionV2().fit(X, mins, played)
    Xc = X.head(5).copy()
    Xc["reported_minutes_limit"] = 15.0
    q = m.predict_quantiles(Xc)
    assert q.max() <= 15.0 + 1e-9


def test_higher_start_probability_increases_expected_minutes():
    X, mins, played = _data()
    m = ConditionalMinutesDistributionV2().fit(X, mins, played)
    lo = m.predict_quantiles(pd.DataFrame([{"player_minutes_ewma": 12, "player_minutes_std_l10": 4,
                                            "player_start_rate_ewma": 0.0}]))
    hi = m.predict_quantiles(pd.DataFrame([{"player_minutes_ewma": 32, "player_minutes_std_l10": 4,
                                            "player_start_rate_ewma": 1.0}]))
    assert hi[0, 3] > lo[0, 3]  # median minutes higher for the starter-like row


def test_dnp_rows_excluded_from_training():
    X, mins, played = _data()
    # if DNP zeros leaked in, median would be dragged toward 0; ensure it stays plausible
    m = ConditionalMinutesDistributionV2().fit(X, mins, played)
    q = m.predict_quantiles(X)
    assert np.median(q[:, 3]) > 8.0
