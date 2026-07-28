"""Starting-role model tests for Opportunity V2 (section 16)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.contracts import STARTER_LABEL_OFFICIAL, STARTER_LABEL_PROXY
from wnba_props_model.opportunity.role_model import StartingRoleModelV2


def _data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    start_rate = rng.uniform(0.0, 1.0, n)
    minutes = 8 + 22 * start_rate + rng.normal(0, 2, n)
    started = rng.binomial(1, np.clip(start_rate, 0.02, 0.98)).astype(bool)
    X = pd.DataFrame({"player_start_rate_ewma": start_rate, "player_minutes_ewma": minutes})
    did_play = pd.Series(np.ones(n, dtype=bool))
    quality = pd.Series([STARTER_LABEL_OFFICIAL] * n)
    return X, pd.Series(started), did_play, quality


def test_start_probability_in_bounds_and_monotone():
    X, started, played, quality = _data()
    m = StartingRoleModelV2().fit(X, started, played, quality)
    p = m.predict_start_probability(X)
    assert p.min() >= 0.001 and p.max() <= 0.999
    lo = m.predict_start_probability(pd.DataFrame([{"player_start_rate_ewma": 0.05, "player_minutes_ewma": 9}]))
    hi = m.predict_start_probability(pd.DataFrame([{"player_start_rate_ewma": 0.95, "player_minutes_ewma": 30}]))
    assert hi[0] > lo[0]


def test_proxy_labels_rejected_when_disallowed():
    X, started, played, _ = _data()
    quality = pd.Series([STARTER_LABEL_PROXY] * len(X))
    with pytest.raises(ValueError):
        StartingRoleModelV2(allow_proxy_labels=False).fit(X, started, played, quality)


def test_proxy_labels_used_when_allowed():
    X, started, played, _ = _data()
    quality = pd.Series([STARTER_LABEL_PROXY] * len(X))
    m = StartingRoleModelV2(allow_proxy_labels=True, proxy_label_weight=0.25).fit(X, started, played, quality)
    assert m.predict_start_probability(X).shape[0] == len(X)


def test_only_appearances_used():
    X, started, played, quality = _data()
    played = played.copy()
    played.iloc[:100] = False  # DNP rows must be excluded from training
    m = StartingRoleModelV2().fit(X, started, played, quality)
    assert m._fitted
