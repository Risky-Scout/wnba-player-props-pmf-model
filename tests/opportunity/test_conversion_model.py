"""Tests for hierarchical baseline rates + Beta conversion (sections 14, 20)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.baseline_rates import HierarchicalLaggedRate
from wnba_props_model.opportunity.conversion_model import HierarchicalBetaConversionModel


def _synthetic_conversion():
    rng = np.random.default_rng(0)
    rows = []
    # two positions with different true 3P%, several players each
    true_p = {"G": 0.40, "F": 0.30}
    for pos, p in true_p.items():
        for pid in range(1, 11):
            attempts = int(rng.integers(20, 120))
            makes = int(rng.binomial(attempts, p))
            rows.append({"position": pos, "role_bucket": "starter", "team_id": 1,
                         "player_id": f"{pos}{pid}", "makes": makes, "attempts": attempts})
    return pd.DataFrame(rows), true_p


HIER = [("position",), ("position", "role_bucket"), ("team_id", "role_bucket"), ("player_id",)]


def test_hierarchical_rate_recovers_group_level():
    df, true_p = _synthetic_conversion()
    m = HierarchicalLaggedRate().fit(df, "makes", "attempts", HIER, prior_strength=50.0)
    g = m.predict(pd.DataFrame([{"position": "G", "role_bucket": "starter", "team_id": 1, "player_id": "G1"}]))
    f = m.predict(pd.DataFrame([{"position": "F", "role_bucket": "starter", "team_id": 1, "player_id": "F1"}]))
    assert g[0] > f[0]
    assert abs(g[0] - true_p["G"]) < 0.08
    assert abs(f[0] - true_p["F"]) < 0.08


def test_unknown_group_falls_back_to_global():
    df, _ = _synthetic_conversion()
    m = HierarchicalLaggedRate().fit(df, "makes", "attempts", HIER, prior_strength=50.0)
    unseen = m.predict(pd.DataFrame([{"position": "C", "role_bucket": "x", "team_id": 99, "player_id": "ZZ"}]))
    global_rate = df["makes"].sum() / df["attempts"].sum()
    assert abs(unseen[0] - global_rate) < 1e-9


def test_low_exposure_player_shrinks_toward_parent():
    df, true_p = _synthetic_conversion()
    # add a tiny-sample extreme player
    df2 = pd.concat([df, pd.DataFrame([{"position": "G", "role_bucket": "starter", "team_id": 1,
                                        "player_id": "SMALL", "makes": 3, "attempts": 3}])], ignore_index=True)
    m = HierarchicalLaggedRate().fit(df2, "makes", "attempts", HIER, prior_strength=50.0)
    small = m.predict(pd.DataFrame([{"position": "G", "role_bucket": "starter", "team_id": 1,
                                     "player_id": "SMALL"}]))[0]
    # 3/3=1.0 raw but must shrink well below 1 toward the guard rate (~0.40)
    assert small < 0.7


def test_beta_posterior_mean_and_uncertainty():
    df, _ = _synthetic_conversion()
    cm = HierarchicalBetaConversionModel().fit(df, "makes", "attempts", HIER, prior_strength=50.0)
    inf = pd.DataFrame([
        {"position": "G", "role_bucket": "starter", "team_id": 1, "player_id": "G1"},   # well observed
        {"position": "C", "role_bucket": "x", "team_id": 99, "player_id": "ZZ"},          # unseen -> global
    ])
    post = cm.predict_posterior(inf)
    assert 0.0 < post[0].mean < 1.0
    # concentration >= prior_strength for every row
    for bp in post:
        assert (bp.alpha + bp.beta) >= 50.0 - 1e-6
    # well-observed guard should have tighter (larger concentration) posterior than unseen fallback
    assert (post[0].alpha + post[0].beta) > (post[1].alpha + post[1].beta) - 1e-6
