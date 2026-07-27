"""Team environment model tests for Opportunity V2 (section 18)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.team_environment import TeamEnvironmentModelV2, box_possessions


def _team_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pace = rng.normal(96, 5, n)
    fga_l = rng.normal(70, 4, n)
    rows = pd.DataFrame({
        "pace_ewma": pace,
        "fga_ewma": fga_l,
        "possessions": np.clip(pace + rng.normal(0, 2, n), 60, None),
        "fga": np.clip(fga_l + rng.normal(0, 3, n), 40, None),
        "fg3a": np.clip(0.35 * fga_l + rng.normal(0, 2, n), 5, None),
        "fta": np.clip(0.2 * fga_l + rng.normal(0, 2, n), 2, None),
    })
    return rows


def test_box_possessions_formula():
    assert abs(box_possessions(np.array([80.0]), np.array([20.0]),
                               np.array([10.0]), np.array([12.0]))[0] - (80 + 0.44 * 20 - 10 + 12)) < 1e-9


def test_fits_available_targets_and_predicts_nonnegative():
    df = _team_frame()
    m = TeamEnvironmentModelV2(reconcile_possessions=False).fit(df, ["pace_ewma", "fga_ewma"])
    assert m.target_available["possessions"] and m.target_available["fga"]
    # tracking targets absent -> unavailable, not fabricated
    assert not m.target_available["potential_assists"]
    pred = m.predict(df.head(20))
    assert (pred["possessions"] >= 0).all()
    assert (pred["fga"] >= 0).all()
    assert not bool(pred["potential_assists_available"].iloc[0])


def test_possession_reconciliation_averages_two_teams():
    # Build two teams per game with known differing possession features; reconciled should equalize.
    df = _team_frame(n=300, seed=1)
    m = TeamEnvironmentModelV2(reconcile_possessions=True).fit(df, ["pace_ewma", "fga_ewma"])
    infer = pd.DataFrame({
        "game_id": [1, 1],
        "team_id": [10, 20],
        "opponent_team_id": [20, 10],
        "pace_ewma": [100.0, 90.0],
        "fga_ewma": [72.0, 68.0],
    })
    pred = m.predict(infer)
    # after reconciliation both teams share the same possession estimate
    assert abs(pred["possessions"].iloc[0] - pred["possessions"].iloc[1]) < 1e-6
