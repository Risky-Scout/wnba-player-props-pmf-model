"""Team-coherent opportunity-share tests for Opportunity V2 (section 19)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.share_model import TeamOpportunityShareModel

GROUP = ("game_id", "team_id", "prediction_cutoff_utc")
CUT = pd.Timestamp("2026-05-08T22:00:00Z")


def _training(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(60):
        base = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}
        total = int(rng.integers(60, 90))
        # allocate opportunities roughly by baseline
        for pid, b in base.items():
            num = rng.binomial(total, b) / 4  # per-player numerator
            rows.append({
                "game_id": g, "team_id": 1, "prediction_cutoff_utc": CUT,
                "player_id": pid, "baseline_share": b, "p_active": 1.0,
                "team_total": total / 4.0, "num": num,
                "feat_min": b * 30, "feat_role": 1.0,
            })
    return pd.DataFrame(rows)


def test_shares_sum_to_one_per_team_group():
    df = _training()
    m = TeamOpportunityShareModel("touch_share", "num", "team_total", "baseline_share",
                                  ["feat_min", "feat_role"]).fit(df)
    shares = m.predict_team_normalized_shares(df)
    df = df.assign(share=shares)
    sums = df.groupby(list(GROUP))["share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_unavailable_star_redistributes_to_active_baseline():
    df = _training()
    m = TeamOpportunityShareModel("touch_share", "num", "team_total", "baseline_share",
                                  ["feat_min", "feat_role"]).fit(df)
    # single team-game with A unavailable (p_active ~ 0). deltas won't be exactly zero from the
    # learned model, but with A removed the renormalized baseline dominates; check A gets ~0 share
    # and B/C/D keep their relative ordering summing to 1.
    infer = pd.DataFrame([
        {"game_id": 999, "team_id": 1, "prediction_cutoff_utc": CUT, "player_id": "A",
         "baseline_share": 0.40, "p_active": 0.0, "feat_min": 12.0, "feat_role": 1.0},
        {"game_id": 999, "team_id": 1, "prediction_cutoff_utc": CUT, "player_id": "B",
         "baseline_share": 0.30, "p_active": 1.0, "feat_min": 9.0, "feat_role": 1.0},
        {"game_id": 999, "team_id": 1, "prediction_cutoff_utc": CUT, "player_id": "C",
         "baseline_share": 0.20, "p_active": 1.0, "feat_min": 6.0, "feat_role": 1.0},
        {"game_id": 999, "team_id": 1, "prediction_cutoff_utc": CUT, "player_id": "D",
         "baseline_share": 0.10, "p_active": 1.0, "feat_min": 3.0, "feat_role": 1.0},
    ])
    shares = m.predict_team_normalized_shares(infer)
    assert abs(shares.sum() - 1.0) < 1e-9
    assert shares[0] < 1e-6                    # inactive A gets ~0
    assert shares[1] > shares[2] > shares[3]   # B>C>D ordering preserved


def test_missing_group_column_raises():
    df = _training()
    m = TeamOpportunityShareModel("touch_share", "num", "team_total", "baseline_share",
                                  ["feat_min", "feat_role"]).fit(df)
    with pytest.raises(RuntimeError):
        m.predict_team_normalized_shares(df.drop(columns=["team_id"]))


def test_analytical_renormalization_when_no_learned_signal():
    # baseline renormalization identity: A out, deltas ~ 0 -> B:0.5 C:0.333 D:0.166
    from wnba_props_model.opportunity.share_model import TeamOpportunityShareModel as SM
    df = _training()
    m = SM("s", "num", "team_total", "baseline_share", ["feat_role"]).fit(df)
    # feat_role constant -> model predicts (near) constant delta; renormalized baseline dominates
    infer = pd.DataFrame([
        {"game_id": 7, "team_id": 1, "prediction_cutoff_utc": CUT, "player_id": p,
         "baseline_share": b, "p_active": (0.0 if p == "A" else 1.0), "feat_role": 1.0}
        for p, b in [("A", 0.40), ("B", 0.30), ("C", 0.20), ("D", 0.10)]
    ])
    shares = m.predict_team_normalized_shares(infer)
    assert shares[0] < 1e-6
    # B:C:D should be ~ 0.30:0.20:0.10 -> 0.5:0.333:0.166
    assert abs(shares[1] - 0.5) < 0.05
    assert abs(shares[2] - 1 / 3) < 0.05
    assert abs(shares[3] - 1 / 6) < 0.05
