"""End-to-end smoke test for Opportunity V2 (section 34 smoke)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.bundle import OpportunityModelBundleV2
from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)
from wnba_props_model.opportunity.pmf_builders import settled_over_probability


def _slate(seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-05-01T23:00:00Z")
    rows, games = [], []
    for g in range(28):
        gid = 2000 + g
        tip = base + pd.Timedelta(days=g)
        games.append({"game_id": gid, "game_date": tip})
        for p in range(1, 13):
            team = 1 if p <= 6 else 2
            # realistic DNP rate throughout, plus one high-usage player who goes out late
            played = (rng.random() > 0.12) and not (p == 1 and g >= 20)
            mins = float(rng.uniform(10, 33)) if played else 0.0
            fg3a = rng.poisson(5) if played else 0
            rows.append({
                "game_id": gid, "player_id": p, "team_id": team,
                "opponent_team_id": 2 if team == 1 else 1,
                "game_date": tip.tz_convert(None), "minutes": mins, "did_play": played,
                "fga": rng.poisson(11) if played else 0, "fg3a": fg3a, "fta": rng.poisson(3) if played else 0,
                "fg3m": rng.binomial(fg3a, 0.36) if fg3a else 0, "pts": rng.poisson(13) if played else 0,
                "reb": rng.poisson(5) if played else 0, "ast": rng.poisson(3) if played else 0,
                "turnover": rng.poisson(2) if played else 0, "stl": rng.poisson(1) if played else 0,
                "blk": rng.poisson(1) if played else 0, "oreb": rng.poisson(1) if played else 0,
                "dreb": rng.poisson(4) if played else 0,
                "position": "G" if p % 2 else "F", "started_proxy": bool(played and mins > 24),
            })
    return pd.DataFrame(rows), pd.DataFrame(games)


def test_end_to_end_two_fold_smoke_is_deterministic():
    pg, games = _slate()
    frame, manifest = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig(minimum_history_games=1))
    assert manifest["forbidden_market_columns_found"] == []
    frame = frame[frame["player_games_played_prior"] >= 3].reset_index(drop=True)
    cut = pd.Timestamp("2026-05-20T00:00:00Z")
    train = frame[frame["game_date"] < cut]
    val = frame[frame["game_date"] >= cut]

    def run():
        b = OpportunityModelBundleV2().fit(train, train)
        return b.predict_active_pmfs(val, None, ["fg3m", "pts"])

    p1, p2 = run(), run()
    # determinism under same seed
    assert np.allclose(p1["active_pmf_mean"].to_numpy(), p2["active_pmf_mean"].to_numpy(), atol=1e-9)
    # every expected candidate row has a normalized PMF; candidate id fixed
    assert set(p1["candidate_id"]) == {"OPP_V2_RAW"}
    for js in p1["active_pmf_json"]:
        arr = np.array(json.loads(js))
        assert abs(arr.sum() - 1.0) < 1e-6
    # settlement is push-safe and well-formed
    for js in p1[p1["stat"] == "fg3m"]["active_pmf_json"].head(10):
        over, under, push = settled_over_probability(np.array(json.loads(js)), 1.5)
        assert abs((over + under) - 1.0) < 1e-6 and push == 0.0
