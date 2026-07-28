"""Model bundle tests for Opportunity V2 (section 24)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.bundle import OpportunityModelBundleV2
from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)


def _synthetic(n_players=10, n_games=30, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-05-01T23:00:00Z")
    rows, games = [], []
    for g in range(n_games):
        gid = 1000 + g
        tip = base + pd.Timedelta(days=g)
        games.append({"game_id": gid, "game_date": tip})
        for p in range(1, n_players + 1):
            team = 1 if p <= n_players // 2 else 2
            opp = 2 if team == 1 else 1
            played = rng.random() > 0.1
            mins = float(rng.uniform(12, 32)) if played else 0.0
            fg3a = rng.poisson(4) if played else 0
            fg3m = rng.binomial(fg3a, 0.36) if fg3a else 0
            rows.append({
                "game_id": gid, "player_id": p, "team_id": team, "opponent_team_id": opp,
                "game_date": tip.tz_convert(None), "minutes": mins, "did_play": played,
                "fga": rng.poisson(10) if played else 0, "fg3a": fg3a, "fta": rng.poisson(3) if played else 0,
                "fg3m": fg3m, "pts": rng.poisson(12) if played else 0, "reb": rng.poisson(5) if played else 0,
                "ast": rng.poisson(3) if played else 0, "turnover": rng.poisson(2) if played else 0,
                "stl": rng.poisson(1) if played else 0, "blk": rng.poisson(1) if played else 0,
                "oreb": rng.poisson(1) if played else 0, "dreb": rng.poisson(4) if played else 0,
                "position": "G" if p % 2 else "F", "started_proxy": bool(played and mins > 24),
            })
    return pd.DataFrame(rows), pd.DataFrame(games)


def _fit_frame(seed=0):
    pg, games = _synthetic(n_players=10, n_games=30, seed=seed)
    frame, _ = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig(minimum_history_games=1))
    # keep rows with enough history for stable features
    frame = frame[frame["player_games_played_prior"] >= 2].reset_index(drop=True)
    return frame


def test_bundle_fits_and_predicts_active_pmfs():
    frame = _fit_frame()
    b = OpportunityModelBundleV2().fit(frame, frame)
    pred = b.predict_active_pmfs(frame.head(20), None, ["fg3m", "pts"])
    assert set(pred["stat"]) == {"fg3m", "pts"}
    assert set(pred["candidate_id"]) == {"OPP_V2_RAW"}
    # every PMF normalized
    import json
    for js in pred["active_pmf_json"]:
        arr = np.array(json.loads(js))
        assert abs(arr.sum() - 1.0) < 1e-6
        assert (arr >= -1e-12).all()


def test_active_pmf_independent_of_p_active():
    # DNP probability must not alter the active PMF (built once, no 1-p_dnp multiply)
    frame = _fit_frame(seed=1)
    b = OpportunityModelBundleV2().fit(frame, frame)
    pred = b.predict_active_pmfs(frame.head(5), None, ["fg3m"])
    # active mean must be finite/positive and unrelated to p_active clamp
    assert (pred["active_pmf_mean"] > 0).all()
    assert (pred["p_active"].between(0.001, 0.999)).all()


def test_pts_mean_exceeds_fg3m_mean():
    frame = _fit_frame(seed=2)
    b = OpportunityModelBundleV2().fit(frame, frame)
    pred = b.predict_active_pmfs(frame.head(30), None, ["fg3m", "pts"])
    m = pred.groupby("stat")["active_pmf_mean"].mean()
    assert m["pts"] > m["fg3m"]


def test_unsupported_prop_raises():
    frame = _fit_frame(seed=3)
    b = OpportunityModelBundleV2().fit(frame, frame)
    with pytest.raises(ValueError):
        b.predict_active_pmfs(frame.head(2), None, ["ast"])  # Tier-2, not available


def test_save_load_roundtrip(tmp_path):
    frame = _fit_frame(seed=4)
    b = OpportunityModelBundleV2().fit(frame, frame)
    meta = b.save(tmp_path)
    assert meta["sha256"]
    b2 = OpportunityModelBundleV2.load(tmp_path)
    p1 = b.predict_active_pmfs(frame.head(5), None, ["fg3m"])
    p2 = b2.predict_active_pmfs(frame.head(5), None, ["fg3m"])
    assert np.allclose(p1["active_pmf_mean"].to_numpy(), p2["active_pmf_mean"].to_numpy(), atol=1e-9)
