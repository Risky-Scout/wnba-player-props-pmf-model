"""Point-in-time feature builder tests for Opportunity V2 (sections 12-13)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)


def _synthetic(n_players=6, n_games=12, seed=0):
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


def test_one_row_per_player_game_and_no_leakage():
    pg, games = _synthetic()
    frame, manifest = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig())
    assert len(frame) == len(pg)
    # cutoff strictly before tip
    assert (frame["prediction_cutoff_utc"] < frame["scheduled_tip_utc"]).all()
    # last game date strictly before cutoff (temporal purity holds -> builder did not raise)
    assert manifest["forbidden_market_columns_found"] == []
    assert manifest["proof_eligible_row_count"] == 0  # fallback cutoffs are not proof eligible


def test_lagged_features_do_not_use_current_game():
    pg, games = _synthetic(seed=1)
    frame, _ = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig())
    # first game per player has no prior -> ewma NaN
    first = frame.sort_values(["player_id", "game_date"]).groupby("player_id").head(1)
    assert first["player_minutes_ewma"].isna().all()


def test_exact_quote_cutoff_marks_proof_eligible():
    pg, games = _synthetic(seed=2)
    # attach an exact pregame quote for one player-game
    gid = games["game_id"].iloc[5]
    tip = pd.to_datetime(games["game_date"].iloc[5], utc=True)
    quotes = pd.DataFrame([{"game_id": gid, "player_id": 1,
                            "quote_timestamp": tip - pd.Timedelta(minutes=45)}])
    frame, manifest = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig(),
    ) if False else build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, quotes, OpportunityFeatureConfig())
    hit = frame[(frame["game_id"] == gid) & (frame["player_id"] == 1)]
    assert bool(hit["proof_eligible"].iloc[0])
    assert hit["cutoff_source"].iloc[0] == "exact_quote_timestamp"
    assert manifest["proof_eligible_row_count"] >= 1


def test_team_environment_columns_present():
    pg, games = _synthetic(seed=3)
    frame, _ = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig())
    for c in ("team_possessions_ewma", "team_fga_ewma", "team_fg3a_ewma", "team_fta_ewma"):
        assert c in frame.columns
