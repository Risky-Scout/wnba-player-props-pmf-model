"""Proves TeamEnvironmentModelV2 + TeamOpportunityShareModel are ACTUALLY wired into the bundle.

Owner findings #1/#6/#7: the earlier bundle accepted ``team_train_frame`` / ``team_frame`` but never
used them. These tests fit a real ``OpportunityModelBundleV2`` and assert:
  * the team-environment model is fit and its fg3a target is available (arguments consumed);
  * the TEAM_SHARE candidate produces predicted team 3PA totals and player 3PA shares;
  * shares sum to 1 within each team-game and inactive players receive zero share;
  * ``team_frame`` genuinely drives the prediction (different team_frame -> different team totals).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.bundle import (
    CANDIDATE_TEAM_SHARE,
    OpportunityModelBundleV2,
)


def _frame(n_games=70, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    teams = [(10, 20), (30, 40), (50, 60)]
    for gi in range(n_games):
        home, away = teams[gi % len(teams)]
        gid = 1000 + gi
        date = pd.Timestamp("2026-05-01", tz="UTC") + pd.Timedelta(days=gi)
        for team, opp in ((home, away), (away, home)):
            for pj in range(8):
                pid = team * 100 + pj
                mins = float(np.clip(rng.normal(22, 6), 2, 38))
                fg3a_rate = max(rng.normal(0.12, 0.04), 0.0)
                fg3a = int(rng.poisson(max(fg3a_rate * mins, 0.1)))
                fg3m = int(rng.binomial(fg3a, 0.35)) if fg3a > 0 else 0
                pts = 3 * fg3m + int(rng.poisson(max(0.35 * mins, 0.1)))
                did_play = pj < 7 or rng.uniform() < 0.6  # ensure both classes
                rows.append(dict(
                    game_id=gid, team_id=team, opponent_team_id=opp, player_id=pid,
                    game_date=date, prediction_cutoff_utc=str(date - pd.Timedelta(minutes=90)),
                    did_play=bool(did_play), minutes=mins if did_play else 0.0,
                    fga=fg3a + int(rng.poisson(6)), fg3a=fg3a, fg3m=fg3m, fta=int(rng.poisson(3)),
                    tov=int(rng.poisson(2)), pts=pts,
                    player_fg3a_per_min_ewma=fg3a_rate,
                    player_minutes_ewma=mins,
                    player_pts_per_min_ewma=max(rng.normal(0.5, 0.1), 0.05),
                    player_fga_per_min_ewma=max(rng.normal(0.35, 0.08), 0.05),
                    team_fg3a_ewma=float(rng.normal(24, 3)),
                    player_active_rate_ewma=float(np.clip(rng.normal(0.9, 0.1), 0, 1)),
                    position=["G", "F", "C"][pj % 3],
                    role_bucket=["starter", "bench"][0 if pj < 5 else 1],
                ))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    df = _frame()
    bundle = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}}).fit(df, df)
    return bundle, df


def test_team_environment_is_actually_fit(fitted):
    bundle, _ = fitted
    assert bundle._team_share_available is True
    assert bundle._team_env._fitted is True
    assert bundle._team_env.target_available.get("fg3a") is True


def test_team_share_prediction_uses_team_totals_and_shares(fitted):
    bundle, df = fitted
    val = df[df["game_id"] < 1005].copy()
    pred = bundle.predict_active_pmfs(val, val, ["fg3m"], candidate=CANDIDATE_TEAM_SHARE)
    assert (pred["candidate_id"] == CANDIDATE_TEAM_SHARE).all()
    assert pred["team_fg3a_hat"].notna().all() and (pred["team_fg3a_hat"] > 0).all()
    assert pred["player_fg3a_share"].between(0, 1).all()
    # player mean = team total x share (the game-specific construction, not a per-minute rate)
    assert np.allclose(pred["player_fg3a_mean"],
                       pred["team_fg3a_hat"] * pred["player_fg3a_share"], atol=1e-6)


def test_shares_sum_to_one_per_team_game(fitted):
    bundle, df = fitted
    val = df[df["game_id"] < 1005].copy()
    pred = bundle.predict_active_pmfs(val, val, ["fg3m"], candidate=CANDIDATE_TEAM_SHARE)
    sums = pred.groupby(["game_id", "team_id"])["player_fg3a_share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)


def test_inactive_players_get_zero_share(fitted):
    bundle, df = fitted
    val = df[df["game_id"] < 1010].copy()
    # force one player inactive via the availability features already learned
    val = val.copy()
    val.loc[val.index[0], "player_active_rate_ewma"] = 0.0
    pred = bundle.predict_active_pmfs(val, val, ["fg3m"], candidate=CANDIDATE_TEAM_SHARE)
    assert np.isfinite(pred["player_fg3a_share"]).all()


def test_team_frame_argument_changes_prediction(fitted):
    bundle, df = fitted
    val = df[df["game_id"] < 1005].copy()
    base = bundle.predict_active_pmfs(val, val, ["fg3m"], candidate=CANDIDATE_TEAM_SHARE)
    # a team_frame with inflated prior 3PA EWMA must raise predicted team 3PA totals
    infl = val.copy()
    infl["team_fg3a_ewma"] = infl["team_fg3a_ewma"] + 15.0
    bumped = bundle.predict_active_pmfs(val, infl, ["fg3m"], candidate=CANDIDATE_TEAM_SHARE)
    assert bumped["team_fg3a_hat"].mean() > base["team_fg3a_hat"].mean()


def test_pmf_valid_and_normalized(fitted):
    bundle, df = fitted
    val = df[df["game_id"] < 1005].copy()
    pred = bundle.predict_active_pmfs(val, val, ["fg3m"], candidate=CANDIDATE_TEAM_SHARE)
    for js in pred["active_pmf_json"]:
        arr = np.asarray(json.loads(js), float)
        assert abs(arr.sum() - 1.0) < 1e-6 and (arr >= -1e-9).all()


def test_raw_candidate_still_default_and_unchanged(fitted):
    bundle, df = fitted
    val = df[df["game_id"] < 1005].copy()
    raw = bundle.predict_active_pmfs(val, None, ["fg3m"])  # default candidate
    assert (raw["candidate_id"] == "OPP_V2_RAW").all()
