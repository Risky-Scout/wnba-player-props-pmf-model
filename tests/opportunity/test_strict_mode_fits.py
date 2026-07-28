"""Strict-mode fit behavior (owner directive item 5).

In strict (certified) mode, a GENUINE submodel fit failure must RAISE a structured BundleFitError
rather than silently degrading to "unavailable". Only when strict_mode is False may the bundle fall
back, and it must record a labeled non-certifiable reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.bundle import BundleFitError, OpportunityModelBundleV2


def _frame(n_games=70, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    teams = [(10, 20), (30, 40)]
    for gi in range(n_games):
        home, away = teams[gi % len(teams)]
        gid = 4000 + gi
        date = pd.Timestamp("2026-05-01", tz="UTC") + pd.Timedelta(days=gi)
        for team, opp in ((home, away), (away, home)):
            for pj in range(8):
                pid = team * 100 + pj
                mins = float(np.clip(rng.normal(24, 6), 3, 38))
                fg3a = int(rng.poisson(max(0.12 * mins, 0.1)))
                fg3m = int(rng.binomial(fg3a, 0.35)) if fg3a > 0 else 0
                fg2a = int(rng.poisson(max(0.25 * mins, 0.1)))
                fg2m = int(rng.binomial(fg2a, 0.48)) if fg2a > 0 else 0
                fta = int(rng.poisson(max(0.10 * mins, 0.1)))
                ftm = int(rng.binomial(fta, 0.80)) if fta > 0 else 0
                pts = 2 * fg2m + 3 * fg3m + ftm
                did_play = pj < 7 or rng.uniform() < 0.6
                rows.append(dict(
                    game_id=gid, team_id=team, opponent_team_id=opp, player_id=pid,
                    game_date=date, prediction_cutoff_utc=str(date - pd.Timedelta(minutes=90)),
                    did_play=bool(did_play), minutes=mins if did_play else 0.0,
                    fga=fg2a + fg3a, fg3a=fg3a, fg3m=fg3m, fg2a=fg2a, fg2m=fg2m,
                    fta=fta, ftm=ftm, tov=int(rng.poisson(2)), pts=pts,
                    player_fg3a_per_min_ewma=max(rng.normal(0.12, 0.03), 0.0),
                    player_minutes_ewma=mins,
                    player_pts_per_min_ewma=max(rng.normal(0.5, 0.1), 0.05),
                    player_fga_per_min_ewma=max(rng.normal(0.37, 0.08), 0.05),
                    team_fg3a_ewma=float(rng.normal(24, 3)),
                    player_active_rate_ewma=float(np.clip(rng.normal(0.9, 0.1), 0, 1)),
                    position=["G", "F", "C"][pj % 3],
                    role_bucket=["starter", "bench"][0 if pj < 5 else 1],
                ))
    return pd.DataFrame(rows)


def _recon(df):
    d = df[df["did_play"]]
    return pd.DataFrame({
        "game_id": d["game_id"].to_numpy(), "player_id": d["player_id"].to_numpy(),
        "FG2M": d["fg2m"].to_numpy(), "FG2A": d["fg2a"].to_numpy(),
        "FG3M": d["fg3m"].to_numpy(), "FG3A": d["fg3a"].to_numpy(),
        "FTM": d["ftm"].to_numpy(), "FTA": d["fta"].to_numpy(),
    })


def _boom(*a, **k):
    raise RuntimeError("boom")


def test_strict_pts_decomp_raises_on_genuine_fit_failure():
    df = _frame()
    b = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}, "strict_mode": True})
    b._conv_2p.fit = _boom  # genuine failure inside _fit_pts_decomposition
    with pytest.raises(BundleFitError, match="pts_decomposition"):
        b.fit(df, df, pts_recon_labels=_recon(df))


def test_nonstrict_pts_decomp_falls_back_and_labels_non_certifiable():
    df = _frame()
    b = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}, "strict_mode": False})
    b._conv_2p.fit = _boom
    b.fit(df, df, pts_recon_labels=_recon(df))  # must NOT raise
    assert b._pts_decomp_available is False
    assert any("pts_decomposition" in r for r in b.non_certifiable_reasons)
    assert "non_certifiable_fallback" in b._pts_decomp_reason


def test_strict_team_share_raises_on_genuine_fit_failure():
    df = _frame()
    b = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}, "strict_mode": True})
    b._team_env.fit = _boom  # genuine failure inside _fit_team_share
    with pytest.raises(BundleFitError, match="team_share"):
        b.fit(df, df)


def test_nonstrict_team_share_falls_back_and_labels_non_certifiable():
    df = _frame()
    b = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}, "strict_mode": False})
    b._team_env.fit = _boom
    b.fit(df, df)  # must NOT raise
    assert b._team_share_available is False
    assert any("team_share" in r for r in b.non_certifiable_reasons)


def test_default_config_is_strict():
    b = OpportunityModelBundleV2({})
    assert b._strict is True
