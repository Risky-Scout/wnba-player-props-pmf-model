"""Leakage guard tests for strictly-lagged PBP opportunity features.

Confirms that (a) every EWMA feature attached to game *g* uses ONLY the player's games strictly
before *g*, and (b) an intentionally leaked (unshifted) feature is caught by the guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.data.pbp_features import (
    PBPFeatureConfig,
    assert_no_leakage,
    build_pbp_features,
)

# one player, five chronological games with increasing 3PA so leakage would be obvious.
_DATES = ["2026-05-08", "2026-05-11", "2026-05-14", "2026-05-17", "2026-05-20"]
BOX = pd.DataFrame([
    {"game_id": g, "player_id": 7, "player_name": "Test Player", "team_id": 1,
     "game_date": d, "minutes": 30.0, "did_play": True}
    for g, d in zip(range(101, 106), _DATES)
])
PARSED = pd.DataFrame([
    {"game_id": g, "player_id": 7, "fg3a": fa, "fg3m": fm, "fga": fa + 3, "fg2a": 3,
     "ast": a, "reb": 4, "oreb": 1, "dreb": 3, "stl": 1, "blk": 0, "tov": 2,
     "fta": 2, "ftm": 2, "fgm": fm + 1, "fg2m": 1, "pts": 3 * fm + 4, "poss_proxy": fa + 5}
    for g, fa, fm, a in zip(range(101, 106), [2, 4, 6, 8, 10], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
])


def test_first_game_has_no_prior_feature():
    feats = build_pbp_features(PARSED, BOX, PBPFeatureConfig(minimum_history_games=0))
    feats = feats.sort_values("game_date").reset_index(drop=True)
    # the earliest game has no prior -> lag EWMA filled to 0 (or 0.35 for the 3P% prior).
    first = feats.iloc[0]
    assert first["player_fg3a_per_min_ewma"] == 0.0
    assert first["player_games_played_prior"] == 0


def test_feature_uses_only_strictly_prior_games():
    cfg = PBPFeatureConfig()
    feats = build_pbp_features(PARSED, BOX, cfg).sort_values("game_date").reset_index(drop=True)
    # game index 2 (third game): fg3a/min EWMA must equal the EWMA over games 0,1 only.
    per_min = pd.Series([2 / 30.0, 4 / 30.0])
    expected = per_min.ewm(halflife=cfg.ewma_halflife_games, adjust=True).mean().iloc[-1]
    got = float(feats.iloc[2]["player_fg3a_per_min_ewma"])
    assert np.isclose(expected, got, rtol=1e-9)


def test_leakage_guard_passes_on_clean_features():
    audit = assert_no_leakage(PARSED, BOX, PBPFeatureConfig(), n_spot_checks=50)
    assert audit["leakage_free"] is True
    assert audit["mismatches"] == 0


def test_leakage_guard_catches_same_game_leak(monkeypatch):
    """If the builder used the CURRENT game (unshifted EWMA), the guard must raise."""
    import wnba_props_model.data.pbp_features as M

    orig = M._ewma_prior

    def leaky(series, halflife):  # no shift(1) -> includes the current game
        return series.ewm(halflife=halflife, adjust=True).mean()

    monkeypatch.setattr(M, "_ewma_prior", leaky)
    with pytest.raises(AssertionError):
        assert_no_leakage(PARSED, BOX, PBPFeatureConfig(), n_spot_checks=50)
    monkeypatch.setattr(M, "_ewma_prior", orig)
