"""P3 Defect #2 — eFG% must be (FGM + 0.5*FG3M)/FGA (was FGA in the numerator)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.features.build_features import (
    _build_shot_quality_features, FEATURE_SCHEMA_VERSION,
)


def _fixture(fgm, fg3m, fga, n=14):
    """A player with constant FGM/FG3M/FGA across n games so the shifted rolling means
    equal those constants for the later games."""
    dates = pd.date_range("2026-05-01", periods=n, freq="D")
    gids = [f"g{i}" for i in range(n)]
    stats = pd.DataFrame({
        "player_id": [1] * n, "game_id": gids, "game_date": dates,
        "fgm": [fgm] * n, "fga": [fga] * n, "fta": [2] * n,
    })
    wide = pd.DataFrame({
        "player_id": [1] * n, "game_id": gids, "game_date": dates,
        "player_fg3m_mean_l10": [fg3m] * n,
        "player_pts_mean_l10": [float(2 * fgm + fg3m)] * n,
        "player_fga_mean_season": [fga] * n,
        "player_fgm_mean_season": [fgm] * n,
        "player_fg3m_mean_season": [fg3m] * n,
    })
    return wide, stats


def test_efg_numerical_fixture_fgm5_fg3m2_fga10_is_060():
    wide, stats = _fixture(fgm=5, fg3m=2, fga=10)
    out = _build_shot_quality_features(wide, stats)
    # last row: shifted rolling means == constants -> eFG = (5 + 0.5*2)/10 = 0.60
    efg = out["player_efg_pct_l10"].dropna()
    assert len(efg) > 0
    assert abs(float(efg.iloc[-1]) - 0.60) < 1e-9


def test_efg_in_plausible_range():
    # A high-volume shooter: FGM=8, FG3M=3, FGA=15 -> (8+1.5)/15 = 0.6333
    wide, stats = _fixture(fgm=8, fg3m=3, fga=15)
    out = _build_shot_quality_features(wide, stats)
    efg = out["player_efg_pct_l10"].dropna()
    assert abs(float(efg.iloc[-1]) - (9.5 / 15)) < 1e-9
    # eFG must be a plausible shooting percentage, never > ~1.5 and never negative
    assert (efg >= 0).all() and (efg <= 1.5).all()


def test_efg_uses_fgm_not_fga_in_numerator():
    # If FGA were (wrongly) in the numerator, eFG would be (10 + 1)/10 = 1.1, not 0.6.
    wide, stats = _fixture(fgm=5, fg3m=2, fga=10)
    out = _build_shot_quality_features(wide, stats)
    efg = float(out["player_efg_pct_l10"].dropna().iloc[-1])
    assert efg < 1.0  # correct (0.60), not the buggy 1.10
    assert FEATURE_SCHEMA_VERSION == "2"
