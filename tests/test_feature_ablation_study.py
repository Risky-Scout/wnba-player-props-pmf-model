"""Tests for the feature-ablation study harness.

Covered:
* opponent-defense feature group is strictly lagged (leakage guard passes on a
  synthetic round-robin, first-game values are NaN, and second-game values equal
  the strictly-prior recomputation);
* nested rolling-origin CV integrity (outer train precedes val; inner folds are
  a subset of outer-train dates and never intersect the outer eval block);
* group assignment excludes forward-only and market-derived columns and routes
  the remaining columns to the expected groups.

All tests use SYNTHETIC data so they run in CI without the gitignored parquets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.ablation import (
    AblationConfig,
    assert_nested_cv_integrity,
    assign_groups,
    make_expanding_folds,
)
from wnba_props_model.ablation.feature_ablation import _numeric_feature_columns
from wnba_props_model.ablation.opponent_defense import (
    OppDefConfig,
    assert_no_opponent_defense_leakage,
    build_opponent_defense_features,
)


def _synthetic_box(n_dates: int = 16, seed: int = 0) -> pd.DataFrame:
    """Two-team-per-day round-robin over 4 teams; deterministic per-game stats."""
    rng = np.random.default_rng(seed)
    teams = [10, 20, 30, 40]
    rows = []
    gid = 1000
    base = pd.Timestamp("2026-05-01")
    for d in range(n_dates):
        date = base + pd.Timedelta(days=d)
        # rotate the pairing so every team accrues history
        pairs = [(teams[0], teams[1]), (teams[2], teams[3])] if d % 2 == 0 else \
                [(teams[0], teams[2]), (teams[1], teams[3])]
        for home, away in pairs:
            gid += 1
            for team, opp in ((home, away), (away, home)):
                for pl in range(5):
                    rows.append({
                        "game_id": gid, "player_id": team * 100 + pl, "team_id": team,
                        "opponent_team_id": opp, "game_date": date,
                        "pts": float(rng.integers(0, 20)), "reb": float(rng.integers(0, 12)),
                        "ast": float(rng.integers(0, 8)), "fg3m": float(rng.integers(0, 5)),
                        "fg3a": float(rng.integers(0, 9)), "stl": float(rng.integers(0, 4)),
                        "blk": float(rng.integers(0, 3)), "turnover": float(rng.integers(0, 5)),
                        "fga": float(rng.integers(3, 20)), "fta": float(rng.integers(0, 8)),
                        "minutes": float(rng.integers(10, 35)), "did_play": True,
                    })
    return pd.DataFrame(rows)


def test_opponent_defense_leakage_guard_passes():
    box = _synthetic_box()
    res = assert_no_opponent_defense_leakage(box, OppDefConfig(), n_spot_checks=100)
    assert res["leakage_free"] is True
    assert res["rows_checked"] > 0
    assert res["mismatches"] == 0


def test_opponent_defense_first_game_is_nan_and_strictly_prior():
    """First game of each defending team has no prior -> NaN; the second game's
    allowed-EWMA must equal exactly what the opponent allowed in its first game."""
    box = _synthetic_box()
    feats = build_opponent_defense_features(box, OppDefConfig())

    # reconstruct allowed-by-defending-team = opponent players' pts summed per game
    team_tot = box.groupby(["game_id", "team_id"], as_index=False).agg(
        pts=("pts", "sum"), game_date=("game_date", "first"),
        opponent_team_id=("opponent_team_id", "first"))
    # allowed by team T in game g = pts scored by the team T faced
    off = team_tot.rename(columns={"team_id": "off_team", "pts": "allowed"})
    dg = team_tot.rename(columns={"team_id": "def_team", "opponent_team_id": "off_team"})[
        ["game_id", "def_team", "off_team", "game_date"]].merge(
        off[["game_id", "off_team", "allowed"]], on=["game_id", "off_team"], how="left")

    checked_first = checked_second = 0
    for def_team, g in dg.sort_values("game_date").groupby("def_team"):
        g = g.reset_index(drop=True)
        first_gid = g.loc[0, "game_id"]
        # a player on the offense facing def_team in def_team's first game has NaN oppdef ewma
        row_first = feats[(feats["game_id"] == first_gid) &
                          (feats["opponent_team_id"] == def_team)]
        assert row_first["oppdef_pts_allowed_ewma"].isna().all()
        checked_first += 1
        if len(g) >= 2:
            second_gid = g.loc[1, "game_id"]
            expected = float(g.loc[0, "allowed"])  # strictly-prior EWMA of a single game = that game
            row_second = feats[(feats["game_id"] == second_gid) &
                               (feats["opponent_team_id"] == def_team)]
            got = row_second["oppdef_pts_allowed_ewma"].dropna().unique()
            assert len(got) == 1 and np.isclose(got[0], expected)
            checked_second += 1
    assert checked_first >= 3 and checked_second >= 3


def test_injected_opponent_defense_leakage_is_detected(monkeypatch):
    """Positive control: if the builder leaks (uses non-shifted EWMA incl. the
    current game), the brute-force guard must raise."""
    import wnba_props_model.ablation.opponent_defense as od

    real_build = od.build_opponent_defense_features

    def leaky_build(box, config=None):
        feats = real_build(box, config)
        # corrupt one team's allowed-ewma to a clearly-wrong (future-informed) value
        mask = feats["oppdef_pts_allowed_ewma"].notna()
        feats.loc[mask, "oppdef_pts_allowed_ewma"] = feats.loc[mask, "oppdef_pts_allowed_ewma"] + 999.0
        return feats

    monkeypatch.setattr(od, "build_opponent_defense_features", leaky_build)
    with pytest.raises(AssertionError):
        od.assert_no_opponent_defense_leakage(_synthetic_box(), OppDefConfig(), n_spot_checks=50)


def test_make_expanding_folds_train_precedes_val():
    dates = pd.date_range("2026-05-01", periods=60, freq="D").to_numpy()
    folds = make_expanding_folds(dates, n_folds=5, min_train_dates=12)
    assert len(folds) == 5
    for tr, va in folds:
        assert max(tr) < min(va)              # no lookahead
        assert len(tr) >= 12
        assert not (tr & va)                   # disjoint


def test_nested_cv_integrity_enforced():
    dates = pd.date_range("2026-05-01", periods=60, freq="D").to_numpy()
    cfg = AblationConfig()
    res = assert_nested_cv_integrity(dates, cfg)
    assert res["integrity_ok"] is True
    assert res["outer_folds"] >= 2
    assert res["inner_checks"] > 0

    # explicit check: inner folds never see the outer eval block
    outer = make_expanding_folds(dates, cfg.n_outer, cfg.min_train_dates)
    for tr, va in outer:
        inner = make_expanding_folds(np.array(sorted(tr)), cfg.n_inner, cfg.min_inner_train_dates)
        for itr, iva in inner:
            assert (itr | iva).issubset(tr)
            assert not (iva & va)


def test_group_assignment_excludes_forward_only_and_market():
    cols = {
        "opp_reb_allowed_mean_l5": 1.0, "team_pace_proxy_l5": 1.0,
        "player_rest_days": 1.0, "player_pts_std_l10": 1.0, "player_pts_mean_l5": 1.0,
        "starter_rate_l5": 1.0, "pbp_pts_per_min_ewma": 1.0,
        # excluded:
        "teammate_535_is_out": 1.0, "projected_usage_given_absences": 1.0,
        "player_market_p_over_prev": 1.0, "blowout_probability": 1.0,
        # current-game / id (dropped):
        "game_id": 1, "actual_pts": 1.0, "fga": 1.0,
    }
    df = pd.DataFrame({k: [v, v + 1] for k, v in cols.items()})
    feats = _numeric_feature_columns(df)
    assert "teammate_535_is_out" not in feats
    assert "projected_usage_given_absences" not in feats
    assert "player_market_p_over_prev" not in feats
    assert "blowout_probability" not in feats
    assert "game_id" not in feats and "actual_pts" not in feats and "fga" not in feats
    assert "opp_reb_allowed_mean_l5" in feats

    groups = assign_groups(feats, pbp_cols=["pbp_pts_per_min_ewma"])
    assert "pbp_pts_per_min_ewma" in groups["player_pbp_rate"]
    assert "opp_reb_allowed_mean_l5" in groups["opponent_defense"]
    assert "team_pace_proxy_l5" in groups["pace_env"]
    assert "player_rest_days" in groups["schedule"]
    assert "player_pts_std_l10" in groups["dispersion"]
    assert "starter_rate_l5" in groups["role"]


def test_game_total_excluded_from_pure_study():
    """Contract fix: the current-game Vegas columns the legacy regex missed
    (game_total / game_spread_home / implied_team_total) must NOT enter a pure study."""
    cols = {
        "player_pts_mean_l5": 1.0, "opp_reb_allowed_mean_l5": 1.0,
        "game_total": 210.0, "game_spread_home": -3.5, "implied_team_total": 84.0,
        "blowout_risk": 0.0, "predicted_spread_abs": 3.5, "close_game_indicator": 1.0,
    }
    df = pd.DataFrame({k: [v, v + 1] for k, v in cols.items()})
    pure = _numeric_feature_columns(df, contract="pure_compact")
    for leaked in ("game_total", "game_spread_home", "implied_team_total",
                   "blowout_risk", "predicted_spread_abs", "close_game_indicator"):
        assert leaked not in pure, f"{leaked} leaked into pure study"
    assert "player_pts_mean_l5" in pure


def test_game_total_admitted_only_in_game_context_stacked():
    df = pd.DataFrame({"player_pts_mean_l5": [1.0, 2.0], "game_total": [210.0, 205.0]})
    stacked = _numeric_feature_columns(df, contract="game_context_stacked")
    assert "game_total" in stacked
    assert "player_pts_mean_l5" in stacked


def test_provenance_audit_flags_legacy_regex_leak():
    from wnba_props_model.ablation.feature_ablation import audit_forward_only_and_market
    audit = audit_forward_only_and_market(
        ["player_pts_mean_l5", "game_total", "game_spread_home", "implied_team_total"])
    # these three are exactly what the legacy market regex failed to catch
    assert set(audit["leaked_into_pure_by_legacy_regex"]) == {
        "game_total", "game_spread_home", "implied_team_total"}
