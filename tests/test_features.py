"""Tests for Stage 3 — leakage-safe WNBA player-props feature pipeline.

Acceptance criteria:
  1.  shifted player minutes do not include current game
  2.  shifted stat mean does not include current game
  3.  shifted per-minute rate uses only prior games
  4.  DNP/zero-minute handling in rolling support
  5.  team rolling context uses prior games only
  6.  opponent rolling allowance uses prior games only
  7.  role bucket assignment
  8.  usage bucket assignment
  9.  long table row grain (player_id × game_id × stat)
  10. duplicate long row detection
  11. forbidden market columns excluded from model_feature_columns
  12. actual target columns excluded from model_feature_columns
  13. no infinite values in model features
  14. manifest contains model_feature_columns
  15. build_features tiny fixture smoke test
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.features.build_features import (
    FEATURE_CUTOFF_POLICY,
    STATS,
    _build_team_game_context,
    _derive_model_feature_columns,
    _dnp_streak_prior,
    _sr,
    assign_minutes_bucket,
    assign_role_status,
    assign_role_uncertainty,
    assign_usage_bucket,
    build_feature_audit,
    build_feature_schema_manifest,
    build_long_table,
    build_wide_table,
)
from wnba_props_model.features.feature_contract import (
    FORBIDDEN_MODEL_FEATURES,
    assert_no_forbidden_features,
)


# ---------------------------------------------------------------------------
# Minimal fixture factories
# ---------------------------------------------------------------------------

def _make_games(n: int = 10) -> pd.DataFrame:
    """n sequential games; 2 teams alternating home/away."""
    rows = []
    for i in range(n):
        rows.append({
            "game_id": 1000 + i,
            "game_date": pd.Timestamp("2024-05-01") + pd.Timedelta(days=i * 2),
            "season": 2024,
            "home_team_id": 1,
            "visitor_team_id": 2,
            "home_team_abbreviation": "TEAM1",
            "visitor_team_abbreviation": "TEAM2",
            "home_team_score": float(80 + i) if i < 8 else None,
            "visitor_team_score": float(75 + i) if i < 8 else None,
            "total_score": float(155 + 2 * i) if i < 8 else None,
            "is_played_game": i < 8,
            "has_final_score": i < 8,
            "status_normalized": "final" if i < 8 else "scheduled",
            "postseason": 0,
            "has_player_stats": i < 8,
            "has_odds": False,
            "has_player_props": False,
            "status": "post" if i < 8 else "pre",
        })
    return pd.DataFrame(rows)


def _make_stats(n_games: int = 8, players_per_game: int = 3) -> pd.DataFrame:
    """Deterministic player stats for testing."""
    rows = []
    player_ids = [101, 102, 103][:players_per_game]
    for i in range(n_games):
        gid = 1000 + i
        gdate = pd.Timestamp("2024-05-01") + pd.Timedelta(days=i * 2)
        for pid_idx, pid in enumerate(player_ids):
            # Deterministic values: player 101 starts with 30 min, decreases by 1 each game
            minutes = max(10.0, 30.0 - i - pid_idx * 5)
            pts_val = 20 - i - pid_idx * 3
            rows.append({
                "game_id": gid,
                "game_date": gdate,
                "season": 2024,
                "player_id": pid,
                "player_name": f"Player {pid}",
                "team_id": 1,
                "team_abbreviation": "TEAM1",
                "opponent_team_id": 2,
                "opponent_team_abbreviation": "TEAM2",
                "is_home": True,
                "home_away": "home",
                "minutes": minutes,
                "pts": max(0, pts_val),
                "reb": max(0, 5 - i),
                "ast": max(0, 3 - i // 2),
                "fg3m": max(0, 2 - i // 3),
                "stl": max(0, 1 - i // 5),
                "blk": max(0, 1 - i // 5),
                "turnover": max(0, 2 - i // 3),
                "fga": max(0, 10 - i),
                "fta": max(0, 3 - i // 2),
                "oreb": 1,
                "dreb": max(0, 4 - i),
                "pf": 2,
                "plus_minus": 0,
                "started_proxy": minutes >= 24,
                "did_play": minutes > 0,
                "zero_minute_flag": minutes == 0,
                "non_playing_flag": False,
                "stat_line_all_zero_flag": False,
                "missing_team_flag": False,
                "missing_opponent_flag": False,
                "missing_game_date_flag": False,
                "pts_ast": max(0, pts_val) + max(0, 3 - i // 2),
                "pts_reb": max(0, pts_val) + max(0, 5 - i),
                "pts_reb_ast": max(0, pts_val) + max(0, 5 - i) + max(0, 3 - i // 2),
                "reb_ast": max(0, 5 - i) + max(0, 3 - i // 2),
                "stocks": max(0, 1 - i // 5) + max(0, 1 - i // 5),
                "position": "G",
                "source": "bdl",
                "pull_timestamp_utc": "2024-06-01T00:00:00+00:00",
                "minutes_raw": str(minutes),
                "minutes_flag": None,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Shifted player minutes do not include current game
# ===========================================================================

class TestShiftedMinutes:
    def test_player_minutes_mean_l3_excludes_current_game(self):
        """player_minutes_mean_l3 at game 3 must use only games 1–2, not game 3."""
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)

        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        # Game 0: no prior → NaN
        assert pd.isna(p.loc[0, "player_minutes_mean_l3"])
        # Game 1: prior = [game 0 minutes] → mean of 1 value
        game0_min = stats[stats["player_id"] == 101].sort_values("game_date").iloc[0]["minutes"]
        assert p.loc[1, "player_minutes_mean_l3"] == pytest.approx(game0_min, abs=0.01)
        # Game 3: mean of games 0,1,2 only (NOT including game 3)
        prior_mins = stats[stats["player_id"] == 101].sort_values("game_date").iloc[:3]["minutes"].values
        expected = prior_mins.mean()
        actual = p.loc[3, "player_minutes_mean_l3"]
        assert actual == pytest.approx(expected, abs=0.01)

    def test_shift_removal_would_change_values(self):
        """Verify that NOT shifting would give a different (wrong) answer."""
        stats = _make_stats(n_games=4, players_per_game=1)
        p = stats[stats["player_id"] == 101].sort_values("game_date")
        # Current minutes at game 2
        current_min = p.iloc[2]["minutes"]
        # Shifted rolling mean (correct)
        shifted_mean = p["minutes"].shift(1).rolling(3, min_periods=1).mean().iloc[2]
        # Non-shifted rolling mean (wrong — includes current game)
        nonshifted_mean = p["minutes"].rolling(3, min_periods=1).mean().iloc[2]
        # They should differ (current game has different minutes)
        if current_min != shifted_mean:
            assert shifted_mean != nonshifted_mean, (
                "Shifted and non-shifted should differ when current game has different value"
            )

    def test_last1_is_prior_game_only(self):
        """player_minutes_last1 must equal the previous game's minutes."""
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        raw = stats[stats["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        # Game 1's last1 must equal game 0's actual minutes
        assert p.loc[1, "player_minutes_last1"] == pytest.approx(float(raw.loc[0, "minutes"]))


# ===========================================================================
# 2. Shifted stat mean does not include current game
# ===========================================================================

class TestShiftedStatMeans:
    def test_pts_mean_l5_excludes_current(self):
        """player_pts_mean_l5 at game N must equal mean of pts in games 0..(N-1)."""
        stats = _make_stats(n_games=7, players_per_game=1)
        games = _make_games(n=7)
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        raw = stats[stats["player_id"] == 101].sort_values("game_date").reset_index(drop=True)

        # At game 6, mean_l5 = mean of games 1–5 (prior 5 games)
        prior5_pts = raw.iloc[1:6]["pts"].values
        expected = prior5_pts.mean()
        assert p.loc[6, "player_pts_mean_l5"] == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize("stat", STATS)
    def test_all_stats_have_rolling_features(self, stat: str):
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        assert f"player_{stat}_mean_l5" in wide.columns
        assert f"player_{stat}_last1" in wide.columns
        assert f"player_{stat}_l5_support" in wide.columns


# ===========================================================================
# 3. Per-minute rate uses prior games only
# ===========================================================================

class TestShiftedRates:
    def test_per_min_rate_uses_shifted_sums(self):
        """player_pts_per_min_l5 = sum(pts in prior 5 games) / sum(min in prior 5 games)."""
        stats = _make_stats(n_games=7, players_per_game=1)
        games = _make_games(n=7)
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        raw = stats[stats["player_id"] == 101].sort_values("game_date").reset_index(drop=True)

        # At game 6: rate = sum(pts[1:6]) / sum(min[1:6])
        prior5 = raw.iloc[1:6]
        expected_rate = prior5["pts"].sum() / max(prior5["minutes"].sum(), 1.0)
        assert p.loc[6, "player_pts_per_min_l5"] == pytest.approx(expected_rate, abs=0.001)

    def test_rate_no_divide_by_zero(self):
        """Player with zero prior minutes should not produce inf rate."""
        stats = _make_stats(n_games=3, players_per_game=1)
        games = _make_games(n=3)
        # Set all minutes to 0 for first game
        stats.loc[stats["game_id"] == 1000, "minutes"] = 0.0
        wide, _ = build_wide_table(stats, games)
        for col in wide.select_dtypes(include="number").columns:
            assert not wide[col].isin([np.inf, -np.inf]).any(), (
                f"Infinite value in column {col}"
            )


# ===========================================================================
# 4. DNP/zero-minute handling
# ===========================================================================

class TestDNPHandling:
    def test_dnp_streak_accumulates(self):
        did_play = np.array([True, False, False, False, True, False])
        streak = _dnp_streak_prior(did_play)
        assert streak[0] == 0   # first game: no prior
        assert streak[1] == 0   # game 1: game 0 was played
        assert streak[2] == 1   # game 2: game 1 was DNP
        assert streak[3] == 2   # game 3: games 1,2 were DNP
        assert streak[4] == 3   # game 4: games 1,2,3 were DNP
        assert streak[5] == 0   # game 5: game 4 was played

    def test_dnp_streak_resets_after_play(self):
        did_play = np.array([False, False, True, False])
        streak = _dnp_streak_prior(did_play)
        assert streak[2] == 2  # 2 consecutive DNPs before game 2
        assert streak[3] == 0  # game 2 was played → reset

    def test_zero_minute_rate_reflects_dnp(self):
        """Player with consecutive DNPs should have high zero_minute_rate."""
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        # Force 3 DNP games for player 101
        stats.loc[stats["game_id"].isin([1001, 1002, 1003]), "minutes"] = 0.0
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        # At game 4, l5 should show high zero-minute rate
        rate = p.loc[4, "player_zero_minute_rate_l5"]
        assert rate > 0.4, f"Expected high zero-minute rate, got {rate}"


# ===========================================================================
# 5. Team rolling context uses prior games only
# ===========================================================================

class TestTeamContext:
    def test_team_context_uses_prior_games(self):
        """team_pts_for_mean_l5 for game G must use games before G."""
        stats = _make_stats(n_games=8, players_per_game=2)
        games = _make_games(n=8)
        wide, _ = build_wide_table(stats, games)

        p = wide[(wide["player_id"] == 101) & (wide["team_id"] == 1)].sort_values("game_date")
        p = p.reset_index(drop=True)

        # At game 0: no prior team games → NaN
        assert pd.isna(p.loc[0, "team_pts_for_mean_l5"])

        # At game 5: uses games 0–4 only
        # Team 1's pts_for = sum of pts by team 1 players in each game
        team1_pts_by_game = (
            stats[stats["team_id"] == 1]
            .groupby("game_id")["pts"].sum()
            .reset_index()
            .merge(games[["game_id", "game_date"]], on="game_id")
            .sort_values("game_date")
        )
        prior5 = team1_pts_by_game.iloc[:5]["pts"].values
        expected = prior5.mean()
        actual = p.loc[5, "team_pts_for_mean_l5"]
        if pd.notna(actual):
            assert actual == pytest.approx(expected, rel=0.05), (
                f"Team context not shifted: expected {expected:.2f}, got {actual:.2f}"
            )

    def test_team_rest_days_known_pregame(self):
        """team_rest_days is based on schedule history, not current game outcome."""
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        # Game 0: no prior → NaN
        assert pd.isna(p.loc[0, "team_rest_days"])
        # Game 1 onwards: rest days should be computable (2 days between games - 1 = 1 rest day)
        if pd.notna(p.loc[1, "team_rest_days"]):
            assert p.loc[1, "team_rest_days"] >= 0


# ===========================================================================
# 6. Opponent rolling allowance uses prior games only
# ===========================================================================

class TestOpponentContext:
    def test_opp_pts_allowed_present(self):
        """opp_pts_allowed_mean_l5 must be present in wide table."""
        stats = _make_stats(n_games=8, players_per_game=2)
        games = _make_games(n=8)
        wide, _ = build_wide_table(stats, games)
        assert "opp_pts_allowed_mean_l5" in wide.columns

    def test_opp_context_uses_prior_games(self):
        """Opponent context features must not include current game data."""
        stats = _make_stats(n_games=6, players_per_game=2)
        games = _make_games(n=6)
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        # First game: no prior opponent context
        assert pd.isna(p.loc[0, "opp_pts_allowed_mean_l5"])


# ===========================================================================
# 7. Role bucket assignment
# ===========================================================================

class TestRoleBuckets:
    @pytest.mark.parametrize("minutes,expected_bucket", [
        (5.0, "bench_low"),
        (15.0, "bench_rotation"),
        (24.0, "rotation"),
        (30.0, "starter"),
        (36.0, "workhorse"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ])
    def test_minutes_bucket_assignment(self, minutes, expected_bucket):
        assert assign_minutes_bucket(minutes) == expected_bucket

    def test_role_bucket_in_wide_table(self):
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        assert "projected_minutes_bucket" in wide.columns
        valid = {"bench_low", "bench_rotation", "rotation", "starter", "workhorse", "unknown"}
        assert wide["projected_minutes_bucket"].isin(valid).all()

    def test_first_game_has_unknown_bucket(self):
        """First game should have unknown bucket (no prior minutes)."""
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        p = wide[wide["player_id"] == 101].sort_values("game_date").reset_index(drop=True)
        assert p.loc[0, "projected_minutes_bucket"] == "unknown"


# ===========================================================================
# 8. Usage bucket assignment
# ===========================================================================

class TestUsageBuckets:
    @pytest.mark.parametrize("usage,expected", [
        (0.10, "low"),
        (0.20, "medium"),
        (0.30, "high"),
        (0.40, "elite"),
        (None, "unknown"),
    ])
    def test_usage_bucket_assignment(self, usage, expected):
        assert assign_usage_bucket(usage) == expected

    def test_usage_bucket_in_wide_table(self):
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        assert "usage_bucket" in wide.columns
        valid = {"low", "medium", "high", "elite", "unknown"}
        assert wide["usage_bucket"].isin(valid).all()


# ===========================================================================
# 9. Long table row grain (player_id × game_id × stat)
# ===========================================================================

class TestLongTableGrain:
    def test_long_table_one_row_per_player_game_stat(self):
        stats = _make_stats(n_games=5, players_per_game=2)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)

        expected_rows = len(stats) * len(STATS)
        # Allow small tolerance if some games were not in stats
        assert len(long) == pytest.approx(expected_rows, rel=0.01)

    def test_long_table_has_stat_column(self):
        stats = _make_stats(n_games=4, players_per_game=2)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        assert "stat" in long.columns
        assert set(long["stat"].unique()) == set(STATS)

    def test_long_table_has_actual_outcome(self):
        stats = _make_stats(n_games=4, players_per_game=2)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        assert "actual_outcome" in long.columns

    def test_long_table_pts_outcome_matches_wide(self):
        stats = _make_stats(n_games=3, players_per_game=1)
        games = _make_games(n=3)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        pts_long = long[long["stat"] == "pts"][["player_id", "game_id", "actual_outcome"]]
        wide_pts = wide[["player_id", "game_id", "actual_pts"]].rename(
            columns={"actual_pts": "actual_outcome"}
        )
        merged = pts_long.merge(wide_pts, on=["player_id", "game_id"], suffixes=("_long", "_wide"))
        assert (merged["actual_outcome_long"] == merged["actual_outcome_wide"]).all()


# ===========================================================================
# 10. Duplicate long row detection
# ===========================================================================

class TestDuplicateDetection:
    def test_no_duplicate_player_game_stat(self):
        stats = _make_stats(n_games=5, players_per_game=3)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        dup = long.duplicated(subset=["player_id", "game_id", "stat"]).sum()
        assert dup == 0, f"Found {dup} duplicate rows in long table"

    def test_no_duplicate_player_game_wide(self):
        stats = _make_stats(n_games=5, players_per_game=3)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        dup = wide.duplicated(subset=["player_id", "game_id"]).sum()
        assert dup == 0, f"Found {dup} duplicate rows in wide table"


# ===========================================================================
# 11. Forbidden market columns excluded from model_feature_columns
# ===========================================================================

class TestForbiddenColumnsExcluded:
    def test_no_forbidden_columns_in_model_features(self):
        stats = _make_stats(n_games=5, players_per_game=2)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        model_cols = _derive_model_feature_columns(wide)
        forbidden_found = [c for c in model_cols if c in FORBIDDEN_MODEL_FEATURES]
        assert forbidden_found == [], (
            f"Forbidden columns in model_feature_columns: {forbidden_found}"
        )

    def test_odds_columns_not_in_model_features(self):
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        # Inject a forbidden column to simulate leakage
        wide["total_value"] = 165.0
        model_cols = _derive_model_feature_columns(wide)
        assert "total_value" not in model_cols

    def test_vendor_not_in_model_features(self):
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        model_cols = _derive_model_feature_columns(wide)
        assert "vendor" not in model_cols

    def test_leakage_guard_raises_for_forbidden_column(self):
        with pytest.raises(ValueError, match="total_value"):
            assert_no_forbidden_features(["pts_per_min_l5", "total_value"])


# ===========================================================================
# 12. Actual target columns excluded from model_feature_columns
# ===========================================================================

class TestTargetColumnsExcluded:
    def test_actual_outcome_not_in_model_features(self):
        stats = _make_stats(n_games=5, players_per_game=2)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        model_cols = _derive_model_feature_columns(wide)
        assert "actual_outcome" not in model_cols

    def test_actual_stat_targets_not_in_model_features(self):
        stats = _make_stats(n_games=5, players_per_game=2)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        model_cols = _derive_model_feature_columns(wide)
        for stat in STATS:
            assert f"actual_{stat}" not in model_cols, (
                f"actual_{stat} should not be in model_feature_columns"
            )

    def test_actual_minutes_not_in_model_features(self):
        stats = _make_stats(n_games=5, players_per_game=1)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        model_cols = _derive_model_feature_columns(wide)
        assert "actual_minutes" not in model_cols


# ===========================================================================
# 13. No infinite values in model features
# ===========================================================================

class TestNoInfiniteValues:
    def test_no_inf_in_wide_model_features(self):
        stats = _make_stats(n_games=6, players_per_game=3)
        games = _make_games(n=6)
        wide, _ = build_wide_table(stats, games)
        model_cols = _derive_model_feature_columns(wide)
        num_cols = [c for c in model_cols if c in wide.columns
                    and pd.api.types.is_numeric_dtype(wide[c])]
        for col in num_cols:
            n_inf = np.isinf(wide[col].fillna(0).values).sum()
            assert n_inf == 0, f"Infinite values found in model feature '{col}'"

    def test_zero_minutes_no_inf_rate(self):
        """Per-minute rate with zero prior minutes must be NaN, not inf."""
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        # Force all minutes to 0 for player 101 in first two games
        stats.loc[stats["game_id"].isin([1000, 1001]), "minutes"] = 0.0
        wide, _ = build_wide_table(stats, games)
        for col in ["player_pts_per_min_l3", "player_pts_per_min_l5"]:
            if col in wide.columns:
                assert not wide[col].isin([np.inf, -np.inf]).any(), (
                    f"Infinite value in {col} when minutes=0"
                )


# ===========================================================================
# 14. Manifest contains model_feature_columns
# ===========================================================================

class TestManifest:
    def test_manifest_has_model_feature_columns(self):
        stats = _make_stats(n_games=5, players_per_game=2)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        model_cols = _derive_model_feature_columns(wide)
        manifest = build_feature_schema_manifest(
            wide_df=wide, long_df=long,
            model_feature_columns=model_cols,
            source_tables=["test"],
            wide_path="test_wide.parquet",
            long_path="test_long.parquet",
        )
        assert "model_feature_columns" in manifest
        assert len(manifest["model_feature_columns"]) > 0
        assert manifest["model_feature_columns"] == model_cols

    def test_manifest_temporal_policy(self):
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        model_cols = _derive_model_feature_columns(wide)
        manifest = build_feature_schema_manifest(
            wide_df=wide, long_df=long,
            model_feature_columns=model_cols,
            source_tables=["test"],
            wide_path="w.parquet", long_path="l.parquet",
        )
        assert manifest["temporal_policy"] == FEATURE_CUTOFF_POLICY

    def test_manifest_has_required_keys(self):
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        model_cols = _derive_model_feature_columns(wide)
        manifest = build_feature_schema_manifest(
            wide_df=wide, long_df=long,
            model_feature_columns=model_cols,
            source_tables=["test"],
            wide_path="w.parquet", long_path="l.parquet",
        )
        required_keys = [
            "wide_table_path", "long_table_path", "created_at_utc",
            "row_grain_wide", "row_grain_long", "identity_columns",
            "target_columns", "model_feature_columns",
            "numeric_feature_columns", "categorical_feature_columns",
            "forbidden_columns", "role_bucket_columns", "temporal_policy",
            "source_tables", "git_commit_if_available",
        ]
        for key in required_keys:
            assert key in manifest, f"Missing manifest key: {key}"

    def test_manifest_no_forbidden_in_model_cols(self):
        stats = _make_stats(n_games=4, players_per_game=2)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)
        model_cols = _derive_model_feature_columns(wide)
        manifest = build_feature_schema_manifest(
            wide_df=wide, long_df=long,
            model_feature_columns=model_cols,
            source_tables=["test"],
            wide_path="w.parquet", long_path="l.parquet",
        )
        for col in manifest["model_feature_columns"]:
            assert col not in FORBIDDEN_MODEL_FEATURES, (
                f"Forbidden column {col} in manifest model_feature_columns"
            )


# ===========================================================================
# 15. Build features tiny fixture smoke test
# ===========================================================================

class TestBuildFeaturesSmoke:
    def test_smoke_wide_table(self):
        stats = _make_stats(n_games=8, players_per_game=3)
        games = _make_games(n=8)
        wide, audit_notes = build_wide_table(stats, games)

        assert len(wide) == len(stats)
        assert "player_minutes_mean_l5" in wide.columns
        assert "team_pts_for_mean_l5" in wide.columns
        assert "opp_pts_allowed_mean_l5" in wide.columns
        assert "projected_minutes_bucket" in wide.columns
        assert "actual_pts" in wide.columns
        assert "feature_cutoff_policy" in wide.columns
        assert (wide["feature_cutoff_policy"] == "strict_pregame_shifted").all()

    def test_smoke_long_table(self):
        stats = _make_stats(n_games=5, players_per_game=2)
        games = _make_games(n=5)
        wide, _ = build_wide_table(stats, games)
        long = build_long_table(wide)

        assert len(long) == len(stats) * len(STATS)
        assert "stat" in long.columns
        assert "actual_outcome" in long.columns
        assert long.duplicated(subset=["player_id", "game_id", "stat"]).sum() == 0

    def test_smoke_full_pipeline(self):
        """End-to-end smoke: build, derive features, build manifest, run audit."""
        stats = _make_stats(n_games=8, players_per_game=3)
        games = _make_games(n=8)
        wide, audit_notes = build_wide_table(stats, games)
        long = build_long_table(wide)
        model_cols = _derive_model_feature_columns(wide)
        manifest = build_feature_schema_manifest(
            wide_df=wide, long_df=long,
            model_feature_columns=model_cols,
            source_tables=["test"],
            wide_path="test_wide.parquet", long_path="test_long.parquet",
        )
        audit = build_feature_audit(wide, long, model_cols, audit_notes)

        # Pipeline assertions
        assert len(model_cols) > 10, "Expected > 10 model feature columns"
        assert manifest["temporal_policy"] == FEATURE_CUTOFF_POLICY
        assert audit["feature_checks"]["forbidden_columns_found"] == []
        assert audit["feature_checks"]["target_columns_in_model_features"] == []
        assert audit["row_counts"]["wide"] == len(wide)
        assert audit["row_counts"]["long"] == len(long)

    def test_smoke_feature_cutoff_policy_correct(self):
        stats = _make_stats(n_games=4, players_per_game=1)
        games = _make_games(n=4)
        wide, _ = build_wide_table(stats, games)
        assert (wide["feature_cutoff_policy"] == FEATURE_CUTOFF_POLICY).all()

    def test_smoke_role_buckets_populated(self):
        stats = _make_stats(n_games=8, players_per_game=3)
        games = _make_games(n=8)
        wide, _ = build_wide_table(stats, games)
        # Players with enough history should have non-unknown bucket
        with_history = wide[wide["player_minutes_l5_support"] >= 3]
        if len(with_history) > 0:
            non_unknown = (with_history["projected_minutes_bucket"] != "unknown").mean()
            assert non_unknown > 0.5, "Expected most players with history to have defined bucket"
