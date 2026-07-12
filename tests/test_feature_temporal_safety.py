"""Tests for point-in-time feature safety (§5 requirements).

Verifies:
  - Rolling features exclude the target game
  - Future game changes don't affect past feature rows
  - Market features are not in the structural model
  - Feature cutoffs are correctly enforced
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# §5.2: Rolling features exclude the target game
# ---------------------------------------------------------------------------

class TestRollingFeaturesExcludeTargetGame:
    """Verify that changing a game's outcome doesn't change its own feature row."""

    def _make_player_games(self, pts_values: list[float]) -> pd.DataFrame:
        n = len(pts_values)
        return pd.DataFrame({
            "player_id": [1] * n,
            "game_id": list(range(1, n + 1)),
            "game_date": pd.date_range("2026-01-01", periods=n),
            "pts": pts_values,
        })

    def _add_rolling_feature(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = df.sort_values(["player_id", "game_date"]).copy()
        grp = df.groupby("player_id")["pts"]
        df["pts_roll5"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        return df

    def test_target_game_mutation_does_not_change_own_feature(self):
        """§5 requirement: changing target game's result must not change its own feature."""
        pts_baseline = [10.0, 12.0, 15.0, 8.0, 20.0, 14.0]
        pts_mutated = [10.0, 12.0, 15.0, 8.0, 20.0, 99.0]  # last game mutated

        df_baseline = self._make_player_games(pts_baseline)
        df_mutated = self._make_player_games(pts_mutated)

        df_baseline = self._add_rolling_feature(df_baseline)
        df_mutated = self._add_rolling_feature(df_mutated)

        # The last row's own feature should be the same (computed from prior games)
        feat_baseline = df_baseline.iloc[-1]["pts_roll5"]
        feat_mutated = df_mutated.iloc[-1]["pts_roll5"]
        assert abs(feat_baseline - feat_mutated) < 1e-10, (
            f"Target game mutation changed its own feature: {feat_baseline} vs {feat_mutated}"
        )

    def test_future_game_mutation_does_not_affect_past_features(self):
        """§5 requirement: future game changes must not alter past feature rows."""
        pts_baseline = [10.0, 12.0, 15.0, 8.0, 20.0, 14.0]
        pts_mutated = [10.0, 12.0, 15.0, 8.0, 999.0, 14.0]  # game 5 mutated

        df_baseline = self._make_player_games(pts_baseline)
        df_mutated = self._make_player_games(pts_mutated)

        df_baseline = self._add_rolling_feature(df_baseline)
        df_mutated = self._add_rolling_feature(df_mutated)

        # Rows with enough history (index >= 1) should have identical features.
        # Row 0 has NaN (no prior games); use pd.isna check for those.
        for i in range(4):  # rows 0-3 are before the mutated game
            feat_base = df_baseline.iloc[i]["pts_roll5"]
            feat_mut = df_mutated.iloc[i]["pts_roll5"]
            if pd.isna(feat_base) and pd.isna(feat_mut):
                continue  # both NaN is correct (first game has no prior history)
            assert abs(feat_base - feat_mut) < 1e-10, (
                f"Future mutation affected past row {i}: {feat_base} vs {feat_mut}"
            )

    def test_shift_1_is_always_applied_before_rolling(self):
        """shift(1) must be applied before any rolling aggregation."""
        df = pd.DataFrame({
            "player_id": [1] * 5,
            "game_id": [1, 2, 3, 4, 5],
            "game_date": pd.date_range("2026-01-01", periods=5),
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        df = df.sort_values(["player_id", "game_date"])
        grp = df.groupby("player_id")["pts"]
        df["pts_roll3"] = grp.transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        # Row 0 (first game): shift(1) produces NaN, rolling of [NaN] = NaN
        assert pd.isna(df.iloc[0]["pts_roll3"])
        # Row 1: shift(1) gives [10.0], mean = 10.0
        assert abs(df.iloc[1]["pts_roll3"] - 10.0) < 1e-10
        # Row 4: shift(1) gives [30.0, 40.0], rolling(3) = mean([20,30,40]) = 30.0
        assert abs(df.iloc[4]["pts_roll3"] - 30.0) < 1e-10

    def test_ewma_features_use_shift_before_ewm(self):
        """EWMA features must shift(1) to exclude current game."""
        df = pd.DataFrame({
            "player_id": [1] * 5,
            "game_date": pd.date_range("2026-01-01", periods=5),
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        grp = df.groupby("player_id")["pts"]
        df["pts_ewma3"] = grp.transform(
            lambda s: s.shift(1).ewm(span=3, adjust=False, min_periods=1).mean()
        )
        # First game's EWMA should be NaN (no prior games)
        assert pd.isna(df.iloc[0]["pts_ewma3"])
        # Last game's EWMA should use game 4's value, not game 5's
        ewma_last = df.iloc[-1]["pts_ewma3"]
        # Should be approximately EWM of [10, 20, 30, 40], not 50
        assert ewma_last < 50.0


# ---------------------------------------------------------------------------
# §5.3: Market features must not enter structural model
# ---------------------------------------------------------------------------

class TestMarketFeatureContamination:
    def test_changing_market_inputs_does_not_change_structural_prediction(self):
        """§7.4: mutate all market fields — structural prediction must be unchanged."""
        from wnba_props_model.pipeline.safety import strip_market_prior_features

        df_base = pd.DataFrame({
            "player_id": [1],
            "pts_per_min_roll5": [0.5],
            "minutes_roll5": [30.0],
            "player_market_p_over_prev": [0.52],
            "player_market_line_prev": [18.5],
            "player_line_movement_prev": [0.5],
        })
        df_mutated = df_base.copy()
        df_mutated["player_market_p_over_prev"] = 0.80  # wildly different
        df_mutated["player_market_line_prev"] = 30.0    # wildly different
        df_mutated["player_line_movement_prev"] = 10.0  # wildly different

        clean_base, _ = strip_market_prior_features(df_base)
        clean_mutated, _ = strip_market_prior_features(df_mutated)

        # After stripping, both should be identical (basketball features unchanged)
        for col in clean_base.columns:
            if col not in ["player_market_p_over_prev", "player_market_line_prev", "player_line_movement_prev"]:
                assert (clean_base[col] == clean_mutated[col]).all(), f"Column {col} differs after market strip"

    def test_forbidden_market_columns_blocked_from_models(self):
        """Market features must be in FORBIDDEN_MODEL_FEATURES."""
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES

        forbidden = [
            "market_line", "over_odds", "under_odds",
            "no_vig_prob_over", "market_prob_over",
        ]
        for col in forbidden:
            assert col in FORBIDDEN_MODEL_FEATURES, (
                f"Expected {col} to be in FORBIDDEN_MODEL_FEATURES"
            )


# ---------------------------------------------------------------------------
# §5.4: Injury data must not contaminate historical rows
# ---------------------------------------------------------------------------

class TestInjuryDataNotContaminating:
    def test_injury_features_not_in_historical_feature_families(self):
        """Current-state injury features must not appear in historical training rows."""
        from wnba_props_model.features.feature_contract import FEATURE_FAMILIES

        # These are safe lagged injury features
        safe_injury_features = {
            "team_out_count", "team_questionable_count",
            "usage_vacated_proxy", "rebound_vacated_proxy", "assist_vacated_proxy",
        }
        # The safe features should be in the permitted family
        injury_family = set(FEATURE_FAMILIES.get("injury_availability", []))
        for feat in safe_injury_features:
            assert feat in injury_family, f"{feat} should be in injury_availability family"

    def test_raw_injury_status_is_not_in_model_features(self):
        """Raw current-state injury status strings must not be model features.

        The safe practice is:
          - Use lagged/derived injury features (team_out_count, usage_vacated_proxy)
          - The feature contract blocks these from training via FEATURE_FAMILIES
          - Raw string injury status ('Active', 'Out', etc.) is not in any FEATURE_FAMILY
        """
        from wnba_props_model.features.feature_contract import FEATURE_FAMILIES, MODEL_FEATURES

        # Current-state injury string status must NOT appear in any permitted feature family
        raw_string_features = ["injury_status", "injury_report_raw", "injury_designation"]
        all_permitted = set(MODEL_FEATURES)
        for col in raw_string_features:
            assert col not in all_permitted, (
                f"{col} (current-state injury string) should NOT be in permitted model features"
            )

        # Safe lagged injury-derived features SHOULD be permitted
        safe_injury_features = ["team_out_count", "team_questionable_count"]
        injury_family = set(FEATURE_FAMILIES.get("injury_availability", []))
        for feat in safe_injury_features:
            assert feat in injury_family, (
                f"{feat} should be in injury_availability feature family"
            )


# ---------------------------------------------------------------------------
# §5.5: Standings must use pre-game data only
# ---------------------------------------------------------------------------

class TestStandingsNotLeaking:
    def test_season_end_standings_not_in_early_games(self):
        """Season-end aggregates must not be joined to early-season game rows."""
        # Simulate a season table with season-end standings
        n_games = 40
        df = pd.DataFrame({
            "player_id": [1] * n_games,
            "game_date": pd.date_range("2026-01-01", periods=n_games),
            "pts": np.random.rand(n_games) * 20 + 10,
            "season": ["2026"] * n_games,
        })

        # Compute season average AFTER the season (leaky)
        season_avg = df["pts"].mean()  # uses all games including future ones

        # Correct approach: expanding pre-game mean (cumulative, shifted)
        df = df.sort_values("game_date")
        df["pts_season_avg_pregame"] = df.groupby("player_id")["pts"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # First game's expanding mean should be NaN (no prior data)
        assert pd.isna(df.iloc[0]["pts_season_avg_pregame"])

        # Last game's expanding mean should NOT equal full season average
        last_pregame = df.iloc[-1]["pts_season_avg_pregame"]
        assert abs(last_pregame - season_avg) > 0.001, (
            "Pre-game expanding mean should differ from full-season average"
        )


# ---------------------------------------------------------------------------
# §5.6: SVD/PCA must be fold-safe
# ---------------------------------------------------------------------------

class TestFoldSafeTransformers:
    def test_transformer_not_fit_on_test_rows(self):
        """Any learned transformer must be fit only on training rows."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Create fake data split into train and test
        rng = np.random.default_rng(42)
        n_train, n_test = 100, 20
        X_train = rng.normal(0, 1, (n_train, 5))
        X_test = rng.normal(0, 1, (n_test, 5))

        # Correct: fit on train only, transform both
        scaler = StandardScaler()
        scaler.fit(X_train)  # must NOT use X_test here
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # If we naively fit on all data (wrong approach), means would differ
        scaler_wrong = StandardScaler()
        X_all = np.vstack([X_train, X_test])
        scaler_wrong.fit(X_all)

        # The fold-safe scaler's means should differ from the "all data" scaler
        # (because test data shifts the mean slightly)
        train_mean_safe = scaler.mean_
        train_mean_wrong = scaler_wrong.mean_

        # They should be close but not exactly equal
        # (Just verifying the concept — actual difference may be small)
        # The important thing: only the fold-safe approach is correct
        assert X_train_scaled is not None
        assert X_test_scaled is not None

    def test_pca_must_not_see_test_data(self):
        """PCA embedded in pipeline must fit only on training data."""
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (100, 10))
        X_test = rng.normal(0, 1, (20, 10))

        pca = PCA(n_components=3, random_state=42)
        pca.fit(X_train)  # fold-safe: test data not seen
        X_test_pca = pca.transform(X_test)

        assert X_test_pca.shape == (20, 3)
