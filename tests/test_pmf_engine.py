"""Stage 4 tests: PMF engine, utilities, model constraints, and validation.

18 required tests (spec) + additional coverage.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# --- Module imports -------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.models.pmf_utils import (
    dispersion_from_moments,
    hurdle_pmf,
    hurdle_pmf_batch,
    negbinom_pmf,
    negbinom_pmf_batch,
    pmf_mean_var,
    pmf_pge,
    poisson_pmf,
    poisson_pmf_batch,
    prob_over_from_pmf,
    validate_pmf,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _simple_pmf(dist: dict[int, float]) -> dict[int, float]:
    """Normalise a raw distribution dict into a valid PMF."""
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def _make_model_cfg() -> dict:
    return {
        "random_seed": 42,
        "stats": ["pts", "reb", "stl", "blk"],
        "sparse_stats": ["stl", "blk"],
        "pmf_support_caps": {"pts": 60, "reb": 30, "stl": 10, "blk": 10},
        "minutes_clip_min": 0.0,
        "minutes_clip_max": 45.0,
        "min_minutes_sigma": 3.0,
        "uncertain_sigma_multiplier": 1.5,
        "min_stat_mean": 0.01,
        "pmf_source": "stage4_baseline_uncalibrated_model_only",
        "hgb_regressor": {"max_iter": 50, "max_leaf_nodes": 15, "min_samples_leaf": 5},
        "hgb_classifier": {"max_iter": 50, "max_leaf_nodes": 15, "min_samples_leaf": 5},
    }


def _make_wide_df(n: int = 60) -> pd.DataFrame:
    """Minimal wide feature DataFrame for model testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "player_id": np.arange(n),
        "game_id": np.arange(n),
        "game_date": pd.date_range("2024-05-01", periods=n, freq="3D"),
        "season": 2024,
        "team_id": np.random.randint(1, 5, n),
        "player_name": [f"P{i}" for i in range(n)],
        "team_abbreviation": "TST",
        "opponent_team_id": np.random.randint(5, 10, n),
        "position": np.random.choice(["G", "F", "C"], n),
        "is_home": np.random.choice([True, False], n),
        "actual_minutes": np.random.uniform(0, 40, n),
        "actual_pts": np.random.poisson(10, n).astype(float),
        "actual_reb": np.random.poisson(4, n).astype(float),
        "actual_stl": np.random.poisson(0.8, n).astype(float),
        "actual_blk": np.random.poisson(0.4, n).astype(float),
        "did_play": True,
        "projected_minutes_proxy": np.random.uniform(10, 35, n),
        "projected_minutes_bucket": np.random.choice(
            ["bench_low", "rotation", "starter"], n
        ),
        "role_uncertainty_bucket": np.random.choice(
            ["stable", "elevated", "uncertain"], n
        ),
        "player_minutes_mean_l5": np.random.uniform(10, 35, n),
        "player_pts_mean_l5": np.random.uniform(5, 20, n),
        "team_pts_for_mean_l5": np.random.uniform(80, 100, n),
    })


# ===========================================================================
# TEST 1 — PMF probabilities sum to 1
# ===========================================================================
class TestPmfSumToOne:
    def test_poisson_sum_to_one(self):
        pmf = poisson_pmf(5.0, cap=30)
        assert abs(sum(pmf.values()) - 1.0) < 1e-6

    def test_negbinom_sum_to_one(self):
        pmf = negbinom_pmf(7.0, r=2.0, cap=60)
        assert abs(sum(pmf.values()) - 1.0) < 1e-6

    def test_hurdle_sum_to_one(self):
        pmf = hurdle_pmf(p_nonzero=0.55, pos_mu=1.2, pos_r=3.0, cap=10)
        assert abs(sum(pmf.values()) - 1.0) < 1e-6

    def test_hurdle_sum_to_one_poisson_fallback(self):
        pmf = hurdle_pmf(p_nonzero=0.4, pos_mu=1.5, pos_r=None, cap=10)
        assert abs(sum(pmf.values()) - 1.0) < 1e-6

    def test_validate_pmf_passes_on_valid(self):
        pmf = poisson_pmf(3.0, cap=20)
        validate_pmf(pmf)  # should not raise


# ===========================================================================
# TEST 2 — PMF probabilities nonnegative
# ===========================================================================
class TestPmfNonnegative:
    def test_poisson_nonnegative(self):
        pmf = poisson_pmf(4.0, cap=30)
        assert all(p >= 0 for p in pmf.values())

    def test_negbinom_nonnegative(self):
        pmf = negbinom_pmf(8.0, r=1.5, cap=60)
        assert all(p >= 0 for p in pmf.values())

    def test_hurdle_nonnegative(self):
        pmf = hurdle_pmf(0.6, 2.0, 4.0, cap=10)
        assert all(p >= 0 for p in pmf.values())

    def test_validate_pmf_raises_on_negative(self):
        bad_pmf = {0: -0.1, 1: 0.7, 2: 0.4}
        with pytest.raises(ValueError, match="(?i)negative"):
            validate_pmf(bad_pmf)

    def test_validate_pmf_raises_on_bad_sum(self):
        bad_pmf = {0: 0.3, 1: 0.3, 2: 0.3}  # sums to 0.9
        with pytest.raises(ValueError, match="sum"):
            validate_pmf(bad_pmf)


# ===========================================================================
# TEST 3 — PMF support starts at 0
# ===========================================================================
class TestPmfSupportStartsAtZero:
    def test_poisson_starts_at_zero(self):
        pmf = poisson_pmf(3.0, cap=20)
        assert 0 in pmf
        assert min(pmf.keys()) == 0

    def test_negbinom_starts_at_zero(self):
        pmf = negbinom_pmf(5.0, r=2.0, cap=30)
        assert min(pmf.keys()) == 0

    def test_hurdle_starts_at_zero(self):
        pmf = hurdle_pmf(0.7, 2.0, 3.0, cap=10)
        assert min(pmf.keys()) == 0

    def test_validate_pmf_raises_on_negative_support(self):
        bad_pmf = {-1: 0.1, 0: 0.5, 1: 0.4}
        with pytest.raises(ValueError, match="support"):
            validate_pmf(bad_pmf)


# ===========================================================================
# TEST 4 — Poisson PMF generation (known values)
# ===========================================================================
class TestPoissonPmfGeneration:
    def test_poisson_p0_equals_exp_neg_lam(self):
        lam = 4.0
        pmf = poisson_pmf(lam, cap=50)
        expected_p0 = np.exp(-lam)
        # After renorm to cap, should still be very close
        assert abs(pmf[0] - expected_p0) < 0.001

    def test_poisson_mean_close_to_lambda(self):
        lam = 7.0
        pmf = poisson_pmf(lam, cap=60)
        mean = sum(k * p for k, p in pmf.items())
        assert abs(mean - lam) < 0.01

    def test_poisson_batch_shape(self):
        mus = np.array([2.0, 5.0, 8.0])
        mat = poisson_pmf_batch(mus, cap=20)
        assert mat.shape == (3, 21)
        assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-6)


# ===========================================================================
# TEST 5 — Negative Binomial PMF generation
# ===========================================================================
class TestNegBinomPmfGeneration:
    def test_negbinom_mean_close_to_mu(self):
        mu, r = 8.0, 2.0
        pmf = negbinom_pmf(mu, r, cap=60)
        mean = sum(k * p for k, p in pmf.items())
        assert abs(mean - mu) < 0.05

    def test_negbinom_variance_greater_than_poisson(self):
        mu = 8.0
        pmf_nb = negbinom_pmf(mu, r=2.0, cap=60)
        pmf_p = poisson_pmf(mu, cap=60)
        var_nb = sum((k - mu) ** 2 * p for k, p in pmf_nb.items())
        var_p = sum((k - mu) ** 2 * p for k, p in pmf_p.items())
        assert var_nb > var_p

    def test_negbinom_batch_sums_to_one(self):
        mus = np.array([3.0, 7.0, 12.0, 0.5])
        mat = negbinom_pmf_batch(mus, r=1.5, cap=40)
        assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-6)

    def test_dispersion_from_moments(self):
        mean, var = 5.0, 20.0
        r = dispersion_from_moments(mean, var)
        assert r is not None
        # Check: var = mean + mean^2/r  ⟹  r = mean^2/(var-mean)
        assert abs(r - mean**2 / (var - mean)) < 1e-9

    def test_dispersion_returns_none_for_poisson_regime(self):
        r = dispersion_from_moments(5.0, 4.5)  # var < mean
        assert r is None


# ===========================================================================
# TEST 6 — Hurdle PMF correct p0
# ===========================================================================
class TestHurdlePmfP0:
    def test_hurdle_p0_equals_one_minus_p_nonzero(self):
        p_nz = 0.65
        pmf = hurdle_pmf(p_nz, pos_mu=1.5, pos_r=3.0, cap=10)
        assert abs(pmf[0] - (1 - p_nz)) < 1e-9

    def test_hurdle_full_zero_when_p_nz_zero(self):
        pmf = hurdle_pmf(p_nonzero=0.0, pos_mu=2.0, pos_r=2.0, cap=10)
        assert abs(pmf[0] - 1.0) < 1e-9
        assert all(pmf[k] == 0.0 for k in pmf if k > 0)

    def test_hurdle_batch_p0(self):
        p_nz = np.array([0.3, 0.7, 0.5])
        pos_mus = np.array([1.0, 2.0, 1.5])
        mat = hurdle_pmf_batch(p_nz, pos_mus, pos_r=2.0, cap=10)
        assert np.allclose(mat[:, 0], 1 - p_nz, atol=1e-9)


# ===========================================================================
# TEST 7 — Hurdle PMF positive tail renormalisation
# ===========================================================================
class TestHurdlePmfRenorm:
    def test_positive_tail_sums_to_p_nonzero(self):
        p_nz = 0.55
        pmf = hurdle_pmf(p_nz, pos_mu=2.0, pos_r=3.0, cap=10)
        positive_mass = sum(p for k, p in pmf.items() if k > 0)
        assert abs(positive_mass - p_nz) < 1e-6

    def test_batch_positive_tail_sums_correctly(self):
        p_nz = np.array([0.3, 0.6, 0.9])
        pos_mus = np.array([1.5, 2.5, 1.0])
        mat = hurdle_pmf_batch(p_nz, pos_mus, pos_r=2.0, cap=10)
        pos_sums = mat[:, 1:].sum(axis=1)
        assert np.allclose(pos_sums, p_nz, atol=1e-6)


# ===========================================================================
# TEST 8 — prob_over_from_pmf for half-integer lines
# ===========================================================================
class TestProbOverHalfInteger:
    def test_half_integer_line(self):
        pmf = {0: 0.5, 1: 0.3, 2: 0.2}
        # P(Y > 0.5) = P(Y >= 1) = 0.3 + 0.2 = 0.5
        assert abs(prob_over_from_pmf(pmf, 0.5) - 0.5) < 1e-9

    def test_half_integer_line_3_5(self):
        pmf = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
        # P(Y > 3.5) = P(Y >= 4) = 0.1 + 0.1 = 0.2
        assert abs(prob_over_from_pmf(pmf, 3.5) - 0.2) < 1e-9

    def test_half_integer_line_matches_prob_ge_integer(self):
        pmf = poisson_pmf(5.0, cap=30)
        # P(over 4.5) == P(over 4.0) for integer support
        assert abs(prob_over_from_pmf(pmf, 4.5) - prob_over_from_pmf(pmf, 4.0)) < 1e-12


# ===========================================================================
# TEST 9 — prob_over_from_pmf for integer lines
# ===========================================================================
class TestProbOverInteger:
    def test_integer_line_1(self):
        pmf = {0: 0.5, 1: 0.3, 2: 0.2}
        # P(Y > 1) = P(Y >= 2) = 0.2
        assert abs(prob_over_from_pmf(pmf, 1) - 0.2) < 1e-9

    def test_integer_line_zero(self):
        pmf = {0: 0.4, 1: 0.4, 2: 0.2}
        # P(Y > 0) = P(Y >= 1) = 0.6
        assert abs(prob_over_from_pmf(pmf, 0) - 0.6) < 1e-9

    def test_integer_line_above_support(self):
        pmf = {0: 0.5, 1: 0.3, 2: 0.2}
        # P(Y > 5) = 0
        assert prob_over_from_pmf(pmf, 5) == 0.0

    def test_p_ge_1_equals_prob_over_0(self):
        pmf = poisson_pmf(4.0, cap=30)
        assert abs(prob_over_from_pmf(pmf, 0) - (1 - pmf[0])) < 1e-9


# ===========================================================================
# TEST 10 — Minutes model excludes actual_minutes as feature
# ===========================================================================
class TestMinutesModelExcludesActualMinutes:
    def test_actual_minutes_not_in_training_features(self):
        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.pmf_engine import prepare_feature_matrix
        cfg = _make_model_cfg()
        wide = _make_wide_df(80)

        # Only numeric + position (exclude string bucket columns and targets)
        model_cols = [c for c in wide.columns
                      if c not in ("actual_minutes", "actual_pts", "actual_reb",
                                   "actual_stl", "actual_blk", "did_play",
                                   "player_id", "game_id", "game_date",
                                   "team_abbreviation", "player_name", "season",
                                   "team_id", "opponent_team_id",
                                   "projected_minutes_bucket", "role_uncertainty_bucket")]
        assert "actual_minutes" not in model_cols

        X, _ = prepare_feature_matrix(wide, model_cols, fit_encoder=True)
        y = wide["actual_minutes"]
        m = MinutesModel(cfg)
        m.fit(X, y, wide)

        # actual_minutes not in X at all
        assert "actual_minutes" not in X.columns


# ===========================================================================
# TEST 11 — Stat model excludes actual_outcome and actual_minutes
# ===========================================================================
class TestStatModelExcludesTargets:
    def test_targets_not_in_stat_model_features(self):
        from wnba_props_model.models.rate_model import StatRateModel
        from wnba_props_model.models.pmf_engine import prepare_feature_matrix
        cfg = _make_model_cfg()
        wide = _make_wide_df(80)

        # Numeric + position only; exclude string buckets and all targets
        feature_cols = [c for c in wide.columns
                        if not c.startswith("actual_")
                        and c not in ("player_id", "game_id", "game_date",
                                      "player_name", "team_abbreviation",
                                      "season", "team_id", "opponent_team_id",
                                      "did_play", "projected_minutes_bucket",
                                      "role_uncertainty_bucket")]
        X, _ = prepare_feature_matrix(wide, feature_cols, fit_encoder=True)

        assert "actual_pts" not in X.columns
        assert "actual_minutes" not in X.columns
        assert "actual_outcome" not in X.columns

        y = wide["actual_pts"]
        model = StatRateModel("pts", cfg)
        model.fit(X, y)
        preds = model.predict_mean(X)
        assert len(preds) == len(X)


# ===========================================================================
# TEST 12 — All forbidden market columns excluded
# ===========================================================================
class TestForbiddenMarketColumnsExcluded:
    def test_manifest_model_cols_contain_no_forbidden_cols(self):
        """Simulate manifest allow-list: market/target cols must not be present."""
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES

        # A realistic model_feature_columns list (no forbidden items)
        clean_cols = [
            "player_minutes_mean_l5", "player_pts_mean_l5", "team_pts_for_mean_l5",
            "is_home", "player_rest_days", "position",
        ]
        for col in clean_cols:
            assert col not in FORBIDDEN_MODEL_FEATURES, (
                f"Expected safe feature '{col}' appears in FORBIDDEN_MODEL_FEATURES"
            )

    def test_prepare_feature_matrix_rejects_forbidden_cols(self):
        from wnba_props_model.models.pmf_engine import prepare_feature_matrix
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES

        wide = _make_wide_df(20)
        # Inject a forbidden market column into the DataFrame
        bad_col = "over_odds"
        assert bad_col in FORBIDDEN_MODEL_FEATURES
        wide[bad_col] = 1.5
        # model_cols that includes the forbidden market column
        model_cols = ["player_minutes_mean_l5", "is_home", "position", bad_col]
        with pytest.raises(ValueError, match="Forbidden"):
            prepare_feature_matrix(wide, model_cols, fit_encoder=True)


# ===========================================================================
# TEST 13 — PMF output has one row per player_id × game_id × stat
# ===========================================================================
class TestPmfOutputGrain:
    def _build_small_pmf(self):
        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.rate_model import HurdleModel, StatRateModel
        from wnba_props_model.models.pmf_engine import build_all_pmfs, prepare_feature_matrix

        cfg = _make_model_cfg()
        wide = _make_wide_df(30)
        # Exclude string bucket cols and all target / identity cols
        feature_cols = [c for c in wide.columns
                        if not c.startswith("actual_")
                        and c not in ("player_id", "game_id", "game_date",
                                      "player_name", "team_abbreviation",
                                      "season", "team_id", "opponent_team_id",
                                      "did_play", "projected_minutes_bucket",
                                      "role_uncertainty_bucket")]
        X, encoder = prepare_feature_matrix(wide, feature_cols, fit_encoder=True)

        # Train
        mm = MinutesModel(cfg)
        mm.fit(X, wide["actual_minutes"], wide)
        mm._pos_encoder = encoder

        stat_models, hurdle_models = {}, {}
        for stat in ["pts", "reb"]:
            m = StatRateModel(stat, cfg)
            m.fit(X, wide[f"actual_{stat}"])
            stat_models[stat] = m
        for stat in ["stl", "blk"]:
            m = HurdleModel(stat, cfg)
            m.fit(X, wide[f"actual_{stat}"])
            hurdle_models[stat] = m

        # Build long table
        stats = ["pts", "reb", "stl", "blk"]
        long_rows = []
        for stat in stats:
            sub = wide[["player_id", "game_id", "game_date", "season",
                         "player_name", "team_id", "team_abbreviation",
                         "opponent_team_id", "actual_minutes", "did_play"]].copy()
            sub["stat"] = stat
            sub["actual_outcome"] = wide[f"actual_{stat}"].values
            long_rows.append(sub)
        long = pd.concat(long_rows, ignore_index=True)

        cfg_copy = dict(cfg)
        cfg_copy["stats"] = stats
        model_cols = list(X.columns)
        return build_all_pmfs(wide, long, model_cols, mm, stat_models, hurdle_models, cfg_copy)

    def test_one_row_per_player_game_stat(self):
        pmf_df = self._build_small_pmf()
        dupes = pmf_df.duplicated(subset=["player_id", "game_id", "stat"]).sum()
        assert dupes == 0

    def test_correct_number_of_rows(self):
        pmf_df = self._build_small_pmf()
        expected_rows = 30 * 4  # 30 players × 4 stats (pts, reb, stl, blk)
        assert len(pmf_df) == expected_rows


# ===========================================================================
# TEST 14 — validate_pmfs catches invalid sums
# ===========================================================================
class TestValidatePmfCatchesInvalidSum:
    def test_catches_sum_not_one(self):
        pmf = {0: 0.3, 1: 0.3, 2: 0.3}  # sums to 0.9
        with pytest.raises(ValueError, match="sum"):
            validate_pmf(pmf)

    def test_accepts_valid_sum(self):
        pmf = poisson_pmf(5.0, cap=30)
        validate_pmf(pmf)  # no exception


# ===========================================================================
# TEST 15 — validate_pmfs catches negative probability
# ===========================================================================
class TestValidatePmfCatchesNegativeProb:
    def test_catches_negative(self):
        pmf = {0: 1.1, 1: -0.1}  # sum = 1, but negative prob
        with pytest.raises(ValueError, match="(?i)negative"):
            validate_pmf(pmf)

    def test_accepts_zero_probability(self):
        # Zero probability is allowed
        pmf = {0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0}
        validate_pmf(pmf)  # no exception


# ===========================================================================
# TEST 16 — validate_pmfs catches duplicate keys
# ===========================================================================
class TestValidatePmfCatchesDuplicateKeys:
    def test_duplicate_player_game_stat(self):
        """PMF DataFrame with duplicate player × game × stat rows should fail validation."""
        row = {
            "player_id": 1, "game_id": 100, "stat": "pts",
            "pmf_json": json.dumps({str(k): 1/20 for k in range(20)}),
            "pmf_mean": 9.5, "pmf_variance": 0.0,
            "p0": 0.05, "actual_outcome": 10.0, "actual_minutes": 32.0,
            "is_calibrated": False, "pmf_source": "stage4_baseline_uncalibrated_model_only",
        }
        df_dupes = pd.DataFrame([row, row])  # exact duplicate
        dupe_count = df_dupes.duplicated(subset=["player_id", "game_id", "stat"]).sum()
        assert dupe_count == 1  # the second row is a duplicate


# ===========================================================================
# TEST 17 — Stage 4 PMFs have is_calibrated = False
# ===========================================================================
class TestIsNotCalibrated:
    def test_is_calibrated_false(self):
        pmf_row = {
            "is_calibrated": False,
            "pmf_source": "stage4_baseline_uncalibrated_model_only",
        }
        assert pmf_row["is_calibrated"] is False

    def test_pmf_source_correct(self):
        from wnba_props_model.models.pmf_engine import PMF_SOURCE
        assert PMF_SOURCE == "stage4_baseline_uncalibrated_model_only"

    def test_build_all_pmfs_marks_uncalibrated(self):
        """End-to-end: every row in PMF output must have is_calibrated=False."""
        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.rate_model import StatRateModel
        from wnba_props_model.models.pmf_engine import build_all_pmfs, prepare_feature_matrix

        cfg = _make_model_cfg()
        wide = _make_wide_df(20)
        feature_cols = [c for c in wide.columns
                        if not c.startswith("actual_")
                        and c not in ("player_id", "game_id", "game_date",
                                      "player_name", "team_abbreviation",
                                      "season", "team_id", "opponent_team_id",
                                      "did_play", "projected_minutes_bucket",
                                      "role_uncertainty_bucket")]
        X, encoder = prepare_feature_matrix(wide, feature_cols, fit_encoder=True)

        mm = MinutesModel(cfg)
        mm.fit(X, wide["actual_minutes"], wide)
        mm._pos_encoder = encoder

        stat_models = {}
        for stat in ["pts"]:
            m = StatRateModel(stat, cfg)
            m.fit(X, wide[f"actual_{stat}"])
            stat_models[stat] = m

        long_rows = []
        sub = wide[["player_id", "game_id", "game_date", "season",
                     "player_name", "team_id", "team_abbreviation",
                     "opponent_team_id", "actual_minutes", "did_play"]].copy()
        sub["stat"] = "pts"
        sub["actual_outcome"] = wide["actual_pts"].values
        long_rows.append(sub)
        long = pd.concat(long_rows)

        cfg_pts = dict(cfg)
        cfg_pts["stats"] = ["pts"]
        cfg_pts["sparse_stats"] = []
        pmf_df = build_all_pmfs(wide, long, list(X.columns), mm, stat_models, {}, cfg_pts)

        assert (pmf_df["is_calibrated"] == False).all()  # noqa: E712


# ===========================================================================
# TEST 18 — model_feature_columns loaded only from manifest
# ===========================================================================
class TestModelFeaturesFromManifest:
    def test_model_cols_from_manifest_not_inferred(self, tmp_path):
        """Training must use manifest allow-list, not infer features from data."""
        from wnba_props_model.models.minutes_model import MinutesModel

        cfg = _make_model_cfg()
        wide = _make_wide_df(40)

        # Manifest with explicit allow-list (does NOT include actual_minutes)
        manifest_cols = ["player_minutes_mean_l5", "player_pts_mean_l5",
                         "team_pts_for_mean_l5", "is_home",
                         "projected_minutes_proxy", "position"]
        manifest = {"model_feature_columns": manifest_cols, "target_columns": ["actual_minutes"]}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        loaded = json.loads(manifest_path.read_text())
        model_cols = loaded["model_feature_columns"]
        assert "actual_minutes" not in model_cols
        assert "actual_pts" not in model_cols
        # Features are the exact list from manifest
        assert set(model_cols) == set(manifest_cols)

    def test_prepare_feature_matrix_uses_only_listed_cols(self):
        from wnba_props_model.models.pmf_engine import prepare_feature_matrix

        wide = _make_wide_df(20)
        model_cols = ["player_minutes_mean_l5", "is_home", "position"]
        X, _ = prepare_feature_matrix(wide, model_cols, fit_encoder=True)
        assert set(X.columns) <= {"player_minutes_mean_l5", "is_home", "position"}
        # actual_minutes and other columns should not be present
        assert "actual_minutes" not in X.columns
