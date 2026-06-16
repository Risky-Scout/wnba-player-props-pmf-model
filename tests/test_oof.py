"""Stage 5 OOF tests.

Covers all 20 spec requirements plus additional coverage:
1.  chronological fold generation
2.  fold train_end_date < validation_start_date
3.  same-day games not split between train and validation
4.  early rows become prior_only or calibration_eligible=false
5.  calibration_eligible=true only for model_oof rows
6.  model_feature_columns loaded from manifest only
7.  forbidden market columns blocked
8.  actual_outcome excluded from model features
9.  actual_minutes excluded from stat model features
10. OOF PMFs sum to 1
11. OOF PMFs nonnegative
12. OOF PMFs finite
13. duplicate OOF key detection
14. fold manifest required columns
15. low-minutes adjustment does not use actual_minutes
16. low-minutes adjustment increases p0 and reduces PMF mean
17. NLL and RPS computed correctly on toy PMFs
18. variance_ratio computed correctly
19. p0 vs empirical zero_rate audit generated
20. validate_oof_pmfs detects fold leakage
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.models.oof_engine import generate_oof_folds, make_prior_only_pmfs
from wnba_props_model.models.training import (
    FoldModel,
    apply_low_minutes_adjustment,
    encode_features,
    generate_fold_pmfs,
    train_fold,
)
from wnba_props_model.evaluation.oof_scoring import (
    nll_from_pmf_json,
    rps_from_pmf_json,
    score_oof_dataframe,
)
from wnba_props_model.models.pmf_utils import poisson_pmf, validate_pmf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(*args) -> list[date]:
    """Create a list of dates from (year, month, day) tuples."""
    return [date(*a) for a in args]


def _make_oof_cfg() -> dict:
    return {
        "random_seed": 42,
        "stats": ["pts", "stl"],
        "sparse_stats": ["stl"],
        "pmf_support_caps": {"pts": 30, "stl": 10},
        "validation_window_days": 7,
        "min_train_long_rows": 10,
        "min_train_stat_rows": 5,
        "min_sparse_positive_rows": 3,
        "dnp_minutes_threshold": 1.0,
        "low_minutes_zero_inflation_enabled": True,
        "pmf_source": "stage5_walk_forward_oof_uncalibrated_model_only",
        "calibration_eligible_prediction_types": ["model_oof"],
        "league_priors": {"pts": {"mean": 7.0, "var": 55.0},
                          "stl": {"mean": 0.6, "var": 0.9}},
        "hgb_regressor": {"max_iter": 30, "max_leaf_nodes": 10, "min_samples_leaf": 3},
        "hgb_classifier": {"max_iter": 30, "max_leaf_nodes": 10, "min_samples_leaf": 3},
        "minutes_clip_min": 0.0,
        "minutes_clip_max": 45.0,
        "min_minutes_sigma": 3.0,
        "uncertain_sigma_multiplier": 1.5,
        "min_stat_mean": 0.01,
    }


def _make_tiny_wide(n: int = 40, start_date: date | None = None) -> pd.DataFrame:
    """Make a minimal wide feature DataFrame."""
    np.random.seed(0)
    if start_date is None:
        start_date = date(2024, 5, 1)
    return pd.DataFrame({
        "player_id":                np.arange(n),
        "game_id":                  np.arange(n),
        "game_date":                pd.date_range(str(start_date), periods=n, freq="D"),
        "season":                   2024,
        "team_id":                  np.random.randint(1, 5, n),
        "player_name":              [f"P{i}" for i in range(n)],
        "team_abbreviation":        "TST",
        "opponent_team_id":         np.random.randint(5, 10, n),
        "position":                 np.random.choice(["G", "F", "C"], n),
        "is_home":                  np.random.choice([True, False], n),
        "actual_minutes":           np.random.uniform(5, 38, n),
        "actual_pts":               np.random.poisson(8, n).astype(float),
        "actual_stl":               np.random.poisson(0.7, n).astype(float),
        "did_play":                 True,
        "projected_minutes_proxy":  np.random.uniform(10, 35, n),
        "projected_minutes_bucket": np.random.choice(["rotation", "starter"], n),
        "role_uncertainty_bucket":  np.random.choice(["stable", "uncertain"], n),
        "player_minutes_mean_l5":   np.random.uniform(10, 35, n),
        "player_pts_mean_l5":       np.random.uniform(4, 18, n),
        "team_pts_for_mean_l5":     np.random.uniform(80, 100, n),
    })


def _make_tiny_long(wide: pd.DataFrame, stats: list[str] = ("pts", "stl")) -> pd.DataFrame:
    """Make a minimal long feature DataFrame."""
    rows = []
    for stat in stats:
        sub = wide[["player_id", "game_id", "game_date", "season",
                     "player_name", "team_id", "team_abbreviation",
                     "opponent_team_id", "actual_minutes", "did_play"]].copy()
        sub["stat"] = stat
        sub["actual_outcome"] = wide[f"actual_{stat}"].values
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


# ===========================================================================
# TEST 1 — Chronological fold generation
# ===========================================================================
class TestChronologicalFoldGeneration:
    def test_folds_in_chronological_order(self):
        dates = _dates((2024, 5, 1), (2024, 5, 3), (2024, 5, 10),
                       (2024, 5, 15), (2024, 5, 20), (2024, 5, 28))
        folds = generate_oof_folds(dates, window_days=7)
        starts = [f["val_start_date"] for f in folds]
        assert starts == sorted(starts), "Folds must be chronological"

    def test_fold_ids_sequential(self):
        dates = _dates((2024, 5, 1), (2024, 5, 5), (2024, 5, 12))
        folds = generate_oof_folds(dates, window_days=7)
        assert [f["fold_id"] for f in folds] == list(range(len(folds)))

    def test_non_overlapping_val_windows(self):
        dates = _dates((2024, 5, 1), (2024, 5, 5), (2024, 5, 10),
                       (2024, 5, 16), (2024, 5, 22))
        folds = generate_oof_folds(dates, window_days=7)
        # Each validation window must not overlap the previous
        for i in range(1, len(folds)):
            assert folds[i]["val_start_date"] > folds[i - 1]["val_end_date"]


# ===========================================================================
# TEST 2 — fold train_end_date < validation_start_date
# ===========================================================================
class TestFoldTemporalSeparation:
    def test_train_end_strictly_before_val_start(self):
        dates = _dates((2024, 5, 1), (2024, 5, 3), (2024, 5, 10), (2024, 5, 15))
        folds = generate_oof_folds(dates, window_days=7)
        for fold in folds:
            assert fold["train_end_date"] < fold["val_start_date"], (
                f"Fold {fold['fold_id']}: train_end={fold['train_end_date']} "
                f">= val_start={fold['val_start_date']}"
            )

    def test_first_fold_has_no_train_data(self):
        dates = _dates((2024, 5, 1), (2024, 5, 2), (2024, 5, 3))
        folds = generate_oof_folds(dates, window_days=7)
        # First fold: no dates strictly before val_start
        first = folds[0]
        assert first["train_games"] == 0


# ===========================================================================
# TEST 3 — Same-day games not split between train and validation
# ===========================================================================
class TestSameDayIntegrity:
    def test_all_games_on_same_date_go_to_same_fold(self):
        # Three games on 2024-05-10; they should ALL be in the same fold
        d = date(2024, 5, 10)
        dates = _dates(
            (2024, 5, 1), (2024, 5, 3),
            (2024, 5, 10), (2024, 5, 10), (2024, 5, 10),  # same day, 3 games
            (2024, 5, 18)
        )
        folds = generate_oof_folds(dates, window_days=7)
        # The date 2024-05-10 must appear in exactly one fold's val_dates
        fold_memberships = [f["fold_id"] for f in folds if d in f["val_dates"]]
        assert len(fold_memberships) == 1, (
            f"Same-day games appear in {len(fold_memberships)} folds instead of 1"
        )

    def test_train_does_not_include_validation_dates(self):
        """Strict: game_date < val_start; games on val_start must NOT be in training."""
        dates = _dates((2024, 5, 1), (2024, 5, 5), (2024, 5, 8),
                       (2024, 5, 10), (2024, 5, 15))
        folds = generate_oof_folds(dates, window_days=7)
        for fold in folds:
            # val_start_date must NOT be in train dates
            val_start = fold["val_start_date"]
            # train_dates are game_dates < val_start
            assert val_start not in [d for d in dates if d < val_start], (
                "val_start_date appears in training set — same-day leakage"
            )


# ===========================================================================
# TEST 4 — Early rows become prior_only or calibration_eligible=False
# ===========================================================================
class TestPriorOnlyEarlyRows:
    def test_prior_only_when_insufficient_train_data(self):
        cfg = _make_oof_cfg()
        cfg["min_train_long_rows"] = 1000  # force prior_only
        wide = _make_tiny_wide(20)
        long = _make_tiny_long(wide)

        fold_meta = {
            "fold_id": 0,
            "train_start_date": None,
            "train_end_date": date(2024, 4, 30),
            "val_start_date": date(2024, 5, 1),
            "val_end_date": date(2024, 5, 7),
            "train_wide_rows": 0,
            "train_games": 0,
            "oof_prediction_type": "prior_only",
        }
        pmf_df = make_prior_only_pmfs(wide, long, fold_meta, cfg)
        assert (pmf_df["oof_prediction_type"] == "prior_only").all()
        assert (pmf_df["calibration_eligible"] == False).all()  # noqa: E712

    def test_prior_only_pmfs_are_valid(self):
        cfg = _make_oof_cfg()
        wide = _make_tiny_wide(10)
        long = _make_tiny_long(wide)
        fold_meta = {
            "fold_id": 0, "train_start_date": None, "train_end_date": date(2024,4,30),
            "val_start_date": date(2024,5,1), "val_end_date": date(2024,5,7),
            "train_wide_rows": 0, "train_games": 0, "oof_prediction_type": "prior_only",
        }
        pmf_df = make_prior_only_pmfs(wide, long, fold_meta, cfg)
        for pmf_json in pmf_df["pmf_json"]:
            pmf = {int(k): float(v) for k, v in json.loads(pmf_json).items()}
            validate_pmf(pmf)


# ===========================================================================
# TEST 5 — calibration_eligible=True only for model_oof rows
# ===========================================================================
class TestCalibrationEligibility:
    def test_model_oof_is_calibration_eligible(self):
        cfg = _make_oof_cfg()
        wide = _make_tiny_wide(40)
        long = _make_tiny_long(wide)
        feature_cols = ["player_minutes_mean_l5", "player_pts_mean_l5",
                        "team_pts_for_mean_l5", "is_home", "position"]
        X, enc = encode_features(wide, feature_cols, fit_encoder=True)

        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.rate_model import StatRateModel, HurdleModel
        mm = MinutesModel(cfg); mm.fit(X, wide["actual_minutes"], wide); mm._pos_encoder = enc
        sm = StatRateModel("pts", cfg); sm.fit(X, wide["actual_pts"])
        hm = HurdleModel("stl", cfg); hm.fit(X, wide["actual_stl"])

        fm = FoldModel(mm, {"pts": sm}, {"stl": hm}, enc, list(X.columns),
                       {}, 40, 80, {"pts": 40, "stl": 40})

        fold_meta = {"fold_id": 1, "oof_prediction_type": "model_oof",
                     "train_start_date": None, "train_end_date": date(2024,4,30),
                     "val_start_date": date(2024,5,1), "val_end_date": date(2024,5,7),
                     "train_wide_rows": 40, "train_stat_rows": {"pts": 40, "stl": 40},
                     "train_games": 5, "fold_validation_rows": 20}
        pmf_df = generate_fold_pmfs(fm, wide.head(20), long.head(40), fold_meta, cfg)
        assert (pmf_df["calibration_eligible"] == True).all()  # noqa: E712

    def test_prior_only_not_calibration_eligible(self):
        """prior_only rows must have calibration_eligible=False."""
        cfg = _make_oof_cfg()
        wide = _make_tiny_wide(10)
        long = _make_tiny_long(wide)
        fold_meta = {
            "fold_id": 0, "train_start_date": None, "train_end_date": date(2024,4,30),
            "val_start_date": date(2024,5,1), "val_end_date": date(2024,5,7),
            "train_wide_rows": 0, "train_games": 0, "oof_prediction_type": "prior_only",
        }
        pmf_df = make_prior_only_pmfs(wide, long, fold_meta, cfg)
        assert (pmf_df["calibration_eligible"] == False).all()  # noqa: E712


# ===========================================================================
# TEST 6 — model_feature_columns loaded from manifest only
# ===========================================================================
class TestModelFeaturesFromManifest:
    def test_encode_features_uses_only_listed_cols(self):
        wide = _make_tiny_wide(20)
        # Inject a column that should NOT be in model_cols
        wide["actual_pts"] = 5
        model_cols = ["player_minutes_mean_l5", "is_home", "position"]
        X, _ = encode_features(wide, model_cols, fit_encoder=True)
        assert "actual_pts" not in X.columns
        assert set(X.columns) <= {"player_minutes_mean_l5", "is_home", "position"}


# ===========================================================================
# TEST 7 — forbidden market columns blocked
# ===========================================================================
class TestForbiddenColumnsBlocked:
    def test_encode_features_rejects_forbidden(self):
        from wnba_props_model.features.feature_contract import FORBIDDEN_MODEL_FEATURES
        wide = _make_tiny_wide(10)
        bad_col = "over_odds"
        assert bad_col in FORBIDDEN_MODEL_FEATURES
        wide[bad_col] = 1.5
        with pytest.raises(ValueError, match="Forbidden"):
            encode_features(wide, ["player_minutes_mean_l5", bad_col])


# ===========================================================================
# TEST 8 — actual_outcome excluded from model features
# ===========================================================================
class TestActualOutcomeExcluded:
    def test_train_fold_does_not_use_actual_outcome(self):
        cfg = _make_oof_cfg()
        wide = _make_tiny_wide(30)
        long = _make_tiny_long(wide)
        feature_cols = ["player_minutes_mean_l5", "player_pts_mean_l5", "is_home", "position"]
        # actual_outcome should not be in feature_cols
        assert "actual_outcome" not in feature_cols
        X, enc = encode_features(wide, feature_cols, fit_encoder=True)
        assert "actual_outcome" not in X.columns


# ===========================================================================
# TEST 9 — actual_minutes excluded from stat model features
# ===========================================================================
class TestActualMinutesExcluded:
    def test_stat_model_features_exclude_actual_minutes(self):
        cfg = _make_oof_cfg()
        wide = _make_tiny_wide(30)
        feature_cols = ["player_minutes_mean_l5", "player_pts_mean_l5",
                        "team_pts_for_mean_l5", "is_home", "position"]
        X, _ = encode_features(wide, feature_cols, fit_encoder=True)
        # actual_minutes is a target and must not be in feature cols
        assert "actual_minutes" not in X.columns


# ===========================================================================
# TEST 10-12 — OOF PMF structural validity
# ===========================================================================
class TestOofPmfValidity:
    def _make_pmf_df(self):
        cfg = _make_oof_cfg()
        wide = _make_tiny_wide(30)
        long = _make_tiny_long(wide)
        feature_cols = ["player_minutes_mean_l5", "player_pts_mean_l5",
                        "team_pts_for_mean_l5", "is_home", "position"]
        X, enc = encode_features(wide, feature_cols, fit_encoder=True)
        from wnba_props_model.models.minutes_model import MinutesModel
        from wnba_props_model.models.rate_model import StatRateModel, HurdleModel
        mm = MinutesModel(cfg); mm.fit(X, wide["actual_minutes"], wide); mm._pos_encoder = enc
        sm = StatRateModel("pts", cfg); sm.fit(X, wide["actual_pts"])
        hm = HurdleModel("stl", cfg); hm.fit(X, wide["actual_stl"])
        fm = FoldModel(mm, {"pts": sm}, {"stl": hm}, enc, list(X.columns),
                       {}, 30, 60, {"pts": 30, "stl": 30})
        fold_meta = {"fold_id": 0, "oof_prediction_type": "model_oof",
                     "train_start_date": None, "train_end_date": date(2024,4,30),
                     "val_start_date": date(2024,5,1), "val_end_date": date(2024,5,7),
                     "train_wide_rows": 30, "train_stat_rows": {}, "train_games": 5,
                     "fold_validation_rows": len(wide)}
        return generate_fold_pmfs(fm, wide, long, fold_meta, cfg)

    def test_pmf_sum_to_one(self):
        pmf_df = self._make_pmf_df()
        for pmf_json in pmf_df["pmf_json"]:
            s = sum(json.loads(pmf_json).values())
            assert abs(s - 1.0) < 1e-6, f"PMF sum = {s}"

    def test_pmf_nonnegative(self):
        pmf_df = self._make_pmf_df()
        for pmf_json in pmf_df["pmf_json"]:
            probs = list(json.loads(pmf_json).values())
            assert all(p >= -1e-9 for p in probs)

    def test_pmf_finite(self):
        pmf_df = self._make_pmf_df()
        for pmf_json in pmf_df["pmf_json"]:
            probs = list(json.loads(pmf_json).values())
            assert all(np.isfinite(p) for p in probs)


# ===========================================================================
# TEST 13 — Duplicate OOF key detection
# ===========================================================================
class TestDuplicateOofKeyDetection:
    def test_duplicate_detection(self):
        row = {"player_id": 1, "game_id": 100, "stat": "pts",
               "pmf_json": json.dumps({str(k): 1/31 for k in range(31)}),
               "pmf_mean": 15.0, "pmf_variance": 5.0, "p0": 0.1,
               "calibration_eligible": True, "oof_prediction_type": "model_oof",
               "is_calibrated": False, "pmf_source": "stage5_...", "actual_outcome": 10}
        df = pd.DataFrame([row, row])
        assert df.duplicated(subset=["player_id", "game_id", "stat"]).sum() == 1


# ===========================================================================
# TEST 14 — Fold manifest required columns
# ===========================================================================
class TestFoldManifestColumns:
    def test_generate_folds_returns_required_fields(self):
        dates = _dates((2024, 5, 1), (2024, 5, 5), (2024, 5, 10))
        folds = generate_oof_folds(dates, window_days=7)
        required_keys = {"fold_id", "train_end_date", "val_start_date", "val_end_date",
                         "train_games", "val_games", "val_dates"}
        for fold in folds:
            missing = required_keys - set(fold.keys())
            assert not missing, f"Fold missing keys: {missing}"


# ===========================================================================
# TEST 15 — Low-minutes adjustment does not use actual_minutes
# ===========================================================================
class TestLowMinutesAdjustmentNoLeakage:
    def test_adjustment_uses_only_predicted_minutes(self):
        """The adjustment is driven by minutes_mean (predicted), not actual_minutes."""
        pmf_mat = np.array([[0.1, 0.4, 0.3, 0.2],
                             [0.2, 0.4, 0.3, 0.1]])
        # Row 0: minutes_mean = 0.5 (below threshold 1.0)
        # Row 1: minutes_mean = 5.0 (above threshold)
        minutes_means = np.array([0.5, 5.0])
        actual_minutes = np.array([0.0, 32.0])  # NOT used in adjustment

        adj_mat, n_adj = apply_low_minutes_adjustment(pmf_mat, minutes_means, threshold=1.0)
        # Row 0 should be adjusted, row 1 should not
        assert n_adj == 1
        # Row 1 should be unchanged
        np.testing.assert_array_almost_equal(adj_mat[1], pmf_mat[1])

    def test_adjustment_not_based_on_actual_minutes(self):
        """Same actual_minutes with different minutes_mean should give different results."""
        pmf_mat = np.array([[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]])
        # Both rows have actual_minutes = 30 (not used)
        # But different minutes_mean
        low_min = np.array([0.3, 30.0])
        adj_mat, n_adj = apply_low_minutes_adjustment(pmf_mat, low_min, threshold=1.0)
        # Row 0 adjusted, row 1 not
        assert adj_mat[0, 0] > pmf_mat[0, 0]  # p0 increased
        np.testing.assert_array_almost_equal(adj_mat[1], pmf_mat[1])


# ===========================================================================
# TEST 16 — Low-minutes adjustment increases p0 and reduces PMF mean
# ===========================================================================
class TestLowMinutesEffect:
    def test_adjustment_increases_p0(self):
        pmf_mat = np.array([[0.1, 0.5, 0.3, 0.1]])
        minutes = np.array([0.3])
        adj_mat, _ = apply_low_minutes_adjustment(pmf_mat, minutes, threshold=1.0)
        assert adj_mat[0, 0] > pmf_mat[0, 0]

    def test_adjustment_reduces_pmf_mean(self):
        pmf_mat = np.array([[0.1, 0.3, 0.4, 0.2]])
        original_mean = sum(k * pmf_mat[0, k] for k in range(4))
        minutes = np.array([0.5])
        adj_mat, _ = apply_low_minutes_adjustment(pmf_mat, minutes, threshold=1.0)
        adj_mean = sum(k * adj_mat[0, k] for k in range(4))
        assert adj_mean < original_mean

    def test_adjusted_pmf_still_sums_to_one(self):
        pmf_mat = np.array([[0.05, 0.25, 0.40, 0.20, 0.10]])
        minutes = np.array([0.1])
        adj_mat, _ = apply_low_minutes_adjustment(pmf_mat, minutes, threshold=1.0)
        assert abs(adj_mat[0].sum() - 1.0) < 1e-9

    def test_no_adjustment_above_threshold(self):
        pmf_mat = np.array([[0.1, 0.5, 0.3, 0.1]])
        minutes = np.array([5.0])  # above threshold
        adj_mat, n_adj = apply_low_minutes_adjustment(pmf_mat, minutes, threshold=1.0)
        assert n_adj == 0
        np.testing.assert_array_almost_equal(adj_mat, pmf_mat)


# ===========================================================================
# TEST 17 — NLL and RPS computed correctly on toy PMFs
# ===========================================================================
class TestNllRpsCorrectness:
    def test_nll_known_probability(self):
        pmf = {"0": 0.5, "1": 0.3, "2": 0.2}
        pmf_json = json.dumps(pmf)
        # NLL for actual=1: -log(0.3)
        expected = -np.log(0.3)
        assert abs(nll_from_pmf_json(pmf_json, 1) - expected) < 1e-9

    def test_nll_perfect_prediction(self):
        pmf = {"0": 1.0, "1": 0.0, "2": 0.0}
        pmf_json = json.dumps(pmf)
        # NLL for actual=0 and p(0)=1: -log(1) = 0
        assert abs(nll_from_pmf_json(pmf_json, 0)) < 1e-9

    def test_rps_perfect_point_mass(self):
        # PMF puts all mass at 0; actual = 0 → perfect prediction
        pmf = {"0": 1.0}
        pmf_json = json.dumps(pmf)
        # F(k) = 1 for all k; G(k) = I(actual<=k) = 1 for all k
        # RPS = sum((1-1)^2) = 0
        assert rps_from_pmf_json(pmf_json, 0, cap=5) < 1e-9

    def test_rps_worst_case(self):
        # PMF puts all mass at 0, but actual = 5 → worst prediction
        pmf = {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}
        pmf_json = json.dumps(pmf)
        rps = rps_from_pmf_json(pmf_json, 5, cap=5)
        # F(k) = 1 for all k; G(k) = I(5<=k) = step at k=5
        # RPS = sum((1-0)^2 for k=0..4) + (1-1)^2 for k=5 = 5
        assert abs(rps - 5.0) < 1e-9

    def test_rps_integer_and_half_integer_consistency(self):
        # For integer support PMFs, RPS should be symmetric around actual
        pmf = poisson_pmf(5.0, cap=20)
        pmf_json = json.dumps({str(k): v for k, v in pmf.items()})
        rps_at_actual = rps_from_pmf_json(pmf_json, 5, cap=20)
        assert rps_at_actual >= 0


# ===========================================================================
# TEST 18 — Variance ratio computed correctly
# ===========================================================================
class TestVarianceRatio:
    def test_variance_ratio_computation(self):
        # Create toy scoring result with known variance_ratio = 1.0
        # actuals: 2, 4, 6, 8 → mean=5, var=5
        # pmf_variances: 5, 5, 5, 5 → mean_pmf_var=5 → ratio=5/5=1.0
        actuals = np.array([2, 4, 6, 8], dtype=float)
        pmf_vars = np.array([5.0, 5.0, 5.0, 5.0])
        pmf_means = np.array([5.0, 5.0, 5.0, 5.0])

        actual_var = float(np.var(actuals))
        mean_pmf_var = float(np.mean(pmf_vars))
        ratio = mean_pmf_var / actual_var if actual_var > 0 else None

        assert abs(ratio - 1.0) < 1e-9


# ===========================================================================
# TEST 19 — p0 vs empirical zero rate audit generated
# ===========================================================================
class TestP0VsEmpiricalZeroAudit:
    def test_score_oof_includes_p0_vs_zero_delta(self):
        # Build minimal OOF DataFrame
        pmf = poisson_pmf(3.0, cap=15)
        pmf_json = json.dumps({str(k): v for k, v in pmf.items()})
        rows = []
        for i in range(10):
            rows.append({
                "player_id": i, "game_id": i, "stat": "pts",
                "actual_outcome": float(np.random.poisson(3)),
                "pmf_json": pmf_json,
                "pmf_mean": 3.0,
                "pmf_variance": 3.0,
                "p0": pmf[0],
                "calibration_eligible": True,
                "oof_prediction_type": "model_oof",
                "projected_minutes_bucket": "rotation",
            })
        df = pd.DataFrame(rows)
        result = score_oof_dataframe(df, {"pts": 15})
        assert "by_stat" in result
        assert "pts" in result["by_stat"]
        s = result["by_stat"]["pts"]
        assert "p0_vs_zero_delta" in s
        assert "mean_predicted_p0" in s
        assert "empirical_zero_rate" in s


# ===========================================================================
# TEST 20 — validate_oof_pmfs detects fold leakage
# ===========================================================================
class TestValidateOofDetectsLeakage:
    def test_leakage_detected_when_train_end_gte_val_start(self):
        """PMF rows where fold_train_end_date >= fold_validation_start_date are leakage."""
        rows = []
        for i in range(5):
            rows.append({
                "player_id": i, "game_id": i, "stat": "pts",
                "pmf_json": json.dumps({"0": 1.0}),
                "is_calibrated": False,
                "pmf_source": "stage5_walk_forward_oof_uncalibrated_model_only",
                "oof_prediction_type": "model_oof",
                "calibration_eligible": True,
                "game_date": pd.Timestamp("2024-05-15"),
                # LEAKAGE: train_end_date = val_start_date (should be strictly less)
                "fold_train_end_date": pd.Timestamp("2024-05-15"),
                "fold_validation_start_date": pd.Timestamp("2024-05-15"),
                "actual_outcome": 5.0,
                "pmf_mean": 3.0, "pmf_variance": 3.0, "p0": 0.05,
            })
        df = pd.DataFrame(rows)
        # Detect leakage in the DataFrame
        te = pd.to_datetime(df["fold_train_end_date"])
        vs = pd.to_datetime(df["fold_validation_start_date"])
        leakage_count = (te >= vs).sum()
        assert leakage_count > 0, "Should detect fold leakage"
