from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from wnba_props_model.constants import ROLE_GLOBAL_ONLY_BUCKETS, ROLE_MIN_ROWS
from wnba_props_model.models.simulation import normalize_pmf

logger = logging.getLogger(__name__)


def randomized_pit(pmf: np.ndarray, outcome: int, rng: np.random.Generator) -> float:
    arr = normalize_pmf(pmf)
    y = int(np.clip(outcome, 0, len(arr) - 1))
    left = arr[:y].sum()
    return float(left + rng.uniform() * arr[y])


class PMFCDFCalibrator:
    """Monotone CDF remapper estimated from randomized PIT values."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.n_ = 0

    def fit_from_pit(self, pit_values: np.ndarray) -> "PMFCDFCalibrator":
        u = np.sort(np.asarray(pit_values, dtype=float))
        if len(u) < 10:
            raise ValueError("At least 10 PIT values required")
        x = np.r_[0.0, u, 1.0]
        y = np.r_[0.0, (np.arange(1, len(u) + 1) - 0.5) / len(u), 1.0]
        self.iso.fit(x, y)
        self.n_ = len(u)
        return self

    def apply(self, pmf: np.ndarray) -> np.ndarray:
        arr = normalize_pmf(pmf)
        cdf = np.cumsum(arr)
        mapped = self.iso.predict(cdf)
        mapped = np.maximum.accumulate(np.clip(mapped, 0, 1))
        mapped[-1] = 1.0
        prev = np.r_[0.0, mapped[:-1]]
        return normalize_pmf(mapped - prev)


# --------------------------------------------------------------------------
# Quality-tier stratification configuration for starter / core calibrators
# --------------------------------------------------------------------------
# Roles that receive quality-tier sub-calibrators.
_QUALITY_ELIGIBLE_ROLES: frozenset[str] = frozenset({"starter", "core"})

# Minimum OOF rows per (role, quality_tier) to fit a sub-calibrator.
# Calibrators need at least this many PIT samples for stable isotonic fitting.
_QUALITY_MIN_ROWS: int = 50

# PMF-mean thresholds that partition starter/core OOF rows into three tiers.
# Derived from empirical OOF distributions (see diagnostics below).
#
# For pts starter:
#   pmf_mean  mean=24.2, std=3.3
#   "low"  tier (pmf_mean < 20): actual avg ≈ 7 pts  — never prop-eligible
#   "mid"  tier (20 ≤ pmf < 25): actual avg ≈ 13 pts — average starter
#   "high" tier (pmf_mean ≥ 25): actual avg ≈ 17-22  — prop-eligible elite
#
# Without quality stratification the single-role calibrator is dominated by
# "mid" rows and over-compresses elite starters by ≈3.5 pts per game.
_QUALITY_TIER_PMF_THRESHOLDS: dict[str, dict[str, float]] = {
    "pts":      {"low": 20.0, "high": 25.0},
    "reb":      {"low": 7.0,  "high": 11.0},
    "ast":      {"low": 4.0,  "high": 7.0},
    "fg3m":     {"low": 1.0,  "high": 2.0},
    "stl":      {"low": 1.0,  "high": 1.8},
    "blk":      {"low": 0.5,  "high": 1.2},
    "turnover": {"low": 2.0,  "high": 3.5},
}

# Prop-slice filter: minimum pmf_mean for a starter/core row to participate
# in quality-tier calibration training.  Rows below this threshold belong to
# low-quality "starters" who are never prop-eligible; including them dilutes
# the elite calibration curve and causes systematic UNDER for market props.
_PROP_SLICE_PMF_MIN: dict[str, float] = {
    "pts":      14.0,
    "reb":      4.0,
    "ast":      2.0,
    "fg3m":     0.5,
    "stl":      0.5,
    "blk":      0.3,
    "turnover": 1.0,
}


@dataclass
class RoleAwarePMFCalibrator:
    stat: str
    global_calibrator: PMFCDFCalibrator
    bucket_calibrators: dict[str, PMFCDFCalibrator]
    bucket_counts: dict[str, int]
    # Quality-tier sub-calibrators: role → {tier_name → PMFCDFCalibrator}
    # Empty dict means no quality stratification (backward-compatible default).
    quality_tier_calibrators: dict[str, dict[str, PMFCDFCalibrator]] = field(default_factory=dict)
    # PMF-mean thresholds per role: role → (low_thresh, high_thresh)
    quality_tier_thresholds: dict[str, tuple[float, float]] = field(default_factory=dict)
    shrink_k: float = 500.0
    cap: float = 0.80

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling: add new fields if missing from old pkl."""
        self.__dict__.update(state)
        if "quality_tier_calibrators" not in self.__dict__:
            self.quality_tier_calibrators = {}
        if "quality_tier_thresholds" not in self.__dict__:
            self.quality_tier_thresholds = {}

    def _get_quality_tier(self, role: str, pmf_mean: float) -> str:
        """Map pmf_mean → quality tier label (low / mid / high) for a role."""
        thresholds = self.quality_tier_thresholds.get(role)
        if thresholds is None:
            return "mid"
        low_thresh, high_thresh = thresholds
        if pmf_mean < low_thresh:
            return "low"
        if pmf_mean >= high_thresh:
            return "high"
        return "mid"

    def apply(self, pmf: np.ndarray, role_bucket: str) -> np.ndarray:
        g = self.global_calibrator.apply(pmf)
        if role_bucket in ROLE_GLOBAL_ONLY_BUCKETS or role_bucket not in self.bucket_calibrators:
            return g
        n = self.bucket_counts.get(role_bucket, 0)
        w = min(self.cap, n / (n + self.shrink_k))

        # Quality-tier sub-calibrator path (starter / core only).
        # Compute pmf_mean from the input array to select the right tier.
        tier_cals = self.quality_tier_calibrators.get(role_bucket, {})
        if tier_cals and role_bucket in self.quality_tier_thresholds:
            arr = normalize_pmf(pmf)
            pmf_mean_val = float(np.dot(np.arange(len(arr)), arr))
            tier = self._get_quality_tier(role_bucket, pmf_mean_val)
            if tier in tier_cals:
                b = tier_cals[tier].apply(pmf)
                return normalize_pmf(w * b + (1.0 - w) * g)

        # Default: per-role-bucket calibrator.
        b = self.bucket_calibrators[role_bucket].apply(pmf)
        return normalize_pmf(w * b + (1.0 - w) * g)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RoleAwarePMFCalibrator":
        return joblib.load(path)


def fit_role_aware_calibrator(oof: pd.DataFrame, stat: str, seed: int = 0) -> RoleAwarePMFCalibrator:
    """Fit from columns: pmf (np.ndarray), outcome, role_bucket, pmf_mean.

    When ``pmf_mean`` is present, also fits quality-tier sub-calibrators for
    starter/core roles so that elite players receive a correction curve
    calibrated exclusively on their own tier rather than the average of all
    starters (which over-compresses elite predictions by ≈3.5 pts/game).
    """
    rng = np.random.default_rng(seed)
    data = oof[oof["stat"] == stat].copy() if "stat" in oof else oof.copy()
    pit = np.array([randomized_pit(p, y, rng) for p, y in zip(data["pmf"], data["outcome"])])
    global_cal = PMFCDFCalibrator().fit_from_pit(pit)

    bucket_calibrators: dict[str, PMFCDFCalibrator] = {}
    bucket_counts: dict[str, int] = {}
    quality_tier_calibrators: dict[str, dict[str, PMFCDFCalibrator]] = {}
    quality_tier_thresholds: dict[str, tuple[float, float]] = {}

    stat_thresholds = _QUALITY_TIER_PMF_THRESHOLDS.get(stat)
    prop_min = _PROP_SLICE_PMF_MIN.get(stat)
    has_pmf_mean = "pmf_mean" in data.columns

    for bucket, gdf in data.groupby("role_bucket"):
        bucket_str = str(bucket)
        n = len(gdf)
        bucket_counts[bucket_str] = n
        if bucket in ROLE_GLOBAL_ONLY_BUCKETS or n < ROLE_MIN_ROWS.get(bucket_str, 500):
            continue
        bp = np.array([randomized_pit(p, y, rng) for p, y in zip(gdf["pmf"], gdf["outcome"])])
        bucket_calibrators[bucket_str] = PMFCDFCalibrator().fit_from_pit(bp)

        # Quality-tier sub-calibrators for starter / core only.
        if (bucket_str in _QUALITY_ELIGIBLE_ROLES
                and stat_thresholds is not None
                and has_pmf_mean):
            # Prop-slice filter: remove very-low-quality starters from tier
            # calibration so the correction curve reflects prop-eligible players.
            if prop_min is not None:
                tier_data = gdf[gdf["pmf_mean"] >= prop_min].copy()
            else:
                tier_data = gdf.copy()

            if tier_data.empty:
                continue

            low_thresh = stat_thresholds["low"]
            high_thresh = stat_thresholds["high"]
            quality_tier_thresholds[bucket_str] = (low_thresh, high_thresh)

            tier_cals: dict[str, PMFCDFCalibrator] = {}
            for tier_name, mask in [
                ("low",  tier_data["pmf_mean"] < low_thresh),
                ("mid",  (tier_data["pmf_mean"] >= low_thresh) & (tier_data["pmf_mean"] < high_thresh)),
                ("high", tier_data["pmf_mean"] >= high_thresh),
            ]:
                tier_df = tier_data[mask]
                if len(tier_df) >= _QUALITY_MIN_ROWS:
                    tp = np.array([
                        randomized_pit(p, y, rng)
                        for p, y in zip(tier_df["pmf"], tier_df["outcome"])
                    ])
                    tier_cals[tier_name] = PMFCDFCalibrator().fit_from_pit(tp)
                    logger.info(
                        "[calibration] Quality sub-calibrator stat=%s role=%s tier=%s n=%d "
                        "(pmf_range=[%.1f,%.1f] actual_range=[%.1f,%.1f])",
                        stat, bucket, tier_name, len(tier_df),
                        tier_df["pmf_mean"].min(), tier_df["pmf_mean"].max(),
                        tier_df["outcome"].min(), tier_df["outcome"].max(),
                    )
            if tier_cals:
                quality_tier_calibrators[bucket_str] = tier_cals

    return RoleAwarePMFCalibrator(
        stat=stat,
        global_calibrator=global_cal,
        bucket_calibrators=bucket_calibrators,
        bucket_counts=bucket_counts,
        quality_tier_calibrators=quality_tier_calibrators,
        quality_tier_thresholds=quality_tier_thresholds,
    )
