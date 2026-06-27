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

# PMF-mean thresholds set to "auto": computed at fit-time from OOF p33/p67
# quantiles of prop-eligible rows. This ensures thresholds stay in sync with
# the actual model output range across retraining cycles, preventing the
# previous bug where hardcoded thresholds (pts: 20/25) were unreachable by
# the model (actual model_mean ≈ 14 for pts starters) causing all predictions
# to fall into the "low" tier and receive over-compressed calibration.
_QUALITY_TIER_PMF_THRESHOLDS: dict[str, str] = {
    "pts":      "auto",
    "reb":      "auto",
    "ast":      "auto",
    "fg3m":     "auto",
    "stl":      "auto",
    "blk":      "auto",
    "turnover": "auto",
}

# Fallback thresholds used when OOF data is insufficient for auto-computation.
_FALLBACK_QUALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "pts":      {"low": 10.0, "high": 15.0},
    "reb":      {"low": 3.5,  "high": 6.0},
    "ast":      {"low": 1.5,  "high": 3.0},
    "fg3m":     {"low": 0.5,  "high": 1.2},
    "stl":      {"low": 0.4,  "high": 1.0},
    "blk":      {"low": 0.3,  "high": 0.7},
    "turnover": {"low": 1.5,  "high": 2.5},
}

# Prop-slice filter: minimum pmf_mean for a starter/core row to participate
# in quality-tier calibration training.  Rows below this threshold belong to
# low-quality "starters" who are never prop-eligible; including them dilutes
# the elite calibration curve and causes systematic UNDER for market props.
# Lowered from previous values (pts: 14→7) because the model's output range
# for prop-eligible players spans ~7-15 pts (not 14-25 as previously assumed).
_PROP_SLICE_PMF_MIN: dict[str, float] = {
    "pts":      7.0,
    "reb":      2.0,
    "ast":      1.0,
    "fg3m":     0.3,
    "stl":      0.3,
    "blk":      0.2,
    "turnover": 0.8,
}


def _compute_auto_thresholds(
    pmf_means: np.ndarray, min_rows: int = 100,
) -> dict[str, float] | None:
    """Compute quality-tier thresholds from OOF pmf_mean distribution.

    Returns {"low": p33, "high": p67} or None if insufficient data.
    Uses the 33rd and 67th percentiles of the prop-eligible distribution,
    ensuring each tier captures roughly one-third of the data for stable
    isotonic fitting.
    """
    valid = pmf_means[~np.isnan(pmf_means)]
    if len(valid) < min_rows:
        return None
    return {
        "low": float(np.percentile(valid, 33)),
        "high": float(np.percentile(valid, 67)),
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
            # #region agent log
            import json as _jc, time as _tc
            _raw_m = pmf_mean_val
            _thresh = self.quality_tier_thresholds.get(role_bucket, ())
            try:
                with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
                    _f.write(_jc.dumps({"sessionId": "94807e", "hypothesisId": "H1", "location": "calibration.py:apply", "message": "quality_tier_selected", "data": {"stat": self.stat, "role": role_bucket, "raw_pmf_mean": round(_raw_m, 3), "tier_selected": tier, "tier_exists_in_cal": tier in tier_cals, "thresholds": list(_thresh), "available_tiers": list(tier_cals.keys())}, "timestamp": int(_tc.time() * 1000)}) + "\n")
            except Exception:
                pass
            # #endregion agent log
            if tier in tier_cals:
                b = tier_cals[tier].apply(pmf)
                # #region agent log
                try:
                    _cal_arr = normalize_pmf(w * b + (1.0 - w) * g)
                    _ks = np.arange(len(_cal_arr))
                    _cal_m = float(_ks @ _cal_arr)
                    with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
                        _f.write(_jc.dumps({"sessionId": "94807e", "hypothesisId": "H1", "location": "calibration.py:apply_tier", "message": "tier_calibration_applied", "data": {"stat": self.stat, "role": role_bucket, "tier": tier, "raw_mean": round(_raw_m, 3), "cal_mean": round(_cal_m, 3), "compression_ratio": round(_cal_m / max(_raw_m, 0.01), 3)}, "timestamp": int(_tc.time() * 1000)}) + "\n")
                except Exception:
                    pass
                # #endregion agent log
                return normalize_pmf(w * b + (1.0 - w) * g)

        # Default: per-role-bucket calibrator.
        b = self.bucket_calibrators[role_bucket].apply(pmf)
        # #region agent log
        try:
            import json as _jd, time as _td
            _arr = normalize_pmf(pmf)
            _ks2 = np.arange(len(_arr))
            _raw_m2 = float(_ks2 @ _arr)
            _cal_arr2 = normalize_pmf(w * b + (1.0 - w) * g)
            _cal_m2 = float(_ks2 @ _cal_arr2)
            with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
                _f.write(_jd.dumps({"sessionId": "94807e", "hypothesisId": "H1", "location": "calibration.py:apply_bucket", "message": "bucket_calibration_applied", "data": {"stat": self.stat, "role": role_bucket, "raw_mean": round(_raw_m2, 3), "cal_mean": round(_cal_m2, 3), "compression_ratio": round(_cal_m2 / max(_raw_m2, 0.01), 3), "weight_w": round(w, 3)}, "timestamp": int(_td.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion agent log
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
                and stat_thresholds != ""
                and has_pmf_mean):
            # Prop-slice filter: remove very-low-quality starters from tier
            # calibration so the correction curve reflects prop-eligible players.
            if prop_min is not None:
                tier_data = gdf[gdf["pmf_mean"] >= prop_min].copy()
            else:
                tier_data = gdf.copy()

            if tier_data.empty:
                continue

            # Auto-compute thresholds from OOF data if configured
            if isinstance(stat_thresholds, str) and stat_thresholds == "auto":
                auto = _compute_auto_thresholds(tier_data["pmf_mean"].values)
                if auto is not None:
                    low_thresh = auto["low"]
                    high_thresh = auto["high"]
                    print(
                        f"[calibration] Auto thresholds stat={stat} role={bucket} "
                        f"low={low_thresh:.2f} high={high_thresh:.2f} "
                        f"(from {len(tier_data)} prop-eligible rows, "
                        f"p33={low_thresh:.2f} p67={high_thresh:.2f})"
                    )
                else:
                    fallback = _FALLBACK_QUALITY_THRESHOLDS.get(stat, {"low": 10.0, "high": 15.0})
                    low_thresh = fallback["low"]
                    high_thresh = fallback["high"]
                    print(
                        f"[calibration] Fallback thresholds stat={stat} role={bucket} "
                        f"low={low_thresh:.2f} high={high_thresh:.2f} "
                        f"(insufficient OOF data for auto)"
                    )
            elif isinstance(stat_thresholds, dict):
                low_thresh = stat_thresholds["low"]
                high_thresh = stat_thresholds["high"]
            else:
                fallback = _FALLBACK_QUALITY_THRESHOLDS.get(stat, {"low": 10.0, "high": 15.0})
                low_thresh = fallback["low"]
                high_thresh = fallback["high"]

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
                    print(
                        f"[calibration] Quality sub-calibrator stat={stat} role={bucket} "
                        f"tier={tier_name} n={len(tier_df)} "
                        f"pmf_range=[{tier_df['pmf_mean'].min():.1f}, {tier_df['pmf_mean'].max():.1f}] "
                        f"actual_range=[{tier_df['outcome'].min():.1f}, {tier_df['outcome'].max():.1f}]"
                    )
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
