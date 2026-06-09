from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from wnba_props_model.constants import ROLE_GLOBAL_ONLY_BUCKETS, ROLE_MIN_ROWS
from wnba_props_model.models.simulation import normalize_pmf


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


@dataclass
class RoleAwarePMFCalibrator:
    stat: str
    global_calibrator: PMFCDFCalibrator
    bucket_calibrators: dict[str, PMFCDFCalibrator]
    bucket_counts: dict[str, int]
    shrink_k: float = 500.0
    cap: float = 0.80

    def apply(self, pmf: np.ndarray, role_bucket: str) -> np.ndarray:
        g = self.global_calibrator.apply(pmf)
        if role_bucket in ROLE_GLOBAL_ONLY_BUCKETS or role_bucket not in self.bucket_calibrators:
            return g
        n = self.bucket_counts.get(role_bucket, 0)
        w = min(self.cap, n / (n + self.shrink_k))
        b = self.bucket_calibrators[role_bucket].apply(pmf)
        return normalize_pmf(w * b + (1.0 - w) * g)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "RoleAwarePMFCalibrator":
        return joblib.load(path)


def fit_role_aware_calibrator(oof: pd.DataFrame, stat: str, seed: int = 0) -> RoleAwarePMFCalibrator:
    """Fit from columns: pmf (np.ndarray), outcome, role_bucket."""
    rng = np.random.default_rng(seed)
    data = oof[oof["stat"] == stat].copy() if "stat" in oof else oof.copy()
    pit = np.array([randomized_pit(p, y, rng) for p, y in zip(data["pmf"], data["outcome"])])
    global_cal = PMFCDFCalibrator().fit_from_pit(pit)

    bucket_calibrators: dict[str, PMFCDFCalibrator] = {}
    bucket_counts: dict[str, int] = {}
    for bucket, gdf in data.groupby("role_bucket"):
        n = len(gdf)
        bucket_counts[str(bucket)] = n
        if bucket in ROLE_GLOBAL_ONLY_BUCKETS or n < ROLE_MIN_ROWS.get(str(bucket), 500):
            continue
        bp = np.array([randomized_pit(p, y, rng) for p, y in zip(gdf["pmf"], gdf["outcome"])])
        bucket_calibrators[str(bucket)] = PMFCDFCalibrator().fit_from_pit(bp)
    return RoleAwarePMFCalibrator(stat, global_cal, bucket_calibrators, bucket_counts)
