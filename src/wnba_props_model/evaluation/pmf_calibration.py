"""Isotonic Distributional Regression (IDR) for full PMF calibration.

Calibrating only P(over) wastes the model's PMF capability and can produce
incoherent distributions. IDR calibrates the FULL CDF at every threshold,
guaranteeing monotone, calibrated distributions.

References:
  Allen et al. (2025). In-sample calibration yields conformal calibration guarantees.
  https://arxiv.org/abs/2503.03841

  Lipiecki et al. (2024). Isotonic distributional regression for day-ahead
  electricity prices. Energy Economics.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)


class PMFCalibrator:
    """Isotonic Distributional Regression for full PMF calibration.

    For each threshold k in {0, 1, ..., max_val}:
      1. Compute model CDF: F(k) = P(stat <= k) from raw PMF
      2. Compute empirical CDF: F_emp(k) = fraction of actuals <= k in OOF
      3. Fit isotonic regression: F_cal(k) = iso(F_raw(k))

    The resulting calibrated CDF is:
      - Monotone (non-decreasing) by construction
      - Calibrated at every quantile (not just P(over))
      - Coherent: P(over) + P(under) = 1 by construction
    """

    def __init__(self, max_val: int = 60) -> None:
        self.max_val = max_val
        # {(stat, role_bucket): {threshold_k: IsotonicRegression | None}}
        self.calibrators: dict[tuple, dict[int, Optional[IsotonicRegression]]] = {}

    def fit(
        self,
        oof_pmfs: list[dict[int, float]],
        oof_actuals: list[int],
        stat: str,
        role_bucket: str,
        min_samples: int = 30,
    ) -> "PMFCalibrator":
        """Fit IDR on OOF PMFs and actual outcomes.

        Args:
            oof_pmfs: list of dicts {value: probability} — one per OOF row
            oof_actuals: list of ints — actual stat values
            stat: stat name (pts, reb, ast, ...)
            role_bucket: player role bucket for stratification
            min_samples: skip if fewer than this many OOF rows
        """
        if len(oof_pmfs) < min_samples:
            log.debug("PMFCalibrator: skipping %s/%s — only %d OOF rows", stat, role_bucket, len(oof_pmfs))
            return self

        calibrators: dict[int, Optional[IsotonicRegression]] = {}
        actuals_arr = np.array(oof_actuals)

        for k in range(self.max_val + 1):
            # Model CDF at threshold k
            model_cdfs = np.array([
                sum(p for v, p in pmf.items() if v <= k)
                for pmf in oof_pmfs
            ])
            empirical = (actuals_arr <= k).astype(float)

            if model_cdfs.std() > 1e-6 and 0 < empirical.mean() < 1:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(model_cdfs, empirical)
                calibrators[k] = iso
            else:
                calibrators[k] = None

        self.calibrators[(stat, role_bucket)] = calibrators
        log.info("PMFCalibrator: fitted %s/%s on %d OOF rows", stat, role_bucket, len(oof_pmfs))
        return self

    def calibrate_pmf(
        self,
        raw_pmf: dict[int, float],
        stat: str,
        role_bucket: str,
    ) -> dict[int, float]:
        """Apply IDR calibration to convert raw PMF to calibrated PMF.

        Steps:
          1. Compute raw CDF
          2. Apply isotonic regression at each threshold
          3. Convert calibrated CDF back to PMF (differences)
          4. Normalize
        """
        key = (stat, role_bucket)
        if key not in self.calibrators:
            return raw_pmf

        cals = self.calibrators[key]
        max_k = min(self.max_val, max(raw_pmf.keys(), default=0))

        # Step 1: Raw CDF
        raw_cdf: dict[int, float] = {}
        cum = 0.0
        for k in range(max_k + 1):
            cum += raw_pmf.get(k, 0.0)
            raw_cdf[k] = min(cum, 1.0)

        # Step 2: Calibrated CDF
        cal_cdf: dict[int, float] = {}
        for k in range(max_k + 1):
            iso = cals.get(k)
            if iso is not None:
                cal_cdf[k] = float(iso.predict(np.array([raw_cdf[k]]))[0])
            else:
                cal_cdf[k] = raw_cdf[k]

        # Step 3: CDF → PMF (differences)
        cal_pmf: dict[int, float] = {}
        prev = 0.0
        for k in range(max_k + 1):
            cal_pmf[k] = max(0.0, cal_cdf.get(k, 0.0) - prev)
            prev = cal_cdf.get(k, 0.0)

        # Tail mass
        tail = max(0.0, 1.0 - prev)
        if tail > 0.0:
            cal_pmf[max_k] = cal_pmf.get(max_k, 0.0) + tail

        # Step 4: Normalize
        total = sum(cal_pmf.values())
        if total > 0:
            cal_pmf = {k: v / total for k, v in cal_pmf.items()}
        return cal_pmf

    def has_calibrator(self, stat: str, role_bucket: str) -> bool:
        return (stat, role_bucket) in self.calibrators

    def calibrate_p_over(
        self,
        raw_pmf: dict[int, float],
        line: float,
        stat: str,
        role_bucket: str,
    ) -> float:
        """Calibrate and return P(over line) from the calibrated PMF."""
        cal = self.calibrate_pmf(raw_pmf, stat, role_bucket)
        return sum(p for k, p in cal.items() if k > line)
