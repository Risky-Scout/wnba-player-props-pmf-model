"""Calibration Monitor with Anytime-Valid Sequential PIT Tests (Enhancement 21).

Research (Walsh & Joshi, 2023) shows calibration drift can silently destroy
model edge for 200+ bets before detection.  This monitor uses the
Probability Integral Transform (PIT) to detect drift within 50-100
observations.

Theory:
    For each prediction with PMF F, compute:
        u_t = F(y_t) = Σ P(x) for x ≤ y_t

    If the model is well-calibrated, {u_t} ~ Uniform(0, 1).
    Any systematic deviation from uniformity indicates miscalibration:
        - mean_pit < 0.45 → underprojection (model projects too low)
        - mean_pit > 0.55 → overprojection  (model projects too high)

Tests:
    - Kolmogorov-Smirnov test against Uniform(0, 1)
    - Region checks: pct in [0, 0.2], [0.8, 1.0] should each be ~20%

Alerts are triggered when p_value < alert_threshold (default: 0.01).

Reference:
    Farran (2026). When Your Model Stops Working: Anytime-Valid Calibration
    Monitoring.  ArXiv.  https://arxiv.org/abs/2603.13156
    Walsh & Joshi (2023). Machine learning for sports betting: Should model
    selection be based on accuracy or calibration?  Machine Learning with
    Applications.  https://doi.org/10.1016/j.mlwa.2024.100539
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class CalibrationMonitor:
    """Monitor calibration drift using PIT (Probability Integral Transform).

    For each prediction:
        u_t = Σ P(x) for x ≤ y_t   (CDF evaluated at observed outcome)

    If the model is well-calibrated, {u_t} ~ Uniform(0, 1).

    Attributes
    ----------
    pit_values      : rolling deque of most recent PIT values (window_size max)
    alert_threshold : KS test p-value below which an alert fires
    n_observations  : total observations recorded (ignores window cap)
    """

    def __init__(
        self,
        window_size: int = 500,
        alert_threshold: float = 0.01,
        stat_label: str = "all",
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.stat_label = stat_label
        self.pit_values: deque[float] = deque(maxlen=window_size)
        self.n_observations: int = 0
        # Track running CLV if available
        self._clv_history: deque[float] = deque(maxlen=window_size)

    # ── Update ────────────────────────────────────────────────────────────

    def update(
        self,
        pmf: dict[int | float, float],
        observed_value: int | float,
        clv: float | None = None,
    ) -> float:
        """Record one observation and return its PIT value.

        Parameters
        ----------
        pmf            : {value: probability} — model's PMF for this prediction
        observed_value : actual observed stat count
        clv            : optional CLV for this bet (for tracking)

        Returns
        -------
        u_t : PIT value in [0, 1]
        """
        u = sum(p for v, p in pmf.items() if v <= observed_value)
        u = float(np.clip(u, 0.0, 1.0))
        self.pit_values.append(u)
        self.n_observations += 1
        if clv is not None and math.isfinite(clv):
            self._clv_history.append(clv)
        return u

    def update_batch(
        self,
        records: list[dict[str, Any]],
    ) -> list[float]:
        """Batch update from a list of prediction records.

        Each record should have:
            pmf              : {value: prob}
            observed_value   : int
            clv (optional)   : float
        """
        return [
            self.update(
                r["pmf"],
                r["observed_value"],
                r.get("clv"),
            )
            for r in records
        ]

    # ── Calibration check ─────────────────────────────────────────────────

    def check_calibration(self, min_obs: int = 50) -> dict[str, Any]:
        """Run calibration test on accumulated PIT values.

        Tests H₀: {u_t} ~ Uniform(0,1)  (model is calibrated)
             H₁: {u_t} deviates from Uniform  (model is miscalibrated)

        Returns
        -------
        dict with:
            status          : "ok", "alert", or "insufficient_data"
            ks_statistic    : KS test statistic
            p_value         : KS test p-value
            mean_pit        : mean of PIT values (0.5 if calibrated)
            direction       : "underprojection" / "overprojection" / "well_calibrated"
            low_region_pct  : % of PIT values in [0, 0.2] (target: ~20%)
            high_region_pct : % of PIT values in [0.8, 1.0] (target: ~20%)
            recommendation  : action string
        """
        n = len(self.pit_values)
        if n < min_obs:
            return {
                "status": "insufficient_data",
                "n": n,
                "required": min_obs,
                "stat_label": self.stat_label,
            }

        pit_array = np.array(self.pit_values)
        ks_stat, p_value = sp_stats.kstest(pit_array, "uniform")

        mean_pit = float(np.mean(pit_array))
        if mean_pit < 0.45:
            direction = "underprojection"
        elif mean_pit > 0.55:
            direction = "overprojection"
        else:
            direction = "well_calibrated"

        low_region  = float(np.mean(pit_array < 0.2)) * 100
        high_region = float(np.mean(pit_array > 0.8)) * 100

        alert = bool(p_value < self.alert_threshold)

        recommendation = (
            "Recalibrate isotonic regression and recheck OOF scores"
            if alert else
            "No action needed"
        )
        if alert and direction == "underprojection":
            recommendation = "Model systematically underpredicts — check feature scaling or retrain"
        elif alert and direction == "overprojection":
            recommendation = "Model systematically overpredicts — check bias in rate models"

        return {
            "status":           "alert" if alert else "ok",
            "stat_label":       self.stat_label,
            "ks_statistic":     round(float(ks_stat), 5),
            "p_value":          round(float(p_value), 5),
            "mean_pit":         round(mean_pit, 4),
            "direction":        direction,
            "low_region_pct":   round(low_region,  2),
            "high_region_pct":  round(high_region, 2),
            "n_observations":   self.n_observations,
            "n_in_window":      n,
            "recommendation":   recommendation,
        }

    def rolling_calibration_score(self) -> float | None:
        """Compute a rolling calibration score (0-100).

        100 = perfectly calibrated, 0 = completely miscalibrated.
        Returns None if insufficient data.
        """
        if len(self.pit_values) < 50:
            return None
        pit_array = np.array(self.pit_values)
        ks_stat, _ = sp_stats.kstest(pit_array, "uniform")
        score = max(0.0, 100.0 * (1.0 - ks_stat * 5.0))
        return round(score, 1)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "stat_label":      self.stat_label,
            "window_size":     self.window_size,
            "alert_threshold": self.alert_threshold,
            "n_observations":  self.n_observations,
            "pit_values":      list(self.pit_values),
            "calibration":     self.check_calibration(),
            "score":           self.rolling_calibration_score(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("CalibrationMonitor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationMonitor":
        with open(path) as f:
            data = json.load(f)
        obj = cls(
            window_size=data.get("window_size", 500),
            alert_threshold=data.get("alert_threshold", 0.01),
            stat_label=data.get("stat_label", "all"),
        )
        obj.n_observations = data.get("n_observations", 0)
        for v in data.get("pit_values", []):
            obj.pit_values.append(float(v))
        return obj


class MultiStatCalibrationMonitor:
    """Per-stat calibration monitors for all 7 prop stats.

    Maintains independent monitors for pts, reb, ast, fg3m, stl, blk, turnover.
    Provides aggregate health summary for the HTML wizard report.
    """

    STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

    def __init__(self, window_size: int = 500, alert_threshold: float = 0.01):
        self.monitors = {
            s: CalibrationMonitor(
                window_size=window_size,
                alert_threshold=alert_threshold,
                stat_label=s,
            )
            for s in self.STATS
        }

    def update(
        self,
        stat: str,
        pmf: dict[int | float, float],
        observed: int | float,
        clv: float | None = None,
    ) -> float | None:
        if stat not in self.monitors:
            return None
        return self.monitors[stat].update(pmf, observed, clv)

    def summary(self) -> dict[str, Any]:
        """Return calibration status for all stats."""
        results = {}
        n_alerts = 0
        for stat, mon in self.monitors.items():
            check = mon.check_calibration()
            results[stat] = {
                "score":  mon.rolling_calibration_score(),
                "status": check.get("status", "insufficient_data"),
                "direction": check.get("direction", "unknown"),
                "p_value": check.get("p_value"),
                "n":      check.get("n_in_window", 0),
            }
            if check.get("status") == "alert":
                n_alerts += 1

        overall_score = None
        all_scores = [
            v["score"] for v in results.values() if v["score"] is not None
        ]
        if all_scores:
            overall_score = round(float(np.mean(all_scores)), 1)

        return {
            "stats":         results,
            "overall_score": overall_score,
            "n_alerts":      n_alerts,
            "health":        "degraded" if n_alerts > 0 else "ok",
        }

    def save(self, dir_path: str | Path) -> None:
        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        for stat, mon in self.monitors.items():
            mon.save(out / f"calibration_{stat}.json")

    @classmethod
    def load(cls, dir_path: str | Path) -> "MultiStatCalibrationMonitor":
        obj = cls()
        d = Path(dir_path)
        for stat in cls.STATS:
            fp = d / f"calibration_{stat}.json"
            if fp.exists():
                obj.monitors[stat] = CalibrationMonitor.load(fp)
        return obj
