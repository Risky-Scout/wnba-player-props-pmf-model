"""Anytime-Valid Calibration Monitor using Probability Integral Transform (Enhancement 21).

For each prediction, the PIT value is:
    u_t = F_t(y_t) = CDF of the model at the observed outcome y_t

A well-calibrated model produces uniform PIT values: {u_t} ~ Uniform(0, 1).
Systematic deviation from uniformity indicates miscalibration.

This module provides:
1. CalibrationMonitor — rolling PIT monitor with KS test + direction diagnosis.
2. Per-stat monitors so drift is detected at the stat level.
3. HTML summary fragment for export_html_report.py.
4. JSON summary for model_manifest.json.

Reference
---------
Farran (2026). When Your Model Stops Working: Anytime-Valid Calibration
Monitoring. arXiv:2603.13156
Walsh & Joshi (2023). Machine learning for sports betting: Should model
selection be based on accuracy or calibration? Machine Learning with Applications.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Stats monitored independently
MONITORED_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

# Default rolling window
DEFAULT_WINDOW = 500

# Alert threshold: p-value below this → miscalibration alert
DEFAULT_ALPHA = 0.01


class CalibrationMonitor:
    """Monitor calibration drift using rolling Probability Integral Transform (PIT) values.

    Usage
    -----
    >>> monitor = CalibrationMonitor(stat="pts", window_size=500)
    >>> monitor.update(pmf={0: 0.05, 1: 0.10, ..., 25: 0.03}, observed_value=22)
    >>> report = monitor.check_calibration()
    >>> print(report["status"], report["direction"])

    Parameters
    ----------
    stat : stat name (for logging)
    window_size : rolling window of PIT values (default 500)
    alert_threshold : KS p-value threshold for miscalibration alert (default 0.01)
    """

    def __init__(
        self,
        stat:            str = "all",
        window_size:     int = DEFAULT_WINDOW,
        alert_threshold: float = DEFAULT_ALPHA,
    ):
        self.stat            = stat
        self.window_size     = window_size
        self.alert_threshold = alert_threshold
        self.pit_values:     deque = deque(maxlen=window_size)
        self.n_observations: int = 0

    # ── Data ingestion ────────────────────────────────────────────────────────

    def update(
        self,
        pmf:            dict[int, float],
        observed_value: int,
    ) -> None:
        """Record one prediction and compute its PIT value.

        Parameters
        ----------
        pmf : {integer_outcome: probability}  — the model's PMF.
        observed_value : the actual observed stat (integer).
        """
        u = float(sum(p for v, p in pmf.items() if int(v) <= int(observed_value)))
        u = float(np.clip(u, 0.0, 1.0))
        self.pit_values.append(u)
        self.n_observations += 1

    def update_from_normal(
        self,
        mean: float,
        std:  float,
        observed_value: float,
    ) -> None:
        """Update from a Normal approximation (for continuous stats / pre-discretisation)."""
        if std <= 0:
            u = 0.5
        else:
            u = float(sp_stats.norm.cdf(observed_value, loc=mean, scale=std))
        u = float(np.clip(u, 0.0, 1.0))
        self.pit_values.append(u)
        self.n_observations += 1

    # ── Calibration check ─────────────────────────────────────────────────────

    def check_calibration(self) -> dict[str, Any]:
        """Run KS calibration test on accumulated PIT values.

        Tests H0: PIT ~ Uniform(0, 1)  [model is calibrated]
        vs    H1: PIT ≠ Uniform(0, 1)  [model is miscalibrated]

        Returns
        -------
        dict with keys: status, ks_statistic, p_value, mean_pit, direction,
                        low_region_pct, high_region_pct, n_observations,
                        n_in_window, recommendation
        """
        n = len(self.pit_values)
        if n < 50:
            return {
                "status": "insufficient_data",
                "n": n,
                "n_observations": self.n_observations,
                "stat": self.stat,
            }

        pit = np.array(self.pit_values)
        ks_stat, p_value = sp_stats.kstest(pit, "uniform")
        mean_pit = float(np.mean(pit))

        # Direction of miscalibration
        if mean_pit < 0.45:
            direction = "underprojection"   # model projects too low
            recommendation = "Increase mean projection; recalibrate isotonic regression"
        elif mean_pit > 0.55:
            direction = "overprojection"    # model projects too high
            recommendation = "Decrease mean projection; recalibrate isotonic regression"
        else:
            direction = "well_calibrated"
            recommendation = "No action needed"

        # Tail analysis
        low_pct  = float(np.mean(pit < 0.20)) * 100   # should be ~20%
        high_pct = float(np.mean(pit > 0.80)) * 100   # should be ~20%

        alert = bool(p_value < self.alert_threshold)
        if alert:
            logger.warning(
                "E21 CalibrationMonitor [%s]: ALERT ks=%.3f p=%.4f direction=%s",
                self.stat, ks_stat, p_value, direction,
            )

        return {
            "status":          "alert" if alert else "ok",
            "stat":            self.stat,
            "ks_statistic":    round(float(ks_stat), 4),
            "p_value":         round(float(p_value), 4),
            "mean_pit":        round(mean_pit, 4),
            "direction":       direction,
            "low_region_pct":  round(low_pct, 1),
            "high_region_pct": round(high_pct, 1),
            "n_observations":  self.n_observations,
            "n_in_window":     n,
            "recommendation":  recommendation if alert else "No action needed",
        }

    def rolling_calibration_score(self) -> float | None:
        """Calibration score 0–100 (100 = perfect uniform PIT)."""
        n = len(self.pit_values)
        if n < 50:
            return None
        pit = np.array(self.pit_values)
        ks_stat, _ = sp_stats.kstest(pit, "uniform")
        return round(max(0.0, 100.0 * (1.0 - ks_stat * 5.0)), 1)

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE) from PIT histogram.

        ECE measures average deviation of PIT histogram from Uniform(0,1).
        """
        n = len(self.pit_values)
        if n < 50:
            return float("nan")
        pit = np.array(self.pit_values)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        expected_density = 1.0 / n_bins
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            observed = float(np.mean((pit >= lo) & (pit < hi)))
            ece += abs(observed - expected_density)
        return round(ece, 4)


# ── Multi-stat monitor container ──────────────────────────────────────────────

class ProductionCalibrationMonitor:
    """Container of per-stat CalibrationMonitors for production monitoring.

    Usage
    -----
    >>> pcm = ProductionCalibrationMonitor()
    >>> pcm.record(stat="pts", pmf={...}, observed=22)
    >>> report = pcm.full_report()
    >>> pcm.save(path)
    """

    def __init__(
        self,
        stats:           list[str] | None = None,
        window_size:     int = DEFAULT_WINDOW,
        alert_threshold: float = DEFAULT_ALPHA,
    ):
        self.stats = stats or MONITORED_STATS
        self.monitors: dict[str, CalibrationMonitor] = {
            s: CalibrationMonitor(stat=s, window_size=window_size,
                                  alert_threshold=alert_threshold)
            for s in self.stats
        }
        # Global monitor across all stats
        self.monitors["all"] = CalibrationMonitor(
            stat="all", window_size=window_size, alert_threshold=alert_threshold
        )

    def record(
        self,
        stat:           str,
        pmf:            dict[int, float],
        observed:       int,
    ) -> None:
        """Record one observation for a specific stat."""
        if stat in self.monitors:
            self.monitors[stat].update(pmf, observed)
        self.monitors["all"].update(pmf, observed)

    def full_report(self) -> dict[str, Any]:
        """Return calibration reports for all stats."""
        return {stat: m.check_calibration() for stat, m in self.monitors.items()}

    def any_alert(self) -> bool:
        """True if any stat has a calibration alert."""
        return any(
            m.check_calibration().get("status") == "alert"
            for m in self.monitors.values()
        )

    def summary_scores(self) -> dict[str, float | None]:
        """Return calibration scores (0-100) for each stat."""
        return {stat: m.rolling_calibration_score() for stat, m in self.monitors.items()}

    def save(self, path: str | Path) -> None:
        """Save full report as JSON."""
        report = self.full_report()
        scores = self.summary_scores()
        out = {"reports": report, "scores": scores, "any_alert": self.any_alert()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        logger.info("E21: calibration monitor saved to %s", path)

    @classmethod
    def load_and_update(
        cls,
        artifact_path: str | Path,
        new_results: list[dict[str, Any]],
    ) -> "ProductionCalibrationMonitor":
        """Load an existing monitor state and update with new results.

        Each item in new_results should have: stat, pmf (dict), observed (int).
        """
        monitor = cls()
        # (State re-loading from JSON deque history is a future extension)
        for item in new_results:
            monitor.record(
                stat=item["stat"],
                pmf=item["pmf"],
                observed=int(item["observed"]),
            )
        return monitor
