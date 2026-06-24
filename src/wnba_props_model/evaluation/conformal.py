"""Conformal prediction intervals for player props.

Conformal prediction provides distribution-free coverage guarantees.
For a 90% interval, the actual stat value will fall inside the predicted
interval at least 90% of the time under exchangeability.

Used to FLAG props where the model has no edge:
  - If the line falls INSIDE the conformal interval → no edge (model too uncertain)
  - If the line falls OUTSIDE the conformal interval → edge exists

References:
  Datta et al. (2025). Conformal Prediction = Bayes? https://arxiv.org/abs/2512.23308
  Marx et al. (2022). Modular Conformal Calibration. https://arxiv.org/abs/2206.11468
  Hullman et al. (2025). Conformal Prediction and Human Decision Making.
    https://arxiv.org/abs/2503.11709
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class ConformalPropPredictor:
    """Split conformal prediction intervals for player props.

    Coverage guarantee: P(actual in interval) >= 1 - alpha
    under exchangeability (reasonable within a WNBA season).

    Usage:
        If the prop line falls inside the conformal interval,
        there is NO edge — the model is too uncertain to be confident
        in either direction.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        # {(stat, role): conformal quantile}
        self.quantiles: dict[tuple[str, str], float] = {}

    def fit(
        self,
        oof_predictions: np.ndarray,
        oof_actuals: np.ndarray,
        stat: str,
        role: str,
        min_samples: int = 20,
    ) -> "ConformalPropPredictor":
        """Compute conformal quantile from OOF residuals.

        Uses split conformal prediction: residuals = |actual - predicted|
        The conformal quantile is the ceil((n+1)*(1-alpha))/n empirical quantile.
        """
        oof_predictions = np.asarray(oof_predictions, dtype=float)
        oof_actuals = np.asarray(oof_actuals, dtype=float)
        residuals = np.abs(oof_actuals - oof_predictions)
        n = len(residuals)
        if n < min_samples:
            log.debug("ConformalPropPredictor: too few samples for %s/%s (%d)", stat, role, n)
            return self
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_idx = max(0, min(q_idx, n - 1))
        self.quantiles[(stat, role)] = float(np.sort(residuals)[q_idx])
        log.debug(
            "ConformalPropPredictor: %s/%s quantile=%.2f (n=%d)",
            stat, role, self.quantiles[(stat, role)], n,
        )
        return self

    def predict_interval(
        self,
        predicted_value: float,
        stat: str,
        role: str,
    ) -> tuple[float, float]:
        """Return (lower, upper) conformal prediction interval."""
        key = (stat, role)
        if key not in self.quantiles:
            q = 5.0  # fallback half-width
        else:
            q = self.quantiles[key]
        return predicted_value - q, predicted_value + q

    def check_edge(
        self,
        predicted_value: float,
        line: float,
        stat: str,
        role: str,
    ) -> tuple[bool, str]:
        """Is the line outside the conformal interval?

        Returns:
            has_edge (bool): True if line is outside the interval
            direction (str): "over", "under", or "none"
        """
        lower, upper = self.predict_interval(predicted_value, stat, role)
        if line > upper:
            return True, "under"
        elif line < lower:
            return True, "over"
        return False, "none"

    def edge_confidence(
        self,
        predicted_value: float,
        line: float,
        stat: str,
        role: str,
    ) -> float:
        """How far outside the conformal interval is the line? (0 = no edge)

        Returns a normalized distance: 0 means the line is at the interval edge,
        >0 means the line is outside (stronger edge).
        """
        lower, upper = self.predict_interval(predicted_value, stat, role)
        if line > upper:
            return float(line - upper)
        elif line < lower:
            return float(lower - line)
        return 0.0

    def coverage_summary(self) -> dict[str, float]:
        """Return the fitted quantile for each (stat, role) bucket."""
        return {f"{s}/{r}": q for (s, r), q in self.quantiles.items()}
