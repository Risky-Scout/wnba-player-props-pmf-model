"""Role-aware post-hoc correction for the minutes model.

Fits per-role multiplicative bias corrections and isotonic DNP calibration
on OOF residuals, then applies them at prediction time.

The fitted object is serialized to artifacts/models/ during weekly_calibration
and loaded during inference via load_minutes_correction().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RoleAwareMinutesCorrection:
    """Post-hoc correction for the minutes model by role bucket.

    Fits per-role multiplicative bias corrections and isotonic DNP calibration
    on OOF residuals, then applies them at prediction time.
    """

    ROLE_SIGMA_FLOORS = {
        "starter": 3.0,
        "core": 3.5,
        "rotation": 4.0,
        "bench": 5.0,
        "bench_low": 5.5,
        "bench_rotation": 4.5,
        "fringe": 6.0,
        "inactive_risk": 7.0,
        "workhorse": 3.0,
    }

    def __init__(self) -> None:
        self.role_mean_correction: dict[str, float] = {}
        self.role_dnp_calibrators: dict[str, IsotonicRegression] = {}
        self.fitted: bool = False

    def fit(
        self,
        minutes_pred: np.ndarray,
        actual_minutes: np.ndarray,
        role_buckets: np.ndarray,
        p_dnp_pred: np.ndarray,
    ) -> "RoleAwareMinutesCorrection":
        """Fit corrections from OOF predictions and actuals.

        Parameters
        ----------
        minutes_pred:   Model-predicted playing minutes (n,)
        actual_minutes: Actual minutes played (n,); 0 = DNP
        role_buckets:   Role bucket string for each row (n,)
        p_dnp_pred:     Model-predicted P(DNP) (n,)
        """
        unique_roles = np.unique(role_buckets[~pd.isnull(role_buckets)])
        for role in unique_roles:
            mask = role_buckets == role
            if mask.sum() < 30:
                continue
            pred = minutes_pred[mask]
            actual = actual_minutes[mask]

            # Multiplicative mean correction on non-DNP games only
            nonzero = actual > 0
            if nonzero.sum() > 20:
                pred_nz = pred[nonzero]
                actual_nz = actual[nonzero]
                ratio = np.median(actual_nz) / np.median(pred_nz)
                self.role_mean_correction[role] = float(np.clip(ratio, 0.5, 1.5))
            else:
                self.role_mean_correction[role] = 1.0

            # DNP isotonic calibration
            dnp_actual = (actual == 0).astype(int)
            dnp_pred_role = p_dnp_pred[mask]
            if dnp_actual.sum() > 10 and (1 - dnp_actual).sum() > 10:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(dnp_pred_role, dnp_actual)
                self.role_dnp_calibrators[role] = iso

        self.fitted = True
        return self

    def correct(
        self,
        minutes_mean: np.ndarray,
        minutes_sigma: np.ndarray,
        p_dnp: np.ndarray,
        role_buckets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Apply fitted corrections to minutes predictions.

        Returns a dict with keys: minutes_mean, minutes_sigma, p_dnp.
        """
        corrected_mean = minutes_mean.copy().astype(float)
        corrected_sigma = minutes_sigma.copy().astype(float)
        corrected_dnp = p_dnp.copy().astype(float)

        for role in np.unique(role_buckets):
            if pd.isnull(role):
                continue
            mask = role_buckets == role

            if role in self.role_mean_correction:
                corrected_mean[mask] *= self.role_mean_correction[role]

            sigma_floor = self.ROLE_SIGMA_FLOORS.get(str(role), 4.0)
            corrected_sigma[mask] = np.maximum(corrected_sigma[mask], sigma_floor)

            if role in self.role_dnp_calibrators:
                corrected_dnp[mask] = self.role_dnp_calibrators[role].predict(
                    p_dnp[mask]
                )

        corrected_dnp = np.clip(corrected_dnp, 0.0, 0.95)
        return {
            "minutes_mean": corrected_mean,
            "minutes_sigma": corrected_sigma,
            "p_dnp": corrected_dnp,
        }

    def save(self, path: str | Path) -> None:
        """Serialize to disk via joblib."""
        import joblib  # noqa: PLC0415
        joblib.dump(self, path)
        logger.info("Saved RoleAwareMinutesCorrection to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RoleAwareMinutesCorrection":
        """Load from disk via joblib."""
        import joblib  # noqa: PLC0415
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RoleAwareMinutesCorrection, got {type(obj)}")
        return obj


def load_minutes_correction(
    model_dir: str | Path,
    filename: str = "minutes_correction.pkl",
) -> "RoleAwareMinutesCorrection | None":
    """Load a fitted RoleAwareMinutesCorrection from model_dir, or return None."""
    path = Path(model_dir) / filename
    if not path.exists():
        return None
    try:
        return RoleAwareMinutesCorrection.load(path)
    except Exception as exc:
        logger.warning("Failed to load minutes correction from %s: %s", path, exc)
        return None
