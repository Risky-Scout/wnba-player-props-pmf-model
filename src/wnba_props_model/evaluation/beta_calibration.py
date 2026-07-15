"""Beta calibration for binary P(over) predictions.

Three-parameter Beta calibration (Manokhin & Grønhaug, 2026) is superior to
isotonic regression because:
  1. It includes the identity map as a special case (a=b=1, c=0)
  2. When the model is already calibrated, it leaves predictions untouched
  3. Isotonic regression ALWAYS distorts already-calibrated predictions

Reference: Manokhin, V., & Grønhaug, D. (2026). Classifier Calibration at Scale.
ArXiv: https://arxiv.org/abs/2601.19944
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class BetaCalibrator:
    """Three-parameter Beta calibration for binary P(over).

    Calibration map: c(s) = 1 / (1 + exp(-c) * s^a / (1-s)^b)

    When a=b=1, c=0:  c(s) = s  (identity — no-op for calibrated models).
    This is the KEY advantage over isotonic regression.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0
        self.fitted = False

    def _cal_map(self, s: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        s = np.clip(s, 1e-8, 1 - 1e-8)
        logit = c + a * np.log(s) - b * np.log(1 - s)
        return 1.0 / (1.0 + np.exp(-logit))

    def _neg_ll(self, params: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
        a, b, c = params
        if a <= 0 or b <= 0:
            return 1e10
        p = self._cal_map(s, a, b, c)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        scores = np.clip(scores, 1e-8, 1 - 1e-8)
        result = minimize(
            self._neg_ll,
            x0=[1.0, 1.0, 0.0],
            args=(scores, labels),
            method="L-BFGS-B",
            bounds=[(0.01, 20.0), (0.01, 20.0), (-10.0, 10.0)],
        )
        if result.success:
            self.a, self.b, self.c = result.x
        self.fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if not self.fitted:
            return scores
        return self._cal_map(scores, self.a, self.b, self.c)

    def is_identity(self, tol: float = 0.15) -> bool:
        """Return True if calibration is approximately a no-op."""
        return (
            abs(self.a - 1.0) < tol
            and abs(self.b - 1.0) < tol
            and abs(self.c) < tol
        )

    @property
    def n_params(self) -> int:
        return 3

    def __repr__(self) -> str:
        return (
            f"BetaCalibrator(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f}, "
            f"fitted={self.fitted}, identity={self.is_identity()})"
        )


class PlattCalibrator:
    """Two-parameter Platt scaling — fallback for small calibration sets (<50 OOF samples).

    Less flexible than Beta but only 2 parameters = less overfitting risk.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        labels = np.asarray(labels, dtype=float)
        lr = LogisticRegression(C=1e6, solver="saga", max_iter=5000)
        lr.fit(scores, labels)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        self.fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if not self.fitted:
            return scores
        logit = self.a * scores + self.b
        return 1.0 / (1.0 + np.exp(-logit))

    @property
    def n_params(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"PlattCalibrator(a={self.a:.3f}, b={self.b:.3f}, fitted={self.fitted})"


def select_calibrator(n_oof_samples: int) -> str:
    """Select calibration method based on OOF sample count.

    Decision tree:
        n >= 500: Compare Beta, Isotonic, Platt on ECCE-MAD; pick best
        50 <= n < 500: Beta calibration (more sample-efficient + identity-safe)
        n < 50: Platt scaling (2 params only, minimal overfitting)
    """
    if n_oof_samples >= 500:
        return "comprehensive"
    elif n_oof_samples >= 50:
        return "beta"
    else:
        return "platt"


def fit_best_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str | None = None,
) -> BetaCalibrator | PlattCalibrator:
    """Fit the best calibrator given the number of samples."""
    if method is None:
        method = select_calibrator(len(scores))
    if method in ("beta", "comprehensive"):
        return BetaCalibrator().fit(scores, labels)
    else:
        return PlattCalibrator().fit(scores, labels)
