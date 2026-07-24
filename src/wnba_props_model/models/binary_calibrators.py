"""Binary P(over) calibrators with a uniform registry-compatible interface (W0.5).

Each calibrator implements ``fit(p, y)`` and ``predict(X)`` where ``X`` is array-like of
shape (n, 1) holding the raw model P(over). ``predict`` returns calibrated probabilities in
(0, 1). The ``.predict([[p]])`` contract matches BinaryCalibrationRegistry so a fitted
calibrator can be joblib-dumped and loaded fail-closed at delivery/proof time.

Families:
  * IdentityCalibrator - passthrough (the null candidate).
  * PlattCalibrator    - logistic regression on logit(p) (Platt scaling).
  * BetaCalibrator     - Kull et al. beta calibration: logistic on [log p, log(1-p)].
  * IsotonicCalibrator - monotonic isotonic regression (out_of_bounds='clip').
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _as_p(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return np.clip(arr.ravel(), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


class IdentityCalibrator:
    """Null calibrator: returns the input probability unchanged."""
    family = "identity"

    def fit(self, p, y):  # noqa: ARG002
        return self

    def predict(self, X):
        return _as_p(X)


class PlattCalibrator:
    """Platt scaling: sigmoid(a * logit(p) + b), fit by logistic regression."""
    family = "platt"

    def __init__(self):
        self._lr = None

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        z = _logit(_as_p(p)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._lr.fit(z, np.asarray(y, dtype=int))
        return self

    def predict(self, X):
        z = _logit(_as_p(X)).reshape(-1, 1)
        return np.clip(self._lr.predict_proba(z)[:, 1], _EPS, 1.0 - _EPS)


class BetaCalibrator:
    """Beta calibration (Kull, Silva Filho, Flach 2017): logistic on [log p, log(1-p)]."""
    family = "beta"

    def __init__(self):
        self._lr = None

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        pp = _as_p(p)
        feats = np.column_stack([np.log(pp), np.log(1.0 - pp)])
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._lr.fit(feats, np.asarray(y, dtype=int))
        return self

    def predict(self, X):
        pp = _as_p(X)
        feats = np.column_stack([np.log(pp), np.log(1.0 - pp)])
        return np.clip(self._lr.predict_proba(feats)[:, 1], _EPS, 1.0 - _EPS)


class IsotonicCalibrator:
    """Monotonic isotonic regression from raw P(over) to realized over-rate."""
    family = "isotonic"

    def __init__(self):
        self._iso = None

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(_as_p(p), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return np.clip(self._iso.predict(_as_p(X)), _EPS, 1.0 - _EPS)


CALIBRATOR_FAMILIES = {
    "identity": IdentityCalibrator,
    "platt": PlattCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}
