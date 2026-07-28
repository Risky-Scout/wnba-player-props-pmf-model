"""Availability model for Opportunity V2: P(player is active | pregame context).

Trained on all eligible historical rows with target ``did_play``. Its output is used ONLY for the
optional unconditional availability mixture and for active-roster / vacated-share renormalization; it
must NEVER be multiplied into the active PMF. Records its training feature contract and fails closed
on missing inference features.
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .contracts import status_prior

# Preferred features (section 15). Whatever subset is present at fit time becomes the required
# inference contract; forward-only status features are used automatically once available.
PREFERRED_FEATURES: tuple[str, ...] = (
    "availability_status_code", "availability_status_age_hours", "availability_conflict_flag",
    "reported_minutes_limit", "has_reported_minutes_limit",
    "player_active_rate_ewma", "player_dnp_streak_prior", "player_days_since_last_game",
    "team_expected_active_count", "team_out_count", "team_questionable_count",
)

_MIN_P, _MAX_P = 0.001, 0.999


class AvailabilityModelV2:
    VERSION = "opportunity_v2_availability_v1"

    def __init__(self, *, random_state: int = 42) -> None:
        self._random_state = random_state
        self._model: HistGradientBoostingClassifier | None = None
        self.feature_names_: list[str] = []
        self.feature_kinds_: dict[str, str] = {}
        self.training_cutoff_utc: str | None = None
        self.model_hash: str | None = None
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        did_play: pd.Series,
        sample_weight: np.ndarray | None = None,
        *,
        feature_columns: list[str] | None = None,
        training_cutoff_utc: str | None = None,
    ) -> "AvailabilityModelV2":
        if feature_columns is None:
            feature_columns = [c for c in PREFERRED_FEATURES if c in X.columns]
        if not feature_columns:
            raise ValueError("AvailabilityModelV2.fit: no usable feature columns present")
        self.feature_names_ = list(feature_columns)
        self.feature_kinds_ = {c: X[c].dtype.kind for c in self.feature_names_}
        y = pd.Series(did_play).astype("boolean").astype(float)
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        if y.nunique(dropna=True) < 2:
            raise ValueError("AvailabilityModelV2.fit: target did_play has a single class")
        model = HistGradientBoostingClassifier(random_state=self._random_state)
        model.fit(Xn, y.astype(int), sample_weight=sample_weight)
        self._model = model
        self.training_cutoff_utc = training_cutoff_utc
        self.model_hash = hashlib.sha256(pickle.dumps(model)).hexdigest()
        self._fitted = True
        return self

    def _check_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"AvailabilityModelV2.predict: missing required feature(s) {missing}")
        return X[self.feature_names_].apply(pd.to_numeric, errors="coerce")

    def predict_active_probability(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("AvailabilityModelV2.predict before fit")
        Xn = self._check_features(X)
        proba = self._model.predict_proba(Xn)[:, 1]
        return np.clip(proba, _MIN_P, _MAX_P)

    @staticmethod
    def status_prior_fallback(status_normalized: pd.Series) -> np.ndarray:
        """Diagnostic-only fallback active probability from normalized status (not a trained output)."""
        return np.array([status_prior(s) for s in status_normalized], dtype=float)
