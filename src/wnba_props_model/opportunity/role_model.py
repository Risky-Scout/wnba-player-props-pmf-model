"""Starting-role model for Opportunity V2: P(player starts | active, pregame context).

Trained only on appearances (``did_play == True``). Prefers official starter labels; proxy
(minutes-derived) labels may be used only when explicitly enabled and are down-weighted. Never uses
target-game actual minutes as a feature.
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .contracts import STARTER_LABEL_OFFICIAL

PREFERRED_FEATURES: tuple[str, ...] = (
    "projected_starter_snapshot", "confirmed_starter_snapshot", "lineup_status_code",
    "player_start_rate_ewma", "player_minutes_ewma", "vacated_minutes_share",
    "same_position_absence_weight", "higher_usage_teammate_absence_weight",
    "last_game_starters_available_count",
)

_MIN_P, _MAX_P = 0.001, 0.999


class StartingRoleModelV2:
    VERSION = "opportunity_v2_role_v1"

    def __init__(self, *, allow_proxy_labels: bool = False, proxy_label_weight: float = 0.25,
                 random_state: int = 42) -> None:
        self._model: HistGradientBoostingClassifier | None = None
        self.feature_names_: list[str] = []
        self.allow_proxy_labels = allow_proxy_labels
        self.proxy_label_weight = float(proxy_label_weight)
        self._random_state = random_state
        self.model_hash: str | None = None
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        actual_started: pd.Series,
        did_play: pd.Series,
        label_quality: pd.Series,
        sample_weight: np.ndarray | None = None,
        *,
        feature_columns: list[str] | None = None,
    ) -> "StartingRoleModelV2":
        if feature_columns is None:
            feature_columns = [c for c in PREFERRED_FEATURES if c in X.columns]
        if not feature_columns:
            raise ValueError("StartingRoleModelV2.fit: no usable feature columns present")
        self.feature_names_ = list(feature_columns)

        played = pd.Series(did_play).astype("boolean").fillna(False).to_numpy()
        quality = pd.Series(label_quality).astype("string").fillna("").to_numpy()
        official = quality == STARTER_LABEL_OFFICIAL
        usable = played & (official | (self.allow_proxy_labels & ~official))
        if int(usable.sum()) == 0:
            raise ValueError("StartingRoleModelV2.fit: no usable (appeared + labeled) training rows")

        y = pd.Series(actual_started).astype("boolean").fillna(False).astype(int).to_numpy()
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")

        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w * np.where(official, 1.0, self.proxy_label_weight)

        if len(np.unique(y[usable])) < 2:
            raise ValueError("StartingRoleModelV2.fit: single-class starter target")
        model = HistGradientBoostingClassifier(random_state=self._random_state)
        model.fit(Xn[usable], y[usable], sample_weight=w[usable])
        self._model = model
        self.model_hash = hashlib.sha256(pickle.dumps(model)).hexdigest()
        self._fitted = True
        return self

    def predict_start_probability(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("StartingRoleModelV2.predict before fit")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"StartingRoleModelV2.predict: missing feature(s) {missing}")
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        return np.clip(self._model.predict_proba(Xn)[:, 1], _MIN_P, _MAX_P)
