"""Component opportunity-rate models for Opportunity V2 (per-minute fallback / diagnostic).

Predicts an opportunity COUNT mean for a proposed minutes value from a strictly-lagged per-minute
rate, with training-only overdispersion and a MINUTES-SENSITIVE zero probability
(P(O>0|M)=1-exp(-h*M)). The team-share model is the principal opportunity source; this model is the
fallback/comparator and is never averaged with it without a globally frozen policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .contracts import DATA_TIER_BOX


@dataclass
class OpportunityPrediction:
    mean: np.ndarray
    dispersion_r: np.ndarray | None
    p_zero: np.ndarray | None
    source_tier: np.ndarray


class OpportunityRateModel:
    VERSION = "opportunity_v2_rate_v1"

    def __init__(self, *, zero_heavy_threshold: float = 0.20, source_tier: int = DATA_TIER_BOX,
                 random_state: int = 42) -> None:
        self._model: HistGradientBoostingRegressor | None = None
        self.feature_names_: list[str] = []
        self._dispersion_r: float | None = None
        self._zero_heavy = False
        self._zero_threshold = float(zero_heavy_threshold)
        self._source_tier = int(source_tier)
        self._random_state = random_state
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        opportunity_count: pd.Series,
        actual_minutes: pd.Series,
        did_play: pd.Series,
        sample_weight: np.ndarray | None = None,
        *,
        feature_columns: list[str] | None = None,
    ) -> "OpportunityRateModel":
        if feature_columns is None:
            feature_columns = [c for c in X.columns if X[c].dtype.kind in "fiub"]
        if not feature_columns:
            raise ValueError("OpportunityRateModel.fit: no usable feature columns")
        self.feature_names_ = list(feature_columns)
        played = pd.Series(did_play).astype("boolean").fillna(False).to_numpy()
        mins = pd.to_numeric(actual_minutes, errors="coerce").to_numpy()
        cnt = pd.to_numeric(opportunity_count, errors="coerce").to_numpy()
        mask = played & np.isfinite(mins) & (mins > 0) & np.isfinite(cnt) & (cnt >= 0)
        if int(mask.sum()) < 30:
            raise ValueError("OpportunityRateModel.fit: insufficient appearance rows")
        target = np.log1p(cnt[mask] / mins[mask])
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        sw = None if sample_weight is None else np.asarray(sample_weight)[mask]
        model = HistGradientBoostingRegressor(loss="squared_error", random_state=self._random_state)
        model.fit(Xn[mask], target, sample_weight=sw)
        self._model = model

        # Training-only NB2 overdispersion estimated CONDITIONAL on the minutes-scaled predicted mean
        # (method of moments on Pearson residuals): E[(y-mu)^2] = mu + mu^2/r. Conditioning on mu
        # removes the minutes-driven variance that would otherwise inflate the tail.
        rate_tr = np.clip(np.expm1(model.predict(Xn[mask])), 0.0, None)
        mu = rate_tr * mins[mask]
        resid2 = (cnt[mask] - mu) ** 2
        num = float(np.sum(mu ** 2))
        den = float(np.sum(resid2 - mu))
        # Floor r to keep the count tail physically plausible (avoids absurd heavy tails from noise).
        self._dispersion_r = float(np.clip(num / den, 2.0, 1e6)) if den > 1e-9 and num > 0 else None

        # Zero-inflation policy for sparse components.
        zero_frac = float(np.mean(cnt[mask] == 0))
        self._zero_heavy = zero_frac >= self._zero_threshold
        self._fitted = True
        return self

    def _rate(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"OpportunityRateModel.predict: missing feature(s) {missing}")
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        return np.clip(np.expm1(self._model.predict(Xn)), 0.0, None)

    def predict_for_minutes(self, X: pd.DataFrame, minutes: np.ndarray) -> OpportunityPrediction:
        if not self._fitted or self._model is None:
            raise RuntimeError("OpportunityRateModel.predict before fit")
        rate = self._rate(X)  # per-minute hazard/rate
        m = np.asarray(minutes, dtype=float)
        mean = rate * m
        n = len(rate)
        disp = None if self._dispersion_r is None else np.full(n, self._dispersion_r)
        if self._zero_heavy:
            # Minutes-sensitive nonzero probability: P(O>0|M) = 1 - exp(-rate*M).
            p_zero = np.exp(-np.clip(rate * m, 0.0, 700.0))
        else:
            p_zero = None
        tier = np.full(n, self._source_tier, dtype=int)
        return OpportunityPrediction(mean=mean, dispersion_r=disp, p_zero=p_zero, source_tier=tier)
