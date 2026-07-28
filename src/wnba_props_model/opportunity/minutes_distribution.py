"""Conditional (active-only) minutes distribution for Opportunity V2.

Quantile-regression forest of minutes GIVEN the player appears. Produces a monotone set of
quantiles, a piecewise-linear inverse CDF, and deterministic equal-probability samples used to
average prop PMFs. DNP (zero-minute) rows are excluded from training; the distribution is never
multiplied by ``1 - p_dnp``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PREFERRED_FEATURES: tuple[str, ...] = (
    "player_minutes_ewma", "player_minutes_std_l10", "player_start_rate_ewma",
    "predicted_start_probability", "reported_minutes_limit", "has_reported_minutes_limit",
    "vacated_minutes_share", "same_position_absence_weight", "team_expected_active_count",
    "team_out_count", "team_questionable_count", "team_back_to_back", "blowout_proxy",
)


class ConditionalMinutesDistributionV2:
    VERSION = "opportunity_v2_minutes_v1"
    QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

    def __init__(self, *, minimum_minutes: float = 0.5, maximum_minutes: float = 60.0,
                 random_state: int = 42) -> None:
        self._models: dict[float, HistGradientBoostingRegressor] = {}
        self.feature_names_: list[str] = []
        self.minimum_minutes = float(minimum_minutes)
        self.maximum_minutes = float(maximum_minutes)
        self._random_state = random_state
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        actual_minutes: pd.Series,
        did_play: pd.Series,
        sample_weight: np.ndarray | None = None,
        *,
        feature_columns: list[str] | None = None,
    ) -> "ConditionalMinutesDistributionV2":
        if feature_columns is None:
            feature_columns = [c for c in PREFERRED_FEATURES if c in X.columns]
        if not feature_columns:
            raise ValueError("ConditionalMinutesDistributionV2.fit: no usable feature columns")
        self.feature_names_ = list(feature_columns)
        played = pd.Series(did_play).astype("boolean").fillna(False).to_numpy()
        mins = pd.to_numeric(actual_minutes, errors="coerce").to_numpy()
        mask = played & np.isfinite(mins) & (mins > 0)  # appearances only, no DNP zeros
        if int(mask.sum()) < len(self.QUANTILES) * 5:
            raise ValueError("ConditionalMinutesDistributionV2.fit: insufficient appearance rows")
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        sw = None if sample_weight is None else np.asarray(sample_weight)[mask]
        for q in self.QUANTILES:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                                  random_state=self._random_state)
            model.fit(Xn[mask], mins[mask], sample_weight=sw)
            self._models[q] = model
        self._fitted = True
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ConditionalMinutesDistributionV2.predict before fit")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"ConditionalMinutesDistributionV2.predict: missing feature(s) {missing}")
        Xn = X[self.feature_names_].apply(pd.to_numeric, errors="coerce")
        cols = [self._models[q].predict(Xn) for q in self.QUANTILES]
        q_mat = np.column_stack(cols)
        # Monotone repair across quantiles, then clip to configured minutes range.
        q_mat = np.maximum.accumulate(q_mat, axis=1)
        q_mat = np.clip(q_mat, self.minimum_minutes, self.maximum_minutes)
        # Apply reported minutes caps if present.
        if "reported_minutes_limit" in X.columns:
            cap = pd.to_numeric(X["reported_minutes_limit"], errors="coerce").to_numpy()
            has_cap = np.isfinite(cap)
            if has_cap.any():
                q_mat[has_cap] = np.minimum(q_mat[has_cap], cap[has_cap][:, None])
        return q_mat

    def deterministic_samples(self, X: pd.DataFrame, n_samples: int = 21) -> tuple[np.ndarray, np.ndarray]:
        """Equal-probability samples from the piecewise-linear inverse CDF (weights all 1/n)."""
        if n_samples < 2:
            raise ValueError("deterministic_samples: n_samples must be >= 2")
        q_mat = self.predict_quantiles(X)
        levels = np.array(self.QUANTILES, dtype=float)
        u = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
        samples = np.empty((q_mat.shape[0], n_samples), dtype=float)
        for i in range(q_mat.shape[0]):
            samples[i] = np.interp(u, levels, q_mat[i])  # clamps outside [0.05,0.95] to endpoints
        samples = np.clip(samples, self.minimum_minutes, self.maximum_minutes)
        weights = np.full(n_samples, 1.0 / n_samples, dtype=float)
        # Sanity: sample mean must sit between predicted q05 and q95 (implied distribution bounds).
        smean = samples.mean(axis=1)
        lo, hi = q_mat[:, 0], q_mat[:, -1]
        if not np.all((smean >= lo - 1e-6) & (smean <= hi + 1e-6)):
            raise AssertionError("deterministic_samples: sample mean outside implied quantile range")
        return samples, weights
