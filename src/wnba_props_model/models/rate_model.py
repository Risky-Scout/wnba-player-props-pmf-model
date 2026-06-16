"""Stage 4 stat rate / count models.

Two model types:
1. StatRateModel   – standard regressor for non-sparse stats (pts, reb, ast, fg3m, turnover).
2. HurdleModel     – two-stage model for sparse stats (stl, blk):
   - Stage A: HistGradientBoostingClassifier → P(Y > 0)
   - Stage B: HistGradientBoostingRegressor  → E[Y | Y > 0] on positive rows only

Neither model uses actual_outcome or actual_minutes as a feature.
Neither model uses any market/forbidden columns.

Dispersion (NegBinom r parameter) is estimated from training data residuals and
stored for PMF generation.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from wnba_props_model.models.pmf_utils import dispersion_from_moments


class StatRateModel:
    """Predicts expected count E[Y] for a single stat.

    Training: fit on did_play=True rows to avoid conflating DNP zeros with
    genuine zero-count played games.
    """

    VERSION = "stage4_baseline_v1"

    def __init__(self, stat: str, cfg: dict[str, Any]) -> None:
        self.stat = stat
        self.cfg = cfg
        self._model: HistGradientBoostingRegressor | None = None
        self._dispersion_r: float | None = None  # NegBinom r; None means Poisson
        self._global_mean: float = 0.0
        self._global_var: float = 0.0
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatRateModel":
        """Fit regressor on did_play rows.

        Args:
            X: Feature matrix (model_feature_columns, numeric, NaN allowed).
            y: actual_{stat} values for did_play=True rows.
        """
        seed = self.cfg.get("random_seed", 42)
        hgb_kw = self.cfg.get("hgb_regressor", {})
        self._model = HistGradientBoostingRegressor(
            max_iter=hgb_kw.get("max_iter", 200),
            max_leaf_nodes=hgb_kw.get("max_leaf_nodes", 31),
            learning_rate=hgb_kw.get("learning_rate", 0.1),
            min_samples_leaf=hgb_kw.get("min_samples_leaf", 20),
            random_state=seed,
        )
        # Drop all-NaN columns to prevent sklearn BinMapper crash on early-season data
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)
        self._model.fit(X, y)

        # Estimate NegBinom dispersion from empirical moments
        self._global_mean = float(y.mean())
        self._global_var = float(y.var())
        self._dispersion_r = dispersion_from_moments(self._global_mean, self._global_var)
        self._fitted = True
        return self

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Predict E[Y], clipped to >= min_stat_mean."""
        if not self._fitted or self._model is None:
            raise RuntimeError(f"StatRateModel({self.stat}) not fitted")
        if hasattr(self, "_usable_cols"):
            X = X.reindex(columns=self._usable_cols)
        min_mean = self.cfg.get("min_stat_mean", 0.01)
        return np.clip(self._model.predict(X), min_mean, None)

    @property
    def dispersion_r(self) -> float | None:
        return self._dispersion_r

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "version": self.VERSION,
            "global_mean": self._global_mean,
            "global_var": self._global_var,
            "dispersion_r": self._dispersion_r,
            "pmf_type": "negbinom" if self._dispersion_r is not None else "poisson",
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "StatRateModel":
        return joblib.load(path)


class HurdleModel:
    """Hurdle model for sparse stats (stl, blk).

    Stage A: P(Y > 0) via binary classifier.
    Stage B: E[Y | Y > 0] via regressor fitted only on positive rows.

    PMF generation:
        p0 = 1 - P(Y > 0)
        positive tail from NegBinom(pos_mu, pos_r) scaled to P(Y > 0)
    """

    VERSION = "stage4_baseline_v1"

    def __init__(self, stat: str, cfg: dict[str, Any]) -> None:
        self.stat = stat
        self.cfg = cfg
        self._clf: HistGradientBoostingClassifier | None = None
        self._reg: HistGradientBoostingRegressor | None = None
        self._pos_dispersion_r: float | None = None
        self._pos_mean: float = 0.0
        self._pos_var: float = 0.0
        self._n_pos: int = 0
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HurdleModel":
        """Fit hurdle model on did_play rows.

        Args:
            X: Feature matrix (model_feature_columns, numeric, NaN allowed).
            y: actual_{stat} values for did_play=True rows.
        """
        seed = self.cfg.get("random_seed", 42)
        clf_kw = self.cfg.get("hgb_classifier", {})
        reg_kw = self.cfg.get("hgb_regressor", {})

        # Drop all-NaN columns to prevent sklearn BinMapper crash on early-season data
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)

        # Stage A: binary classifier P(Y > 0)
        y_binary = (y > 0).astype(int)
        self._clf = HistGradientBoostingClassifier(
            max_iter=clf_kw.get("max_iter", 200),
            max_leaf_nodes=clf_kw.get("max_leaf_nodes", 31),
            learning_rate=clf_kw.get("learning_rate", 0.1),
            min_samples_leaf=clf_kw.get("min_samples_leaf", 20),
            random_state=seed,
        )
        self._clf.fit(X, y_binary)

        # Stage B: regressor E[Y | Y > 0] on positive rows
        pos_mask = y > 0
        self._n_pos = int(pos_mask.sum())
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]

        if self._n_pos >= 10:
            self._reg = HistGradientBoostingRegressor(
                max_iter=reg_kw.get("max_iter", 200),
                max_leaf_nodes=reg_kw.get("max_leaf_nodes", 31),
                learning_rate=reg_kw.get("learning_rate", 0.1),
                min_samples_leaf=reg_kw.get("min_samples_leaf", 20),
                random_state=seed,
            )
            self._reg.fit(X_pos, y_pos)
            self._pos_mean = float(y_pos.mean())
            self._pos_var = float(y_pos.var())
            self._pos_dispersion_r = dispersion_from_moments(self._pos_mean, self._pos_var)
        else:
            # Fall back to global positive-count mean if too few positive samples
            self._pos_mean = float(y_pos.mean()) if self._n_pos > 0 else 1.0
            self._pos_var = float(y_pos.var()) if self._n_pos > 1 else 0.5
            self._pos_dispersion_r = dispersion_from_moments(self._pos_mean, self._pos_var)

        self._fitted = True
        return self

    def predict(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_nonzero, pos_mu) arrays.

        p_nonzero: P(Y > 0)
        pos_mu:    E[Y | Y > 0]
        """
        if not self._fitted or self._clf is None:
            raise RuntimeError(f"HurdleModel({self.stat}) not fitted")

        # Align inference to the exact column set used at fit time.
        # Missing columns are filled with NaN — HGB handles NaN natively.
        if hasattr(self, "_usable_cols"):
            X = X.reindex(columns=self._usable_cols)

        # P(Y > 0)
        p_nz = self._clf.predict_proba(X)[:, 1]
        p_nz = np.clip(p_nz, 0.0, 1.0)

        # E[Y | Y > 0]
        min_mean = self.cfg.get("min_stat_mean", 0.01)
        if self._reg is not None:
            pos_mu = np.clip(self._reg.predict(X), min_mean, None)
        else:
            pos_mu = np.full(len(X), max(self._pos_mean, min_mean))

        return p_nz, pos_mu

    @property
    def pos_dispersion_r(self) -> float | None:
        return self._pos_dispersion_r

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "version": self.VERSION,
            "n_positive_rows": self._n_pos,
            "pos_mean": self._pos_mean,
            "pos_var": self._pos_var,
            "pos_dispersion_r": self._pos_dispersion_r,
            "has_reg": self._reg is not None,
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "HurdleModel":
        return joblib.load(path)
