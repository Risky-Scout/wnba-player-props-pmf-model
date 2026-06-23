"""Multi-Task Stat Rate Model with Shared Representations (Enhancement 13).

Instead of training 7 independent HGB models (pts, reb, ast, fg3m, stl, blk,
turnover), this module trains a multi-output chained model where each stat's
prediction is augmented with the predictions of all higher-variance stats.

Architecture
------------
1. Shared chain: stats sorted by variance (highest → lowest).  For stat i,
   the feature matrix is augmented with predictions from stats 0…i-1.  This
   enables cross-stat knowledge transfer: the blk model sees the reb
   prediction as an extra feature.

2. Private residual: each stat also has its own private HGB trained on the
   residual (actual − shared_pred).  This captures idiosyncratic patterns
   not explained by the shared component.

3. Final prediction = shared + private.

4. Residual correlation matrix: computed from training residuals.  Used to
   replace hand-coded copula correlations with empirically learned ones.

References
----------
He & Choi (2025). Stacked ensemble model for NBA game outcome prediction.
Scientific Reports.
Terner & Franks (2020). Modeling Player and Team Performance in Basketball.
Annual Review of Statistics and Its Application.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


class MultiTaskStatRateModel(BaseEstimator, RegressorMixin):
    """Multi-task stat rate prediction with shared + private HGB components.

    Trains a chain of shared HGB models (ordered by stat variance) augmented
    with each other's predictions, plus private residual models per stat.

    Parameters
    ----------
    shared_depth : int   max depth for shared models (default 5)
    shared_iter  : int   boosting iterations for shared models (default 200)
    private_depth: int   max depth for private residual models (default 3)
    private_iter : int   boosting iterations for private models (default 100)
    stats        : list  target stat names (default STATS)
    """

    def __init__(
        self,
        shared_depth: int = 5,
        shared_iter:  int = 200,
        private_depth: int = 3,
        private_iter:  int = 100,
        stats: list[str] | None = None,
    ):
        self.shared_depth  = shared_depth
        self.shared_iter   = shared_iter
        self.private_depth = private_depth
        self.private_iter  = private_iter
        self.stats = stats or STATS

        self.shared_models:       dict[str, tuple[HistGradientBoostingRegressor, int]] = {}
        self.private_models:      dict[str, HistGradientBoostingRegressor] = {}
        self.residual_correlation: np.ndarray | None = None
        self._ordered_stats:      list[str] = []
        self._n_base_features:    int = 0
        self._is_fitted = False

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, Y_dict: dict[str, np.ndarray]) -> "MultiTaskStatRateModel":
        """Fit shared chain + private residual models.

        Parameters
        ----------
        X      : (n_samples, n_features) feature matrix
        Y_dict : {stat_name: target_vector (n_samples,)}
        """
        self._n_base_features = X.shape[1]

        # Order stats by variance (highest → lowest) to maximise information
        # flow through the shared chain
        available = [s for s in self.stats if s in Y_dict]
        self._ordered_stats = sorted(
            available,
            key=lambda s: float(np.var(Y_dict[s])),
            reverse=True,
        )

        # ── Step 1: Shared chain ─────────────────────────────────────────────
        shared_preds_train: dict[str, np.ndarray] = {}
        for i, stat in enumerate(self._ordered_stats):
            X_aug = self._augment_X(X, shared_preds_train, i)
            model = HistGradientBoostingRegressor(
                max_iter=self.shared_iter,
                max_depth=self.shared_depth,
                random_state=42,
                min_samples_leaf=10,
                l2_regularization=0.1,
            )
            try:
                model.fit(X_aug, Y_dict[stat])
                shared_preds_train[stat] = model.predict(X_aug)
            except Exception as e:
                logger.warning("E13: shared model for %s failed: %s", stat, e)
                shared_preds_train[stat] = np.full(len(Y_dict[stat]), np.mean(Y_dict[stat]))
                model = None
            self.shared_models[stat] = (model, i)

        # ── Step 2: Private residual models ──────────────────────────────────
        residuals: dict[str, np.ndarray] = {}
        for stat in self._ordered_stats:
            res = Y_dict[stat] - shared_preds_train[stat]
            residuals[stat] = res
            priv = HistGradientBoostingRegressor(
                max_iter=self.private_iter,
                max_depth=self.private_depth,
                random_state=42,
                min_samples_leaf=10,
            )
            try:
                priv.fit(X, res)
            except Exception as e:
                logger.warning("E13: private model for %s failed: %s", stat, e)
            self.private_models[stat] = priv

        # ── Step 3: Residual correlation matrix ──────────────────────────────
        if len(self._ordered_stats) > 1:
            res_matrix = np.column_stack([residuals[s] for s in self._ordered_stats])
            with np.errstate(invalid="ignore"):
                self.residual_correlation = np.nan_to_num(np.corrcoef(res_matrix.T))

        self._is_fitted = True
        logger.info(
            "E13 MultiTaskStatRateModel: fitted %d stats, residual corr shape=%s",
            len(self._ordered_stats),
            self.residual_correlation.shape if self.residual_correlation is not None else "N/A",
        )
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(
        self, X: np.ndarray, stat: str | None = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict per-minute rate for one stat or all stats."""
        if not self._is_fitted:
            raise RuntimeError("MultiTaskStatRateModel not fitted")
        if stat is not None:
            return self._predict_single(X, stat)
        return {s: self._predict_single(X, s) for s in self._ordered_stats}

    def _predict_single(self, X: np.ndarray, stat: str) -> np.ndarray:
        if stat not in self.shared_models:
            raise ValueError(f"Stat '{stat}' not in trained model; available: {self._ordered_stats}")

        model, order = self.shared_models[stat]
        # Re-build predictions from lower-order stats
        prior_preds: dict[str, np.ndarray] = {}
        for s, (m, o) in sorted(self.shared_models.items(), key=lambda x: x[1][1]):
            if o < order and m is not None:
                X_aug_prior = self._augment_X(X, prior_preds, o)
                prior_preds[s] = m.predict(X_aug_prior)

        X_aug = self._augment_X(X, prior_preds, order)
        shared_pred = model.predict(X_aug) if model is not None else np.zeros(len(X))

        if stat in self.private_models:
            priv_pred = self.private_models[stat].predict(X)
            return shared_pred + priv_pred
        return shared_pred

    # ── Utilities ────────────────────────────────────────────────────────────

    def _augment_X(
        self,
        X: np.ndarray,
        prior_preds: dict[str, np.ndarray],
        order: int,
    ) -> np.ndarray:
        """Concatenate base features with predictions of already-fitted stats."""
        if order == 0 or not prior_preds:
            return X
        aug = np.column_stack([prior_preds[s] for s in self._ordered_stats
                               if s in prior_preds])
        return np.column_stack([X, aug])

    def get_residual_correlation(self) -> dict[str, Any]:
        """Return the learned residual correlation matrix as a dict."""
        if self.residual_correlation is None:
            return {}
        idx = self._ordered_stats
        return {
            (idx[i], idx[j]): float(self.residual_correlation[i, j])
            for i in range(len(idx))
            for j in range(len(idx))
        }

    def get_correlation_matrix(self) -> np.ndarray | None:
        """Alias for copula_correlation_matrix (backward-compat)."""
        return self.copula_correlation_matrix()

    def copula_correlation_matrix(self) -> np.ndarray | None:
        """Return correlation matrix formatted for copula input (ordered by STATS)."""
        if self.residual_correlation is None:
            return None
        n = len(STATS)
        mat = np.eye(n)
        for i, si in enumerate(STATS):
            for j, sj in enumerate(STATS):
                if si in self._ordered_stats and sj in self._ordered_stats:
                    ri = self._ordered_stats.index(si)
                    rj = self._ordered_stats.index(sj)
                    mat[i, j] = self.residual_correlation[ri, rj]
        return mat
