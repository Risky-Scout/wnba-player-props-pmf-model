"""Multi-Task Stat Rate Model with Shared Representations (Enhancement 13).

Instead of 7 independent HGB models, trains a multi-output model with:
    - Shared HGB: chained multi-output feature transformation (stats sorted
      by variance, each subsequent model appended predictions of prior ones)
    - Private residual HGB per stat: captures stat-specific signal missed
      by the shared layer
    - Final prediction = shared + private
    - Byproduct: residual_correlation matrix for use as a data-learned
      replacement for the hand-coded Gaussian copula correlations

Reference:
    He & Choi (2025). Stacked ensemble model for NBA game outcome prediction.
    Scientific Reports. https://www.nature.com/articles/s41598-025-13657-1
    Terner & Franks (2020). Modeling Player and Team Performance in Basketball.
    Annual Review of Statistics and Its Application.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


class MultiTaskStatRateModel(BaseEstimator, RegressorMixin):
    """Multi-task stat rate prediction with shared and private components.

    Architecture
    ------------
    1. Shared chained HGB: for each stat (ordered by variance, descending),
       train a HGB model where the features are augmented with predictions
       from all already-fitted shared models.  This enables cross-stat
       knowledge transfer: the rebound model sees predicted points, the
       block model sees predicted rebounds and points, etc.

    2. Private residual HGB: trained on the residual Y - shared_pred for
       each stat.  Captures stat-specific patterns not explained by the
       shared layer.

    3. Final prediction: shared_pred + private_pred

    4. Residual correlation: computed from training residuals.  Stored as
       ``self.residual_correlation`` (7×7 numpy array, same order as STATS).
       This replaces hand-coded copula correlations with data-learned ones.
    """

    def __init__(
        self,
        shared_depth: int = 5,
        shared_iter: int = 200,
        private_depth: int = 3,
        private_iter: int = 100,
        stats: list[str] | None = None,
    ):
        self.shared_depth = shared_depth
        self.shared_iter = shared_iter
        self.private_depth = private_depth
        self.private_iter = private_iter
        self.stats = stats or STATS

        # (model, chain_order) tuples keyed by stat name
        self.shared_models: dict[str, tuple[HistGradientBoostingRegressor, int]] = {}
        self.private_models: dict[str, HistGradientBoostingRegressor] = {}
        self.residual_correlation: Optional[np.ndarray] = None
        self._stats_by_var: list[str] = []

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, Y_dict: dict[str, np.ndarray]) -> "MultiTaskStatRateModel":
        """Fit shared + private models.

        Parameters
        ----------
        X      : feature matrix (n_samples, n_features)
        Y_dict : {stat_name: target_vector} for each stat
        """
        available_stats = [s for s in self.stats if s in Y_dict]
        if not available_stats:
            raise ValueError("No matching stats in Y_dict.")

        # Sort by variance (highest first) — determines chain order
        self._stats_by_var = sorted(
            available_stats, key=lambda s: np.nanvar(Y_dict[s]), reverse=True
        )

        shared_preds_train: dict[str, np.ndarray] = {}

        for i, stat in enumerate(self._stats_by_var):
            X_aug = self._augment(X, shared_preds_train, self._stats_by_var[:i])

            model = HistGradientBoostingRegressor(
                max_iter=self.shared_iter,
                max_depth=self.shared_depth,
                random_state=42,
            )
            model.fit(X_aug, Y_dict[stat])
            shared_preds_train[stat] = model.predict(X_aug)
            self.shared_models[stat] = (model, i)
            logger.debug("Shared model fitted for stat=%s (chain_order=%d)", stat, i)

        # Private residual models
        residuals: dict[str, np.ndarray] = {}
        for stat in self._stats_by_var:
            res = Y_dict[stat] - shared_preds_train[stat]
            residuals[stat] = res
            priv = HistGradientBoostingRegressor(
                max_iter=self.private_iter,
                max_depth=self.private_depth,
                random_state=42,
            )
            priv.fit(X, res)
            self.private_models[stat] = priv

        # Residual correlation matrix (learned copula replacement)
        stat_order = [s for s in STATS if s in residuals]
        res_matrix = np.column_stack([residuals[s] for s in stat_order])
        self.residual_correlation = np.corrcoef(res_matrix.T)
        logger.info(
            "MultiTaskStatRateModel fitted for %d stats. "
            "Residual correlation range: [%.3f, %.3f]",
            len(available_stats),
            float(np.nanmin(np.tril(self.residual_correlation, -1))),
            float(np.nanmax(np.tril(self.residual_correlation, -1))),
        )
        return self

    # ── Predict ──────────────────────────────────────────────────────────

    def predict(
        self, X: np.ndarray, stat: str | None = None
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Predict per-minute rate.

        Parameters
        ----------
        X    : feature matrix (n_samples, n_features)
        stat : if given, returns only that stat's predictions (1D array);
               otherwise returns a dict {stat: predictions}.
        """
        if stat is not None:
            return self._predict_single(X, stat)
        return {s: self._predict_single(X, s) for s in self._stats_by_var}

    def _predict_single(self, X: np.ndarray, stat: str) -> np.ndarray:
        if stat not in self.shared_models:
            raise ValueError(f"Stat '{stat}' not in trained models. "
                             f"Available: {list(self.shared_models)}")
        model, order = self.shared_models[stat]

        # Reproduce the same augmentation as during training
        prior_preds: dict[str, np.ndarray] = {}
        for s, (_, o) in sorted(self.shared_models.items(), key=lambda x: x[1][1]):
            if o < order:
                prior_preds[s] = self._predict_shared_raw(X, s, prior_preds)

        X_aug = self._augment(X, prior_preds, list(prior_preds.keys()))
        shared_pred = model.predict(X_aug)

        if stat in self.private_models:
            return shared_pred + self.private_models[stat].predict(X)
        return shared_pred

    def _predict_shared_raw(
        self, X: np.ndarray, stat: str, prior: dict[str, np.ndarray]
    ) -> np.ndarray:
        model, order = self.shared_models[stat]
        prior_stats = [s for s, (_, o) in sorted(self.shared_models.items(), key=lambda x: x[1][1]) if o < order]
        X_aug = self._augment(X, prior, prior_stats)
        return model.predict(X_aug)

    @staticmethod
    def _augment(
        X: np.ndarray, preds: dict[str, np.ndarray], ordered_keys: list[str]
    ) -> np.ndarray:
        """Append already-predicted shared outputs to the feature matrix."""
        if not ordered_keys:
            return X
        aug_cols = [preds[s] for s in ordered_keys if s in preds]
        if not aug_cols:
            return X
        aug = np.column_stack(aug_cols)
        return np.column_stack([X, aug])

    # ── Utilities ────────────────────────────────────────────────────────

    def get_correlation_matrix(self) -> np.ndarray | None:
        """Return the 7×7 residual correlation matrix (STATS order)."""
        return self.residual_correlation

    def get_correlation_for_pair(self, stat_a: str, stat_b: str) -> float:
        """Return the learned correlation between two stats."""
        if self.residual_correlation is None:
            return 0.0
        stat_order = [s for s in STATS if s in self.shared_models]
        if stat_a not in stat_order or stat_b not in stat_order:
            return 0.0
        i, j = stat_order.index(stat_a), stat_order.index(stat_b)
        return float(self.residual_correlation[i, j])
