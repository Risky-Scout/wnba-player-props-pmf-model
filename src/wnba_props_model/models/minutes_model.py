"""Stage 4 minutes model.

Predicts expected playing time (minutes_mean) and uncertainty (minutes_sigma)
for each player-game row.

Key design:
- Trained on all rows (including DNP rows with 0 minutes).
- HistGradientBoostingRegressor handles NaN features natively.
- Sigma estimated from training residuals, grouped by role/uncertainty bucket.
- Minimum sigma enforced to prevent overconfident projections.
- actual_minutes is NEVER included as a model feature.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class MinutesModel:
    """Predicts (minutes_mean, minutes_sigma) for each player-game row."""

    VERSION = "stage4_baseline_v1"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._model: HistGradientBoostingRegressor | None = None
        self._sigma_lookup: dict[tuple[str, str], float] = {}
        self._global_sigma: float = cfg.get("min_minutes_sigma", 3.0)
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata_df: pd.DataFrame,
    ) -> "MinutesModel":
        """Fit minutes regressor and estimate role-stratified sigma.

        Args:
            X: Feature matrix (model_feature_columns, already numeric).
            y: Target = actual_minutes (all rows, including DNP zeros).
            metadata_df: Wide table rows aligned with X, for sigma stratification.
                Must contain 'projected_minutes_bucket' and 'role_uncertainty_bucket'.
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
        # sklearn's BinMapper raises "window shape cannot be larger than input
        # array shape" on all-NaN columns (common in early-season data).
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)
        self._model.fit(X, y)

        # --- Residual-based sigma estimation -----------------------------------
        y_pred = np.clip(self._model.predict(X), 0.0, self.cfg.get("minutes_clip_max", 45.0))
        residuals = y.values - y_pred
        global_std = float(np.std(residuals))
        self._global_sigma = max(global_std, self.cfg.get("min_minutes_sigma", 3.0))

        # Stratify by (projected_minutes_bucket × role_uncertainty_bucket)
        bucket_df = pd.DataFrame({
            "residual": residuals,
            "min_bucket": metadata_df["projected_minutes_bucket"].values
                          if "projected_minutes_bucket" in metadata_df.columns
                          else "unknown",
            "unc_bucket": metadata_df["role_uncertainty_bucket"].values
                          if "role_uncertainty_bucket" in metadata_df.columns
                          else "unknown",
        })
        for (mb, ub), grp in bucket_df.groupby(["min_bucket", "unc_bucket"]):
            if len(grp) >= 10:
                self._sigma_lookup[(str(mb), str(ub))] = max(
                    float(np.std(grp["residual"])),
                    self.cfg.get("min_minutes_sigma", 3.0),
                )

        self._fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (minutes_mean, minutes_sigma) arrays.

        minutes_mean: predicted expected minutes, clipped to [clip_min, clip_max].
        minutes_sigma: estimated residual std for this player's role category,
                       with minimum enforced and multiplier for uncertain roles.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("MinutesModel not fitted")

        # Align inference columns to the usable set determined at fit time
        if hasattr(self, "_usable_cols"):
            avail = [c for c in self._usable_cols if c in X.columns]
            X = X[avail]

        y_pred = np.clip(
            self._model.predict(X),
            self.cfg.get("minutes_clip_min", 0.0),
            self.cfg.get("minutes_clip_max", 45.0),
        )

        min_bucket = metadata_df.get("projected_minutes_bucket", pd.Series(["unknown"] * len(X)))
        unc_bucket = metadata_df.get("role_uncertainty_bucket", pd.Series(["unknown"] * len(X)))
        min_sigma = self.cfg.get("min_minutes_sigma", 3.0)
        unc_mult = self.cfg.get("uncertain_sigma_multiplier", 1.5)

        sigmas = np.array([
            self._sigma_lookup.get((str(mb), str(ub)), self._global_sigma)
            for mb, ub in zip(min_bucket, unc_bucket)
        ], dtype=float)
        sigmas = np.clip(sigmas, min_sigma, None)

        # Increase sigma for uncertain/injury-dependent roles
        uncertain_mask = np.isin(unc_bucket.values, ["uncertain", "injury_dependent"])
        sigmas = np.where(uncertain_mask, sigmas * unc_mult, sigmas)

        return y_pred, sigmas

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "global_sigma": self._global_sigma,
            "n_sigma_buckets": len(self._sigma_lookup),
            "sigma_by_bucket": {f"{k[0]}_{k[1]}": v for k, v in self._sigma_lookup.items()},
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MinutesModel":
        return joblib.load(path)
