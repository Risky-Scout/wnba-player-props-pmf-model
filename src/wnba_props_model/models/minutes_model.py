"""Stage 4 minutes model.

Predicts expected playing time and uncertainty for each player-game row.

Key design:
- Mean regressor: HistGradientBoostingRegressor fitted on all rows (including DNPs).
- Quantile regressors: 5 quantile HGBRs (q10, q25, q50, q75, q90) for uncertainty.
- DNP model: LogisticRegression predicting P(did_not_play).
- Sigma estimated from IQR of quantile predictions (IQR/1.35 ≈ std for normal).
- Minimum sigma enforced to prevent overconfident projections.
- actual_minutes is NEVER included as a model feature.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


class MinutesModel:
    """Predicts (minutes_mean, minutes_sigma, p_dnp) for each player-game row."""

    VERSION = "stage4_baseline_v2"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._model: HistGradientBoostingRegressor | None = None
        self._quantile_models: dict[float, HistGradientBoostingRegressor] = {}
        self._dnp_model: Pipeline | None = None
        self._sigma_lookup: dict[tuple[str, str], float] = {}
        self._global_sigma: float = cfg.get("min_minutes_sigma", 3.0)
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata_df: pd.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> "MinutesModel":
        """Fit mean regressor, quantile regressors, DNP model, and sigma lookup.

        Args:
            X: Feature matrix (model_feature_columns, already numeric).
            y: Target = actual_minutes (all rows, including DNP zeros).
            metadata_df: Wide table rows aligned with X, for sigma stratification.
                Must contain 'projected_minutes_bucket' and 'role_uncertainty_bucket'.
                If 'did_play' column is present it is used to fit the DNP model.
            sample_weight: Optional per-sample weights (e.g. temporal decay).
        """
        seed = self.cfg.get("random_seed", 42)
        hgb_kw = self.cfg.get("hgb_regressor", {})
        clip_max = self.cfg.get("minutes_clip_max", 45.0)

        # Drop all-NaN columns (common in early-season data)
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)

        # --- Mean regressor --------------------------------------------------
        self._model = HistGradientBoostingRegressor(
            max_iter=hgb_kw.get("max_iter", 200),
            max_leaf_nodes=hgb_kw.get("max_leaf_nodes", 31),
            learning_rate=hgb_kw.get("learning_rate", 0.1),
            min_samples_leaf=hgb_kw.get("min_samples_leaf", 20),
            early_stopping=hgb_kw.get("early_stopping", False),
            n_iter_no_change=hgb_kw.get("n_iter_no_change", 10),
            tol=hgb_kw.get("tol", 1e-7),
            random_state=seed,
        )
        self._model.fit(X, y, sample_weight=sample_weight)

        # --- Quantile regressors (q10, q25, q50, q75, q90) ------------------
        self._quantile_models = {}
        for q in _QUANTILES:
            qm = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=hgb_kw.get("max_iter", 200),
                max_leaf_nodes=hgb_kw.get("max_leaf_nodes", 31),
                learning_rate=hgb_kw.get("learning_rate", 0.1),
                min_samples_leaf=hgb_kw.get("min_samples_leaf", 20),
                early_stopping=hgb_kw.get("early_stopping", False),
                n_iter_no_change=hgb_kw.get("n_iter_no_change", 10),
                tol=hgb_kw.get("tol", 1e-7),
                random_state=seed,
            )
            qm.fit(X, y, sample_weight=sample_weight)
            self._quantile_models[q] = qm

        # --- DNP logistic regression (P(did_not_play)) -----------------------
        did_play_col = metadata_df.get("did_play", None) if metadata_df is not None else None
        if did_play_col is None and "did_play" in metadata_df.columns:
            did_play_col = metadata_df["did_play"]
        if did_play_col is not None:
            dnp_y = (did_play_col.values == 0).astype(int)
            if len(np.unique(dnp_y)) >= 2:
                self._dnp_model = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(solver="saga", max_iter=1000, class_weight="balanced",
                                               random_state=seed)),
                ])
                self._dnp_model.fit(X, dnp_y)
            else:
                # All same class — skip DNP model (everyone plays or no one plays)
                self._dnp_model = None
        else:
            self._dnp_model = None

        # --- Residual-based sigma fallback (legacy) --------------------------
        y_pred = np.clip(self._model.predict(X), 0.0, clip_max)
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

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        metadata_df: pd.DataFrame,  # noqa: ARG002 — kept for API symmetry
    ) -> np.ndarray:
        """Return (n, 5) array of quantile minute predictions [q10..q90], clipped [0, 42]."""
        if not self._fitted:
            raise RuntimeError("MinutesModel not fitted")
        X_aligned = X.reindex(columns=self._usable_cols)
        clip_max = min(self.cfg.get("minutes_clip_max", 45.0), 42.0)
        _qm = getattr(self, "_quantile_models", {}) or {}
        cols = []
        for q in _QUANTILES:
            if q in _qm:
                raw = _qm[q].predict(X_aligned)
            else:
                raw = self._model.predict(X_aligned)  # fallback
            cols.append(np.clip(raw, 0.0, clip_max))
        return np.column_stack(cols)  # (n, 5)

    def predict(
        self,
        X: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (minutes_mean, minutes_sigma, p_dnp) arrays.

        minutes_mean  : predicted expected minutes, clipped to [clip_min, clip_max].
        minutes_sigma : IQR-derived std when quantile models are available, else
                        residual-std lookup from training, with minimum enforced.
        p_dnp         : P(player does not play), from LogisticRegression.
                        Returns zeros if DNP model was not fitted.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("MinutesModel not fitted")

        if hasattr(self, "_usable_cols"):
            X = X.reindex(columns=self._usable_cols)

        y_pred = np.clip(
            self._model.predict(X),
            self.cfg.get("minutes_clip_min", 0.0),
            self.cfg.get("minutes_clip_max", 45.0),
        )

        # --- Sigma from IQR of quantile predictions -------------------------
        min_sigma = self.cfg.get("min_minutes_sigma", 3.0)
        unc_mult  = self.cfg.get("uncertain_sigma_multiplier", 1.5)

        _qm = getattr(self, "_quantile_models", {}) or {}
        if _qm:
            clip_max = min(self.cfg.get("minutes_clip_max", 45.0), 42.0)
            q25_pred = np.clip(_qm[0.25].predict(X), 0.0, clip_max)
            q75_pred = np.clip(_qm[0.75].predict(X), 0.0, clip_max)
            sigmas = np.maximum((q75_pred - q25_pred) / 1.35, min_sigma)
        else:
            # Legacy fallback: role-stratified residual sigma
            min_bucket = metadata_df.get("projected_minutes_bucket",
                                         pd.Series(["unknown"] * len(X)))
            unc_bucket = metadata_df.get("role_uncertainty_bucket",
                                         pd.Series(["unknown"] * len(X)))
            sigmas = np.array([
                self._sigma_lookup.get((str(mb), str(ub)), self._global_sigma)
                for mb, ub in zip(min_bucket, unc_bucket)
            ], dtype=float)
            sigmas = np.clip(sigmas, min_sigma, None)

        # Increase sigma for uncertain/injury-dependent roles
        if "role_uncertainty_bucket" in metadata_df.columns:
            unc_bucket = metadata_df["role_uncertainty_bucket"]
            uncertain_mask = np.isin(unc_bucket.values, ["uncertain", "injury_dependent"])
            sigmas = np.where(uncertain_mask, sigmas * unc_mult, sigmas)

        # --- P(DNP) ----------------------------------------------------------
        _dnp = getattr(self, "_dnp_model", None)
        if _dnp is not None:
            p_dnp = _dnp.predict_proba(X)[:, 1].astype(float)
        else:
            p_dnp = np.zeros(len(y_pred), dtype=float)

        return y_pred, sigmas, p_dnp

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "global_sigma": self._global_sigma,
            "n_sigma_buckets": len(self._sigma_lookup),
            "sigma_by_bucket": {f"{k[0]}_{k[1]}": v for k, v in self._sigma_lookup.items()},
            "quantile_models_fitted": len(self._quantile_models),
            "dnp_model_fitted": self._dnp_model is not None,
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MinutesModel":
        return joblib.load(path)
