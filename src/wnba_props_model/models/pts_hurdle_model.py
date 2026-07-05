"""Minutes-conditional hurdle model for PTS with three components:
1. Structural zeros: DNP + garbage-time (< 3 min)
2. Sampling zeros: played meaningful minutes but scored 0
3. Positive count: HGBR trained on pts > 0 games only

Addresses the ~19.1pp P(0) gap observed in OOF diagnostics by explicitly
modelling the zero-inflation structure conditional on playing time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


class PtsHurdleModel:
    """Minutes-conditional hurdle model for PTS with three components:
    1. Structural zeros: DNP + garbage-time (< 3 min)
    2. Sampling zeros: played meaningful minutes but scored 0
    3. Positive count: HGBR trained on pts > 0 games only
    """

    MIN_MINUTES_FOR_COUNTING = 3.0

    def __init__(self) -> None:
        self.garbage_zero_model: Optional[LogisticRegression] = None
        self.sampling_zero_model: Optional[LogisticRegression] = None
        self.pos_model: Optional[HistGradientBoostingRegressor] = None
        self.role_dispersion: dict[str, float] = {}
        self.role_zero_rates: dict[str, dict[str, float]] = {}
        self.garbage_features_: list[str] = []
        self.sampling_features_: list[str] = []
        self.count_features_: list[str] = []
        self._fallback_pos_mean: float = 8.0

    def _compute_role_zero_rates(self, df: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, float]]:
        rates: dict[str, dict[str, float]] = {}
        dnp_mask = (
            df["actual_minutes"] == 0
            if "actual_minutes" in df.columns
            else pd.Series(False, index=df.index)
        )
        garbage_mask = (
            (df.get("actual_minutes", pd.Series(0, index=df.index)) >= 0) &
            (df.get("actual_minutes", pd.Series(0, index=df.index)) < self.MIN_MINUTES_FOR_COUNTING)
        )
        meaningful_mask = (
            df.get("actual_minutes", pd.Series(30, index=df.index)) >= self.MIN_MINUTES_FOR_COUNTING
        )

        for role in df.get("role_bucket", pd.Series(dtype=str)).unique():
            if pd.isna(role):
                continue
            role_mask = df["role_bucket"] == role
            not_dnp = role_mask & (~dnp_mask)
            if not_dnp.sum() > 20:
                garbage_not_dnp = garbage_mask & not_dnp
                p_garbage = garbage_not_dnp.sum() / not_dnp.sum()
            else:
                p_garbage = 0.05
            meaningful_role = meaningful_mask & role_mask
            if meaningful_role.sum() > 20:
                p_sampling_zero = float(((y == 0) & meaningful_role).sum() / meaningful_role.sum())
            else:
                p_sampling_zero = 0.01
            rates[role] = {
                "p_garbage": float(p_garbage),
                "p_sampling_zero": float(p_sampling_zero),
            }
        return rates

    def fit(self, df: pd.DataFrame, y_col: str = "pts") -> "PtsHurdleModel":
        y = df[y_col].values

        self.role_zero_rates = self._compute_role_zero_rates(df, y)

        dnp_mask = (df.get("actual_minutes", pd.Series(0, index=df.index)) == 0).values
        garbage_mask = (
            (df.get("actual_minutes", pd.Series(0, index=df.index)) >= 0) &
            (df.get("actual_minutes", pd.Series(0, index=df.index)) < self.MIN_MINUTES_FOR_COUNTING)
        ).values
        meaningful_mask = (
            df.get("actual_minutes", pd.Series(30, index=df.index)) >= self.MIN_MINUTES_FOR_COUNTING
        ).values

        # Garbage-time zero model
        garbage_features = [
            "p_dnp", "minutes_mean", "minutes_sigma",
            "player_usage_rate_season", "role_bucket",
        ]
        available_gf = [c for c in garbage_features if c in df.columns and c != "role_bucket"]
        if "role_bucket" in df.columns:
            X_gf_base = pd.get_dummies(
                df[available_gf + ["role_bucket"]], columns=["role_bucket"], drop_first=False
            )
        else:
            X_gf_base = df[available_gf] if available_gf else pd.DataFrame(index=df.index)
        y_garbage = garbage_mask.astype(int)
        if len(X_gf_base.columns) > 0 and y_garbage.sum() > 20:
            self.garbage_zero_model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000, solver="saga"
            )
            self.garbage_zero_model.fit(X_gf_base.fillna(0), y_garbage)
            self.garbage_features_ = X_gf_base.columns.tolist()

        # Sampling-zero model (played >= 3 min but scored 0)
        sampling_features = [
            "actual_minutes", "player_usage_rate_season",
            "player_pts_mean_l10", "player_pts_mean_season", "role_bucket",
        ]
        available_sf = [c for c in sampling_features if c in df.columns and c != "role_bucket"]
        meaningful_idx = df.index[meaningful_mask]
        if "role_bucket" in df.columns:
            X_sf_base = pd.get_dummies(
                df.loc[meaningful_idx, available_sf + ["role_bucket"]],
                columns=["role_bucket"], drop_first=False,
            )
        else:
            X_sf_base = (
                df.loc[meaningful_idx, available_sf] if available_sf
                else pd.DataFrame(index=meaningful_idx)
            )
        y_sf = (y[meaningful_mask] == 0).astype(int)
        if len(X_sf_base.columns) > 0 and meaningful_mask.sum() > 50:
            self.sampling_zero_model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000, solver="saga"
            )
            self.sampling_zero_model.fit(X_sf_base.fillna(0), y_sf)
            self.sampling_features_ = X_sf_base.columns.tolist()

        # Positive count model
        pos_mask = (y > 0) & meaningful_mask
        count_features = [
            "actual_minutes", "player_pts_mean_l10", "player_pts_mean_season",
            "player_usage_rate_season", "opponent_def_pts_allowed_ratio",
            "game_pace_estimate", "home_flag", "rest_days", "player_pts_season_zscore",
        ]
        available_cf = [c for c in count_features if c in df.columns]
        if available_cf and pos_mask.sum() > 50:
            X_pos = df.loc[pos_mask, available_cf].fillna(0).values
            y_pos = y[pos_mask]
            self.pos_model = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.05, max_depth=6,
                min_samples_leaf=15, l2_regularization=1.0,
                early_stopping=True, n_iter_no_change=10,
                random_state=42,
            )
            self.pos_model.fit(X_pos, y_pos)
            self.count_features_ = available_cf
        else:
            self._fallback_pos_mean = float(y[y > 0].mean()) if (y > 0).any() else 8.0

        # Per-role NB dispersion on positive residuals
        if "role_bucket" in df.columns:
            for role in df["role_bucket"].unique():
                if pd.isna(role):
                    continue
                mask = pos_mask & (df["role_bucket"] == role).values
                pos_y = y[mask]
                if len(pos_y) < 30:
                    continue
                mu = np.mean(pos_y)
                var = np.var(pos_y)
                if var > mu > 0:
                    r = mu ** 2 / (var - mu)
                    self.role_dispersion[role] = max(float(r), 2.0)

        return self

    def predict_p_nonzero(
        self,
        p_dnp: np.ndarray,
        role_buckets: np.ndarray,
        minutes_mean: np.ndarray,
        minutes_sigma: np.ndarray,
        df_row: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Predict P(pts > 0) for each player row."""
        p_garbage = np.zeros(len(p_dnp))
        p_sampling_zero = np.zeros(len(p_dnp))
        for role in np.unique(role_buckets):
            if pd.isna(role):
                continue
            rate = self.role_zero_rates.get(role, {})
            mask = role_buckets == role
            p_garbage[mask] = rate.get("p_garbage", 0.05)
            p_sampling_zero[mask] = rate.get("p_sampling_zero", 0.01)
        p_structural_zero = p_dnp + (1 - p_dnp) * p_garbage
        p_nonzero = (1 - p_structural_zero) * (1 - p_sampling_zero)
        return np.clip(p_nonzero, 0.01, 0.999)

    def build_pmf(
        self,
        p_nonzero: float,
        pos_mu: float,
        r: float,
        cap: int = 60,
    ) -> np.ndarray:
        """Build truncated-at-zero NegBinom PMF for PTS."""
        pmf = np.zeros(cap + 1)
        pmf[0] = 1.0 - p_nonzero
        nb_p_zero = (r / (r + pos_mu)) ** r
        denom = 1.0 - nb_p_zero
        if denom < 1e-9:
            pmf[0] = 1.0
            return pmf
        for k in range(1, cap + 1):
            pmf[k] = p_nonzero * nbinom.pmf(k, r, r / (r + pos_mu)) / denom
        total = pmf.sum()
        if total > 0:
            pmf /= total
        return pmf
