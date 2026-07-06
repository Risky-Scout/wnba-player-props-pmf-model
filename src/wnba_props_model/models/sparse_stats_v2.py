"""Position-stratified sparse count model for STL and BLK.

Replaces the single-prior HurdleModel with position-aware priors and
minutes-conditional zero-inflation. Addresses BLK overprediction at all roles
by fitting separate zero-inflation and positive-count models per position group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


class PositionStratifiedSparseModel:
    """Position-stratified sparse count model for STL and BLK.

    Replaces the single-prior HurdleModel with position-aware priors and
    minutes-conditional zero-inflation.
    """

    POSITION_LEAGUE_PRIORS: dict[str, dict[str, float]] = {
        "stl": {"G": 0.75, "F": 0.70, "C": 0.55, "default": 0.70},
        "blk": {"G": 0.28, "F": 0.55, "C": 1.05, "default": 0.45},
    }

    POSITION_DISPERSION: dict[str, dict[str, float]] = {
        "stl": {"G": 4.0, "F": 3.5, "C": 3.0, "default": 3.5},
        "blk": {"G": 12.0, "F": 6.0, "C": 4.0, "default": 6.0},
    }

    def __init__(self, stat_name: str = "stl") -> None:
        self.stat_name = stat_name
        self.zero_model: LogisticRegression | None = None
        self.count_model: HistGradientBoostingRegressor | None = None
        self.zero_features_: list[str] = []
        self.count_features_: list[str] = []
        self._pos_mean: float = 0.5

    def _ensure_per_minute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for stat in ("stl", "blk"):
            rate_col = f"player_{stat}_rate_per_minute_season"
            mean_col = f"player_{stat}_mean_season"
            min_col = "player_minutes_mean_season"
            if mean_col in df.columns and min_col in df.columns and rate_col not in df.columns:
                df[rate_col] = df[mean_col] / df[min_col].clip(lower=1)
        return df

    def _build_zero_features(self, df: pd.DataFrame) -> list[str]:
        features = [
            "minutes_mean", "minutes_sigma", "p_dnp",
            f"player_{self.stat_name}_rate_per_minute_season",
            "player_position_encoded", "player_usage_rate_season",
            "player_height_inches",
        ]
        if self.stat_name == "stl":
            features.append("opponent_turnover_rate_season")
        elif self.stat_name == "blk":
            features.extend(["opponent_fg_pct_allowed_season", "opponent_fg3a_rate_season"])
        return [c for c in features if c in df.columns]

    def _build_count_features(self, df: pd.DataFrame) -> list[str]:
        features = [
            "minutes_mean",
            f"player_{self.stat_name}_mean_l10",
            f"player_{self.stat_name}_mean_season",
            f"player_{self.stat_name}_rate_per_minute_season",
            "player_position_encoded", "player_usage_rate_season",
            "home_flag", "rest_days",
        ]
        return [c for c in features if c in df.columns]

    def fit(self, df: pd.DataFrame, y_col: str | None = None) -> "PositionStratifiedSparseModel":
        if y_col is None:
            y_col = self.stat_name
        df = self._ensure_per_minute_features(df)
        y = df[y_col].values

        zero_features = self._build_zero_features(df)
        self.zero_features_ = zero_features
        if zero_features:
            X_zero = df[zero_features].fillna(0).values
            y_zero = (y > 0).astype(int)
            self.zero_model = LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=10000, solver="saga", tol=1e-3
            )
            self.zero_model.fit(X_zero, y_zero)

        pos_mask = y > 0
        count_features = self._build_count_features(df)
        self.count_features_ = count_features
        if count_features and pos_mask.sum() > 50:
            X_pos = df.loc[pos_mask, count_features].fillna(0).values
            y_pos = y[pos_mask]
            self.count_model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=4,
                min_samples_leaf=10, l2_regularization=1.0,
                early_stopping=True, n_iter_no_change=10,
                random_state=42,
            )
            self.count_model.fit(X_pos, y_pos)
        else:
            self._pos_mean = float(y[y > 0].mean()) if (y > 0).any() else 0.5

        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        df = self._ensure_per_minute_features(df)
        X_zero = (
            df[self.zero_features_].fillna(0).values
            if self.zero_features_
            else np.zeros((len(df), 1))
        )
        if self.zero_model is not None:
            p_nz = self.zero_model.predict_proba(X_zero)[:, 1]
        else:
            p_nz = np.full(len(df), 0.5)
        p_nz = np.clip(p_nz, 0.01, 0.99)

        if self.count_model is not None and self.count_features_:
            X_count = df[self.count_features_].fillna(0).values
            pos_mus = self.count_model.predict(X_count)
        else:
            pos_mus = np.full(len(df), self._pos_mean)
        pos_mus = np.clip(pos_mus, 0.1, None)

        disp_map = self.POSITION_DISPERSION.get(self.stat_name, self.POSITION_DISPERSION["blk"])
        if "player_position_encoded" in df.columns:
            pos_codes = df["player_position_encoded"].fillna("F").astype(str)
            r_arr = np.array([disp_map.get(p, disp_map["default"]) for p in pos_codes])
        else:
            r_arr = np.full(len(df), disp_map["default"])

        return {"p_nz": p_nz, "pos_mus": pos_mus, "role_rs": r_arr}

    def build_pmf(
        self,
        p_nz: float,
        pos_mu: float,
        r: float,
        cap: int = 20,
    ) -> np.ndarray:
        """Build truncated-at-zero NegBinom PMF."""
        pmf = np.zeros(cap + 1)
        pmf[0] = 1.0 - p_nz
        nb_p_zero = (r / (r + pos_mu)) ** r
        denom = 1.0 - nb_p_zero
        if denom < 1e-9:
            pmf[0] = 1.0
            return pmf
        for k in range(1, cap + 1):
            pmf[k] = p_nz * nbinom.pmf(k, r, r / (r + pos_mu)) / denom
        total = pmf.sum()
        if total > 0:
            pmf /= total
        return pmf
