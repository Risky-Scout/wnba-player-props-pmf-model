import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import joblib


class ZINBHurdleModel:
    """
    Zero-Inflated Negative Binomial model for sparse count stats (stl, blk).

    Architecture:
    1. Zero-inflation model: P(Y > 0) via logistic regression, minutes-primary.
       MUST use saga solver — lbfgs causes convergence failures on this dataset.
    2. Count model: HGBR predicting E[Y | Y > 0]
    3. NB dispersion r fitted per role bucket

    Compatible interface with HurdleModel: implements .predict(X) → (p_nz, pos_mus),
    .pos_dispersion_r, and ._pos_var for drop-in compatibility with hurdle_pmf_batch.
    """

    def __init__(self, stat: str = 'stl'):
        self.stat = stat
        self.zero_model = None
        self.count_model = None
        self.role_dispersion: dict = {}
        self.global_dispersion = 5.0
        self.zero_features_: list = []
        self.count_features_: list = []
        self._pos_mean: float = 1.0
        # HurdleModel-compatible attributes
        self.pos_dispersion_r: float = 5.0
        self._pos_var: float = 1.0

    def get_training_summary(self) -> dict:
        return {
            "stat": self.stat,
            "model_type": "ZINBHurdleModel",
            "global_dispersion": self.global_dispersion,
            "pos_dispersion_r": self.pos_dispersion_r,
            "zero_features": self.zero_features_,
            "count_features": self.count_features_,
        }

    def _zero_feature_candidates(self) -> list:
        base = [
            'minutes_mean', 'minutes_sigma', 'p_dnp',
            'player_usage_rate_season',
            f'player_{self.stat}_rate_per_minute_season',
        ]
        if self.stat == 'stl':
            base += ['opponent_turnover_rate_season']
        elif self.stat == 'blk':
            base += ['player_height_inches', 'player_position_encoded',
                     'opponent_fg3a_rate_season']
        return base

    def _count_feature_candidates(self) -> list:
        return [
            'minutes_mean',
            f'player_{self.stat}_mean_l10',
            f'player_{self.stat}_mean_season',
            f'player_{self.stat}_season_zscore',
            f'player_{self.stat}_rate_per_minute_season',
            'player_usage_rate_season',
            'home_flag', 'rest_days', 'game_pace_estimate',
        ]

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for stat in ['stl', 'blk']:
            rate_col = f'player_{stat}_rate_per_minute_season'
            mean_col = f'player_{stat}_mean_season'
            min_col = 'player_minutes_mean_season'
            if rate_col not in df.columns and mean_col in df.columns and min_col in df.columns:
                df[rate_col] = df[mean_col] / df[min_col].clip(lower=1)
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray | None = None,
            actual_minutes: np.ndarray | None = None) -> 'ZINBHurdleModel':
        df = self._add_engineered_features(X.copy())
        y_arr = y.fillna(0).values.astype(float)

        self.zero_features_ = [c for c in self._zero_feature_candidates() if c in df.columns]
        if self.zero_features_:
            X_zero = df[self.zero_features_].fillna(0).values
            y_zero = (y_arr > 0).astype(int)
            if len(np.unique(y_zero)) >= 2:
                self.zero_model = LogisticRegression(
                    C=1.0, class_weight='balanced',
                    max_iter=10000, solver='saga', tol=1e-3,
                )
                self.zero_model.fit(X_zero, y_zero,
                                    sample_weight=sample_weight)

        pos_mask = y_arr > 0
        self.count_features_ = [c for c in self._count_feature_candidates() if c in df.columns]
        if pos_mask.sum() > 50 and self.count_features_:
            X_pos = df.loc[pos_mask, self.count_features_].fillna(0).values
            _sw_pos = sample_weight[pos_mask] if sample_weight is not None else None
            self.count_model = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=5,
                min_samples_leaf=10, early_stopping=True,
                n_iter_no_change=10, random_state=42,
            )
            self.count_model.fit(X_pos, y_arr[pos_mask], sample_weight=_sw_pos)
        else:
            self._pos_mean = float(np.mean(y_arr[y_arr > 0])) if (y_arr > 0).any() else 1.0

        # Fit per-role dispersion if role_bucket is available
        if 'role_bucket' in X.columns:
            for role in X['role_bucket'].unique():
                mask = (X['role_bucket'] == role) & (y_arr > 0)
                pos_y = y_arr[mask]
                if len(pos_y) >= 20:
                    mu = np.mean(pos_y)
                    var = np.var(pos_y)
                    if var > mu > 0:
                        r = mu ** 2 / (var - mu)
                        self.role_dispersion[role] = max(r, 1.0)

        # Set global dispersion from positive outcomes
        if pos_mask.sum() > 10:
            pos_y = y_arr[pos_mask]
            mu = float(np.mean(pos_y))
            var = float(np.var(pos_y))
            if var > mu > 0:
                self.global_dispersion = max(mu ** 2 / (var - mu), 1.0)
        self.pos_dispersion_r = self.global_dispersion
        self._pos_var = float(np.var(y_arr[pos_mask])) if pos_mask.sum() > 1 else 1.0

        return self

    def predict(self, X: pd.DataFrame, role_series=None) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_nz, pos_mus) compatible with hurdle_pmf_batch interface.

        role_series is accepted for API symmetry with HurdleModel but unused."""
        df = self._add_engineered_features(X.copy())
        n = len(df)

        if self.zero_model is not None and self.zero_features_:
            X_z = df[self.zero_features_].fillna(0).values
            p_nz = self.zero_model.predict_proba(X_z)[:, 1]
        else:
            p_nz = np.full(n, 0.5)
        p_nz = np.clip(p_nz, 0.01, 0.99)

        if self.count_model is not None and self.count_features_:
            X_c = df[self.count_features_].fillna(0).values
            pos_mus = np.clip(self.count_model.predict(X_c), 0.1, None)
        else:
            pos_mus = np.full(n, self._pos_mean)

        return p_nz, pos_mus

    def predict_pmf(self, row: pd.Series, cap: int = 10) -> np.ndarray:
        row_df = self._add_engineered_features(pd.DataFrame([row]))

        if self.zero_model is not None and self.zero_features_:
            X_z = row_df[self.zero_features_].fillna(0).values
            p_nz = float(np.clip(self.zero_model.predict_proba(X_z)[0, 1], 0.01, 0.99))
        else:
            p_nz = 0.5

        if self.count_model is not None and self.count_features_:
            X_c = row_df[self.count_features_].fillna(0).values
            pos_mu = float(np.clip(self.count_model.predict(X_c)[0], 0.1, None))
        else:
            pos_mu = self._pos_mean

        role = row.get('role_bucket', 'starter') if hasattr(row, 'get') else 'starter'
        r = self.role_dispersion.get(role, self.global_dispersion)

        pmf = np.zeros(cap + 1)
        pmf[0] = 1.0 - p_nz
        nb_zero_mass = (r / (r + pos_mu)) ** r
        denom = 1.0 - nb_zero_mass
        for k in range(1, cap + 1):
            pmf[k] = p_nz * nbinom.pmf(k, r, r / (r + pos_mu)) / max(denom, 1e-9)
        return pmf / pmf.sum()

    def save(self, path: str):
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path: str) -> 'ZINBHurdleModel':
        data = joblib.load(path)
        obj = cls(stat_name=data.get('stat', 'stl'))
        obj.__dict__.update(data)
        return obj
