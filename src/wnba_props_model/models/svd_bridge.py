import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


class SVDBridgeEstimator:
    """
    Surrogate model that predicts SVD embedding dimensions from leak-free
    player features. Used during OOF generation to close the train/serve
    feature gap.

    During production inference: REAL SVD embeddings are used.
    During OOF inference: BRIDGE-predicted SVD embeddings are used.
    This eliminates the structural gap without data leakage.
    """

    def __init__(self, n_svd_dims: int = 2, min_samples: int = 50):
        self.n_svd_dims = n_svd_dims
        self.min_samples = min_samples
        self.bridge_models: dict = {}
        self.feature_names_: list = []

    def _select_bridge_features(self, df: pd.DataFrame) -> list:
        candidates = [
            'player_usage_rate_season',
            'player_pct_pts_ast_season',
            'player_pct_pts_fgm_season',
            'player_pct_pts_fg3m_season',
            'player_pct_pts_ft_season',
            'player_height_inches',
            'player_position_encoded',
            'player_pts_mean_season',
            'player_reb_mean_season',
            'player_ast_mean_season',
            'player_stl_mean_season',
            'player_blk_mean_season',
            'player_fg3m_mean_season',
            'player_to_mean_season',
            'player_ast_to_ratio_season',
            'player_reb_rate_season',
            'player_fg3a_rate_season',
            'player_pts_season_zscore',
            'player_ast_season_zscore',
            'player_reb_season_zscore',
        ]
        return [c for c in candidates if c in df.columns]

    def fit(self, df: pd.DataFrame, svd_cols: list,
            game_date_col: str = 'game_date') -> 'SVDBridgeEstimator':
        bridge_features = self._select_bridge_features(df)
        self.feature_names_ = bridge_features
        X = df[bridge_features].fillna(0).values

        for svd_col in svd_cols:
            if svd_col not in df.columns:
                print(f"[SVD Bridge] {svd_col} not found, skipping")
                continue
            y = df[svd_col].values
            valid = ~np.isnan(y)
            X_valid, y_valid = X[valid], y[valid]
            if len(y_valid) < self.min_samples:
                print(f"[SVD Bridge] Only {len(y_valid)} valid rows for {svd_col}")
                continue

            if game_date_col in df.columns:
                dates = df[game_date_col].values[valid]
            else:
                dates = np.arange(len(y_valid))
            sort_idx = np.argsort(dates)
            split = int(len(sort_idx) * 0.7)
            train_idx, val_idx = sort_idx[:split], sort_idx[split:]

            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=6,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, n_iter_no_change=10, random_state=42,
            )
            model.fit(X_valid[train_idx], y_valid[train_idx])

            val_pred = model.predict(X_valid[val_idx])
            ss_res = np.sum((y_valid[val_idx] - val_pred) ** 2)
            ss_tot = np.sum((y_valid[val_idx] - np.mean(y_valid)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"[SVD Bridge] {svd_col}: R²={r2:.3f} (n_val={len(val_idx)})")

            model.fit(X_valid, y_valid)  # Refit on full data
            self.bridge_models[svd_col] = model

        return self

    def predict(self, df: pd.DataFrame, use_real_svd: bool = False) -> pd.DataFrame:
        result = df.copy()
        if use_real_svd or not self.feature_names_:
            return result
        X = result[self.feature_names_].fillna(0).values
        for svd_col, model in self.bridge_models.items():
            result[svd_col] = model.predict(X)
            result[f'{svd_col}_bridged'] = 1
        return result

    def save(self, path: str):
        joblib.dump({
            'bridge_models': self.bridge_models,
            'feature_names_': self.feature_names_,
            'n_svd_dims': self.n_svd_dims,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'SVDBridgeEstimator':
        data = joblib.load(path)
        est = cls(n_svd_dims=data['n_svd_dims'])
        est.bridge_models = data['bridge_models']
        est.feature_names_ = data['feature_names_']
        return est
