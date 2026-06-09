from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:  # optional, faster/better if installed
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

from wnba_props_model.constants import QUANTILES


def _make_quantile_regressor(alpha: float, random_state: int = 2026):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            metric="quantile",
            n_estimators=400,
            learning_rate=0.035,
            num_leaves=40,
            max_depth=7,
            feature_fraction=0.75,
            bagging_fraction=0.80,
            bagging_freq=1,
            reg_alpha=0.5,
            reg_lambda=3.0,
            min_child_samples=50,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=350,
        learning_rate=0.035,
        max_depth=3,
        random_state=random_state,
    )


@dataclass
class QuantileModelBundle:
    target: str
    features: list[str]
    quantiles: tuple[float, ...] = QUANTILES
    models: dict[float, Pipeline] | None = None

    def fit(self, df: pd.DataFrame, y_col: str, sample_weight: Iterable[float] | None = None) -> "QuantileModelBundle":
        x = df[self.features]
        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(float)
        self.models = {}
        for q in self.quantiles:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", _make_quantile_regressor(q)),
            ])
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["regressor__sample_weight"] = np.asarray(list(sample_weight))
            model.fit(x, y, **fit_kwargs)
            self.models[q] = model
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> list[dict[float, float]]:
        if not self.models:
            raise RuntimeError("QuantileModelBundle is not fit")
        preds = {}
        for q, model in self.models.items():
            preds[q] = np.asarray(model.predict(df[self.features]), dtype=float)
        out = []
        qs = sorted(preds)
        mat = np.column_stack([preds[q] for q in qs])
        mat = np.maximum.accumulate(mat, axis=1)
        for row in mat:
            out.append({q: max(0.0, float(v)) for q, v in zip(qs, row)})
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "QuantileModelBundle":
        return joblib.load(path)


def train_minutes_model(df: pd.DataFrame, features: list[str]) -> QuantileModelBundle:
    return QuantileModelBundle(target="minutes", features=features).fit(df, "minutes")


def train_rate_model(df: pd.DataFrame, stat: str, features: list[str]) -> QuantileModelBundle:
    y_col = f"{stat}_per_min"
    if y_col not in df:
        df = df.copy()
        df[y_col] = df[stat].astype(float) / np.maximum(df["minutes"].astype(float), 1.0)
    return QuantileModelBundle(target=y_col, features=features).fit(df, y_col)
