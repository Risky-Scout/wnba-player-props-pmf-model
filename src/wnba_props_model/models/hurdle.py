from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline

from wnba_props_model.constants import DOMAIN_MAX
from wnba_props_model.models.simulation import normalize_pmf


def _nbinom_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """Negative-binomial PMF with Var = mu + alpha*mu^2."""
    mu = max(float(mu), 1e-9)
    alpha = max(float(alpha), 1e-6)
    r = 1.0 / alpha
    p = r / (r + mu)
    logp = gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(p) + k * np.log1p(-p)
    return np.exp(logp)


class SparseHurdleModel:
    """Zero-inflated sparse model for steals/blocks.

    Stage 1 estimates p0 = P(Y=0).
    Stage 2 estimates positive conditional mean, then uses a truncated NB tail.
    """

    def __init__(self, stat: str, features: list[str]) -> None:
        if stat not in {"stl", "blk"}:
            raise ValueError("SparseHurdleModel is intended for stl/blk")
        self.stat = stat
        self.features = features
        self.zero_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.pos_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", PoissonRegressor(alpha=0.1, max_iter=1000)),
        ])
        self.alpha_ = 0.75

    def fit(self, df: pd.DataFrame) -> "SparseHurdleModel":
        x = df[self.features]
        y = pd.to_numeric(df[self.stat], errors="coerce").fillna(0).astype(int)
        self.zero_model.fit(x, (y == 0).astype(int))
        pos = y > 0
        if pos.sum() >= 25:
            self.pos_model.fit(x.loc[pos], y.loc[pos])
            mu = y.loc[pos].mean()
            var = y.loc[pos].var()
            self.alpha_ = max(0.05, float((var - mu) / max(mu * mu, 1e-9)))
        else:
            self.pos_model.fit(x, np.maximum(y, 1))
            self.alpha_ = 1.0
        return self

    def predict_pmf(self, df: pd.DataFrame) -> list[np.ndarray]:
        p0s = self.zero_model.predict_proba(df[self.features])[:, 1]
        mus = np.maximum(self.pos_model.predict(df[self.features]), 1e-3)
        k = np.arange(DOMAIN_MAX[self.stat] + 1)
        out = []
        for p0, mu in zip(p0s, mus):
            tail = _nbinom_pmf(k, mu=mu, alpha=self.alpha_)
            tail[0] = 0.0
            tail = normalize_pmf(tail)
            pmf = (1.0 - float(p0)) * tail
            pmf[0] = float(p0)
            out.append(normalize_pmf(pmf))
        return out
