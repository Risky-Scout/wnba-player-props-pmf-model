from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline

from wnba_props_model.constants import DOMAIN_MAX
from wnba_props_model.models.simulation import normalize_pmf

logger = logging.getLogger(__name__)


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
            ("clf", LogisticRegression(solver="saga", max_iter=5000, class_weight=None)),
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


# ---------------------------------------------------------------------------
# Zero-Inflated Negative Binomial (ZINB) — Stage 4 replacement for HurdleModel
# ---------------------------------------------------------------------------

def _zinb_nll(r: float, y: np.ndarray, pi: float, mu: float) -> float:
    """Negative log-likelihood for NegBinom component (used for MLE of r)."""
    r = max(r, 1e-4)
    p = r / (r + mu)
    log_nb_0 = r * np.log(p)
    ll = np.sum(
        np.where(
            y == 0,
            np.log(pi + (1 - pi) * np.exp(log_nb_0) + 1e-300),
            np.log((1 - pi) + 1e-300)
            + gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(p) + y * np.log1p(-p),
        )
    )
    return -ll


class ZINBStatModel:
    """Zero-Inflated Negative Binomial model for sparse count stats (stl, blk).

    Three-stage model:
    Stage 1: LogisticRegression → π (structural zero probability).
    Stage 2: HistGBR Regressor → μ (NegBinom mean for non-structural component).
    Stage 3: Scalar MLE for dispersion r on the full data.

    Interface matches HurdleModel:  .predict(X) → (p_nz, pos_mus)  [for compatibility]
    ZINB interpretation: p_nz = 1 - π,  pos_mus = μ (the NegBinom mean)
    Additional attribute: self._r  (NegBinom dispersion parameter)
    """

    VERSION = "stage4_zinb_v1"

    def __init__(self, stat: str, cfg: dict[str, Any]) -> None:
        if stat not in {"stl", "blk"}:
            raise ValueError("ZINBStatModel is intended for stl/blk")
        self.stat = stat
        self.cfg = cfg
        self._pi_model: Pipeline | None = None
        self._mu_model: HistGradientBoostingRegressor | None = None
        self._r: float = 1.0            # NegBinom dispersion
        self._global_pi: float = 0.5
        self._global_mu: float = 0.3
        self._usable_cols: list[str] = []
        self._fitted = False

        # Compatibility attributes expected by hurdle_pmf_batch dispatch
        self.pos_dispersion_r: float = 1.0   # alias for _r
        self._pos_var: float = 1.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        actual_minutes: np.ndarray | None = None,
    ) -> "ZINBStatModel":
        seed  = self.cfg.get("random_seed", 42)
        hgb_kw = self.cfg.get("hgb_regressor", {})

        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)

        y_arr = y.fillna(0).values.astype(float)
        is_zero = (y_arr == 0).astype(int)

        # Stage 1: π — structural zero probability
        if len(np.unique(is_zero)) >= 2:
            from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
            _base_lr = LogisticRegression(solver="saga", max_iter=5000, random_state=seed)
            # CalibratedClassifierCV with isotonic regression produces well-calibrated
            # P(structural_zero) probabilities. class_weight="balanced" optimizes recall,
            # not calibration — it systematically under-predicts pi for prop-eligible players
            # where non-zero outcomes dominate, inflating p_nz and biasing the mean high by ~20%.
            _cal_lr = CalibratedClassifierCV(_base_lr, method="isotonic", cv=5)
            self._pi_model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", _cal_lr),
            ])
            self._pi_model.fit(X, is_zero)
        self._global_pi = float(is_zero.mean())

        # Stage 2: μ — NegBinom mean (fit on all rows; model predicts unconditional mean)
        self._mu_model = HistGradientBoostingRegressor(
            max_iter=hgb_kw.get("max_iter", 200),
            max_leaf_nodes=hgb_kw.get("max_leaf_nodes", 31),
            learning_rate=hgb_kw.get("learning_rate", 0.1),
            min_samples_leaf=hgb_kw.get("min_samples_leaf", 20),
            early_stopping=hgb_kw.get("early_stopping", False),
            n_iter_no_change=hgb_kw.get("n_iter_no_change", 10),
            tol=hgb_kw.get("tol", 1e-7),
            random_state=seed,
        )
        # Up-weight games where the player played meaningful minutes (prop-eligible context).
        # Garbage-time appearances (low minutes, y≈0 despite non-trivial feature vectors)
        # pull mu down and create a spurious correction in the mu-model.
        if actual_minutes is not None:
            # Up-weight prop-eligible games (minutes >= 10) in mu-model training.
            # Aligns the conditional mean estimate with the production population.
            _min_w = np.clip(actual_minutes / 25.0, 0.05, 1.0)
            _combined_w = _min_w * (sample_weight if sample_weight is not None else np.ones(len(_min_w)))
        else:
            _combined_w = sample_weight
        self._mu_model.fit(X, y_arr, sample_weight=_combined_w)
        self._global_mu = float(y_arr.mean()) or 0.3

        # Stage 3: MLE for dispersion r using global means
        mu_hat = float(np.clip(self._mu_model.predict(X), 1e-9, None).mean())
        pi_hat = self._global_pi
        result = minimize_scalar(
            _zinb_nll, args=(y_arr, pi_hat, mu_hat),
            bounds=(1e-3, 100.0), method="bounded",
        )
        self._r = float(result.x)
        self.pos_dispersion_r = self._r
        self._pos_var = float(np.var(y_arr[y_arr > 0])) if (y_arr > 0).any() else 1.0

        logger.info(
            "ZINBStatModel %s: pi=%.3f mu=%.3f r=%.3f",
            self.stat, pi_hat, mu_hat, self._r,
        )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_nz, pos_mus) compatible with hurdle_pmf_batch interface.

        p_nz = 1 - π  (probability of a non-structural zero, i.e. plays)
        pos_mus = μ   (NegBinom mean; dispersion r stored in self._r)
        """
        if not self._fitted:
            raise RuntimeError("ZINBStatModel not fitted")
        X_aligned = X.reindex(columns=self._usable_cols)
        if self._pi_model is not None:
            pi = self._pi_model.predict_proba(X_aligned)[:, 1]
        else:
            pi = np.full(len(X_aligned), self._global_pi)
        if self._mu_model is not None:
            mu = np.clip(self._mu_model.predict(X_aligned), 1e-9, None)
        else:
            mu = np.full(len(X_aligned), self._global_mu)
        p_nz = np.clip(1.0 - pi, 0.0, 1.0)
        return p_nz, mu

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "model_type": "ZINBStatModel",
            "global_pi": self._global_pi,
            "global_mu": self._global_mu,
            "dispersion_r": self._r,
            "version": self.VERSION,
        }
