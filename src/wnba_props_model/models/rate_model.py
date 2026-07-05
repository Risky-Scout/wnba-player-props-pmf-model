"""Stage 4 stat rate / count models.

Two model types:
1. StatRateModel   – standard regressor for non-sparse stats (pts, reb, ast, fg3m, turnover).
2. HurdleModel     – two-stage model for sparse stats (stl, blk):
   - Stage A: HistGradientBoostingClassifier → P(Y > 0)
   - Stage B: HistGradientBoostingRegressor  → E[Y | Y > 0] on positive rows only

Neither model uses actual_outcome or actual_minutes as a feature.
Neither model uses any market/forbidden columns.

Dispersion (NegBinom r parameter) is estimated from training data residuals and
stored for PMF generation.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from wnba_props_model.models.pmf_utils import dispersion_from_moments


class StatRateModel:
    """Predicts expected count E[Y] for a single stat.

    Training: fit on did_play=True rows to avoid conflating DNP zeros with
    genuine zero-count played games.
    """

    VERSION = "stage4_baseline_v1"

    _ROLE_R_FLOORS: dict[str, dict[str, float]] = {
        "pts":      {"bench": 2.0, "fringe": 1.5, "rotation": 1.0, "core": 0.8, "starter": 0.5},
        "reb":      {"bench": 2.5, "fringe": 2.0, "rotation": 1.5, "core": 1.0, "starter": 0.8},
        "ast":      {"bench": 2.0, "fringe": 1.5, "rotation": 1.0, "core": 0.8, "starter": 0.5},
        "fg3m":     {"bench": 3.0, "fringe": 2.5, "rotation": 2.0, "core": 1.5, "starter": 1.0},
        "stl":      {"bench": 4.0, "fringe": 3.0, "rotation": 2.5, "core": 2.0, "starter": 1.5},
        "blk":      {"bench": 3.0, "fringe": 2.5, "rotation": 2.0, "core": 1.5, "starter": 1.0},
        "turnover": {"bench": 2.5, "fringe": 2.0, "rotation": 1.5, "core": 1.0, "starter": 0.8},
    }

    def __init__(self, stat: str, cfg: dict[str, Any]) -> None:
        self.stat = stat
        self.cfg = cfg
        self._model: HistGradientBoostingRegressor | None = None
        self._dispersion_r: float | None = None  # NegBinom r; None means Poisson
        self._global_mean: float = 0.0
        self._global_var: float = 0.0
        self._fitted = False
        # Part 3: feature-based learned dispersion model (predicts log(r) from features)
        # Trained after main HGB; replaces role-lookup at inference when available.
        self.dispersion_model: HistGradientBoostingRegressor | None = None

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling: add new fields if missing."""
        self.__dict__.update(state)
        if "dispersion_model" not in self.__dict__:
            self.dispersion_model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        context_df: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "StatRateModel":
        """Fit regressor on did_play rows.

        Args:
            X: Feature matrix (model_feature_columns, numeric, NaN allowed).
            y: actual_{stat} values for did_play=True rows.
            context_df: Original wide-table rows (same index as X/y) used for
                computing per-role dispersion. Needs a ``role_bucket`` column.
                When ``use_minutes_offset`` is set in cfg, must contain
                ``actual_minutes``.
            sample_weight: Optional per-sample weights (e.g. temporal decay).
        """
        use_offset = self.cfg.get("use_minutes_offset", False)
        seed = self.cfg.get("random_seed", 42)
        hgb_kw = self.cfg.get("hgb_regressor", {})
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
        # Drop all-NaN columns to prevent sklearn BinMapper crash on early-season data
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)

        # Minutes-offset mode: fit on per-minute rate; scale at prediction time
        if use_offset and context_df is not None and "actual_minutes" in context_df.columns:
            minutes = context_df["actual_minutes"].clip(lower=1.0).reset_index(drop=True)
            y_aligned = y.reset_index(drop=True)
            y_rate = y_aligned / minutes
            self._use_minutes_offset = True
            self._minutes_offset_col = "player_minutes_mean_l5"  # feature col used at predict
            self._model.fit(X, y_rate, sample_weight=sample_weight)
            # Dispersion from rate-scale residuals back-transformed to count scale
            # (use original counts for dispersion estimation, not rate)
            self._global_mean = float(y_aligned.mean())
            self._global_var = float(y_aligned.var())
        else:
            self._use_minutes_offset = False
            self._model.fit(X, y, sample_weight=sample_weight)
            # Global NegBinom dispersion from empirical moments
            self._global_mean = float(y.mean())
            self._global_var = float(y.var())

        self._dispersion_r = dispersion_from_moments(self._global_mean, self._global_var)

        # Per-role dispersion: stratify by role_bucket for fatter tails on stars,
        # narrower tails on bench players (fixes PIT KS underdispersion for pts).
        self._role_dispersion: dict[str, float | None] = {}
        if context_df is not None and "role_bucket" in context_df.columns:
            ctx = context_df.reset_index(drop=True)
            y_aligned = y.reset_index(drop=True)
            for role, grp in ctx.groupby("role_bucket"):
                y_role = y_aligned.loc[grp.index]
                if len(y_role) >= 20:
                    self._role_dispersion[str(role)] = dispersion_from_moments(
                        float(y_role.mean()), float(y_role.var())
                    )

        _floors = self._ROLE_R_FLOORS.get(self.stat, {})
        for _role_key in list(self._role_dispersion.keys()):
            _floor_val = _floors.get(str(_role_key), 1.0)
            if self._role_dispersion[_role_key] is not None:
                self._role_dispersion[_role_key] = max(self._role_dispersion[_role_key], _floor_val)

        # P3.1: Mean-dependent dispersion r(mu) via log-linear fit
        # r_approx(i) = mu_hat(i)^2 / max(y(i) - mu_hat(i), 0.01)
        # log(r_approx) ~ beta0 + beta1 * log(mu_hat)  →  r(mu) = exp(beta0 + beta1*log(mu))
        self._dispersion_slope: float | None = None
        self._dispersion_intercept: float | None = None
        if self.cfg.get("use_mean_dependent_dispersion", False) and self._model is not None:
            try:
                X_pred_fit = X.reindex(columns=self._usable_cols)
                mu_hat = np.clip(self._model.predict(X_pred_fit), 0.01, None)
                y_vals = y.reset_index(drop=True).values.astype(float)
                r_approx = mu_hat ** 2 / np.maximum(y_vals - mu_hat, 0.01)
                r_approx = np.clip(r_approx, 0.1, 100.0)
                log_mu = np.log(np.clip(mu_hat, 0.01, None))
                log_r  = np.log(r_approx)
                beta1, beta0 = np.polyfit(log_mu, log_r, deg=1)
                self._dispersion_slope = float(beta1)
                self._dispersion_intercept = float(beta0)
            except Exception:
                pass  # fall back to global dispersion on any error

        # Part 3: Feature-based dispersion model — predicts log(r) from features.
        # Trained only when mean-dependent dispersion is not already providing a
        # per-instance r.  Gives player-specific, context-specific dispersion that
        # adapts to opponent defense consistency, back-to-back fatigue, etc.
        if self._model is not None and len(X) >= 200:
            try:
                X_pred_disp = X.reindex(columns=self._usable_cols)
                mu_hat = np.clip(self._model.predict(X_pred_disp), 0.01, None)
                y_vals = (y.reset_index(drop=True) if use_offset and context_df is not None
                          and "actual_minutes" in context_df.columns
                          else y.reset_index(drop=True)).values.astype(float)
                r_approx = mu_hat ** 2 / np.maximum(np.abs(y_vals - mu_hat), 0.01)
                r_approx = np.clip(r_approx, 0.3, 15.0)
                log_r = np.log(r_approx)
                disp_mdl = HistGradientBoostingRegressor(
                    max_iter=50, max_leaf_nodes=15,
                    learning_rate=0.05, min_samples_leaf=20,
                    random_state=self.cfg.get("random_seed", 42),
                )
                disp_mdl.fit(X_pred_disp, log_r)
                self.dispersion_model = disp_mdl
            except Exception:
                self.dispersion_model = None

        # P3.3: Bayesian shrinkage prior (Gamma-Poisson empirical Bayes)
        # Correctly uses compute_league_priors_from_data(context_df) → {"pts": mean, ...}
        # then derives alpha, beta from inter-player rate variance in training data.
        self._league_prior_alpha: float | None = None
        self._league_prior_beta: float | None = None
        if self.cfg.get("use_model_ensemble", False) and context_df is not None:
            try:
                from wnba_props_model.models.shrinkage import compute_league_priors_from_data  # noqa: PLC0415
                # 1-arg call: returns {"pts": league_mean_pts, "reb": league_mean_reb, ...}
                priors_dict = compute_league_priors_from_data(context_df)
                league_mu = float(priors_dict.get(self.stat, self._global_mean))
                # Compute inter-player rate variance for Gamma hyperparameters
                actual_col = f"actual_{self.stat}"
                if actual_col in context_df.columns and "player_id" in context_df.columns:
                    per_player_mean = (
                        context_df.dropna(subset=[actual_col])
                        .groupby("player_id")[actual_col]
                        .mean()
                    )
                    inter_var = float(per_player_mean.var()) if len(per_player_mean) >= 5 else None
                else:
                    inter_var = None
                if inter_var is not None and inter_var > 1e-6 and league_mu > 1e-6:
                    self._league_prior_alpha = league_mu ** 2 / inter_var
                    self._league_prior_beta  = league_mu / inter_var
            except Exception:
                pass

        # P5.1: Store feature importances from HGB inner model
        self._feature_importances: dict[str, float] = {}
        if self._model is not None and hasattr(self._model, "feature_importances_"):
            self._feature_importances = dict(
                zip(self._usable_cols, self._model.feature_importances_.tolist())
            )

        self._fitted = True
        return self

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Predict E[Y], clipped to >= min_stat_mean.

        When ``use_minutes_offset`` was active at fit time, the model predicts
        the per-minute rate.  We multiply by the projected minutes feature
        (``player_minutes_mean_l5``) to recover the expected count.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError(f"StatRateModel({self.stat}) not fitted")
        X_pred = X.reindex(columns=getattr(self, "_usable_cols", X.columns))
        min_mean = self.cfg.get("min_stat_mean", 0.01)
        raw = self._model.predict(X_pred)

        if getattr(self, "_use_minutes_offset", False):
            minutes_col = getattr(self, "_minutes_offset_col", "player_minutes_mean_l5")
            if minutes_col in X.columns:
                projected_min = X[minutes_col].values.astype(float)
                projected_min = np.clip(projected_min, 1.0, None)
            else:
                # Fallback: use column-agnostic average if feature not available
                projected_min = np.full(len(raw), max(self._global_mean / 0.5, 20.0))
            return np.clip(raw * projected_min, min_mean, None)

        return np.clip(raw, min_mean, None)

    @property
    def dispersion_r(self) -> float | None:
        return self._dispersion_r

    def get_dispersion(self, role: str, mu: float | None = None) -> float | None:
        """Return dispersion r for NegBinom PMF generation.

        When ``mu`` is provided and mean-dependent dispersion is fitted,
        returns r(mu) = exp(intercept + slope * log(mu)) clamped to [0.5, 50].
        Otherwise falls back to per-role or global r.
        """
        # P3.1: mean-dependent dispersion takes priority when mu is provided
        slope = getattr(self, "_dispersion_slope", None)
        intercept = getattr(self, "_dispersion_intercept", None)
        if (mu is not None and slope is not None and intercept is not None
                and self.cfg.get("use_mean_dependent_dispersion", False)):
            r_mu = np.exp(intercept + slope * np.log(max(mu, 0.01)))
            return float(np.clip(r_mu, 0.5, 50.0))
        # Per-role fallback
        role_disp = getattr(self, "_role_dispersion", {})
        return role_disp.get(role, self._dispersion_r)

    def predict_with_shrinkage(
        self,
        X: pd.DataFrame,
        wide_df: pd.DataFrame,
    ) -> np.ndarray:
        """P3.3: Blend HGB prediction with Gamma-Poisson posterior mean.

        Weight by player support (games played):
            w = clip(support / 10, 0, 1)
            result = w * mu_hgb + (1-w) * mu_bayes
        """
        mu_hgb = self.predict_mean(X)
        alpha = getattr(self, "_league_prior_alpha", None)
        beta  = getattr(self, "_league_prior_beta", None)
        if alpha is None or beta is None:
            return mu_hgb  # no prior available — pure HGB

        support_col = f"player_{self.stat}_l5_support"
        support = wide_df[support_col].fillna(0).astype(float).values if support_col in wide_df.columns else np.zeros(len(mu_hgb))
        w = np.clip(support / 10.0, 0.0, 1.0)

        # Gamma-Poisson posterior mean = (alpha + obs_sum) / (beta + n_games)
        # Approximation: use support count and HGB prediction as obs_sum proxy
        stat_col = f"player_{self.stat}_mean_l5"
        obs_mean = wide_df[stat_col].fillna(float(self._global_mean)).values if stat_col in wide_df.columns else np.full(len(mu_hgb), self._global_mean)
        obs_sum = obs_mean * support
        mu_bayes = (alpha + obs_sum) / (beta + support)
        return w * mu_hgb + (1.0 - w) * mu_bayes

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "version": self.VERSION,
            "global_mean": self._global_mean,
            "global_var": self._global_var,
            "dispersion_r": self._dispersion_r,
            "pmf_type": "negbinom" if self._dispersion_r is not None else "poisson",
            "role_dispersion": getattr(self, "_role_dispersion", {}),
            "dispersion_slope": getattr(self, "_dispersion_slope", None),
            "dispersion_intercept": getattr(self, "_dispersion_intercept", None),
            "feature_importances_top10": dict(
                sorted(
                    getattr(self, "_feature_importances", {}).items(),
                    key=lambda kv: kv[1], reverse=True
                )[:10]
            ),
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "StatRateModel":
        return joblib.load(path)


class HurdleModel:
    """Hurdle model for sparse stats (stl, blk).

    Stage A: P(Y > 0) via binary classifier.
    Stage B: E[Y | Y > 0] via regressor fitted only on positive rows.

    PMF generation:
        p0 = 1 - P(Y > 0)
        positive tail from NegBinom(pos_mu, pos_r) scaled to P(Y > 0)
    """

    VERSION = "stage4_baseline_v1"

    def __init__(self, stat: str, cfg: dict[str, Any]) -> None:
        self.stat = stat
        self.cfg = cfg
        self._clf: HistGradientBoostingClassifier | None = None
        self._reg: HistGradientBoostingRegressor | None = None
        self._pos_dispersion_r: float | None = None
        self._pos_mean: float = 0.0
        self._pos_var: float = 0.0
        self._n_pos: int = 0
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        context_df: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "HurdleModel":
        """Fit hurdle model on did_play rows.

        Args:
            X: Feature matrix (model_feature_columns, numeric, NaN allowed).
            y: actual_{stat} values for did_play=True rows.
            context_df: Unused for HurdleModel (kept for API symmetry with StatRateModel).
            sample_weight: Optional per-sample weights (e.g. temporal decay).
        """
        seed = self.cfg.get("random_seed", 42)
        clf_kw = self.cfg.get("hgb_classifier", {})
        reg_kw = self.cfg.get("hgb_regressor", {})

        # Drop all-NaN columns to prevent sklearn BinMapper crash on early-season data
        all_nan = [c for c in X.columns if X[c].isna().all()]
        if all_nan:
            X = X.drop(columns=all_nan)
        self._usable_cols = list(X.columns)

        # Stage A: binary classifier P(Y > 0)
        y_binary = (y > 0).astype(int)
        self._clf = HistGradientBoostingClassifier(
            max_iter=clf_kw.get("max_iter", 200),
            max_leaf_nodes=clf_kw.get("max_leaf_nodes", 31),
            learning_rate=clf_kw.get("learning_rate", 0.1),
            min_samples_leaf=clf_kw.get("min_samples_leaf", 20),
            early_stopping=clf_kw.get("early_stopping", False),
            n_iter_no_change=clf_kw.get("n_iter_no_change", 10),
            tol=clf_kw.get("tol", 1e-7),
            random_state=seed,
        )
        self._clf.fit(X, y_binary, sample_weight=sample_weight)

        # Stage B: regressor E[Y | Y > 0] on positive rows
        pos_mask = y > 0
        self._n_pos = int(pos_mask.sum())
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

        if self._n_pos >= 10:
            self._reg = HistGradientBoostingRegressor(
                max_iter=reg_kw.get("max_iter", 200),
                max_leaf_nodes=reg_kw.get("max_leaf_nodes", 31),
                learning_rate=reg_kw.get("learning_rate", 0.1),
                min_samples_leaf=reg_kw.get("min_samples_leaf", 20),
                early_stopping=reg_kw.get("early_stopping", False),
                n_iter_no_change=reg_kw.get("n_iter_no_change", 10),
                tol=reg_kw.get("tol", 1e-7),
                random_state=seed,
            )
            self._reg.fit(X_pos, y_pos, sample_weight=sw_pos)
            self._pos_mean = float(y_pos.mean())
            self._pos_var = float(y_pos.var())
            self._pos_dispersion_r = dispersion_from_moments(self._pos_mean, self._pos_var)
        else:
            # Fall back to global positive-count mean if too few positive samples
            self._pos_mean = float(y_pos.mean()) if self._n_pos > 0 else 1.0
            self._pos_var = float(y_pos.var()) if self._n_pos > 1 else 0.5
            self._pos_dispersion_r = dispersion_from_moments(self._pos_mean, self._pos_var)

        self._fitted = True
        return self

    def predict(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_nonzero, pos_mu) arrays.

        p_nonzero: P(Y > 0)
        pos_mu:    E[Y | Y > 0]
        """
        if not self._fitted or self._clf is None:
            raise RuntimeError(f"HurdleModel({self.stat}) not fitted")

        # Align inference to the exact column set used at fit time.
        # Missing columns are filled with NaN — HGB handles NaN natively.
        if hasattr(self, "_usable_cols"):
            X = X.reindex(columns=self._usable_cols)

        # P(Y > 0)
        p_nz = self._clf.predict_proba(X)[:, 1]
        p_nz = np.clip(p_nz, 0.0, 1.0)

        # E[Y | Y > 0]
        min_mean = self.cfg.get("min_stat_mean", 0.01)
        if self._reg is not None:
            pos_mu = np.clip(self._reg.predict(X), min_mean, None)
        else:
            pos_mu = np.full(len(X), max(self._pos_mean, min_mean))

        return p_nz, pos_mu

    @property
    def pos_dispersion_r(self) -> float | None:
        return self._pos_dispersion_r

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "version": self.VERSION,
            "n_positive_rows": self._n_pos,
            "pos_mean": self._pos_mean,
            "pos_var": self._pos_var,
            "pos_dispersion_r": self._pos_dispersion_r,
            "has_reg": self._reg is not None,
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "HurdleModel":
        return joblib.load(path)
