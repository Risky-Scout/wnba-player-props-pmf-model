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
        # Role-stratified HGB models: {"starter": HGB, "bench": HGB, ...}
        self._role_models: dict[str, HistGradientBoostingRegressor] = {}
        # Per-player std for dispersion scaling: {player_id: float}
        self._player_std_map: dict[str, float] = {}
        # Per-role mean std (for scaling denominator): {"starter": float, ...}
        self._role_mean_std: dict[str, float] = {}
        # Part 3: feature-based learned dispersion model (predicts log(r) from features)
        # Trained after main HGB; replaces role-lookup at inference when available.
        self.dispersion_model: HistGradientBoostingRegressor | None = None

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling: add new fields if missing."""
        self.__dict__.update(state)
        if "dispersion_model" not in self.__dict__:
            self.dispersion_model = None
        if "_role_models" not in self.__dict__:
            self._role_models = {}
        if "_player_std_map" not in self.__dict__:
            self._player_std_map = {}
        if "_role_mean_std" not in self.__dict__:
            self._role_mean_std = {}

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
        # Use quantile(0.5) loss so the model targets the conditional median,
        # which matches how sportsbooks set prop lines (not the mean).
        # For right-skewed WNBA stat distributions mean > median, so MSE
        # predictions systematically exceed market lines → 116 UNDER / 22 OVER.
        hgb_loss = hgb_kw.get("loss", "quantile")
        hgb_mdl_kw: dict = dict(
            loss=hgb_loss,
            max_iter=hgb_kw.get("max_iter", 200),
            max_leaf_nodes=hgb_kw.get("max_leaf_nodes", 31),
            learning_rate=hgb_kw.get("learning_rate", 0.1),
            min_samples_leaf=hgb_kw.get("min_samples_leaf", 20),
            early_stopping=hgb_kw.get("early_stopping", False),
            n_iter_no_change=hgb_kw.get("n_iter_no_change", 10),
            tol=hgb_kw.get("tol", 1e-7),
            random_state=seed,
        )
        if hgb_loss == "quantile":
            hgb_mdl_kw["quantile"] = hgb_kw.get("quantile", 0.5)
        self._model = HistGradientBoostingRegressor(**hgb_mdl_kw)
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

        # Role-stratified training: fit a separate HGB for each role_bucket.
        # Fixes structural over-prediction caused by a single all-player model
        # whose mean is pulled toward high-volume starters, inflating bench predictions.
        if self.cfg.get("use_role_stratified_training", False) and context_df is not None and "role_bucket" in context_df.columns:
            ctx_rst = context_df.reset_index(drop=True)
            y_rst = y.reset_index(drop=True)
            X_rst = X.reset_index(drop=True)
            for role_name, role_grp in ctx_rst.groupby("role_bucket"):
                role_idx = role_grp.index
                if len(role_idx) < 30:
                    continue  # not enough data for a separate model
                X_role = X_rst.loc[role_idx].reindex(columns=self._usable_cols)
                y_role = y_rst.loc[role_idx]
                sw_role = sample_weight[role_idx] if sample_weight is not None else None
                if use_offset and context_df is not None and "actual_minutes" in context_df.columns:
                    minutes_role = ctx_rst.loc[role_idx, "actual_minutes"].clip(lower=1.0)
                    y_role_fit = y_role / minutes_role.values
                else:
                    y_role_fit = y_role
                role_mdl = HistGradientBoostingRegressor(**hgb_mdl_kw)
                role_mdl.fit(X_role, y_role_fit, sample_weight=sw_role)
                self._role_models[str(role_name)] = role_mdl

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

        # Per-player dispersion scaling: build player_id → individual std lookup.
        # player_{stat}_std_l10 captures individual volatility; players with
        # higher vol than the role mean get wider NegBinom tails (lower r).
        if self.cfg.get("use_per_player_dispersion", False) and context_df is not None:
            std_col = f"player_{self.stat}_std_l10"
            if "player_id" in context_df.columns and std_col in context_df.columns:
                ctx_ppd = context_df.reset_index(drop=True)
                _ppd_cols = ["player_id", std_col]
                if "role_bucket" in ctx_ppd.columns:
                    _ppd_cols = ["player_id", "role_bucket", std_col]
                ctx_ppd = ctx_ppd[_ppd_cols].dropna(subset=[std_col])
                # Per-player median std (stable across their games in training)
                self._player_std_map = (
                    ctx_ppd.groupby("player_id")[std_col].median().to_dict()
                )
                # Role mean std (denominator for scaling)
                if "role_bucket" in ctx_ppd.columns:
                    self._role_mean_std = (
                        ctx_ppd.groupby("role_bucket")[std_col].median().to_dict()
                    )

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

    def predict_mean(self, X: pd.DataFrame, role_series: "pd.Series | None" = None) -> np.ndarray:
        """Predict E[Y], clipped to >= min_stat_mean.

        When ``use_minutes_offset`` was active at fit time, the model predicts
        the per-minute rate.  We multiply by the projected minutes feature
        (``player_minutes_mean_l5``) to recover the expected count.

        When ``role_series`` is provided and role-stratified models exist,
        routes each player's prediction to the appropriate role-specific model,
        falling back to the global model for unknown roles.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError(f"StatRateModel({self.stat}) not fitted")
        X_pred = X.reindex(columns=getattr(self, "_usable_cols", X.columns))
        min_mean = self.cfg.get("min_stat_mean", 0.01)

        role_models = getattr(self, "_role_models", {})
        if (role_series is not None and role_models
                and self.cfg.get("use_role_stratified_training", False)):
            role_arr = role_series.reset_index(drop=True).values
            raw = self._model.predict(X_pred)  # global as fallback
            for role_name, role_mdl in role_models.items():
                mask = role_arr == role_name
                if mask.any():
                    raw[mask] = role_mdl.predict(X_pred.iloc[mask] if hasattr(X_pred, 'iloc') else X_pred[mask])
        else:
            raw = self._model.predict(X_pred)

        if getattr(self, "_use_minutes_offset", False):
            minutes_col = getattr(self, "_minutes_offset_col", "player_minutes_mean_l5")
            _fallback_min = max(getattr(self, "_global_mean", 1.0) / 0.5, 20.0)
            if minutes_col in X.columns:
                projected_min = X[minutes_col].values.astype(float)
                # Replace NaN/inf with fallback before clip so NaN doesn't propagate
                projected_min = np.where(np.isfinite(projected_min), projected_min, _fallback_min)
                projected_min = np.clip(projected_min, 1.0, None)
            else:
                projected_min = np.full(len(raw), _fallback_min)
            result = raw * projected_min
            # Final safety net: replace any remaining non-finite with floor
            result = np.where(np.isfinite(result), result, min_mean)
            return np.clip(result, min_mean, None)

        return np.clip(raw, min_mean, None)

    @property
    def dispersion_r(self) -> float | None:
        return self._dispersion_r

    def get_dispersion(self, role: str, mu: float | None = None, player_id: str | None = None) -> float | None:
        """Return dispersion r for NegBinom PMF generation.

        Priority order:
        1. Feature-based dispersion model (per-instance r from HGB)
        2. Mean-dependent dispersion r(mu)
        3. Per-player scaling of role r based on individual rolling std
        4. Per-role r
        5. Global r
        """
        # P3.1: mean-dependent dispersion takes priority when mu is provided
        slope = getattr(self, "_dispersion_slope", None)
        intercept = getattr(self, "_dispersion_intercept", None)
        if (mu is not None and slope is not None and intercept is not None
                and self.cfg.get("use_mean_dependent_dispersion", False)):
            r_mu = np.exp(intercept + slope * np.log(max(mu, 0.01)))
            return float(np.clip(r_mu, 0.5, 50.0))
        # Per-player dispersion scaling: adjust role r by individual volatility ratio
        role_disp = getattr(self, "_role_dispersion", {})
        base_r = role_disp.get(role, self._dispersion_r)
        if (player_id is not None
                and self.cfg.get("use_per_player_dispersion", False)
                and base_r is not None):
            player_std = self._player_std_map.get(str(player_id))
            role_mean_std = self._role_mean_std.get(str(role))
            if player_std is not None and role_mean_std is not None and role_mean_std > 0.01:
                # Higher individual std → wider PMF (lower r); scale inversely
                # Clamp ratio to [0.5, 2.0] to avoid extreme adjustments
                vol_ratio = float(np.clip(player_std / role_mean_std, 0.5, 2.0))
                adjusted_r = float(np.clip(base_r / vol_ratio, 0.3, 50.0))
                return adjusted_r
        return base_r

    def predict_with_shrinkage(
        self,
        X: pd.DataFrame,
        wide_df: pd.DataFrame,
        role_series: "pd.Series | None" = None,
    ) -> np.ndarray:
        """P3.3: Blend HGB prediction with Gamma-Poisson posterior mean.

        Weight by player support (games played):
            w = clip(support / 5, 0, 1)  — trusts HGB fully at 5 L5 games
            result = w * mu_hgb + (1-w) * mu_bayes

        NaN fallback for obs_mean uses season mean then HGB prediction,
        NOT global mean — prevents low-season-data players from being
        pulled to league-wide all-player average.
        """
        mu_hgb = self.predict_mean(X, role_series=role_series)
        alpha = getattr(self, "_league_prior_alpha", None)
        beta  = getattr(self, "_league_prior_beta", None)
        if alpha is None or beta is None:
            return mu_hgb  # no prior available — pure HGB

        support_col = f"player_{self.stat}_l5_support"
        support = wide_df[support_col].fillna(0).astype(float).values if support_col in wide_df.columns else np.zeros(len(mu_hgb))
        # Halved divisor: fully trusts HGB at 5 L5 games (was 10).
        w = np.clip(support / 5.0, 0.0, 1.0)

        # Gamma-Poisson posterior mean = (alpha + obs_sum) / (beta + n_games)
        # Fallback chain: L5 mean → season mean → HGB prediction.
        # Using HGB (not global mean) prevents players with missing L5 data
        # from being unfairly collapsed to the all-player league average.
        stat_col = f"player_{self.stat}_mean_l5"
        season_col = f"player_{self.stat}_mean_season"
        if stat_col in wide_df.columns:
            obs_mean_series = wide_df[stat_col]
            if season_col in wide_df.columns:
                obs_mean_series = obs_mean_series.fillna(wide_df[season_col])
            obs_mean = obs_mean_series.fillna(pd.Series(mu_hgb, index=wide_df.index)).values.astype(float)
        else:
            obs_mean = mu_hgb  # pure HGB fallback
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
        self._role_regs: dict[str, HistGradientBoostingRegressor] = {}

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
            # Use quantile(0.5) so E[Y | Y > 0] targets the conditional median,
            # consistent with the StatRateModel fix above.
            reg_loss = reg_kw.get("loss", "quantile")
            reg_mdl_kw: dict = dict(
                loss=reg_loss,
                max_iter=reg_kw.get("max_iter", 200),
                max_leaf_nodes=reg_kw.get("max_leaf_nodes", 31),
                learning_rate=reg_kw.get("learning_rate", 0.1),
                min_samples_leaf=reg_kw.get("min_samples_leaf", 20),
                early_stopping=reg_kw.get("early_stopping", False),
                n_iter_no_change=reg_kw.get("n_iter_no_change", 10),
                tol=reg_kw.get("tol", 1e-7),
                random_state=seed,
            )
            if reg_loss == "quantile":
                reg_mdl_kw["quantile"] = reg_kw.get("quantile", 0.5)
            self._reg = HistGradientBoostingRegressor(**reg_mdl_kw)
            self._reg.fit(X_pos, y_pos, sample_weight=sw_pos)
            self._pos_mean = float(y_pos.mean())
            self._pos_var = float(y_pos.var())
            self._pos_dispersion_r = dispersion_from_moments(self._pos_mean, self._pos_var)

            # Role-stratified Stage B: fit separate positive-tail regressors per role.
            if self.cfg.get("use_role_stratified_training", False) and context_df is not None and "role_bucket" in context_df.columns:
                ctx_pos = context_df.reset_index(drop=True).loc[pos_mask.reset_index(drop=True).values]
                for role_name, role_grp in ctx_pos.groupby("role_bucket"):
                    role_idx_local = role_grp.index
                    if len(role_idx_local) < 15:
                        continue
                    X_rpos = X_pos.reset_index(drop=True).loc[role_idx_local].reindex(columns=self._usable_cols)
                    y_rpos = y_pos.reset_index(drop=True).loc[role_idx_local]
                    sw_rpos = sw_pos[role_idx_local] if sw_pos is not None else None
                    r_reg_mdl = HistGradientBoostingRegressor(**reg_mdl_kw)
                    r_reg_mdl.fit(X_rpos, y_rpos, sample_weight=sw_rpos)
                    self._role_regs[str(role_name)] = r_reg_mdl
        else:
            # Fall back to global positive-count mean if too few positive samples
            self._pos_mean = float(y_pos.mean()) if self._n_pos > 0 else 1.0
            self._pos_var = float(y_pos.var()) if self._n_pos > 1 else 0.5
            self._pos_dispersion_r = dispersion_from_moments(self._pos_mean, self._pos_var)

        self._fitted = True
        return self

    def predict(
        self, X: pd.DataFrame, role_series: "pd.Series | None" = None,
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
        role_regs = getattr(self, "_role_regs", {})
        if (role_series is not None and role_regs
                and self.cfg.get("use_role_stratified_training", False)
                and self._reg is not None):
            pos_mu = np.clip(self._reg.predict(X), min_mean, None)
            role_arr_h = role_series.reset_index(drop=True).values
            for role_name, role_reg in role_regs.items():
                mask_h = role_arr_h == role_name
                if mask_h.any():
                    pos_mu[mask_h] = np.clip(role_reg.predict(X.iloc[mask_h] if hasattr(X, 'iloc') else X[mask_h]), min_mean, None)
        elif self._reg is not None:
            pos_mu = np.clip(self._reg.predict(X), min_mean, None)
        else:
            pos_mu = np.full(len(X), max(self._pos_mean, min_mean))

        return p_nz, pos_mu

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling: add new fields if missing."""
        self.__dict__.update(state)
        if "_role_regs" not in self.__dict__:
            self._role_regs = {}

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
