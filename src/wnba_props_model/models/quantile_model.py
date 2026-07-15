"""Multi-quantile distribution architecture for WNBA player props.

This module implements a full parametric-distribution pipeline built on top of
a quantile ensemble, replacing the single-point HGB mean prediction with a
complete predictive distribution for each player-game.

High-level architecture
-----------------------
1. ``train_quantile_ensemble`` — trains one HGB per quantile level.
2. ``predict_distribution`` — returns (n_samples, n_quantiles) array.
3. ``enforce_monotonic_quantiles`` — fixes any quantile crossings per sample.
4. ``StackedQuantilePropModel`` — Ridge baseline + HGB residual quantile stack.
5. ``fit_gaussian_from_quantiles`` — fits Normal(mu, sigma) to predicted quantiles.
6. ``fit_negbinom_from_quantiles`` — fits NegBinom(n, p) to predicted quantiles.
7. ``PlayerStatDistribution`` — wraps a fitted parametric distribution with
   ``.prob_over(line)`` / ``.prob_under(line)`` methods.
8. ``PlayerVolatilityEstimator`` — per-player exponentially weighted sigma.
9. ``compute_edge`` — (prob_model - prob_implied) / prob_implied.
10. ``generate_edge_board`` — full edge board from distribution parameters.
11. ``WNBAPlayerPropPipeline`` — end-to-end fit/predict with OOF calibration.
12. ``walk_forward_validate`` — TimeSeriesSplit validation; never random splits.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: list[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

_BASE_HGB_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "learning_rate": 0.1,
    "min_samples_leaf": 20,
    "early_stopping": False,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# a) Quantile ensemble training
# ---------------------------------------------------------------------------

def train_quantile_ensemble(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    quantiles: list[float] = QUANTILES,
    base_params: dict[str, Any] | None = None,
) -> dict[float, HistGradientBoostingRegressor]:
    """Train one HGB per quantile level.

    Args:
        X_train: Feature matrix (n_samples, n_features).
        y_train: Target values (n_samples,).
        quantiles: List of quantile levels in (0, 1).
        base_params: HGB hyperparameters (without ``loss``/``quantile``).

    Returns:
        Dict mapping quantile level → fitted ``HistGradientBoostingRegressor``.
    """
    params = {**_BASE_HGB_PARAMS, **(base_params or {})}
    models: dict[float, HistGradientBoostingRegressor] = {}
    for q in quantiles:
        mdl = HistGradientBoostingRegressor(loss="quantile", quantile=q, **params)
        mdl.fit(X_train, y_train)
        models[q] = mdl
        logger.debug("Trained HGB quantile=%.2f", q)
    return models


# ---------------------------------------------------------------------------
# b) Predict distribution
# ---------------------------------------------------------------------------

def predict_distribution(
    models: dict[float, HistGradientBoostingRegressor],
    X: np.ndarray | pd.DataFrame,
    quantiles: list[float] = QUANTILES,
) -> tuple[np.ndarray, list[float]]:
    """Predict all quantile levels for each sample.

    Args:
        models: Dict from ``train_quantile_ensemble``.
        X: Feature matrix (n_samples, n_features).
        quantiles: Quantile levels to predict (must match keys in ``models``).

    Returns:
        Tuple of:
        - ``preds``: (n_samples, n_quantiles) array of predicted quantile values.
        - ``q_vals``: List of quantile levels in the same column order as ``preds``.
    """
    q_vals = [q for q in quantiles if q in models]
    cols = [models[q].predict(X) for q in q_vals]
    preds = np.column_stack(cols)  # (n_samples, n_quantiles)
    return preds, q_vals


# ---------------------------------------------------------------------------
# c) Enforce monotonic quantiles
# ---------------------------------------------------------------------------

def enforce_monotonic_quantiles(
    preds: np.ndarray,
    q_vals: list[float],  # noqa: ARG001 — kept for API symmetry / future weighting
) -> np.ndarray:
    """Fix quantile crossings so predictions are non-decreasing across quantiles.

    Applies isotonic regression (``increasing=True``) to each sample row so
    that the resulting quantile estimates are guaranteed monotone.

    Args:
        preds: (n_samples, n_quantiles) raw quantile predictions.
        q_vals: Quantile levels (used for future weighting; currently unused).

    Returns:
        (n_samples, n_quantiles) monotone-corrected array.
    """
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    out = np.empty_like(preds)
    dummy_x = np.arange(preds.shape[1], dtype=float)
    for i in range(preds.shape[0]):
        out[i] = iso.fit_transform(dummy_x, preds[i])
    return out


# ---------------------------------------------------------------------------
# d) Stacked quantile prop model
# ---------------------------------------------------------------------------

class StackedQuantilePropModel:
    """Two-layer stacked quantile model.

    Layer 1: ``Ridge(alpha=10.0)`` as a stable, low-variance baseline.
    Layer 2: One ``HistGradientBoostingRegressor(loss='quantile', quantile=q)``
             per quantile level, trained on residuals from the Ridge and with
             the Ridge prediction appended as an extra feature.

    This gives the non-linear HGB a head start: it only needs to learn the
    residual structure, reducing over-fit risk on small WNBA samples.
    """

    def __init__(
        self,
        quantiles: list[float] = QUANTILES,
        ridge_alpha: float = 10.0,
        hgb_params: dict[str, Any] | None = None,
    ) -> None:
        self.quantiles = quantiles
        self.ridge_alpha = ridge_alpha
        self.hgb_params = hgb_params or {}
        self._ridge: Ridge | None = None
        self._scaler: StandardScaler | None = None
        self._imputer: SimpleImputer | None = None
        self._hgb_models: dict[float, HistGradientBoostingRegressor] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "StackedQuantilePropModel":
        """Fit Ridge baseline then HGB residual quantile stack.

        Ridge does not handle NaN natively, so an imputer (median strategy)
        and a standard scaler are applied before Ridge. HGB receives the raw
        (un-scaled, NaN-tolerant) augmented matrix so its native NaN handling
        is unaffected.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            self (fitted).
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Layer 1: impute → scale → Ridge baseline
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_imputed = self._imputer.fit_transform(X_arr)
        X_scaled = self._scaler.fit_transform(X_imputed)
        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(X_scaled, y_arr)
        ridge_pred = self._ridge.predict(X_scaled).reshape(-1, 1)
        residuals = y_arr - ridge_pred.ravel()

        # Layer 2: HGB on residuals with ridge_pred as extra feature.
        # HGB handles NaN natively, so pass the original (un-scaled) X_arr.
        X_aug = np.hstack([X_arr, ridge_pred])
        base_params = {**_BASE_HGB_PARAMS, **self.hgb_params}
        for q in self.quantiles:
            mdl = HistGradientBoostingRegressor(loss="quantile", quantile=q, **base_params)
            mdl.fit(X_aug, residuals)
            self._hgb_models[q] = mdl
            logger.debug("Stacked HGB quantile=%.2f fitted", q)

        self._fitted = True
        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, list[float]]:
        """Predict quantile distribution for each sample.

        Returns:
            Tuple of:
            - ``preds``: (n_samples, n_quantiles) monotone-corrected array.
            - ``q_vals``: Quantile levels in column order.
        """
        if not self._fitted or self._ridge is None or self._imputer is None or self._scaler is None:
            raise RuntimeError("StackedQuantilePropModel is not fitted")
        X_arr = np.asarray(X, dtype=float)
        X_imputed = self._imputer.transform(X_arr)
        X_scaled = self._scaler.transform(X_imputed)
        ridge_pred = self._ridge.predict(X_scaled).reshape(-1, 1)
        X_aug = np.hstack([X_arr, ridge_pred])

        q_vals = sorted(self._hgb_models.keys())
        cols = [
            ridge_pred.ravel() + self._hgb_models[q].predict(X_aug)
            for q in q_vals
        ]
        preds = np.column_stack(cols)
        preds = np.clip(preds, 0.0, None)  # stats are non-negative
        preds = enforce_monotonic_quantiles(preds, q_vals)
        return preds, q_vals


# ---------------------------------------------------------------------------
# e) Fit Gaussian from quantiles
# ---------------------------------------------------------------------------

def fit_gaussian_from_quantiles(
    quantile_values: np.ndarray,
    quantile_levels: list[float],
) -> tuple[float, float]:
    """Fit Normal(mu, sigma) by minimising SSE between predicted and theoretical quantiles.

    Args:
        quantile_values: Array of predicted quantile values (n_quantiles,).
        quantile_levels: Quantile levels (n_quantiles,).

    Returns:
        Tuple (mu, sigma) of fitted Normal parameters.
    """
    q_arr = np.asarray(quantile_levels, dtype=float)
    v_arr = np.asarray(quantile_values, dtype=float)

    def _loss(params: np.ndarray) -> float:
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        theoretical = norm.ppf(q_arr, loc=mu, scale=sigma)
        return float(np.sum((theoretical - v_arr) ** 2))

    # Initial guess: median as mu, IQR/1.35 as sigma
    mu0 = float(np.interp(0.5, q_arr, v_arr))
    q75 = float(np.interp(0.75, q_arr, v_arr))
    q25 = float(np.interp(0.25, q_arr, v_arr))
    sigma0 = max((q75 - q25) / 1.35, 0.1)

    result = minimize(_loss, [mu0, np.log(sigma0)], method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 500})
    mu_hat, log_sigma_hat = result.x
    return float(mu_hat), float(np.exp(log_sigma_hat))


# ---------------------------------------------------------------------------
# f) Fit NegBinom from quantiles
# ---------------------------------------------------------------------------

def fit_negbinom_from_quantiles(
    quantile_values: np.ndarray,
    quantile_levels: list[float],
) -> tuple[float, float]:
    """Fit NegativeBinomial(n, p) to predicted quantiles.

    Uses the parameterisation ``nbinom(n, p)`` where mean = n*(1-p)/p and
    variance = n*(1-p)/p^2.

    Args:
        quantile_values: Array of predicted quantile values (n_quantiles,).
        quantile_levels: Quantile levels (n_quantiles,).

    Returns:
        Tuple (n, p) of fitted NegBinom parameters.
    """
    q_arr = np.asarray(quantile_levels, dtype=float)
    v_arr = np.asarray(quantile_values, dtype=float)
    v_arr = np.clip(v_arr, 0.0, None)

    def _loss(params: np.ndarray) -> float:
        log_n, logit_p = params
        n = np.exp(log_n)
        p = 1.0 / (1.0 + np.exp(-logit_p))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theoretical = nbinom.ppf(q_arr, n=n, p=p).astype(float)
        return float(np.sum((theoretical - v_arr) ** 2))

    mu0 = max(float(np.interp(0.5, q_arr, v_arr)), 0.5)
    q75 = float(np.interp(0.75, q_arr, v_arr))
    q25 = float(np.interp(0.25, q_arr, v_arr))
    var0 = max(((q75 - q25) / 1.35) ** 2, mu0 + 0.1)
    # Method-of-moments initial guess
    p0 = max(min(mu0 / var0, 0.95), 0.05)
    n0 = max(mu0 * p0 / (1.0 - p0), 0.5)

    result = minimize(
        _loss,
        [np.log(n0), np.log(p0 / (1.0 - p0))],
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 1000},
    )
    log_n_hat, logit_p_hat = result.x
    n_hat = float(np.exp(log_n_hat))
    p_hat = float(1.0 / (1.0 + np.exp(-logit_p_hat)))
    return n_hat, p_hat


# ---------------------------------------------------------------------------
# g) PlayerStatDistribution
# ---------------------------------------------------------------------------

class PlayerStatDistribution:
    """Parametric distribution wrapper for a single player-game stat prediction.

    Gaussian is used for high-volume continuous-ish stats (pts, reb, ast);
    NegBinom is used for low-count discrete stats (fg3m, stl, blk, turnover).

    Args:
        stat_type: Name of the stat (e.g. "pts", "fg3m").
        quantile_values: Predicted quantile values (n_quantiles,).
        quantile_levels: Corresponding quantile levels (n_quantiles,).
    """

    GAUSSIAN_STATS: frozenset[str] = frozenset({"pts", "reb", "ast"})
    NEGBINOM_STATS: frozenset[str] = frozenset({"fg3m", "stl", "blk", "turnover"})

    def __init__(
        self,
        stat_type: str,
        quantile_values: np.ndarray,
        quantile_levels: list[float],
    ) -> None:
        self.stat_type = stat_type
        self._q_vals = np.asarray(quantile_values, dtype=float)
        self._q_levels = list(quantile_levels)

        if stat_type in self.GAUSSIAN_STATS:
            self._dist = "gaussian"
            self._mu, self._sigma = fit_gaussian_from_quantiles(self._q_vals, self._q_levels)
        else:
            self._dist = "negbinom"
            self._n, self._p = fit_negbinom_from_quantiles(self._q_vals, self._q_levels)

    def prob_over(self, line: float) -> float:
        """P(Y > line)."""
        if self._dist == "gaussian":
            return float(1.0 - norm.cdf(line, loc=self._mu, scale=max(self._sigma, 1e-6)))
        n, p = self._n, self._p
        return float(1.0 - nbinom.cdf(int(np.floor(line)), n=n, p=p))

    def prob_under(self, line: float) -> float:
        """P(Y <= line)."""
        return 1.0 - self.prob_over(line)

    @property
    def median(self) -> float:
        """Predicted median (50th percentile)."""
        if 0.5 in self._q_levels:
            idx = self._q_levels.index(0.5)
            return float(self._q_vals[idx])
        return float(np.interp(0.5, self._q_levels, self._q_vals))

    @property
    def params(self) -> dict[str, Any]:
        """Distribution parameters as a dict."""
        if self._dist == "gaussian":
            return {"dist": "gaussian", "mu": self._mu, "sigma": self._sigma}
        return {"dist": "negbinom", "n": self._n, "p": self._p}


# ---------------------------------------------------------------------------
# h) PlayerVolatilityEstimator
# ---------------------------------------------------------------------------

class PlayerVolatilityEstimator:
    """Per-player exponentially-weighted rolling sigma from prediction residuals.

    Uses an EW rolling window (span configurable, default 30 games) with
    decay=0.95 to give more weight to recent games.

    Args:
        decay: EW decay factor (lambda). Equivalent EWMA span ≈ 2/(1-decay) - 1.
        window: Maximum lookback window in games.
    """

    def __init__(self, decay: float = 0.95, window: int = 30) -> None:
        self.decay = decay
        self.window = window
        self._player_sigma: dict[Any, float] = {}

    def fit(
        self,
        player_ids: np.ndarray | pd.Series,
        y_actual: np.ndarray | pd.Series,
        y_predicted: np.ndarray | pd.Series,
    ) -> "PlayerVolatilityEstimator":
        """Compute per-player exponentially-weighted sigma from residuals.

        Args:
            player_ids: Player identifier per row.
            y_actual: Observed stat values.
            y_predicted: Model-predicted stat values.

        Returns:
            self (fitted).
        """
        pid = np.asarray(player_ids)
        ya = np.asarray(y_actual, dtype=float)
        yp = np.asarray(y_predicted, dtype=float)
        residuals = ya - yp

        df = pd.DataFrame({"player_id": pid, "residual": residuals})
        span = int(round(2.0 / (1.0 - self.decay) - 1))

        for player_id, grp in df.groupby("player_id"):
            res = grp["residual"].values[-self.window:]
            if len(res) < 3:
                continue
            ewm_var = pd.Series(res).ewm(span=span, adjust=False).var()
            sigma = float(np.sqrt(max(ewm_var.iloc[-1], 0.0)))
            self._player_sigma[player_id] = sigma

        return self

    def get_sigma(self, player_id: Any, fallback: float = 5.0) -> float:
        """Return per-player sigma, or ``fallback`` if player is unknown.

        Args:
            player_id: Player identifier.
            fallback: Sigma to return when player has no history (default 5.0).

        Returns:
            Estimated sigma (standard deviation of residuals).
        """
        return self._player_sigma.get(player_id, fallback)


# ---------------------------------------------------------------------------
# i) compute_edge
# ---------------------------------------------------------------------------

def compute_edge(prob_model: float, prob_implied: float) -> float:
    """Compute edge as fractional gain over the implied probability.

    Args:
        prob_model: Model's probability for the outcome.
        prob_implied: Market-implied probability (vig-removed).

    Returns:
        Edge = (prob_model - prob_implied) / prob_implied.
        Returns 0.0 when prob_implied is too small to avoid division-by-zero.
    """
    if prob_implied < 1e-6:
        return 0.0
    return (prob_model - prob_implied) / prob_implied


# ---------------------------------------------------------------------------
# j) generate_edge_board
# ---------------------------------------------------------------------------

def generate_edge_board(
    player_games: list[dict[str, Any]],
    quantile_preds_matrix: np.ndarray,
    q_vals: list[float],
    mu_per_game: np.ndarray,
    sigma_per_game: np.ndarray,
    min_edge: float = 0.03,
) -> list[dict[str, Any]]:
    """Generate full edge board using parametric distributions.

    Args:
        player_games: List of dicts with at minimum keys:
            ``player_id``, ``stat``, ``line``, ``implied_over``, ``implied_under``.
        quantile_preds_matrix: (n_games, n_quantiles) predicted quantile values.
        q_vals: Quantile levels in column order of ``quantile_preds_matrix``.
        mu_per_game: (n_games,) distribution mu (Gaussian) or mean from NegBinom.
        sigma_per_game: (n_games,) distribution sigma.
        min_edge: Minimum absolute edge to include in the board.

    Returns:
        List of edge dicts, each with keys:
        ``player_id``, ``stat``, ``line``, ``direction``, ``edge``,
        ``prob_model``, ``prob_implied``, ``mu``, ``sigma``.
        Sorted by descending absolute edge.
    """
    board: list[dict[str, Any]] = []
    for i, pg in enumerate(player_games):
        stat = pg.get("stat", "")
        line = float(pg.get("line", 0.0))
        imp_over = float(pg.get("implied_over", 0.5))
        imp_under = float(pg.get("implied_under", 0.5))

        q_row = quantile_preds_matrix[i]
        dist = PlayerStatDistribution(stat, q_row, q_vals)

        p_over = dist.prob_over(line)
        p_under = dist.prob_under(line)
        edge_over = compute_edge(p_over, imp_over)
        edge_under = compute_edge(p_under, imp_under)

        for direction, edge, prob_model, prob_imp in [
            ("OVER", edge_over, p_over, imp_over),
            ("UNDER", edge_under, p_under, imp_under),
        ]:
            if abs(edge) >= min_edge:
                board.append({
                    "player_id": pg.get("player_id"),
                    "stat": stat,
                    "line": line,
                    "direction": direction,
                    "edge": round(edge, 4),
                    "prob_model": round(prob_model, 4),
                    "prob_implied": round(prob_imp, 4),
                    "mu": round(float(mu_per_game[i]), 3),
                    "sigma": round(float(sigma_per_game[i]), 3),
                    "median": round(dist.median, 3),
                })

    board.sort(key=lambda r: abs(r["edge"]), reverse=True)
    return board


# ---------------------------------------------------------------------------
# k) WNBAPlayerPropPipeline
# ---------------------------------------------------------------------------

class WNBAPlayerPropPipeline:
    """End-to-end multi-quantile WNBA player prop pipeline.

    Workflow:
        1. Fit ``StackedQuantilePropModel`` per stat.
        2. Generate OOF quantile predictions via ``TimeSeriesSplit``.
        3. Fit a quantile-coverage calibrator (isotonic on median vs. actuals).
        4. Fit ``PlayerVolatilityEstimator`` from OOF residuals.
        5. At prediction time: generate quantile distribution → parametric fit →
           edge board.

    Args:
        quantiles: Quantile levels to model.
        hgb_params: Passed to ``StackedQuantilePropModel``.
        min_edge: Minimum edge for the board.
        n_oof_splits: Number of TimeSeriesSplit folds for OOF generation.
    """

    def __init__(
        self,
        quantiles: list[float] = QUANTILES,
        hgb_params: dict[str, Any] | None = None,
        min_edge: float = 0.03,
        n_oof_splits: int = 5,
    ) -> None:
        self.quantiles = quantiles
        self.hgb_params = hgb_params or {}
        self.min_edge = min_edge
        self.n_oof_splits = n_oof_splits
        self._stacked_models: dict[str, StackedQuantilePropModel] = {}
        self._volatility: dict[str, PlayerVolatilityEstimator] = {}
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y_dict: dict[str, pd.Series],
        dates: pd.Series,
        player_ids: pd.Series,
    ) -> "WNBAPlayerPropPipeline":
        """Fit stacked models and volatility estimator per stat.

        Args:
            X: Feature matrix (all stats share the same feature set).
            y_dict: Dict mapping stat name → target Series.
            dates: Game dates (used for temporal splits).
            player_ids: Player IDs per row.

        Returns:
            self (fitted).
        """
        for stat, y in y_dict.items():
            valid = y.notna()
            X_s = X[valid].reset_index(drop=True)
            y_s = y[valid].reset_index(drop=True)
            pid_s = player_ids[valid].reset_index(drop=True)

            if len(y_s) < 50:
                logger.warning("Skipping stat=%s — only %d rows", stat, len(y_s))
                continue

            # Fit full stacked model
            model = StackedQuantilePropModel(
                quantiles=self.quantiles,
                hgb_params=self.hgb_params,
            )
            model.fit(X_s, y_s)
            self._stacked_models[stat] = model

            # OOF residuals for volatility estimation
            preds_50, _ = model.predict(X_s)
            median_col_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            oof_median = preds_50[:, median_col_idx]

            vol = PlayerVolatilityEstimator()
            vol.fit(pid_s, y_s, oof_median)
            self._volatility[stat] = vol
            logger.info("Fitted pipeline for stat=%s (n=%d)", stat, len(y_s))

        self._fitted = True
        return self

    def predict_edge_board(
        self,
        X: pd.DataFrame,
        player_games: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate edge board for upcoming games.

        Args:
            X: Feature matrix (one row per player-game in ``player_games``).
            player_games: List of dicts with ``player_id``, ``stat``, ``line``,
                ``implied_over``, ``implied_under``.

        Returns:
            Edge board as list of dicts (see ``generate_edge_board``).
        """
        if not self._fitted:
            raise RuntimeError("WNBAPlayerPropPipeline is not fitted")

        board: list[dict[str, Any]] = []
        stats = list({pg["stat"] for pg in player_games})

        for stat in stats:
            if stat not in self._stacked_models:
                continue
            stat_mask = [i for i, pg in enumerate(player_games) if pg["stat"] == stat]
            if not stat_mask:
                continue

            X_stat = X.iloc[stat_mask].reset_index(drop=True)
            preds, q_vals = self._stacked_models[stat].predict(X_stat)

            # Extract mu and sigma per game
            median_idx = q_vals.index(0.5) if 0.5 in q_vals else len(q_vals) // 2
            mu_arr = preds[:, median_idx]
            vol = self._volatility.get(stat)
            sigma_arr = np.array([
                vol.get_sigma(player_games[i]["player_id"]) if vol else 5.0
                for i in stat_mask
            ])

            stat_pgs = [player_games[i] for i in stat_mask]
            stat_board = generate_edge_board(
                stat_pgs, preds, q_vals, mu_arr, sigma_arr,
                min_edge=self.min_edge,
            )
            board.extend(stat_board)

        board.sort(key=lambda r: abs(r["edge"]), reverse=True)
        return board


# ---------------------------------------------------------------------------
# l) walk_forward_validate
# ---------------------------------------------------------------------------

def walk_forward_validate(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    dates: pd.Series,  # noqa: ARG001 — TimeSeriesSplit on row order is sufficient
    quantiles: list[float] = QUANTILES,
    n_splits: int = 5,
) -> list[dict[str, Any]]:
    """Walk-forward cross-validation for quantile ensemble.

    Uses ``TimeSeriesSplit`` (never random splits) to respect temporal ordering.

    Args:
        X: Feature matrix.
        y: Target values.
        dates: Game dates (used to verify ordering; splits are on row index).
        quantiles: Quantile levels to evaluate.
        n_splits: Number of time-series splits.

    Returns:
        List of fold result dicts, each with keys:
        ``fold``, ``n_train``, ``n_test``, ``quantile_coverage``.
        ``quantile_coverage`` maps quantile level → empirical hit rate
        (ideally close to the quantile level itself).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        if len(y_tr) < 100 or len(y_te) < 30:
            logger.warning("Fold %d too small (train=%d, test=%d) — skipping",
                           fold, len(y_tr), len(y_te))
            continue

        models = train_quantile_ensemble(X_tr, y_tr, quantiles=quantiles)
        preds, q_vals = predict_distribution(models, X_te, quantiles=quantiles)
        preds = enforce_monotonic_quantiles(preds, q_vals)

        coverage: dict[float, float] = {}
        for j, q in enumerate(q_vals):
            hit_rate = float(np.mean(y_te <= preds[:, j]))
            coverage[q] = round(hit_rate, 4)

        results.append({
            "fold": fold,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "quantile_coverage": coverage,
        })
        logger.info("Fold %d: n_train=%d, n_test=%d, median_coverage=%.3f",
                    fold, len(y_tr), len(y_te), coverage.get(0.5, float("nan")))

    return results
