"""Adversarial Bookmaker Line Predictor (Enhancement 16).

Trains a SEPARATE HGB model to predict WHERE a prop line will CLOSE given:
    - opening_line
    - model_projection (our model's projected value)
    - model_edge_vs_opening
    - hours_until_game
    - injury status features
    - public betting percentage
    - number of books offering
    - vig/odds features

This model directly targets CLV by predicting the expected closing line,
enabling:
    1. expected_clv = predicted_close - current_line  (> 0 → bet)
    2. optimal timing: bet when line is furthest from predicted close
    3. steam detection: if predicted_close is moving, sharp money is arriving

Reference:
    Getting Precise About Closing Line Value. Unabated.
    https://unabated.com/articles/getting-precise-about-closing-line-value
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Features expected by the line predictor
LINE_PREDICTOR_FEATURES = [
    "opening_line",
    "model_projection",
    "model_edge_vs_opening",     # model_projection - opening_line
    "hours_until_game",
    "public_over_pct",           # fraction of public bets on over
    "book_count",                # number of books offering this prop
    "opening_over_odds",         # vig on over (e.g. -110 → -1.10)
    "opening_under_odds",
    "is_prime_time",             # nationally televised game flag
    "star_player_prop",          # prop is for a star (top-5 usage)
    "injury_key_teammate",       # a key teammate is questionable/out
    "line_move_last_hour",       # how much line moved in past hour
    "model_confidence",          # model's calibrated confidence interval width
]


class LinePredictor:
    """Predict closing prop lines from opening lines and market features.

    Training data: historical props where BOTH opening and closing lines
    are available.  Target = closing_line.

    The model captures information flow into the market: how injuries,
    news, and sharp money move lines over time.
    """

    def __init__(self, max_iter: int = 300, max_depth: int = 4):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            random_state=42,
        )
        self.trained: bool = False
        self._feature_names: list[str] = LINE_PREDICTOR_FEATURES.copy()
        self._train_metrics: dict[str, float] = {}

    # ── Fit ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y_closing: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LinePredictor":
        """Train on historical data with opening and closing lines.

        Parameters
        ----------
        X         : feature matrix (n_samples, n_features).
                    Columns should match LINE_PREDICTOR_FEATURES order.
        y_closing : target closing line values.
        feature_names: column names (for logging/interpretation).
        """
        if feature_names is not None:
            self._feature_names = feature_names

        self.model.fit(X, y_closing)
        self.trained = True

        # Cross-validate for MAE
        try:
            cv_scores = cross_val_score(
                HistGradientBoostingRegressor(
                    max_iter=self.max_iter,
                    max_depth=self.max_depth,
                    random_state=42,
                ),
                X,
                y_closing,
                cv=5,
                scoring="neg_mean_absolute_error",
            )
            self._train_metrics = {
                "cv_mae_mean": float(-cv_scores.mean()),
                "cv_mae_std":  float(cv_scores.std()),
                "n_samples":   int(len(y_closing)),
            }
            logger.info(
                "LinePredictor fitted: CV MAE = %.3f ± %.3f (n=%d)",
                self._train_metrics["cv_mae_mean"],
                self._train_metrics["cv_mae_std"],
                self._train_metrics["n_samples"],
            )
        except Exception as exc:
            logger.warning("CV evaluation failed: %s", exc)

        return self

    # ── Predict ──────────────────────────────────────────────────────────

    def predict_closing_line(
        self, features: np.ndarray
    ) -> tuple[float, tuple[float, float]]:
        """Predict where the line will close.

        Returns (predicted_close, (ci_low, ci_high)).
        CI is approximated via ±0.5 (use quantile regression for production).
        """
        if not self.trained:
            raise RuntimeError(
                "LinePredictor not trained. Call .fit() first or load a checkpoint."
            )
        if features.ndim == 1:
            features = features.reshape(1, -1)
        pred = float(self.model.predict(features)[0])
        # Approximate CI — replace with quantile HGBR in v2
        ci_low  = pred - 0.5
        ci_high = pred + 0.5
        return pred, (ci_low, ci_high)

    def compute_expected_clv(
        self,
        current_line: float,
        features: np.ndarray,
    ) -> dict[str, Any]:
        """Compute expected CLV for betting at the current_line.

        expected_clv = predicted_close - current_line
        If positive, the line is expected to move in our direction.

        Parameters
        ----------
        current_line : the line as it stands NOW (may differ from opening)
        features     : feature vector (shape (n_features,))

        Returns
        -------
        dict with expected_clv, predicted_close, confidence interval,
        optimal_timing recommendation, and steam_flag.
        """
        pred_close, (ci_low, ci_high) = self.predict_closing_line(features)
        expected_clv = pred_close - current_line

        return {
            "predicted_close":        round(pred_close, 3),
            "current_line":           round(current_line, 3),
            "expected_clv":           round(expected_clv, 3),
            "clv_confidence_lower":   round(ci_low - current_line, 3),
            "clv_confidence_upper":   round(ci_high - current_line, 3),
            "optimal_timing":         "now" if abs(expected_clv) > 0.5 else "wait",
            "steam_flag":             abs(expected_clv) > 1.0,
        }

    def batch_expected_clv(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute expected CLV for a batch of prop snapshots.

        Parameters
        ----------
        rows : list of dicts, each containing the keys in LINE_PREDICTOR_FEATURES
               plus a "current_line" key.

        Returns the input rows enriched with CLV fields.
        """
        results = []
        for row in rows:
            feats = np.array(
                [float(row.get(k, 0.0)) for k in self._feature_names],
                dtype=float,
            )
            current = float(row.get("current_line", row.get("opening_line", 0.0)))
            try:
                clv_info = self.compute_expected_clv(current, feats)
            except Exception as exc:
                logger.warning("CLV computation failed for row: %s", exc)
                clv_info = {
                    "predicted_close": math.nan,
                    "expected_clv": math.nan,
                    "optimal_timing": "unknown",
                    "steam_flag": False,
                }
            results.append({**row, **clv_info})
        return results

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model metadata (sklearn checkpoint via joblib)."""
        import joblib  # noqa: PLC0415
        meta = {
            "feature_names": self._feature_names,
            "train_metrics": self._train_metrics,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
        }
        out_dir = Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path) + ".pkl")
        with open(str(path) + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("LinePredictor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LinePredictor":
        """Load a saved LinePredictor."""
        import joblib  # noqa: PLC0415
        meta_path = str(path) + ".meta.json"
        pkl_path  = str(path) + ".pkl"
        obj = cls()
        if Path(pkl_path).exists():
            obj.model = joblib.load(pkl_path)
            obj.trained = True
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            obj._feature_names  = meta.get("feature_names", LINE_PREDICTOR_FEATURES)
            obj._train_metrics  = meta.get("train_metrics", {})
            obj.max_iter        = meta.get("max_iter", 300)
            obj.max_depth       = meta.get("max_depth", 4)
        return obj


def build_line_predictor_features(
    prop: dict[str, Any],
    model_projection: float,
    hours_until_game: float,
) -> np.ndarray:
    """Build feature vector for a single prop snapshot.

    Parameters
    ----------
    prop             : dict with market data (opening_line, odds, book_count, …)
    model_projection : our model's projected value for this prop
    hours_until_game : time until game tip-off in hours

    Returns
    -------
    1D numpy array in LINE_PREDICTOR_FEATURES order.
    """
    opening = float(prop.get("opening_line", prop.get("prop_line", 0.0)))
    return np.array([
        opening,
        model_projection,
        model_projection - opening,        # model_edge_vs_opening
        hours_until_game,
        float(prop.get("public_over_pct", 0.50)),
        float(prop.get("book_count", 1.0)),
        float(prop.get("opening_over_odds", -110)),
        float(prop.get("opening_under_odds", -110)),
        float(prop.get("is_prime_time", 0)),
        float(prop.get("star_player_prop", 0)),
        float(prop.get("injury_key_teammate", 0)),
        float(prop.get("line_move_last_hour", 0.0)),
        float(prop.get("model_confidence", 1.0)),
    ], dtype=float)
