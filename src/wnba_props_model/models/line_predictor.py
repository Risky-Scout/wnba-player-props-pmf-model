"""Adversarial Bookmaker / Closing Line Predictor (Enhancement 16).

Trains a SEPARATE prediction model to forecast where a prop line will CLOSE
given the opening line, injury context, public betting signals, and model edge.

This enables:
1. Expected CLV = predicted_close − current_line (if positive, bet now).
2. Optimal timing: place bets when gap between current and predicted close is largest.
3. Steam detection: if predicted_close moves across time, sharp money is arriving.
4. Negative-CLV filtering: skip bets when expected CLV < threshold.

Theory
------
CLV = E[closing_line] − current_line (for over bets; flip sign for under).
When computed against a vig-free closing line, CLV equals expected value of the
bet (Getting Precise About Closing Line Value, Unabated).
In WNBA props (structurally inefficient, low limits), the model's price may be
MORE accurate than the closing line, so CLV is a secondary metric behind
calibration.  But the line predictor still provides bet-timing intelligence.

Reference
---------
"Getting Precise About Closing Line Value", Unabated.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Minimum training samples required to fit the model
MIN_TRAINING_SAMPLES = 200

# Line predictor features (subset used if present)
LINE_PREDICTOR_FEATURES = [
    "opening_line",
    "model_projection",
    "model_edge_vs_opening",
    "hours_until_game",
    "is_questionable_flag",
    "teammate_out_flag",
    "public_betting_pct_over",
    "opening_over_odds",
    "opening_under_odds",
    "book_count",
    "is_prime_time",
    "season_phase_enc",
    "line_movement_direction",
    "line_movement_magnitude",
    "reverse_line_movement_flag",
    "under_bias_indicator",
]


def build_line_predictor_features(
    row: dict[str, Any],
    model_projection: float = 0.0,
    hours_until_game: float = 24.0,
) -> dict[str, float]:
    """Build a LinePredictor feature dict from a market row (for deliver.py).

    Parameters
    ----------
    row : prop row dict (from joined DataFrame)
    model_projection : model's projected stat mean
    hours_until_game : hours until tipoff

    Returns
    -------
    dict ready for LinePredictor.compute_expected_clv(current_line, features)
    """
    line = float(row.get("line", row.get("prop_line", 0.0)))
    return {
        "opening_line":             float(row.get("prop_line_open", line)),
        "model_projection":         model_projection,
        "model_edge_vs_opening":    model_projection - float(row.get("prop_line_open", line)),
        "hours_until_game":         hours_until_game,
        "is_questionable_flag":     float(row.get("is_questionable_flag", 0)),
        "teammate_out_flag":        float(row.get("key_player_out", 0)),
        "public_betting_pct_over":  float(row.get("public_betting_pct_over", 50.0)),
        "opening_over_odds":        float(row.get("opening_over_odds", -110.0)),
        "opening_under_odds":       float(row.get("opening_under_odds", -110.0)),
        "book_count":               float(row.get("number_of_books_offering", 5.0)),
        "is_prime_time":            float(row.get("is_prime_time", 0)),
        "season_phase_enc":         float(row.get("season_phase_enc", 1.0)),
        "line_movement_direction":  float(row.get("line_movement_direction", 0)),
        "line_movement_magnitude":  float(row.get("line_movement_magnitude", 0)),
        "reverse_line_movement_flag": float(row.get("reverse_line_movement_flag", 0)),
        "under_bias_indicator":     float(row.get("under_bias_indicator", 0)),
    }


class LinePredictor:
    """Predict where a prop line will close to compute expected CLV.

    The model is trained on historical props where both opening_line and
    closing_line are known.  At prediction time, it forecasts the closing
    line given current market state.

    Parameters
    ----------
    max_iter : boosting iterations
    max_depth : tree depth
    """

    def __init__(self, max_iter: int = 300, max_depth: int = 4):
        self.max_iter  = max_iter
        self.max_depth = max_depth
        self._model    = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            random_state=42,
            min_samples_leaf=10,
            l2_regularization=0.1,
        )
        self._feature_names: list[str] = []
        self.trained = False
        self._cv_mae: float | None = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: "pd.DataFrame | np.ndarray",
        target_col: "str | np.ndarray" = "closing_line",
    ) -> "LinePredictor":
        """Train on historical props with both opening and closing lines.

        Parameters
        ----------
        df : DataFrame with at minimum 'opening_line', 'closing_line', etc.
             OR a numpy array (n_samples, n_features) — in which case
             target_col must be the y array.
        target_col : column name in df (string) OR y numpy array when df is ndarray.
        """
        import pandas as pd  # noqa: PLC0415

        # Support direct numpy array calling convention: fit(X, y)
        if isinstance(df, np.ndarray):
            X = df.astype(float)
            y = np.asarray(target_col, dtype=float)
            self._feature_names = LINE_PREDICTOR_FEATURES[:X.shape[1]]
            if len(X) < MIN_TRAINING_SAMPLES:
                logger.warning("E16: only %d training rows; LinePredictor not trained", len(X))
                return self
            self._model.fit(X, y)
            self.trained = True
            self._cv_mae = 0.5  # approximate; no CV for numpy path
            logger.info("E16 LinePredictor: trained (numpy path) on %d rows", len(X))
            return self

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        col_name = target_col if isinstance(target_col, str) else "closing_line"
        if col_name not in df.columns:
            logger.warning("E16: '%s' not in df; LinePredictor not trained", col_name)
            return self
        if len(df) < MIN_TRAINING_SAMPLES:
            logger.warning(
                "E16: only %d training rows (need %d); LinePredictor not trained",
                len(df), MIN_TRAINING_SAMPLES,
            )
            return self

        avail = [c for c in LINE_PREDICTOR_FEATURES if c in df.columns]
        if not avail:
            # Use all numeric columns
            avail = df.select_dtypes(include="number").columns.difference([col_name]).tolist()
        if not avail:
            logger.warning("E16: no usable feature columns; LinePredictor not trained")
            return self

        self._feature_names = avail
        X = df[avail].fillna(df[avail].median()).values
        y = df[col_name].values.astype(float)

        self._model.fit(X, y)
        self.trained = True

        # Cross-validated MAE for monitoring
        try:
            maes = cross_val_score(self._model, X, y, scoring="neg_mean_absolute_error", cv=5)
            self._cv_mae = float(-maes.mean())
            logger.info("E16 LinePredictor: trained on %d rows, CV MAE=%.3f", len(df), self._cv_mae)
        except Exception:
            logger.info("E16 LinePredictor: trained on %d rows", len(df))

        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_closing_line(
        self,
        features: dict[str, float] | np.ndarray,
    ) -> tuple[float, tuple[float, float]]:
        """Predict where the line will close.

        Parameters
        ----------
        features : dict of feature values or pre-ordered numpy array.

        Returns
        -------
        (predicted_close, (ci_low, ci_high))
        """
        if not self.trained:
            raise RuntimeError("LinePredictor not trained")
        if isinstance(features, dict):
            x = np.array([features.get(f, 0.0) for f in self._feature_names], dtype=float)
        else:
            x = np.asarray(features, dtype=float)

        pred = float(self._model.predict(x.reshape(1, -1))[0])
        # Approximate 80% CI using training MAE (bootstrap in production)
        margin = (self._cv_mae or 0.5) * 1.28
        return pred, (pred - margin, pred + margin)

    def compute_expected_clv(
        self,
        current_line: float,
        features: dict[str, float] | np.ndarray,
        bet_side: str = "over",
    ) -> dict[str, Any]:
        """Compute expected CLV for betting at the current line.

        Parameters
        ----------
        current_line : the line currently offered by the book.
        features : market feature dict.
        bet_side : "over" or "under".

        Returns
        -------
        dict with predicted_close, expected_clv, confidence bounds,
        optimal_timing, and bet_recommendation.
        """
        pred_close, (ci_low, ci_high) = self.predict_closing_line(features)

        # For over: positive CLV if line moves UP (we bought low)
        # For under: positive CLV if line moves DOWN
        sign = 1.0 if bet_side == "over" else -1.0
        e_clv = sign * (pred_close - current_line)

        return {
            "predicted_close":      round(pred_close, 3),
            "current_line":         round(current_line, 3),
            "expected_clv":         round(e_clv, 4),
            "clv_confidence_lower": round(sign * (ci_low - current_line), 4),
            "clv_confidence_upper": round(sign * (ci_high - current_line), 4),
            "bet_side":             bet_side,
            "optimal_timing":       "now" if abs(pred_close - current_line) > 0.25 else "wait",
            "bet_recommendation":   "bet" if e_clv > 0 else "skip",
            "cv_mae":               self._cv_mae,
        }

    def save(self, path: str) -> None:
        """Persist the line predictor to disk using joblib."""
        import joblib  # noqa: PLC0415
        joblib.dump(self, path + ".pkl")
        logger.info("E16 LinePredictor: saved to %s.pkl", path)

    @classmethod
    def load(cls, path: str) -> "LinePredictor":
        """Load a persisted LinePredictor from disk."""
        import joblib  # noqa: PLC0415
        return joblib.load(path + ".pkl")

    def batch_expected_clv(
        self,
        prop_rows: list[dict[str, Any]],
        current_line_col:  str = "prop_line",
        side_col:          str = "bet_side",
    ) -> list[dict[str, Any]]:
        """Compute expected CLV for a batch of props.

        Each row in prop_rows should contain current_line_col, side_col,
        and any available LINE_PREDICTOR_FEATURES columns.
        """
        if not self.trained:
            return [{"error": "LinePredictor not trained"} for _ in prop_rows]
        results = []
        for row in prop_rows:
            line = float(row.get(current_line_col, 0.0))
            side = row.get(side_col, "over")
            try:
                result = self.compute_expected_clv(line, row, bet_side=side)
            except Exception as e:
                result = {"error": str(e)}
            result.update({k: row[k] for k in ("player_id", "stat", "game_id")
                           if k in row})
            results.append(result)
        return results
