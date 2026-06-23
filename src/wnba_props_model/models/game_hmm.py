"""Regime-Aware Hidden Markov Model for Game Momentum (Enhancement 22).

Research (Weimer et al., 2023; Calvo et al., 2023) shows that causal
momentum exists after structural game interruptions (TV timeouts, etc.)
and that HMMs can meaningfully capture latent performance regimes.

Latent states (4):
    0 — Normal competitive play     (base production rates)
    1 — High-scoring run            (+15% pts, +10% ast)
    2 — Defensive struggle          (-15% pts, +10% reb)
    3 — Garbage time                (-20% pts, bench players elevated)

The HMM is trained on historical game sequences and fitted with
hmmlearn's GaussianHMM.

Integration with LiveEngine:
    Inside LiveEngine.compute_live_projection():
        state_id, state_name, state_prob = hmm.infer_current_state(recent_events)
        adjustments = hmm.get_adjustment_factors(state_id)
        # Blend: state_prob * adjusted_rate + (1-state_prob) * base_rate
        post_rate_final = state_prob * post_rate * adj + (1-state_prob) * post_rate

Reference:
    Weimer, Steinert-Threlkeld & Coltin (2023). A causal approach for
    detecting team-level momentum in NBA games.
    Journal of Sports Analytics. https://content.iospress.com/...
    Calvo, Armero & Spezia (2023). Can the hot hand phenomenon be modelled?
    A Bayesian hidden Markov approach. Computational Statistics.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── State definitions ─────────────────────────────────────────────────────────

N_STATES = 4
STATE_NAMES = ["normal", "high_scoring", "defensive", "garbage"]

# Per-state production rate adjustment factors
# (multiplied against the Gamma-Poisson posterior per-minute rate)
DEFAULT_EMISSIONS: dict[str, dict[str, float]] = {
    "normal": {
        "pts_adj": 1.00, "reb_adj": 1.00, "ast_adj": 1.00,
        "stl_adj": 1.00, "blk_adj": 1.00, "turnover_adj": 1.00,
        "fg3m_adj": 1.00,
    },
    "high_scoring": {
        "pts_adj": 1.15, "reb_adj": 0.95, "ast_adj": 1.10,
        "stl_adj": 0.95, "blk_adj": 0.90, "turnover_adj": 0.90,
        "fg3m_adj": 1.12,
    },
    "defensive": {
        "pts_adj": 0.85, "reb_adj": 1.10, "ast_adj": 0.90,
        "stl_adj": 1.15, "blk_adj": 1.20, "turnover_adj": 1.15,
        "fg3m_adj": 0.80,
    },
    "garbage": {
        "pts_adj": 0.80, "reb_adj": 1.05, "ast_adj": 0.85,
        "stl_adj": 1.10, "blk_adj": 1.05, "turnover_adj": 1.10,
        "fg3m_adj": 0.75,
    },
}

# WNBA-calibrated transition matrix (row = from, col = to)
# Inertia is strong: ~90% chance of staying in the same state
DEFAULT_TRANSITION = np.array([
    [0.90, 0.05, 0.03, 0.02],   # normal → ...
    [0.15, 0.75, 0.05, 0.05],   # high_scoring → ...
    [0.20, 0.05, 0.70, 0.05],   # defensive → ...
    [0.05, 0.05, 0.05, 0.85],   # garbage → ...
])


class GameRegimeHMM:
    """Regime-aware HMM for within-game momentum / state detection.

    Uses hmmlearn's GaussianHMM when trained on historical data.
    Falls back to rule-based state inference when untrained (cold-start).

    Parameters
    ----------
    n_states : number of latent states (default: 4)
    """

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.state_names = STATE_NAMES[:n_states]
        self.trained: bool = False
        self._model = None
        self._transition = DEFAULT_TRANSITION[:n_states, :n_states].copy()

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, sequences: list[np.ndarray]) -> "GameRegimeHMM":
        """Train HMM on historical game sequences.

        Parameters
        ----------
        sequences : list of 2D arrays, each shape (n_possessions, n_features)
            Features per possession (suggested):
                [cumulative_pts_rate, cumulative_reb_rate,
                 margin, possession_number_frac, timeout_flag]
        """
        try:
            from hmmlearn import hmm as hmmlib  # noqa: PLC0415

            X = np.concatenate(sequences, axis=0)
            lengths = [len(s) for s in sequences]

            model = hmmlib.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            model.fit(X, lengths)
            self._model = model
            self.trained = True
            self._transition = model.transmat_.copy()
            logger.info(
                "GameRegimeHMM trained on %d sequences (%d total possessions).",
                len(sequences), len(X),
            )
        except ImportError:
            logger.warning(
                "hmmlearn not installed. GameRegimeHMM will use rule-based fallback. "
                "Install with: pip install hmmlearn"
            )
        except Exception as exc:
            logger.warning("HMM training failed: %s — using fallback.", exc)

        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def infer_current_state(
        self,
        recent_events: list[dict[str, Any]] | np.ndarray,
        n_events: int = 20,
    ) -> tuple[int, str, float]:
        """Infer the current game regime from recent PBP events.

        Parameters
        ----------
        recent_events : list of event dicts or 2D array (n_events, n_features)
        n_events      : how many recent events to use for inference

        Returns
        -------
        (state_id, state_name, state_probability)
        """
        if self.trained and self._model is not None:
            return self._hmmlearn_infer(recent_events, n_events)
        return self._rule_based_infer(recent_events)

    def _hmmlearn_infer(
        self,
        recent_events: list | np.ndarray,
        n_events: int,
    ) -> tuple[int, str, float]:
        try:
            if isinstance(recent_events, list):
                X = self._events_to_array(recent_events[-n_events:])
            else:
                X = np.atleast_2d(recent_events[-n_events:])
            if X.shape[0] < 2:
                return 0, "normal", 1.0
            _, post_probs = self._model.score_samples(X)
            current_state = int(np.argmax(post_probs[-1]))
            state_prob = float(post_probs[-1, current_state])
            name = self.state_names[current_state] if current_state < len(self.state_names) else "normal"
            return current_state, name, state_prob
        except Exception as exc:
            logger.debug("HMM inference error: %s — using fallback.", exc)
            return self._rule_based_infer(recent_events)

    def _rule_based_infer(
        self,
        recent_events: list | np.ndarray,
    ) -> tuple[int, str, float]:
        """Simple rule-based state inference for cold-start / no-train scenario."""
        if not recent_events or (isinstance(recent_events, np.ndarray) and recent_events.size == 0):
            return 0, "normal", 1.0

        if isinstance(recent_events, list) and recent_events:
            last = recent_events[-1] if isinstance(recent_events[-1], dict) else {}
            margin = abs(float(last.get("margin", 0)))
            clock_remaining = float(last.get("clock_remaining_secs", 300))
            recent_pts_rate = float(last.get("recent_pts_rate", 1.0))
        else:
            return 0, "normal", 1.0

        # Garbage time: big margin late
        if margin > 20 and clock_remaining < 300:
            return 3, "garbage", 0.80
        # High scoring run: elevated recent pts rate
        if recent_pts_rate > 1.20:
            return 1, "high_scoring", 0.70
        # Defensive struggle: depressed pts rate
        if recent_pts_rate < 0.80:
            return 2, "defensive", 0.70
        return 0, "normal", 0.90

    def get_adjustment_factors(self, state_id: int) -> dict[str, float]:
        """Return stat-level production rate adjustment factors for a regime.

        These factors multiply the Gamma-Poisson posterior rate inside
        LiveEngine.compute_live_projection().
        """
        state_name = (
            self.state_names[state_id]
            if state_id < len(self.state_names)
            else "normal"
        )
        return DEFAULT_EMISSIONS.get(state_name, DEFAULT_EMISSIONS["normal"])

    def adjust_live_rate(
        self,
        stat: str,
        base_rate: float,
        recent_events: list[dict[str, Any]],
        blend_weight: float = 0.70,
    ) -> tuple[float, str, float]:
        """Apply regime adjustment to a live per-minute rate.

        Parameters
        ----------
        stat          : stat name (e.g. "pts", "reb")
        base_rate     : Gamma-Poisson posterior per-minute rate
        recent_events : recent PBP event list
        blend_weight  : weight on adjusted rate vs base (default 0.7)

        Returns
        -------
        (adjusted_rate, state_name, state_probability)
        """
        state_id, state_name, state_prob = self.infer_current_state(recent_events)
        adjs = self.get_adjustment_factors(state_id)
        adj_factor = adjs.get(f"{stat}_adj", 1.0)

        adjusted = base_rate * adj_factor
        # Blend: state_prob controls how much we trust the regime signal
        blended = blend_weight * state_prob * adjusted + (1 - blend_weight * state_prob) * base_rate
        return float(blended), state_name, float(state_prob)

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _events_to_array(events: list[dict[str, Any]]) -> np.ndarray:
        """Convert list of PBP event dicts to numpy feature array."""
        rows = []
        for e in events:
            rows.append([
                float(e.get("cumulative_pts_rate", 1.0)),
                float(e.get("cumulative_reb_rate", 0.5)),
                float(e.get("margin", 0)),
                float(e.get("possession_frac", 0.5)),
                float(e.get("timeout_flag", 0)),
            ])
        return np.array(rows, dtype=float)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_states":   self.n_states,
                    "trained":    self.trained,
                    "model":      self._model,
                    "transition": self._transition,
                },
                f,
            )
        logger.info("GameRegimeHMM saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GameRegimeHMM":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n_states=data.get("n_states", N_STATES))
        obj.trained     = data.get("trained", False)
        obj._model      = data.get("model")
        obj._transition = data.get("transition", DEFAULT_TRANSITION)
        return obj
