"""Regime-Aware Hidden Markov Model for Game Momentum (Enhancement 22).

Models within-game momentum as a 4-state HMM:
  State 0 — Normal competitive play   (base rates)
  State 1 — High-scoring run          (+15% pts rate, +10% ast rate)
  State 2 — Defensive struggle        (−15% pts rate, +10% reb rate)
  State 3 — Garbage time              (stars benched, bench elevated)

The Viterbi / forward algorithm infers the current state from recent PBP
observations.  The live engine modulates its Gamma-Poisson posterior rates
using the inferred regime's adjustment factors.

Theory
------
Causal momentum exists after structural interruptions such as TV timeouts
(Weimer et al. 2023).  A Bayesian HMM for hot/cold hand in basketball
shows that latent state models can meaningfully capture performance regimes
(Calvo et al. 2023).

References
----------
Weimer, Steinert-Threlkeld & Coltin (2023). A causal approach for detecting
team-level momentum in NBA games. Journal of Sports Analytics.
Calvo, Armero & Spezia (2023). Can the hot hand phenomenon be modelled?
A Bayesian hidden Markov approach. Computational Statistics.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from hmmlearn import hmm as _hmm_lib
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

N_STATES = 4
STATE_NAMES = ["normal", "high_scoring", "defensive", "garbage"]

# Stat rate adjustment factors per regime
# (multiply the live engine's posterior per-minute rate by these)
DEFAULT_EMISSIONS: dict[str, dict[str, float]] = {
    "normal":       {"pts_adj": 1.00, "reb_adj": 1.00, "ast_adj": 1.00,
                     "stl_adj": 1.00, "blk_adj": 1.00, "turnover_adj": 1.00},
    "high_scoring": {"pts_adj": 1.15, "reb_adj": 0.95, "ast_adj": 1.10,
                     "stl_adj": 0.95, "blk_adj": 0.90, "turnover_adj": 0.90},
    "defensive":    {"pts_adj": 0.85, "reb_adj": 1.10, "ast_adj": 0.90,
                     "stl_adj": 1.10, "blk_adj": 1.15, "turnover_adj": 1.15},
    "garbage":      {"pts_adj": 0.80, "reb_adj": 1.05, "ast_adj": 0.85,
                     "stl_adj": 1.05, "blk_adj": 1.00, "turnover_adj": 1.10},
}

# Default transition matrix (row = from_state, col = to_state)
# Momentum is persistent; regime changes are infrequent
_DEFAULT_TRANSMAT = np.array([
    [0.88, 0.05, 0.05, 0.02],   # normal → ...
    [0.10, 0.82, 0.04, 0.04],   # high_scoring → ...
    [0.10, 0.04, 0.82, 0.04],   # defensive → ...
    [0.05, 0.03, 0.03, 0.89],   # garbage → ...
])

# Default start probabilities (most games start in "normal")
_DEFAULT_STARTPROB = np.array([0.75, 0.10, 0.10, 0.05])


class GameRegimeHMM:
    """Hidden Markov Model for within-game momentum / regime detection.

    Used by the LiveEngine to modulate Gamma-Poisson posterior rates
    based on inferred game state.

    If hmmlearn is available, a full Gaussian HMM is trained on historical
    PBP sequences.  Otherwise, a simplified rule-based state classifier is
    used as a deterministic fallback.

    Parameters
    ----------
    n_states : number of latent states (default 4)
    use_learned_emissions : bool — if True and hmmlearn is available,
        use learned emission parameters from fit(); else use DEFAULT_EMISSIONS.
    """

    def __init__(
        self,
        n_states:               int = N_STATES,
        use_learned_emissions:  bool = True,
    ):
        self.n_states              = n_states
        self.use_learned_emissions = use_learned_emissions
        self.trained               = False

        if _HMMLEARN_AVAILABLE:
            self._model = _hmm_lib.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
                init_params="cm",        # learn covariance + means
                params="cmt",            # fit covariance, means, transitions
            )
            # Warm-start with default transition / start probs
            self._model.startprob_ = _DEFAULT_STARTPROB[:n_states]
            self._model.transmat_  = _DEFAULT_TRANSMAT[:n_states, :n_states]
        else:
            logger.warning(
                "E22: hmmlearn not available; using rule-based regime classifier"
            )
            self._model = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, sequences: list[np.ndarray]) -> "GameRegimeHMM":
        """Train HMM on historical game sequences.

        Parameters
        ----------
        sequences : list of (n_possessions, n_features) arrays.
            Feature vector per possession:
            [cumulative_pts_rate, cumulative_reb_rate, margin,
             possession_number_normalised, timeout_flag]
        """
        if not sequences:
            logger.warning("E22: no sequences provided; HMM not trained")
            return self

        if not _HMMLEARN_AVAILABLE or self._model is None:
            self.trained = False
            return self

        X = np.concatenate(sequences, axis=0)
        lengths = [len(s) for s in sequences]
        try:
            self._model.fit(X, lengths)
            self.trained = True
            logger.info(
                "E22 GameRegimeHMM: fitted on %d possessions across %d games",
                X.shape[0], len(sequences),
            )
        except Exception as e:
            logger.warning("E22: HMM fit failed: %s", e)
            self.trained = False
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer_current_state(
        self,
        recent_events: list[dict[str, Any]] | np.ndarray,
        n_events:      int = 20,
    ) -> tuple[int, str, float]:
        """Infer current game regime from recent play-by-play events.

        Parameters
        ----------
        recent_events : last N events as list of dicts or (N, n_features) array.
            Dict keys: pts_scored, reb, margin, possession_number, timeout_flag.
        n_events : number of recent events to consider.

        Returns
        -------
        (state_id: int, state_name: str, state_probability: float)
        """
        if not self.trained or self._model is None:
            state_id = self._rule_based_state(recent_events)
            return state_id, STATE_NAMES[state_id], 1.0

        X = self._encode_events(recent_events, n_events)
        if X is None or len(X) < 3:
            return 0, "normal", 1.0

        try:
            _, post_probs = self._model.score_samples(X)
            current_state = int(np.argmax(post_probs[-1]))
            state_prob    = float(post_probs[-1, current_state])
            return current_state, STATE_NAMES[current_state], state_prob
        except Exception as e:
            logger.debug("E22 HMM score_samples failed: %s", e)
            return 0, "normal", 1.0

    def get_adjustment_factors(self, state_id: int) -> dict[str, float]:
        """Return production rate multipliers for the inferred regime."""
        if state_id < 0 or state_id >= N_STATES:
            state_id = 0
        return DEFAULT_EMISSIONS.get(STATE_NAMES[state_id], DEFAULT_EMISSIONS["normal"])

    def modulate_rate(
        self,
        base_rate:     float,
        stat:          str,
        state_id:      int,
        state_prob:    float = 1.0,
    ) -> float:
        """Apply regime-aware adjustment to a per-minute rate.

        The blend formula:
            adjusted_rate = state_prob × (base_rate × factor) +
                            (1 − state_prob) × base_rate
                          = base_rate × (1 + state_prob × (factor − 1))

        Partial adjustment weighted by state probability avoids
        overcorrecting when the HMM state estimate is uncertain.
        """
        adj = self.get_adjustment_factors(state_id)
        factor = adj.get(f"{stat}_adj", 1.0)
        blended = base_rate * (1.0 + state_prob * (factor - 1.0))
        return max(0.0, blended)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_events(
        self,
        events: list[dict[str, Any]] | np.ndarray,
        n_events: int,
    ) -> np.ndarray | None:
        """Convert event list to (n, n_features) observation matrix."""
        if isinstance(events, np.ndarray):
            return events[-n_events:].reshape(-1, events.shape[-1] if events.ndim > 1 else 1)

        if not events:
            return None

        rows = []
        for evt in events[-n_events:]:
            rows.append([
                float(evt.get("pts_scored", 0)),
                float(evt.get("reb",        0)),
                float(evt.get("margin",     0)),
                float(evt.get("possession_number", 0)) / 80.0,  # normalise
                float(evt.get("timeout_flag", 0)),
            ])
        return np.array(rows, dtype=float)

    def adjust_live_rate(
        self,
        stat:         str,
        base_rate:    float,
        recent_events: list[dict[str, Any]] | np.ndarray,
        blend_weight: float = 0.70,
    ) -> tuple[float, str, float]:
        """Convenience wrapper used by LiveEngine.

        Parameters
        ----------
        stat          : e.g. "pts", "reb"
        base_rate     : the Gamma-Poisson posterior per-minute rate
        recent_events : recent PBP events for regime inference
        blend_weight  : how much weight to give the regime-adjusted rate
                        (1 − blend_weight goes to the unadjusted base rate)

        Returns
        -------
        (adjusted_rate, state_name, state_probability)
        """
        state_id, state_name, state_prob = self.infer_current_state(recent_events)
        adj_rate = self.modulate_rate(base_rate, stat, state_id, state_prob)
        # Blend with base rate (don't overcorrect)
        blended = blend_weight * adj_rate + (1.0 - blend_weight) * base_rate
        return max(0.0, blended), state_name, state_prob

    def _rule_based_state(self, events: list[dict[str, Any]] | Any) -> int:
        """Deterministic regime classification when HMM is unavailable."""
        if not isinstance(events, list) or not events:
            return 0

        last = events[-1] if events else {}
        margin    = abs(float(last.get("margin",     0)))
        pts_rate  = float(last.get("pts_rate",  last.get("recent_pts_rate",
                         last.get("cumulative_pts_rate", 1.8))))
        poss_norm = float(last.get("possession_frac",
                         last.get("possession_number", 0)) if "possession_frac" in last
                         else last.get("possession_number", 0) / 80.0)

        # Check for clock_remaining_secs to infer late game
        clock_rem = float(last.get("clock_remaining_secs", 9999))
        is_late   = clock_rem < 300 or poss_norm > 0.85  # < 5 min remaining

        if is_late and margin > 15:
            return 3   # garbage time
        if pts_rate > 2.5:
            return 1   # high-scoring
        if pts_rate < 1.5:
            return 2   # defensive struggle
        return 0       # normal
