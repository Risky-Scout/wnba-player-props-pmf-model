"""WNBA Rotation Structure Model — Bimodal Minutes Distribution (Enhancement 19).

WNBA-specific rotation insight:
    - 7-8 player rotations (smaller than NBA's 9-10)
    - Starting unit plays Q1 & Q3; second unit takes Q2; Q4 is game-script
    - Minutes distributions are BIMODAL for starters:
        * 65% chance of 33 min (close game)
        * 25% chance of 28 min (comfortable win)
        * 10% chance of 22 min (blowout garbage time)

The current quantile-regression minutes model misses this bimodality.
This rotation model produces game-script-conditioned minute samples that
are marginalized over scenario probabilities to form the correct bimodal
mixture distribution.

Integration:
    minutes_samples = RotationPattern(role="starter").sample_conditional_minutes(
        scenario_probs={"close_game": 0.65, "comfortable_win": 0.25, "blowout": 0.10}
    )
    Then the PMF engine marginalizes stat production over these minutes samples.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# WNBA role taxonomy
ROLES = ("starter", "sixth_woman", "rotation", "bench")

# Per-scenario, per-role minutes distribution parameters
# Calibrated to WNBA 2022-2024 player-game logs
SCENARIOS: dict[str, dict[str, dict[str, float]]] = {
    "close_game": {           # margin ≤ 5 in Q4
        "starter":     {"mean": 33.5, "std": 2.0, "floor": 28.0, "ceil": 40.0},
        "sixth_woman": {"mean": 24.0, "std": 3.0, "floor": 16.0, "ceil": 33.0},
        "rotation":    {"mean": 16.0, "std": 4.0, "floor":  6.0, "ceil": 26.0},
        "bench":       {"mean":  3.0, "std": 2.0, "floor":  0.0, "ceil":  8.0},
    },
    "comfortable_win": {      # margin 10-20 in Q4
        "starter":     {"mean": 28.0, "std": 3.0, "floor": 20.0, "ceil": 36.0},
        "sixth_woman": {"mean": 22.0, "std": 3.0, "floor": 14.0, "ceil": 30.0},
        "rotation":    {"mean": 18.0, "std": 4.0, "floor":  8.0, "ceil": 28.0},
        "bench":       {"mean":  8.0, "std": 4.0, "floor":  0.0, "ceil": 18.0},
    },
    "blowout": {              # margin > 20 in Q4
        "starter":     {"mean": 22.0, "std": 4.0, "floor": 12.0, "ceil": 30.0},
        "sixth_woman": {"mean": 20.0, "std": 4.0, "floor": 10.0, "ceil": 28.0},
        "rotation":    {"mean": 22.0, "std": 5.0, "floor":  8.0, "ceil": 32.0},
        "bench":       {"mean": 16.0, "std": 6.0, "floor":  2.0, "ceil": 30.0},
    },
}

# Default scenario prior (before game-script information)
DEFAULT_SCENARIO_PROBS: dict[str, float] = {
    "close_game":      0.40,
    "comfortable_win": 0.40,
    "blowout":         0.20,
}


@dataclass
class RotationPattern:
    """WNBA rotation template for a team-role combination.

    Parameters
    ----------
    role     : one of ROLES ("starter", "sixth_woman", "rotation", "bench")
    scenarios: optional override of the SCENARIOS table for this team
    """

    role:      str
    scenarios: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: SCENARIOS
    )

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            logger.warning(
                "Unknown role '%s'. Defaulting to 'rotation'.", self.role
            )
            self.role = "rotation"

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample_minutes(
        self,
        scenario: str,
        n_samples: int = 1_000,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample minutes from the scenario-specific distribution."""
        rng = rng or np.random.default_rng()
        dist = self.scenarios.get(scenario, SCENARIOS["comfortable_win"]).get(
            self.role, SCENARIOS["comfortable_win"]["rotation"]
        )
        samples = rng.normal(dist["mean"], dist["std"], n_samples)
        return np.clip(samples, dist["floor"], dist["ceil"])

    def sample_conditional_minutes(
        self,
        scenario_probs: dict[str, float] | None = None,
        n_samples: int = 1_000,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample minutes conditioned on game-script scenario probabilities.

        Produces the CORRECT bimodal mixture distribution by sampling from
        each scenario weighted by its probability.

        Parameters
        ----------
        scenario_probs : {scenario: probability} — defaults to DEFAULT_SCENARIO_PROBS
        n_samples      : total Monte Carlo samples

        Returns
        -------
        1D array of sampled minutes (length ≤ n_samples due to rounding)
        """
        if scenario_probs is None:
            scenario_probs = DEFAULT_SCENARIO_PROBS.copy()

        rng = rng or np.random.default_rng()
        all_samples: list[np.ndarray] = []

        for scenario, prob in scenario_probs.items():
            n = max(1, int(round(prob * n_samples)))
            all_samples.append(self.sample_minutes(scenario, n, rng))

        if all_samples:
            return np.concatenate(all_samples)
        return np.full(n_samples, self.scenarios["comfortable_win"][self.role]["mean"])

    def minutes_pmf(
        self,
        scenario_probs: dict[str, float] | None = None,
        n_samples: int = 10_000,
        bin_width: float = 1.0,
    ) -> dict[float, float]:
        """Compute the bimodal minutes PMF as a histogram.

        Returns {minutes_bin: probability} for use in stat marginalization.
        """
        samples = self.sample_conditional_minutes(scenario_probs, n_samples)
        rounded = np.round(samples / bin_width) * bin_width
        unique, counts = np.unique(rounded, return_counts=True)
        return {float(u): float(c / len(rounded)) for u, c in zip(unique, counts)}

    def percentiles(
        self,
        scenario_probs: dict[str, float] | None = None,
        n_samples: int = 10_000,
    ) -> dict[str, float]:
        """Return key quantiles of the conditional minutes distribution."""
        samples = self.sample_conditional_minutes(scenario_probs, n_samples)
        return {
            "q10": float(np.percentile(samples, 10)),
            "q25": float(np.percentile(samples, 25)),
            "q50": float(np.percentile(samples, 50)),
            "q75": float(np.percentile(samples, 75)),
            "q90": float(np.percentile(samples, 90)),
            "mean": float(np.mean(samples)),
            "std":  float(np.std(samples)),
        }


# ── Scenario probability estimation ──────────────────────────────────────────

def estimate_scenario_probs(
    pregame_win_prob: float,
    blowout_prob: float,
) -> dict[str, float]:
    """Estimate game-script scenario probabilities from pregame model outputs.

    Parameters
    ----------
    pregame_win_prob : P(home team wins) from game-script model
    blowout_prob     : P(margin > 20 by Q4) from game-script model

    Returns
    -------
    {scenario: probability} summing to 1.0
    """
    # Blowout: applies to either team winning big
    p_blowout = float(np.clip(blowout_prob, 0.0, 1.0))
    # Close game: roughly when win prob is near 50%
    closeness = 1.0 - 2.0 * abs(pregame_win_prob - 0.50)
    p_close = float(np.clip(closeness * (1 - p_blowout), 0.0, 1.0))
    p_comfortable = float(np.clip(1.0 - p_blowout - p_close, 0.0, 1.0))

    total = p_blowout + p_close + p_comfortable
    if total == 0:
        return DEFAULT_SCENARIO_PROBS.copy()

    return {
        "close_game":      round(p_close / total, 4),
        "comfortable_win": round(p_comfortable / total, 4),
        "blowout":         round(p_blowout / total, 4),
    }


def classify_role(projected_minutes: float) -> str:
    """Classify a player's role based on projected minutes."""
    if projected_minutes >= 28:
        return "starter"
    elif projected_minutes >= 20:
        return "sixth_woman"
    elif projected_minutes >= 10:
        return "rotation"
    else:
        return "bench"


def build_rotation_minutes_samples(
    player_features: dict[str, Any],
    n_samples: int = 10_000,
) -> np.ndarray:
    """Build minutes samples for one player using the rotation model.

    Parameters
    ----------
    player_features : dict with:
        projected_minutes    — base minutes projection
        pregame_win_prob     — from game-script model (default 0.50)
        blowout_prob         — from game-script model (default 0.15)
        player_role          — override role classification

    Returns
    -------
    Array of sampled minutes (length = n_samples).
    """
    proj_min     = float(player_features.get("projected_minutes", 20.0))
    win_prob     = float(player_features.get("pregame_win_probability", 0.50))
    blowout_prob = float(player_features.get("blowout_probability", 0.15))

    role = player_features.get("player_role") or classify_role(proj_min)
    scenario_probs = estimate_scenario_probs(win_prob, blowout_prob)

    pattern = RotationPattern(role=role)
    samples = pattern.sample_conditional_minutes(scenario_probs, n_samples)

    # Anchor samples around the model's projected minutes
    # (blend: 70% rotation model, 30% quantile model projection)
    anchor = proj_min
    blend = 0.70
    blended = blend * samples + (1 - blend) * anchor
    return np.clip(blended, 0.0, 40.0)
