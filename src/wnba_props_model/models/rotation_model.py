"""WNBA Rotation Structure Model — Bimodal Minutes (Enhancement 19).

WNBA rosters are 11-12 players deep; rotations are 7-8 players.
Minutes distributions for starters are distinctly BIMODAL:
  • Close game → starters play 33-35 min
  • Blowout     → starters play 20-25 min (bench absorbs Q4)

The current quantile HGB minutes model produces a UNIMODAL distribution
(e.g., q10=22, q50=28, q90=34), which is systematically mis-calibrated for
props near 25 or 32 minutes.

This module replaces the generic quantile distribution with a
GAME-SCRIPT-CONDITIONED BIMODAL MIXTURE:
  P(minutes) = P(close)   * TruncNormal(μ_close,   σ_close,   floor_close)
             + P(comfort) * TruncNormal(μ_comfort, σ_comfort, floor_comfort)
             + P(blowout) * TruncNormal(μ_blowout, σ_blowout, floor_blowout)

Scenario probabilities come from pregame_win_probability, blowout_probability,
and close_game_probability features (already in the pipeline from Enhancement 6
game-script features).

Role classification (starter / sixth_woman / rotation / bench) is derived from
the player's historical minutes distribution.

Reference
---------
WNBA rotation structure modelled on published team depth charts + empirical
minute-distribution analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum WNBA minutes per game (4 × 10-min quarters)
WNBA_MAX_MINUTES = 40.0

# ── WNBA-calibrated scenario parameters by role ───────────────────────────────

SCENARIOS: dict[str, dict[str, dict[str, float]]] = {
    "close_game": {            # Margin < 5 in Q4 → starters stay on
        "starter":     {"mean": 33.5, "std": 2.0, "floor": 28.0},
        "sixth_woman": {"mean": 24.0, "std": 3.0, "floor": 18.0},
        "rotation":    {"mean": 16.0, "std": 4.0, "floor":  8.0},
        "bench":       {"mean":  3.0, "std": 2.0, "floor":  0.0},
    },
    "comfortable_win": {       # Margin 10–20 in Q4 → some starters rest
        "starter":     {"mean": 28.0, "std": 3.0, "floor": 22.0},
        "sixth_woman": {"mean": 22.0, "std": 3.0, "floor": 16.0},
        "rotation":    {"mean": 18.0, "std": 4.0, "floor": 10.0},
        "bench":       {"mean":  8.0, "std": 4.0, "floor":  0.0},
    },
    "blowout": {               # Margin > 20 in Q4 → bench plays 4th
        "starter":     {"mean": 22.0, "std": 4.0, "floor": 15.0},
        "sixth_woman": {"mean": 20.0, "std": 4.0, "floor": 12.0},
        "rotation":    {"mean": 22.0, "std": 5.0, "floor": 10.0},
        "bench":       {"mean": 16.0, "std": 6.0, "floor":  5.0},
    },
}

# Default scenario probability weights (when no pregame script features available)
DEFAULT_SCENARIO_PROBS = {
    "close_game":      0.35,
    "comfortable_win": 0.40,
    "blowout":         0.25,
}

# Role classification thresholds based on season average minutes
ROLE_MINUTE_THRESHOLDS = {
    "starter":     (28.0, WNBA_MAX_MINUTES),
    "sixth_woman": (20.0, 28.0),
    "rotation":    (10.0, 20.0),
    "bench":       (0.0,  10.0),
}


@dataclass
class RotationPattern:
    """WNBA rotation template for a specific player role."""
    role: str
    player_id: int = 0
    season_avg_minutes: float = 20.0
    historical_std: float = 5.0

    def classify_role(self, avg_minutes: float) -> "RotationPattern":
        """Re-classify role based on season average minutes."""
        for role, (lo, hi) in ROLE_MINUTE_THRESHOLDS.items():
            if lo <= avg_minutes < hi:
                self.role = role
                break
        return self

    def sample_minutes(
        self,
        scenario: str,
        n_samples: int = 1_000,
    ) -> np.ndarray:
        """Sample from a truncated-normal distribution for the given scenario."""
        if scenario not in SCENARIOS:
            scenario = "comfortable_win"
        dist  = SCENARIOS[scenario][self.role]
        mean  = dist["mean"]
        std   = dist["std"]
        floor = dist["floor"]
        samples = np.random.normal(mean, std, n_samples)
        return np.clip(samples, floor, WNBA_MAX_MINUTES)

    def sample_conditional_minutes(
        self,
        scenario_probs: dict[str, float],
        n_samples: int = 1_000,
    ) -> np.ndarray:
        """Sample minutes conditioned on game-script scenario probabilities.

        Produces a BIMODAL distribution by mixing scenario-specific
        truncated-normals weighted by their probability.

        Parameters
        ----------
        scenario_probs : {scenario_name: probability}  (should sum ≈ 1)
        n_samples : total Monte Carlo samples
        """
        total = max(sum(scenario_probs.values()), 1e-9)
        normed = {k: v / total for k, v in scenario_probs.items()}

        all_samples: list[np.ndarray] = []
        for scenario, prob in normed.items():
            n = max(1, int(round(prob * n_samples)))
            all_samples.append(self.sample_minutes(scenario, n))

        combined = np.concatenate(all_samples)
        np.random.shuffle(combined)
        return combined[:n_samples]

    def minutes_pmf(
        self,
        scenario_probs: dict[str, float],
        n_samples: int = 2_000,
    ) -> dict[str, Any]:
        """Return the full minutes PMF summary from the bimodal distribution."""
        samples = self.sample_conditional_minutes(scenario_probs, n_samples)
        return {
            "mean":        float(np.mean(samples)),
            "std":         float(np.std(samples)),
            "q10":         float(np.percentile(samples, 10)),
            "q25":         float(np.percentile(samples, 25)),
            "q50":         float(np.percentile(samples, 50)),
            "q75":         float(np.percentile(samples, 75)),
            "q90":         float(np.percentile(samples, 90)),
            "p_over_30":   float(np.mean(samples > 30)),
            "p_under_25":  float(np.mean(samples < 25)),
            "role":        self.role,
            "bimodal":     True,
        }


# ── Scenario probability inference ───────────────────────────────────────────

def estimate_scenario_probs(
    pregame_win_prob: float | None = None,
    blowout_prob:     float | None = None,
    close_game_prob:  float | None = None,
) -> dict[str, float]:
    """Alias for infer_scenario_probs with alternative parameter names."""
    return infer_scenario_probs(
        pregame_win_prob=pregame_win_prob,
        blowout_probability=blowout_prob,
        close_game_probability=close_game_prob,
    )


def infer_scenario_probs(
    pregame_win_prob:    float | None = None,
    blowout_probability: float | None = None,
    close_game_probability: float | None = None,
) -> dict[str, float]:
    """Derive game-script scenario probabilities from pregame features.

    Falls back to DEFAULT_SCENARIO_PROBS when features are missing.
    """
    if all(v is None for v in [pregame_win_prob, blowout_probability, close_game_probability]):
        return DEFAULT_SCENARIO_PROBS.copy()

    # If blowout/close_game probabilities are directly available
    p_blowout = float(blowout_probability)     if blowout_probability     is not None else 0.25
    p_close   = float(close_game_probability)  if close_game_probability  is not None else 0.35

    # Clamp and derive comfortable-win
    p_blowout  = np.clip(p_blowout, 0.0, 0.80)
    p_close    = np.clip(p_close,   0.0, 0.80)
    p_comfort  = max(0.0, 1.0 - p_blowout - p_close)

    total = p_blowout + p_close + p_comfort
    return {
        "close_game":      p_close   / total,
        "comfortable_win": p_comfort / total,
        "blowout":         p_blowout / total,
    }


# ── Pipeline feature injection ────────────────────────────────────────────────

def add_rotation_minutes_features(
    wide: "pd.DataFrame",
    n_samples: int = 2_000,
) -> "pd.DataFrame":
    """Replace unimodal minutes features with bimodal rotation features.

    Adds to wide:
      rotation_minutes_mean, rotation_minutes_std,
      rotation_minutes_q10/q25/q50/q75/q90,
      rotation_p_over_30, rotation_p_under_25, rotation_role
    """
    import pandas as pd  # noqa: PLC0415

    new_rows: list[dict[str, Any]] = []

    for _, row in wide.iterrows():
        avg_mins = float(row.get("player_minutes_l5",
                                 row.get("player_minutes_season", 20.0)) or 20.0)
        pat = RotationPattern(role="rotation", player_id=int(row.get("player_id", 0)),
                               season_avg_minutes=avg_mins)
        pat.classify_role(avg_mins)

        scenario_probs = infer_scenario_probs(
            pregame_win_prob=row.get("pregame_win_probability"),
            blowout_probability=row.get("blowout_probability"),
            close_game_probability=row.get("close_game_probability"),
        )

        pmf = pat.minutes_pmf(scenario_probs, n_samples=n_samples)
        new_rows.append({f"rotation_minutes_{k}": v for k, v in pmf.items()})

    if new_rows:
        result_df = pd.DataFrame(new_rows, index=wide.index)
        wide = pd.concat([wide, result_df], axis=1)
        logger.info("E19: added rotation minutes features for %d players", len(wide))

    return wide
