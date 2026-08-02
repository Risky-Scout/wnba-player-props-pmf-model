"""Public delivery types for the authoritative V6 inference graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PlayerTargetPMF:
    game_id: int
    player_id: int
    player_name: str
    team_id: int
    opponent_id: int
    target: str
    p_active: float
    active_pmf_atoms: list[float]
    overflow_probability: float
    model_probability_over: dict[float, float] = field(default_factory=dict)
    model_probability_under: dict[float, float] = field(default_factory=dict)
    model_probability_push: dict[float, float] = field(default_factory=dict)
    predictive_mean: float = float("nan")
    predictive_variance: float = float("nan")
    calibration_method: str = "identity"
    feature_contract_hash: str = ""
    source: str = "sharp_v6"
    prediction_cutoff: str = ""
    missingness: dict[str, float] = field(default_factory=dict)


@dataclass
class SlatePMFDelivery:
    prediction_timestamp: str
    games: list[dict[str, Any]]
    player_pmfs: list[PlayerTargetPMF]
    atoms_frame: pd.DataFrame
    prices_frame: pd.DataFrame
    participation_frame: pd.DataFrame
    manifest: dict[str, Any]
    combo_frame: pd.DataFrame | None = None
    q1_frame: pd.DataFrame | None = None
    first_basket_frame: pd.DataFrame | None = None
    unsupported: dict[str, str] = field(default_factory=dict)
