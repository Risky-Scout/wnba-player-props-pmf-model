from __future__ import annotations

import logging
import math
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)


def american_to_prob(odds: float | int | None) -> float | None:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    o = float(odds)
    if o < 0:
        return -o / (-o + 100.0)
    if o > 0:
        return 100.0 / (o + 100.0)
    return None


def _multiplicative_no_vig(over_odds: float | int | None, under_odds: float | int | None) -> tuple[float | None, float | None]:
    """Simple multiplicative no-vig (divide by sum of raw implied probs)."""
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    if po is None or pu is None or po + pu <= 0:
        return None, None
    s = po + pu
    return po / s, pu / s


def shin_no_vig_two_way(over_odds: float | int | None, under_odds: float | int | None) -> tuple[float | None, float | None]:
    """Shin's method for two-way market no-vig extraction.

    Theoretically superior to multiplicative: accounts for favourite-longshot
    bias by modelling a mix of informed and uninformed bettors (Shin 1992).
    Falls back to multiplicative if penaltyblog is unavailable or odds are None.
    """
    if over_odds is None or under_odds is None:
        return None, None
    try:
        from penaltyblog.implied import calculate_implied  # type: ignore[import]
        from penaltyblog.implied.models import ImpliedMethod, OddsFormat  # type: ignore[import]

        result = calculate_implied(
            [float(over_odds), float(under_odds)],
            method=ImpliedMethod.SHIN,
            odds_format=OddsFormat.AMERICAN,
        )
        probs = result.probabilities
        if len(probs) >= 2 and all(math.isfinite(p) for p in probs):
            return float(probs[0]), float(probs[1])
    except Exception as exc:  # pragma: no cover
        logger.warning("Shin no-vig failed (%s); falling back to multiplicative", exc)
    return _multiplicative_no_vig(over_odds, under_odds)


def no_vig_two_way(over_odds: float | int | None, under_odds: float | int | None) -> tuple[float | None, float | None]:
    """Primary no-vig extractor using Shin's method (PenaltyBlog methodology)."""
    return shin_no_vig_two_way(over_odds, under_odds)


def prob_over_from_pmf(pmf: Mapping[int, float] | np.ndarray, line: float) -> float:
    if isinstance(pmf, np.ndarray):
        return float(pmf[np.arange(len(pmf)) > float(line)].sum())
    return float(sum(float(p) for k, p in pmf.items() if int(k) > float(line)))


def fair_american(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def binary_logloss(p: float, y: int) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return float(-(y * math.log(p) + (1 - y) * math.log(1 - p)))


def ignorance_score_binary(p: float, y: int) -> float:
    """Log Loss (Ignorance Score) in bits — PenaltyBlog's recommended primary metric.

    Equivalent to binary_logloss / log(2). Preferred over RPS for binary
    over/under markets: more sample-efficient at identifying better models.
    """
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return float(-(y * math.log2(p) + (1 - y) * math.log2(1 - p)))


def brier(p: float, y: int) -> float:
    return float((float(p) - int(y)) ** 2)
