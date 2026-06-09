from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def american_to_prob(odds: float | int | None) -> float | None:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    o = float(odds)
    if o < 0:
        return -o / (-o + 100.0)
    if o > 0:
        return 100.0 / (o + 100.0)
    return None


def no_vig_two_way(over_odds: float | int | None, under_odds: float | int | None) -> tuple[float | None, float | None]:
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    if po is None or pu is None or po + pu <= 0:
        return None, None
    s = po + pu
    return po / s, pu / s


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


def brier(p: float, y: int) -> float:
    return float((float(p) - int(y)) ** 2)
