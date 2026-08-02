"""Pricing engine: fair odds from a count PMF (push-safe), plus an optional margin layer.

Given any active-player count PMF (probability atoms over 0..K) and a line L this returns
p_over_win / p_under_win / p_push, push-safe settled probabilities, and fair decimal + American
odds. A separate margin layer maps fair probabilities to quoted prices WITHOUT modifying the PMF.
Yes/No and categorical (complete normalized vector) markets are also supported.

All identities:
  p_over_win  = P(Y > L)
  p_under_win = P(Y < L)
  p_push      = P(Y = L)  (only when L is an integer; half-point lines cannot push)
  p_over_settled  = p_over_win  / (p_over_win + p_under_win)
  p_under_settled = p_under_win / (p_over_win + p_under_win)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def _normalize(pmf) -> np.ndarray:
    a = np.asarray(pmf, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("pmf must be a non-empty 1-D array of atom probabilities")
    if not np.all(np.isfinite(a)):
        raise ValueError("pmf contains non-finite values")
    if np.any(a < -1e-9):
        raise ValueError(f"pmf contains negative mass (min={a.min()})")
    a = np.clip(a, 0.0, None)
    s = a.sum()
    if s <= 0:
        raise ValueError("pmf total mass is non-positive")
    return a / s


# ---- odds conversions -------------------------------------------------------------
def prob_to_decimal(p: float) -> float:
    p = min(max(float(p), _EPS), 1 - _EPS)
    return 1.0 / p


def decimal_to_american(d: float) -> float:
    d = float(d)
    if not np.isfinite(d) or d <= 1.0:
        return float("nan")
    return round((d - 1.0) * 100.0) if d >= 2.0 else round(-100.0 / (d - 1.0))


def prob_to_american(p: float) -> float:
    return decimal_to_american(prob_to_decimal(p))


def american_to_prob(american: float) -> float:
    a = float(american)
    if not np.isfinite(a) or (-100 < a < 100):
        return float("nan")
    return (100.0 / (a + 100.0)) if a > 0 else (abs(a) / (abs(a) + 100.0))


@dataclass(frozen=True)
class PricedLine:
    market_key: str
    line: float
    p_over_win: float
    p_under_win: float
    p_push: float
    p_over_settled: float
    p_under_settled: float
    fair_decimal_over: float
    fair_decimal_under: float
    fair_american_over: float
    fair_american_under: float
    margin_method: str = "none"
    quoted_decimal_over: float = float("nan")
    quoted_decimal_under: float = float("nan")
    quoted_american_over: float = float("nan")
    quoted_american_under: float = float("nan")
    pricing_status: str = "PRICED"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def price_over_under(pmf, line: float, market_key: str = "", *, margin_method: str = "none",
                     overround: float = 0.0) -> PricedLine:
    """Price a single Over/Under line from a count PMF. Push-safe."""
    a = _normalize(pmf)
    k = np.arange(a.size)
    L = float(line)
    p_over = float(a[k > L].sum())
    p_under = float(a[k < L].sum())
    p_push = float(a[k == L].sum()) if L.is_integer() else 0.0
    denom = max(p_over + p_under, _EPS)
    p_over_s = p_over / denom
    p_under_s = p_under / denom
    fdo, fdu = prob_to_decimal(p_over_s), prob_to_decimal(p_under_s)
    q_do = q_du = float("nan")
    if margin_method != "none":
        q_over, q_under = apply_margin(p_over_s, p_under_s, method=margin_method, overround=overround)
        q_do, q_du = prob_to_decimal(q_over), prob_to_decimal(q_under)
    return PricedLine(
        market_key=market_key, line=L, p_over_win=p_over, p_under_win=p_under, p_push=p_push,
        p_over_settled=p_over_s, p_under_settled=p_under_s,
        fair_decimal_over=fdo, fair_decimal_under=fdu,
        fair_american_over=decimal_to_american(fdo), fair_american_under=decimal_to_american(fdu),
        margin_method=margin_method,
        quoted_decimal_over=q_do, quoted_decimal_under=q_du,
        quoted_american_over=decimal_to_american(q_do) if margin_method != "none" else float("nan"),
        quoted_american_under=decimal_to_american(q_du) if margin_method != "none" else float("nan"))


def price_alternate_ladder(pmf, lines, market_key: str = "", **kw) -> list[PricedLine]:
    """Price a ladder of (base + alternate) lines from ONE PMF -> monotone by construction."""
    return [price_over_under(pmf, L, market_key, **kw) for L in sorted(float(x) for x in lines)]


def apply_margin(p_over: float, p_under: float, *, method: str = "proportional",
                 overround: float = 0.05) -> tuple[float, float]:
    """Map fair (settled) probabilities to quoted probabilities with a target overround.
    Does NOT touch the PMF. Returns quoted (over, under) probabilities summing to 1+overround."""
    po, pu = float(p_over), float(p_under)
    if method == "proportional":
        scale = (1.0 + overround)
        return po * scale, pu * scale
    if method == "power":
        # power method: q_i ∝ p_i^k solved to hit target overround; approximate via bisection
        target = 1.0 + overround
        lo, hi = 0.5, 1.5
        for _ in range(40):
            mid = (lo + hi) / 2
            s = po ** mid + pu ** mid
            if s > target:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
        return po ** k, pu ** k
    raise ValueError(f"unknown margin method: {method}")


# ---- Yes/No and categorical markets ----------------------------------------------
@dataclass(frozen=True)
class YesNoPrice:
    market_key: str
    p_yes: float
    p_no: float
    fair_decimal_yes: float
    fair_decimal_no: float
    fair_american_yes: float
    fair_american_no: float
    pricing_status: str = "PRICED"


def price_yes_no(p_yes: float, market_key: str = "") -> YesNoPrice:
    py = min(max(float(p_yes), 0.0), 1.0)
    pn = 1.0 - py
    return YesNoPrice(market_key=market_key, p_yes=py, p_no=pn,
                      fair_decimal_yes=prob_to_decimal(py), fair_decimal_no=prob_to_decimal(pn),
                      fair_american_yes=prob_to_american(py), fair_american_no=prob_to_american(pn))


def price_categorical(outcome_probs: dict[str, float], market_key: str = "") -> dict:
    """Complete normalized outcome vector + fair price for every category."""
    items = {k: max(float(v), 0.0) for k, v in outcome_probs.items()}
    s = sum(items.values())
    if s <= 0:
        raise ValueError("categorical probabilities sum to non-positive")
    norm = {k: v / s for k, v in items.items()}
    return {
        "market_key": market_key, "pricing_status": "PRICED",
        "categories": {k: {"probability": p, "fair_decimal": prob_to_decimal(p),
                           "fair_american": prob_to_american(p)} for k, p in norm.items()},
        "normalized_sum": float(sum(norm.values())),
    }
