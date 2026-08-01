"""Executable-price and EV helpers for the pick engine."""

from __future__ import annotations

import math


def american_to_decimal(american: float | int) -> float:
    """Convert American odds to decimal odds (total return per 1 unit stake)."""
    a = float(american)
    if not math.isfinite(a):
        raise ValueError(f"non-finite American odds: {american!r}")
    if a >= 100.0:
        return 1.0 + a / 100.0
    if a <= -100.0:
        return 1.0 + 100.0 / abs(a)
    raise ValueError(f"invalid American odds: {american!r}")


def break_even_probability(decimal_odds: float) -> float:
    """Sportsbook break-even win probability: 1 / decimal_odds."""
    d = float(decimal_odds)
    if not math.isfinite(d) or d <= 1.0:
        raise ValueError(f"invalid decimal odds: {decimal_odds!r}")
    return 1.0 / d


def is_integer_line(line: float, tol: float = 1e-9) -> bool:
    return abs(float(line) - round(float(line))) <= tol


def expected_value(
    *,
    p_win: float,
    p_lose: float,
    p_push: float,
    decimal_odds: float,
) -> float:
    """Unit-stake EV with explicit pushes.

    EV = p_win * (decimal_odds - 1) - p_lose
    Push returns the stake and contributes zero P/L.
    """
    for name, v in (("p_win", p_win), ("p_lose", p_lose), ("p_push", p_push)):
        if not math.isfinite(v) or v < -1e-12:
            raise ValueError(f"{name} must be finite and nonnegative; got {v!r}")
    if abs((p_win + p_lose + p_push) - 1.0) > 1e-6:
        raise ValueError(
            f"p_win+p_lose+p_push must equal 1; got {p_win + p_lose + p_push}"
        )
    d = float(decimal_odds)
    if not math.isfinite(d) or d <= 1.0:
        raise ValueError(f"invalid decimal odds: {decimal_odds!r}")
    return float(p_win) * (d - 1.0) - float(p_lose)


def side_settlement_probs(
    *,
    side: str,
    p_over_unc: float,
    p_under_unc: float,
    p_push: float,
    line: float,
) -> tuple[float, float, float]:
    """Return (p_win, p_lose, p_push) for an executable side.

    Half-point lines force p_push = 0.
    """
    s = str(side).strip().lower()
    if s not in {"over", "under"}:
        raise ValueError(f"side must be over/under; got {side!r}")
    push = 0.0 if not is_integer_line(line) else float(p_push)
    if s == "over":
        return float(p_over_unc), float(p_under_unc), push
    return float(p_under_unc), float(p_over_unc), push
