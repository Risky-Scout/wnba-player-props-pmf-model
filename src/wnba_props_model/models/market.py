from __future__ import annotations

import logging
import math
from typing import Mapping

import numpy as np
from scipy import optimize, stats

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


def shin_no_vig_two_way_with_z(
    over_odds: float | int | None,
    under_odds: float | int | None,
) -> tuple[float | None, float | None, float | None]:
    """Shin's method returning (p_over, p_under, z).

    z is Shin's informed-bettor fraction parameter.  Low z = soft market =
    higher confidence in model edge.  Typical range: 0.01 to 0.08.

    Returns (None, None, None) if odds are missing or Shin fails.
    """
    if over_odds is None or under_odds is None:
        return None, None, None
    try:
        from penaltyblog.implied import calculate_implied  # type: ignore[import]
        from penaltyblog.implied.models import ImpliedMethod, OddsFormat  # type: ignore[import]

        result = calculate_implied(
            [float(over_odds), float(under_odds)],
            method=ImpliedMethod.SHIN,
            odds_format=OddsFormat.AMERICAN,
        )
        probs = result.probabilities
        z_param = None
        if hasattr(result, "method_params") and result.method_params:
            z_param = result.method_params.get("z")
        if len(probs) >= 2 and all(math.isfinite(p) for p in probs):
            return float(probs[0]), float(probs[1]), (float(z_param) if z_param is not None else None)
    except Exception as exc:  # pragma: no cover
        logger.debug("shin_no_vig_two_way_with_z failed (%s)", exc)
    # Fallback: multiplicative, z=None
    p_o, p_u = _multiplicative_no_vig(over_odds, under_odds)
    return p_o, p_u, None


def shin_no_vig_two_way(
    over_odds: float | int | None,
    under_odds: float | int | None,
) -> tuple[float | None, float | None]:
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


# ---------------------------------------------------------------------------
# Multi-method implied probability comparison (F5)
# ---------------------------------------------------------------------------

_PB_IMPLIED_METHODS = ["multiplicative", "additive", "power", "shin",
                        "differential_margin_weighting", "odds_ratio", "logarithmic"]


def implied_probs_all_methods(
    over_odds: float | int | None,
    under_odds: float | int | None,
) -> dict[str, tuple[float | None, float | None]]:
    """Compute no-vig implied probabilities using all available PenaltyBlog methods.

    Returns a dict: {method_name: (p_over, p_under)}
    Methods that fail gracefully return (None, None).
    """
    if over_odds is None or under_odds is None:
        return {m: (None, None) for m in _PB_IMPLIED_METHODS}

    results: dict[str, tuple[float | None, float | None]] = {}

    try:
        from penaltyblog.implied import calculate_implied  # noqa: PLC0415
        from penaltyblog.implied.models import OddsFormat  # noqa: PLC0415

        for method_name in _PB_IMPLIED_METHODS:
            try:
                result = calculate_implied(
                    [float(over_odds), float(under_odds)],
                    method=method_name,
                    odds_format=OddsFormat.AMERICAN,
                )
                probs = result.probabilities
                if len(probs) >= 2 and all(math.isfinite(p) for p in probs):
                    results[method_name] = (float(probs[0]), float(probs[1]))
                else:
                    results[method_name] = (None, None)
            except Exception:
                results[method_name] = (None, None)
    except ImportError:
        # penaltyblog unavailable — use multiplicative fallback
        p_o, p_u = _multiplicative_no_vig(over_odds, under_odds)
        for m in _PB_IMPLIED_METHODS:
            results[m] = (p_o, p_u)

    return results


def get_no_vig_prob(
    over_odds: float | int | None,
    under_odds: float | int | None,
    method: str = "shin",
) -> tuple[float | None, float | None]:
    """Get no-vig P(over) and P(under) using the specified method.

    Supported methods: any PenaltyBlog ImpliedMethod name (case-insensitive),
    or 'multiplicative' as a pure-Python fallback.

    Returns (p_over, p_under).  Falls back to Shin on failure.
    """
    if over_odds is None or under_odds is None:
        return None, None

    method_lower = method.lower()

    if method_lower == "shin":
        return shin_no_vig_two_way(over_odds, under_odds)

    if method_lower == "multiplicative":
        return _multiplicative_no_vig(over_odds, under_odds)

    try:
        from penaltyblog.implied import calculate_implied  # noqa: PLC0415
        from penaltyblog.implied.models import OddsFormat  # noqa: PLC0415

        result = calculate_implied(
            [float(over_odds), float(under_odds)],
            method=method_lower,
            odds_format=OddsFormat.AMERICAN,
        )
        probs = result.probabilities
        if len(probs) >= 2 and all(math.isfinite(p) for p in probs):
            return float(probs[0]), float(probs[1])
    except Exception as exc:
        logger.debug("get_no_vig_prob(%s) failed: %s — falling back to Shin", method, exc)

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


# ---------------------------------------------------------------------------
# Kelly Criterion (Phase 4a)
# ---------------------------------------------------------------------------

def kelly_fraction(
    model_prob: float,
    over_odds_american: float | int,
    fractional_kelly: float = 0.25,
) -> float:
    """Full Kelly × fractional_kelly stake fraction.

    f* = (b*p - q) / b, where b = decimal_odds - 1, p = model prob, q = 1-p.

    The fractional multiplier (default 0.25 = quarter Kelly) is standard
    bankroll management: reduces variance while preserving expected value.

    Returns 0.0 if the edge is non-positive (no bet recommended).

    Parameters
    ----------
    model_prob        : model probability for the bet direction
    over_odds_american: American odds for the bet (positive or negative integer)
    fractional_kelly  : Kelly fraction (0.25 = quarter Kelly is recommended)
    """
    o = float(over_odds_american)
    if o > 0:
        decimal_odds = o / 100.0 + 1.0
    elif o < 0:
        decimal_odds = 100.0 / abs(o) + 1.0
    else:
        return 0.0

    b = decimal_odds - 1.0  # net odds per unit wagered
    p = float(model_prob)
    q = 1.0 - p

    full_kelly = (b * p - q) / b
    return float(max(0.0, full_kelly * fractional_kelly))


def kelly_from_edge_and_prob(
    edge: float,
    model_prob: float,
    fractional_kelly: float = 0.25,
) -> float:
    """Compute Kelly fraction from raw edge and model probability.

    edge = model_prob_over - market_prob_over_no_vig

    The market's fair decimal odds are inferred from market_prob = 1 - edge:
        b ≈ 1/market_prob - 1

    This avoids needing raw American odds when only edge is available.
    """
    market_prob = float(model_prob) - float(edge)
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1.0 / market_prob) - 1.0
    p = float(model_prob)
    q = 1.0 - p
    full_kelly = (b * p - q) / b
    return float(max(0.0, full_kelly * fractional_kelly))


# ---------------------------------------------------------------------------
# Market-implied Poisson mean (Phase 4b)
# ---------------------------------------------------------------------------

def market_implied_mean(
    line: float,
    market_prob_over: float,
    max_k: int = 150,
    stat: str | None = None,
) -> float | None:
    """Numerically invert Poisson CDF to find market-implied λ.

    Solves: P(Y > line; λ) = market_prob_over for λ.

    This gives the Poisson mean the market is pricing.  Compare to the
    model's predicted mean: a large discrepancy (|model_λ - market_λ| > 2)
    signals a structural disagreement worth investigating.

    Returns None if the inversion fails or the inputs are invalid.

    Parameters
    ----------
    line             : the prop line (e.g. 17.5)
    market_prob_over : market's no-vig P(Y > line) from Shin extraction
    max_k            : upper cap on the Poisson support for numerical CDF
    stat             : optional stat name for logging context
    """
    if not (0.0 < market_prob_over < 1.0):
        return None
    if line < 0:
        return None

    def objective(lam: float) -> float:
        """Return P(Y > line; λ) - target."""
        if lam <= 0:
            return -market_prob_over
        # P(Y > line; λ) = 1 - CDF(floor(line); λ)
        k_floor = int(math.floor(line))
        p_over  = 1.0 - float(stats.poisson.cdf(k_floor, lam))
        return p_over - market_prob_over

    # Bracket: at λ=0, P(Y > line) ≈ 0.  At λ = large, P(Y > line) ≈ 1.
    lo, hi = 0.01, 100.0
    try:
        result = optimize.brentq(objective, lo, hi, xtol=1e-4, maxiter=100)
        return float(result)
    except ValueError:
        logger.debug("[market_implied_mean] Brentq failed for %s line=%.1f p_over=%.3f",
                     stat or "?", line, market_prob_over)
        return None
