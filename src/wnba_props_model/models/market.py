from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)

# Push-safe settled-probability tolerances (documented, PR 1A B1).
_PMF_SUM_NORMALIZE_TOL = 1e-3   # |sum-1| within this is silently renormalized
_PMF_SUM_MATERIAL_TOL = 1e-3    # |sum-1| beyond this is a material error -> raise
_INTEGER_LINE_TOL = 1e-9        # |line-round(line)| within this counts as an integer line
_PUSH_DEFINED_TOL = 1e-12       # (1 - p_push) must exceed this for settled probs to exist


class UndefinedSettledProbabilityError(ValueError):
    """Raised when settled over/under probabilities are mathematically undefined.

    This happens when the push mass P(Y == line) is effectively one, so the
    denominator (1 - p_push) collapses. The caller must mark the row binary-ineligible
    rather than fabricate a 0.5.
    """


@dataclass(frozen=True)
class SettledProbabilities:
    """Push-safe decomposition of a PMF at a prop line.

    Unconditional probabilities always sum with the push mass to one. Settled
    probabilities condition out the push (integer lines); for half-lines the push is
    zero so settled == unconditional. Settled values are None only when undefined
    (all mass on the push).
    """
    p_over_unconditional: float
    p_under_unconditional: float
    p_push: float
    p_over_settled: float | None
    p_under_settled: float | None


def _pmf_to_dense_array(pmf: "Mapping[int, float] | Sequence[float]") -> np.ndarray:
    """Coerce a PMF (dense sequence or {support: mass} mapping, int or str keys) to a
    dense nonnegative float64 array indexed by integer support, with validation."""
    if isinstance(pmf, Mapping):
        pairs = []
        for k, v in pmf.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                raise ValueError(f"PMF mapping key is not an integer support: {k!r}")
            if ik < 0:
                raise ValueError(f"PMF support must be nonnegative integers; got {ik}")
            pairs.append((ik, float(v)))
        if not pairs:
            raise ValueError("PMF mapping is empty")
        max_k = max(k for k, _ in pairs)
        arr = np.zeros(max_k + 1, dtype=np.float64)
        for k, v in pairs:
            arr[k] += v
    else:
        arr = np.asarray(list(pmf), dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("PMF sequence must be a nonempty 1-D array")

    if not np.all(np.isfinite(arr)):
        raise ValueError("PMF contains non-finite (NaN/inf) values")
    if np.any(arr < 0):
        raise ValueError("PMF contains negative probability mass")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("PMF has non-positive total mass")
    if abs(total - 1.0) > _PMF_SUM_MATERIAL_TOL:
        raise ValueError(f"PMF sum {total:.6f} deviates materially from 1.0")
    if abs(total - 1.0) > 0:
        # Within material tolerance: renormalize the small drift.
        arr = arr / total
    return arr


def settled_probabilities_from_pmf(
    pmf: "Mapping[int, float] | Sequence[float]",
    line: float,
) -> SettledProbabilities:
    """Push-safe over/under/push probabilities for a count PMF at a prop line.

    For any line:
        p_over_unconditional  = P(Y > line)
        p_under_unconditional = P(Y < line)
    Integer line:
        p_push = P(Y == line)
        p_over_settled  = P(Y > line) / (1 - p_push)
        p_under_settled = P(Y < line) / (1 - p_push)
    Half (non-integer) line:
        p_push = 0
        p_over_settled  = p_over_unconditional
        p_under_settled = p_under_unconditional

    Raises ValueError on malformed PMFs or lines, and
    UndefinedSettledProbabilityError when (1 - p_push) collapses.
    """
    if not math.isfinite(line):
        raise ValueError(f"line must be finite; got {line!r}")
    if line < 0:
        raise ValueError(f"line must be nonnegative; got {line!r}")

    arr = _pmf_to_dense_array(pmf)
    k = np.arange(arr.size)
    is_integer_line = abs(line - round(line)) <= _INTEGER_LINE_TOL

    p_over_unc = float(arr[k > line].sum())
    p_under_unc = float(arr[k < line].sum())

    if is_integer_line:
        line_idx = int(round(line))
        p_push = float(arr[line_idx]) if line_idx < arr.size else 0.0
    else:
        p_push = 0.0

    denom = 1.0 - p_push
    if denom <= _PUSH_DEFINED_TOL:
        raise UndefinedSettledProbabilityError(
            f"settled probabilities undefined: p_push={p_push:.12f} at line={line}"
        )
    p_over_settled = p_over_unc / denom
    p_under_settled = p_under_unc / denom

    def _clip01(x: float) -> float:
        # Guard tiny floating error outside [0,1]; a material excursion is a bug.
        if x < -1e-9 or x > 1.0 + 1e-9:
            raise ValueError(f"probability {x} outside [0,1]")
        return float(min(1.0, max(0.0, x)))

    return SettledProbabilities(
        p_over_unconditional=_clip01(p_over_unc),
        p_under_unconditional=_clip01(p_under_unc),
        p_push=_clip01(p_push),
        p_over_settled=_clip01(p_over_settled),
        p_under_settled=_clip01(p_under_settled),
    )


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
        if math.isnan(float(over_odds)) or math.isnan(float(under_odds)):
            return None, None, None
    except (TypeError, ValueError):
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
    logger.debug(
        "shin_no_vig_two_way_with_z: fell back to multiplicative (over=%s, under=%s)",
        over_odds, under_odds,
    )
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
        if math.isnan(float(over_odds)) or math.isnan(float(under_odds)):
            return None, None
    except (TypeError, ValueError):
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
    """DEPRECATED: unconditional P(Y > line). Use settled_probabilities_from_pmf.

    Retained only as a thin backward-compatible wrapper preserving the historical
    UNCONDITIONAL semantics. It must NOT be used for binary sportsbook settlement scoring;
    new sportsbook code must use settled_probabilities_from_pmf(...).p_over_settled, which
    conditions out integer-line push mass (intentionally different when push mass is nonzero).
    """
    import warnings  # noqa: PLC0415
    warnings.warn(
        "prob_over_from_pmf is deprecated; use settled_probabilities_from_pmf(...).p_over_settled "
        "for binary sportsbook scoring (push-safe) or .p_over_unconditional for full-PMF analysis.",
        DeprecationWarning, stacklevel=2,
    )
    return settled_probabilities_from_pmf(pmf, float(line)).p_over_unconditional


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


# ---------------------------------------------------------------------------
# Part 5: Portfolio-Aware Kelly Sizing (Markowitz-Kelly)
# ---------------------------------------------------------------------------

def compute_portfolio_kelly(
    bets: list[dict],
    max_total_exposure: float = 0.15,
) -> list[dict]:
    """Scale individual Kelly fractions for correlated-bet portfolio.

    Applies Markowitz-Kelly: reduce each bet's Kelly fraction so total
    bankroll exposure across correlated bets stays within max_total_exposure.

    Correlation model:
      - Same player_id, different stats → scale factor 0.5 (highly correlated)
      - Same game_id, different players → scale factor 0.8 (weakly correlated)
      - Different games → no adjustment (independent)

    Each input bet dict must contain:
        player_id, stat, game_id, kelly_individual

    Returns the same list with a new field: kelly_portfolio (float | None).
    """
    if not bets:
        return bets

    result = [dict(b) for b in bets]

    # Group by player_id to detect same-player correlated bets
    from collections import defaultdict
    player_bet_counts: dict = defaultdict(int)
    game_bet_counts: dict = defaultdict(int)
    for b in result:
        if b.get("kelly_individual"):
            player_bet_counts[str(b.get("player_id", ""))] += 1
            game_bet_counts[str(b.get("game_id", ""))] += 1

    for b in result:
        kf = b.get("kelly_individual")
        if not kf:
            b["kelly_portfolio"] = None
            continue

        pid = str(b.get("player_id", ""))
        gid = str(b.get("game_id", ""))
        n_player = player_bet_counts.get(pid, 1)
        n_game = game_bet_counts.get(gid, 1)

        # Apply correlation penalty
        scale = 1.0
        if n_player > 1:
            # Multiple bets on same player: 50% scale per additional bet
            scale *= 0.5 ** (n_player - 1)
        elif n_game > 1:
            # Multiple bets in same game (different players): 80% scale
            scale *= 0.8 ** (n_game - 1)

        # Enforce total exposure cap proportionally
        total_raw = sum(
            abs(x.get("kelly_individual") or 0.0) for x in result
        )
        if total_raw > max_total_exposure:
            exposure_scale = max_total_exposure / total_raw
            scale *= exposure_scale

        b["kelly_portfolio"] = round(max(0.0, min(kf * scale, max_total_exposure)), 4)

    return result
