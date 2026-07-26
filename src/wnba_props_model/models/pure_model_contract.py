"""Pure-model information contract (PHASE 0 of the pure-model-supremacy mission).

A *pure* count model is forbidden from consuming ANY current-game market information as a
predictive input or blend. The sportsbook line is permitted ONLY as a query threshold applied
to an independently generated PMF (to compute P(X>line), P(X<line), P(X=line), push-conditioned
P(Over)) -- never as a feature, prior, offset, or blend weight.

This module is the single fail-closed enforcement point:

  * ``FORBIDDEN_MARKET_INPUT_FIELDS`` / ``FORBIDDEN_MARKET_PREFIXES`` -- market-derived columns
    that may never be predictive inputs to a pure model or a pure combo joint.
  * ``PURE_ZERO_WEIGHT_CONFIG_KEYS`` -- config knobs that must be exactly 0.0 for a pure model
    (market prior blend, market probability weight, ...).
  * ``PURE_DISABLED_CONFIG_FLAGS`` -- market-aware nudges that must be OFF for a pure model
    (CLV head, live market calibrators, ...).
  * ``enforce_pure_model_config`` -- returns a normalized pure cfg (forces the above) and stamps
    ``pure_model=True`` / ``market_probability_weight=0.0``.
  * ``assert_pure_model_config`` -- raises ``MarketLeakageError`` if a cfg claims to be pure but
    still carries a nonzero market weight / enabled market nudge.
  * ``assert_pure_feature_columns`` -- raises if a training/inference feature frame carries a
    forbidden market column.
  * ``NOT_PURE_CANDIDATES`` -- diagnostic-only market blends (C4/C5/C6) that are explicitly
    NOT_PURE_MODEL_EVIDENCE / NOT_SUPREMACY_ELIGIBLE / NOT_EDGE_BOARD_ELIGIBLE.

A separate market-aware sensitivity candidate may use at most 15% market weight but is never
supremacy- or edge-board-eligible (see ``MAX_SENSITIVITY_MARKET_WEIGHT``).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Current-game market-derived fields that must NEVER be predictive inputs to a pure model.
FORBIDDEN_MARKET_INPUT_FIELDS: frozenset[str] = frozenset({
    "market_prob_over", "market_prob_under", "market_prob_over_no_vig",
    "market_prob_under_no_vig", "market_prob_over_final",
    "over_odds", "under_odds", "odds_over", "odds_under",
    "consensus_odds", "opening_odds", "closing_odds", "closing_line", "opening_line",
    "line_movement", "line_move", "clv", "closing_line_value",
    "game_total", "game_spread_home", "game_spread", "implied_team_total",
    "implied_total", "vegas_total", "vegas_spread",
    "predicted_spread", "blowout_prob", "close_game_prob", "is_blowout", "is_close_game",
    "market_line", "prop_line", "line",
})
# Prefixes covering families of market-derived columns (e.g. player_market_p_over_prev).
FORBIDDEN_MARKET_PREFIXES: tuple[str, ...] = (
    "player_market_", "market_", "vegas_", "book_", "sportsbook_", "consensus_",
    "closing_", "opening_", "implied_", "clv_", "odds_",
)
# Substrings that make a column market-derived regardless of exact name.
FORBIDDEN_MARKET_SUBSTRINGS: tuple[str, ...] = ("no_vig", "novig", "_odds", "vig")

# Config knobs that must be exactly 0.0 for a pure model.
PURE_ZERO_WEIGHT_CONFIG_KEYS: tuple[str, ...] = (
    "market_prior_lambda", "market_prior_lambda_display", "market_probability_weight",
    "market_blend_weight", "market_weight",
)
# Market-aware nudges that must be disabled for a pure model.
PURE_DISABLED_CONFIG_FLAGS: tuple[str, ...] = (
    "use_clv_head", "use_live_calibrators", "use_market_prior", "use_market_blend",
)

# Diagnostic-only market blends: never pure-model evidence, never supremacy/edge eligible.
NOT_PURE_CANDIDATES: frozenset[str] = frozenset({
    "C4_blend", "C5_role_blend", "C6_market_residual",
})
NOT_PURE_STATUS = "NOT_PURE_MODEL_EVIDENCE / NOT_SUPREMACY_ELIGIBLE / NOT_EDGE_BOARD_ELIGIBLE"

# A market-aware sensitivity candidate may use at most this much market weight (never eligible).
MAX_SENSITIVITY_MARKET_WEIGHT = 0.15


class MarketLeakageError(ValueError):
    """Raised when a pure model/candidate touches forbidden market information."""


def is_forbidden_market_field(name: str) -> bool:
    """True if ``name`` is a market-derived field forbidden as a pure predictive input.

    The line-threshold family (``line`` / ``prop_line`` / ``market_line``) is included: those
    are permitted ONLY post-PMF as a query threshold, never as a predictive feature.
    """
    n = str(name).strip().lower()
    if n in FORBIDDEN_MARKET_INPUT_FIELDS:
        return True
    if any(n.startswith(p) for p in FORBIDDEN_MARKET_PREFIXES):
        return True
    if any(s in n for s in FORBIDDEN_MARKET_SUBSTRINGS):
        return True
    return False


def forbidden_market_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if is_forbidden_market_field(c)]


def assert_pure_feature_columns(columns: Iterable[str], *, context: str = "pure_model") -> None:
    """Fail closed if any predictive feature column is market-derived."""
    bad = forbidden_market_columns(columns)
    if bad:
        raise MarketLeakageError(
            f"{context}: forbidden market-derived predictive input column(s): {sorted(bad)}. "
            "Pure models may use the line ONLY as a post-PMF query threshold.")


def is_pure_model(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("pure_model", False))


def enforce_pure_model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` normalized to the pure-model contract.

    Forces every market weight to 0.0, disables every market-aware nudge, and stamps
    ``pure_model=True`` / ``market_probability_weight=0.0``. Idempotent.
    """
    out = dict(cfg)
    for k in PURE_ZERO_WEIGHT_CONFIG_KEYS:
        out[k] = 0.0
    for k in PURE_DISABLED_CONFIG_FLAGS:
        out[k] = False
    out["pure_model"] = True
    out["market_probability_weight"] = 0.0
    return out


def assert_pure_model_config(cfg: dict[str, Any], *, context: str = "pure_model") -> None:
    """Fail closed if a cfg is marked pure but still carries market weight / enabled nudges."""
    if not is_pure_model(cfg):
        return
    violations = []
    for k in PURE_ZERO_WEIGHT_CONFIG_KEYS:
        v = cfg.get(k, 0.0)
        if v is not None and float(v) != 0.0:
            violations.append(f"{k}={v}")
    for k in PURE_DISABLED_CONFIG_FLAGS:
        if bool(cfg.get(k, False)):
            violations.append(f"{k}=True")
    if float(cfg.get("market_probability_weight", 0.0) or 0.0) != 0.0:
        violations.append(f"market_probability_weight={cfg.get('market_probability_weight')}")
    if violations:
        raise MarketLeakageError(
            f"{context}: config is marked pure_model but violates the pure contract: {violations}")
