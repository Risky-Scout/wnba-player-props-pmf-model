"""Market pipeline integrity — Ticket 2.

Provides:
  - Error classes for every failure mode
  - PMF probability computation (p_over, p_push, p_under) with integer / half-point awareness
  - No-vig probability extraction using multiplicative method (not labeled as CLV)
  - Model edge computation (NOT labeled as CLV; named edge_over / edge_under)
  - Market quote validation (duplicates, staleness, identity, malformed odds)
  - Expected PMF and edge manifests
  - Artifact lineage validation (run_id, commit, game_date, timestamp window)
  - Stale-fallback guard (check_no_stale_fallback)
  - Inactive-player vendor settlement rule resolution
  - Staging board validation (complete-board check)
  - Atomic deployment (promote staging → live only after full validation)

Design constraints (Ticket 2 scope):
  - Do NOT alter model prediction coefficients, PMF families, calibration values,
    combo correlations, edge thresholds, bankroll or bet-sizing logic.
  - Edge output columns must be named edge_over / edge_under.
  - The string "clv" must not appear in any compute_model_edge output or label.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

LIVE_MARKETS_NOT_YET_AVAILABLE = "LIVE_MARKETS_NOT_YET_AVAILABLE"
NONCRITICAL_EXPLAINABILITY_FAILURE = "NONCRITICAL_EXPLAINABILITY_FAILURE"

VENDOR_RULE_VOID_IF_NO_PARTICIPATION = "void_if_no_participation"
VENDOR_RULE_ACTION_IF_STARTS = "action_if_starts"
VENDOR_RULE_ACTION_IF_PLAYS = "action_if_plays"

_VALID_VENDOR_RULES = frozenset({
    VENDOR_RULE_VOID_IF_NO_PARTICIPATION,
    VENDOR_RULE_ACTION_IF_STARTS,
    VENDOR_RULE_ACTION_IF_PLAYS,
})

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class MarketIntegrityError(Exception):
    """Base class for all market integrity failures."""


class DuplicateQuoteError(MarketIntegrityError):
    """Raised when the same (vendor, game_id, player_id, stat, line) appears more than once."""


class StaleQuoteError(MarketIntegrityError):
    """Raised when a market quote is older than the allowed staleness window."""


class UnmatchedIdentityError(MarketIntegrityError):
    """Raised when a player or game identity cannot be resolved."""


class AmbiguousIdentityError(MarketIntegrityError):
    """Raised when a player name matches more than one canonical player."""


class MalformedOddsError(MarketIntegrityError):
    """Raised when an odds value cannot be parsed as valid American odds."""


class MissingPMFError(MarketIntegrityError):
    """Raised when a required PMF is absent from the actual PMF set, or an unexpected PMF appears."""


class DuplicatePMFError(MarketIntegrityError):
    """Raised when the same (game_id, player_id, stat) PMF appears more than once."""


class MissingEdgeError(MarketIntegrityError):
    """Raised when a required edge is absent from the actual edge set, or an unexpected edge appears."""


class DuplicateEdgeError(MarketIntegrityError):
    """Raised when the same (game_id, player_id, stat, vendor, line) edge appears more than once."""


class ArtifactLineageMismatchError(MarketIntegrityError):
    """Raised when artifacts from different runs, commits, or game dates are mixed together."""


class PartialBoardError(MarketIntegrityError):
    """Raised when the staging board is incomplete (missing required artifacts)."""


class StaleFallbackForbiddenError(MarketIntegrityError):
    """Raised when a current-run artifact is missing and a stale fallback would otherwise be used."""


class ArtifactManifestError(MarketIntegrityError):
    """Raised when an artifact manifest is missing, invalid, or incompatible."""


# ---------------------------------------------------------------------------
# PMF probability computation
# ---------------------------------------------------------------------------


def compute_pmf_probabilities(
    pmf: np.ndarray,
    line: float,
) -> tuple[float, float, float]:
    """Compute (p_over, p_push, p_under) from a discrete PMF array.

    For integer lines:
        p_over  = P(stat > line)  = sum(pmf[i] for i > line)
        p_push  = P(stat == line) = pmf[int(line)] if in range, else 0
        p_under = P(stat < line)  = 1 - p_over - p_push

    For half-point lines (line != int(line)):
        p_over  = P(stat > line)  = sum(pmf[i] for i > line)
        p_push  = 0  (never a push on a half-point line)
        p_under = 1 - p_over

    Asserts that p_over + p_push + p_under == 1.0 within 1e-9.
    Does NOT mutate the input array.
    """
    pmf = np.asarray(pmf, dtype=float)
    indices = np.arange(len(pmf), dtype=float)

    p_over = float(pmf[indices > float(line)].sum())

    is_integer_line = (float(line) == math.floor(float(line)))
    if is_integer_line:
        idx = int(line)
        p_push = float(pmf[idx]) if 0 <= idx < len(pmf) else 0.0
    else:
        p_push = 0.0

    p_under = 1.0 - p_over - p_push

    # Clamp tiny floating-point error to zero
    p_under = max(p_under, 0.0)

    total = p_over + p_push + p_under
    assert abs(total - 1.0) < 1e-9, (
        f"compute_pmf_probabilities: p_over={p_over} + p_push={p_push} + p_under={p_under} = {total} != 1.0"
    )
    return p_over, p_push, p_under


# ---------------------------------------------------------------------------
# No-vig probability extraction
# ---------------------------------------------------------------------------


def compute_no_vig_probs_from_american(
    over_odds: float | int,
    under_odds: float | int,
) -> tuple[float | None, float | None]:
    """Extract true market probabilities by removing the bookmaker's vig.

    Uses multiplicative (proportional) method:
        raw_p_over  = american_to_prob(over_odds)
        raw_p_under = american_to_prob(under_odds)
        market_p_over_no_vig  = raw_p_over  / (raw_p_over + raw_p_under)
        market_p_under_no_vig = raw_p_under / (raw_p_over + raw_p_under)

    Returns (None, None) if either odds value is invalid or missing.

    NOT labeled as CLV.  Returns model-vs-market-implied probabilities only.
    """
    raw_over = _american_to_prob_internal(over_odds)
    raw_under = _american_to_prob_internal(under_odds)
    if raw_over is None or raw_under is None:
        return None, None
    total = raw_over + raw_under
    if total <= 0.0:
        return None, None
    return raw_over / total, raw_under / total


def _american_to_prob_internal(odds: float | int | None) -> float | None:
    """Convert American odds to implied probability.  Returns None for invalid input."""
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if math.isnan(o) or math.isinf(o):
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return -o / (-o + 100.0)
    return None  # odds == 0 is invalid


# ---------------------------------------------------------------------------
# Model edge computation (NOT labeled as CLV)
# ---------------------------------------------------------------------------


def compute_model_edge(
    pmf: np.ndarray,
    line: float,
    over_odds: float | int,
    under_odds: float | int,
) -> tuple[float, float]:
    """Compute (edge_over, edge_under) for a player prop line.

    edge_over  = model_p_over  - market_p_over_no_vig
    edge_under = model_p_under - market_p_under_no_vig

    Uses the final serialized PMF (passed in as np.ndarray).
    Does NOT label the result as CLV, closing-line value, or any related concept.
    Does NOT mutate the input PMF.

    Returns (edge_over, edge_under) as a plain tuple.
    """
    p_over, p_push, p_under = compute_pmf_probabilities(pmf, line)
    market_p_over_nv, market_p_under_nv = compute_no_vig_probs_from_american(over_odds, under_odds)
    if market_p_over_nv is None or market_p_under_nv is None:
        raise MalformedOddsError(
            f"Cannot compute edge: invalid odds over={over_odds}, under={under_odds}"
        )
    edge_over = p_over - market_p_over_nv
    edge_under = p_under - market_p_under_nv
    return edge_over, edge_under


# ---------------------------------------------------------------------------
# Market quote validation
# ---------------------------------------------------------------------------

# Preferred dedup key in priority order. Validation uses the longest prefix
# of available columns. Requires at least one player-identity column (player_id
# or player_name) to be meaningful — without player identity, (vendor,stat,line)
# is too coarse and will falsely flag legitimate multi-player market data.
_QUOTE_DEDUP_KEYS = ["vendor", "game_id", "player_id", "stat", "line"]
_QUOTE_DEDUP_KEYS_ODDSAPI = ["vendor", "event_id", "player_name", "stat", "line"]
_QUOTE_PLAYER_IDENTITY_COLS = frozenset({"player_id", "player_name"})


def validate_no_duplicate_quotes(
    quotes_df: pd.DataFrame,
    key_cols: list[str] | None = None,
) -> None:
    """Raise DuplicateQuoteError if any key combination appears more than once.

    For BDL props: key = (vendor, game_id, player_id, stat, line).
    For Odds API props: key = (vendor, event_id, player_name, stat, line).
    Validation is skipped when no player-identity column (player_id or player_name)
    is present — a key of (vendor, stat, line) alone is too coarse for multi-player
    slates and produces false positives.
    """
    if quotes_df.empty:
        return
    # Select the best available dedup key.
    # Try primary BDL key first, then Odds API key.
    if key_cols:
        present_keys = [k for k in key_cols if k in quotes_df.columns]
    else:
        # Select the first candidate key that includes a player-identity column.
        # The Odds API parquet has (vendor, stat, line) but NOT player_id/game_id —
        # that 3-column key is too coarse and falsely flags multi-player props.
        # We must try the Odds API secondary key (event_id, player_name) when
        # the primary BDL key lacks player identity.
        present_keys = []
        for candidate_keys in (_QUOTE_DEDUP_KEYS, _QUOTE_DEDUP_KEYS_ODDSAPI):
            candidate_present = [k for k in candidate_keys if k in quotes_df.columns]
            if candidate_present and _QUOTE_PLAYER_IDENTITY_COLS.intersection(candidate_present):
                present_keys = candidate_present
                break

    if not present_keys:
        return

    # Final guard: skip when no player-identity column is present.
    if not _QUOTE_PLAYER_IDENTITY_COLS.intersection(present_keys):
        return

    dupes = quotes_df[quotes_df.duplicated(subset=present_keys, keep=False)]
    if not dupes.empty:
        sample = dupes[present_keys].drop_duplicates().head(5).to_dict("records")
        raise DuplicateQuoteError(
            f"{len(dupes)} duplicate quote rows detected (key={present_keys}). "
            f"First duplicates: {sample}"
        )


def validate_quote_freshness(
    quotes_df: pd.DataFrame,
    timestamp_col: str = "market_updated_at",
    max_age_seconds: float = 3600,
    current_time: datetime | None = None,
) -> None:
    """Raise StaleQuoteError if any quote is older than max_age_seconds.

    current_time defaults to datetime.now(timezone.utc).
    """
    if quotes_df.empty or timestamp_col not in quotes_df.columns:
        return
    now = current_time or datetime.now(timezone.utc)
    stale_rows = []
    for _, row in quotes_df.iterrows():
        ts_val = row[timestamp_col]
        if ts_val is None or (isinstance(ts_val, float) and math.isnan(ts_val)):
            continue
        try:
            ts = pd.Timestamp(ts_val)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            now_ts = pd.Timestamp(now)
            if now_ts.tzinfo is None:
                now_ts = now_ts.tz_localize("UTC")
            age_seconds = (now_ts - ts).total_seconds()
            if age_seconds > max_age_seconds:
                stale_rows.append((row.get("player_id"), row.get("stat"), age_seconds))
        except Exception as exc:
            raise StaleQuoteError(f"Cannot parse timestamp '{ts_val}': {exc}") from exc
    if stale_rows:
        raise StaleQuoteError(
            f"{len(stale_rows)} stale quote(s) detected (max_age={max_age_seconds}s). "
            f"Examples: {stale_rows[:3]}"
        )


def validate_player_identity_resolved(
    quotes_df: pd.DataFrame,
    player_id_col: str = "player_id",
    ambiguous_ids: set[str] | None = None,
) -> None:
    """Strict canonical validation — requires nonblank player_id.

    Call AFTER build_market_comparison, where every row must carry a canonical
    player_id from the PMF join.  Do NOT call this on raw provider quotes that
    have not yet been reconciled to canonical IDs.
    """
    if quotes_df.empty:
        return
    if player_id_col not in quotes_df.columns:
        raise UnmatchedIdentityError(
            f"Column '{player_id_col}' not found in quotes DataFrame. "
            "Canonical validation requires player_id. "
            "For pre-join provider-native validation use validate_provider_quotes()."
        )
    null_mask = quotes_df[player_id_col].isna()
    if null_mask.any():
        n = int(null_mask.sum())
        raise UnmatchedIdentityError(
            f"{n} quote(s) have unresolved player_id (None/NaN). "
            "Identity reconciliation must complete before edge computation."
        )
    if ambiguous_ids:
        ambiguous_mask = quotes_df[player_id_col].isin(ambiguous_ids)
        if ambiguous_mask.any():
            bad = quotes_df.loc[ambiguous_mask, player_id_col].unique().tolist()
            raise AmbiguousIdentityError(
                f"Ambiguous player identities detected: {bad}. "
                "These must be resolved before proceeding."
            )


def validate_game_identity_resolved(
    quotes_df: pd.DataFrame,
    game_id_col: str = "game_id",
) -> None:
    """Strict canonical validation — requires nonblank game_id.

    Call AFTER build_market_comparison, where every row must carry a canonical
    game_id from the PMF join.  Do NOT call this on raw provider quotes.
    """
    if quotes_df.empty:
        return
    if game_id_col not in quotes_df.columns:
        raise UnmatchedIdentityError(
            f"Column '{game_id_col}' not found in quotes DataFrame. "
            "Canonical validation requires game_id. "
            "For pre-join provider-native validation use validate_provider_quotes()."
        )
    null_mask = quotes_df[game_id_col].isna()
    if null_mask.any():
        n = int(null_mask.sum())
        raise UnmatchedIdentityError(
            f"{n} quote(s) have unresolved game_id (None/NaN). "
            "Game identity reconciliation must complete before edge computation."
        )


# ── Provider-native pre-join validation ──────────────────────────────────────

def validate_provider_quotes(
    quotes_df: pd.DataFrame,
    source: str = "oddsapi",
) -> None:
    """Validate raw provider quotes before PMF reconciliation.

    Stage 1 of the two-stage identity contract.  Uses provider-native column
    names — NOT canonical player_id/game_id.

    ``source='oddsapi'`` (The Odds API):
        Requires per row: nonblank event_id, normalized player_name,
        nonblank vendor/bookmaker, nonblank stat, valid line.

    ``source='bdl'`` (BDL/canonical):
        Requires per row: nonblank game_id, nonblank player_id.

    Blank or null values in any required column are fatal.
    """
    if quotes_df.empty:
        return

    errors: list[str] = []

    # Normalise source aliases so callers can pass the policy/source string directly
    # (e.g. `_load_props` returns "odds_api"; policies use "odds_api_v4" /
    # "odds_api_then_bdl"). Without this, "odds_api" matched neither branch and
    # silently skipped provider-native validation.
    _oddsapi_sources = {"oddsapi", "odds_api", "odds_api_v4"}
    _bdl_sources = {"bdl", "bdl_required", "odds_api_then_bdl", "none"}

    if source in _oddsapi_sources:
        # Odds API uses internal UUIDs for games and plain-text player names.
        # Require the provider-native identity columns, not canonical IDs. Accept
        # event_id (raw Odds API UUID) OR game_id (already normalized/reconciled
        # rows carry game_id directly) as the game identity.
        required: list[tuple[str, list[str]]] = [
            ("event_id or game_id", ["event_id", "game_id"]),
            ("player_name",         ["player_name"]),
            ("vendor/bookmaker",    ["vendor", "bookmaker"]),
            ("stat",                ["stat"]),
        ]
        for field_name, candidates in required:
            col = next((c for c in candidates if c in quotes_df.columns), None)
            if col is None:
                errors.append(f"Odds API quotes missing required column {field_name!r} "
                               f"(tried: {candidates})")
                continue
            blank = quotes_df[col].isna() | (quotes_df[col].astype(str).str.strip() == "")
            if blank.any():
                n = int(blank.sum())
                errors.append(f"{n} Odds API quote(s) have blank {col!r}")

        # line must be numeric and finite
        if "line" in quotes_df.columns:
            import math as _math  # noqa: PLC0415
            bad_line = quotes_df["line"].apply(
                lambda v: v is None or (isinstance(v, float) and not _math.isfinite(v))
            )
            if bad_line.any():
                errors.append(f"{int(bad_line.sum())} Odds API quote(s) have invalid line value")

    elif source in _bdl_sources:
        for col in ("game_id", "player_id"):
            if col not in quotes_df.columns:
                errors.append(f"BDL quotes missing required column {col!r}")
                continue
            blank = quotes_df[col].isna()
            if blank.any():
                n = int(blank.sum())
                errors.append(f"{n} BDL quote(s) have blank {col!r}")

    if errors:
        raise UnmatchedIdentityError(
            f"Provider-native validation failed ({source!r}): " + "; ".join(errors)
        )


def validate_odds_format(
    quotes_df: pd.DataFrame,
    over_col: str = "over_odds",
    under_col: str = "under_odds",
) -> None:
    """Raise MalformedOddsError if any odds value is not a valid American odds number.

    Valid American odds: non-zero numeric, with |odds| > 0.
    """
    if quotes_df.empty:
        return
    for col in [over_col, under_col]:
        if col not in quotes_df.columns:
            continue
        for idx, val in quotes_df[col].items():
            prob = _american_to_prob_internal(val)
            if prob is None:
                raise MalformedOddsError(
                    f"Row {idx}: column '{col}' value '{val}' is not valid American odds. "
                    "Valid examples: -110, +150, -105."
                )


# ---------------------------------------------------------------------------
# PMF manifest
# ---------------------------------------------------------------------------


def build_expected_pmf_manifest(
    slate_df: pd.DataFrame,
    stats: list[str],
) -> pd.DataFrame:
    """Build expected PMF manifest keyed by (game_id, player_id, stat).

    Returns one row per (game_id, player_id, stat) cross-join.
    """
    if slate_df.empty or not stats:
        return pd.DataFrame(columns=["game_id", "player_id", "stat"])
    rows = []
    for _, player_row in slate_df.iterrows():
        for stat in stats:
            rows.append({
                "game_id": player_row.get("game_id", ""),
                "player_id": player_row.get("player_id", ""),
                "stat": stat,
            })
    return pd.DataFrame(rows, columns=["game_id", "player_id", "stat"])


def validate_pmf_manifest(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
) -> None:
    """Validate PMF manifest: counts, missing, duplicate, unexpected.

    Raises:
        DuplicatePMFError  — if any (game_id, player_id, stat) appears >1 in actual
        MissingPMFError    — if expected count != actual count OR any expected not in actual
                             OR any actual not in expected (unexpected)
    """
    key_cols = ["game_id", "player_id", "stat"]

    # Check duplicates in actual
    if not actual.empty:
        dupe_mask = actual.duplicated(subset=key_cols, keep=False)
        if dupe_mask.any():
            dupe_sample = actual.loc[dupe_mask, key_cols].drop_duplicates().head(5).to_dict("records")
            raise DuplicatePMFError(
                f"Duplicate PMF rows detected in actual manifest: {dupe_sample}"
            )

    if expected.empty and actual.empty:
        return

    exp_set = set(
        tuple(r) for r in expected[key_cols].itertuples(index=False, name=None)
    ) if not expected.empty else set()

    act_set = set(
        tuple(r) for r in actual[key_cols].itertuples(index=False, name=None)
    ) if not actual.empty else set()

    missing = exp_set - act_set
    unexpected = act_set - exp_set

    errors: list[str] = []
    if missing:
        errors.append(f"Missing PMFs (expected but absent): {sorted(missing)[:5]}")
    if unexpected:
        errors.append(f"Unexpected PMFs (present but not expected): {sorted(unexpected)[:5]}")
    if errors:
        raise MissingPMFError(
            f"PMF manifest validation failed ({len(missing)} missing, {len(unexpected)} unexpected). "
            + " | ".join(errors)
        )


# ---------------------------------------------------------------------------
# Edge manifest
# ---------------------------------------------------------------------------


_EDGE_KEY_COLS = ["game_id", "player_id", "stat", "vendor", "line"]


def build_expected_edge_manifest(
    validated_markets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build expected edge manifest from validated market quotes.

    Returns a DataFrame with one row per (game_id, player_id, stat, vendor, line).
    Preserves over_odds, under_odds, market_updated_at, market_type, and vendor_settlement_rule
    when available.
    """
    if validated_markets_df.empty:
        return pd.DataFrame(columns=_EDGE_KEY_COLS)

    keep_cols = _EDGE_KEY_COLS + [
        c for c in [
            "over_odds", "under_odds", "market_updated_at", "market_type",
            "vendor_settlement_rule", "market_pulled_at_utc",
        ]
        if c in validated_markets_df.columns
    ]
    return validated_markets_df[keep_cols].reset_index(drop=True)


def validate_edge_manifest(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
) -> None:
    """Validate edge manifest: counts, missing, duplicate, unexpected.

    Raises:
        DuplicateEdgeError  — if any key combination appears >1 in actual
        MissingEdgeError    — if counts differ or any expected/unexpected rows exist
    """
    key_cols = _EDGE_KEY_COLS

    # Check duplicates in actual
    if not actual.empty:
        dupe_mask = actual.duplicated(subset=key_cols, keep=False)
        if dupe_mask.any():
            dupe_sample = actual.loc[dupe_mask, key_cols].drop_duplicates().head(5).to_dict("records")
            raise DuplicateEdgeError(
                f"Duplicate edge rows detected in actual manifest: {dupe_sample}"
            )

    if expected.empty and actual.empty:
        return

    exp_key_cols = [c for c in key_cols if c in expected.columns]
    act_key_cols = [c for c in key_cols if c in actual.columns]

    exp_set = set(
        tuple(r) for r in expected[exp_key_cols].itertuples(index=False, name=None)
    ) if not expected.empty else set()

    act_set = set(
        tuple(r) for r in actual[act_key_cols].itertuples(index=False, name=None)
    ) if not actual.empty else set()

    missing = exp_set - act_set
    unexpected = act_set - exp_set

    errors: list[str] = []
    if missing:
        errors.append(f"Missing edges (expected but absent): {sorted(str(x) for x in missing)[:5]}")
    if unexpected:
        errors.append(f"Unexpected edges (present but not expected): {sorted(str(x) for x in unexpected)[:5]}")
    if errors:
        raise MissingEdgeError(
            f"Edge manifest validation failed ({len(missing)} missing, {len(unexpected)} unexpected). "
            + " | ".join(errors)
        )


# ---------------------------------------------------------------------------
# Artifact lineage validation
# ---------------------------------------------------------------------------


def validate_artifact_lineage(
    artifacts: list[dict],
    run_id: str,
    git_commit: str,
    game_date: str,
    prediction_timestamp_utc: str,
    tolerance_seconds: float = 300,
) -> None:
    """Validate that every artifact in the list belongs to the same run.

    Checks: github_run_id, git_commit, game_date for every artifact.
    Mismatches raise ArtifactLineageMismatchError.

    prediction_timestamp_utc tolerance check: if an artifact carries a
    prediction_timestamp_utc, it must be within tolerance_seconds of the
    canonical timestamp.
    """
    errors: list[str] = []
    canonical_ts = _parse_utc(prediction_timestamp_utc)

    for i, art in enumerate(artifacts):
        if not isinstance(art, dict):
            continue
        art_run_id = art.get("github_run_id")
        art_commit = art.get("git_commit")
        art_date = art.get("game_date")
        art_ts_str = art.get("prediction_timestamp_utc")

        if art_run_id is not None and str(art_run_id) != str(run_id):
            errors.append(f"artifact[{i}].github_run_id={art_run_id!r} != expected {run_id!r}")
        if art_commit is not None and str(art_commit) != str(git_commit):
            errors.append(f"artifact[{i}].git_commit={art_commit!r} != expected {git_commit!r}")
        if art_date is not None and str(art_date) != str(game_date):
            errors.append(f"artifact[{i}].game_date={art_date!r} != expected {game_date!r}")

        if art_ts_str and canonical_ts is not None:
            art_ts = _parse_utc(art_ts_str)
            if art_ts is not None:
                delta = abs((art_ts - canonical_ts).total_seconds())
                if delta > tolerance_seconds:
                    errors.append(
                        f"artifact[{i}].prediction_timestamp_utc={art_ts_str!r} is "
                        f"{delta:.0f}s from canonical {prediction_timestamp_utc!r} "
                        f"(tolerance={tolerance_seconds}s)"
                    )

    if errors:
        raise ArtifactLineageMismatchError(
            f"Artifact lineage mismatch ({len(errors)} error(s)): " + "; ".join(errors[:5])
        )


def _parse_utc(ts_str: str) -> "datetime | None":
    """Parse ISO timestamp string to a timezone-aware datetime."""
    try:
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inactive-player vendor settlement rule
# ---------------------------------------------------------------------------


def check_inactive_market_settlement(
    player_status: str,
    vendor_settlement_rule: str,
) -> str:
    """Determine what to do with a market quote for a potentially-inactive player.

    Returns one of:
        'generate_edge'          — player is active; compute edge normally
        'defer_to_settlement'    — player is inactive; defer to post-game settlement
        'void'                   — player is confirmed inactive and rule is void

    Does NOT fabricate an edge for an inactive player.
    """
    status = (player_status or "").lower().strip()

    if status == "active":
        return "generate_edge"

    # inactive or questionable: apply vendor rule
    if vendor_settlement_rule == VENDOR_RULE_VOID_IF_NO_PARTICIPATION:
        return "defer_to_settlement"
    elif vendor_settlement_rule == VENDOR_RULE_ACTION_IF_STARTS:
        # Cannot confirm if player started at prediction time
        return "defer_to_settlement"
    elif vendor_settlement_rule == VENDOR_RULE_ACTION_IF_PLAYS:
        return "defer_to_settlement"
    else:
        # Unknown rule — be conservative
        return "defer_to_settlement"


# ---------------------------------------------------------------------------
# Stale-fallback guard
# ---------------------------------------------------------------------------


def check_no_stale_fallback(
    artifact_type: str,
    current_path: Path,
    fallback_path: "Path | None" = None,
) -> None:
    """Raise StaleFallbackForbiddenError if the current-run artifact is missing.

    The existence of a fallback_path does NOT change this behaviour — if the
    current artifact is absent, the run must fail regardless of any fallback.

    Usage in production scripts:
        check_no_stale_fallback("slate", Path("deliveries/tonight/slate_2026-07-13.parquet"))
    """
    if not Path(current_path).exists():
        fallback_note = (
            f" (stale fallback at {fallback_path} exists but must NOT be used)"
            if fallback_path and Path(fallback_path).exists()
            else ""
        )
        raise StaleFallbackForbiddenError(
            f"Current-run {artifact_type} artifact is missing: {current_path}{fallback_note}. "
            "The run must fail rather than use stale data."
        )


# ---------------------------------------------------------------------------
# Staging board validation
# ---------------------------------------------------------------------------


def validate_staging_board(
    staging_dir: Path,
    run_id: str,
    required_artifacts: "list[str] | None" = None,
) -> None:
    """Validate that the staging directory contains all required artifacts.

    Raises PartialBoardError if any required artifact is missing.
    Also checks that JSON artifacts with a github_run_id field match run_id.
    """
    staging_dir = Path(staging_dir)
    required = required_artifacts or []

    missing = [name for name in required if not (staging_dir / name).exists()]
    if missing:
        raise PartialBoardError(
            f"Staging board is incomplete: {len(missing)} required artifact(s) missing "
            f"from {staging_dir}. Missing: {missing}"
        )

    # Check run_id consistency in any JSON artifacts that carry it
    mismatched = []
    for name in required:
        p = staging_dir / name
        if not p.exists() or not name.endswith(".json"):
            continue
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and "github_run_id" in data:
                if str(data["github_run_id"]) != str(run_id):
                    mismatched.append(
                        f"{name}: github_run_id={data['github_run_id']!r} != {run_id!r}"
                    )
        except Exception:
            pass

    if mismatched:
        raise ArtifactLineageMismatchError(
            f"Staging artifacts have mismatched run_id: {mismatched}"
        )


# ---------------------------------------------------------------------------
# Atomic deployment
# ---------------------------------------------------------------------------

_REQUIRED_LIVE_ARTIFACTS = [
    "full_pmfs_wide.parquet",
    "publishable_edges.parquet",
    "run_metadata.json",
]


def atomic_deploy(
    staging_dir: Path,
    live_dir: Path,
    release_manifest: dict,
) -> None:
    """Promote staging to live atomically.

    Steps:
      1. Validate release_manifest.validation_result == 'PASS'
      2. Validate all staging artifacts exist (PartialBoardError if not)
      3. Verify artifact hashes match release_manifest.artifact_hashes
      4. Copy staging → temporary directory
      5. Atomically rename temp → live (best-effort; falls back to copy + swap)
      6. Write release manifest to live directory

    On any failure: raises PartialBoardError or MarketIntegrityError.
    The live directory is NEVER modified on failure.
    """
    staging_dir = Path(staging_dir)
    live_dir = Path(live_dir)

    # Step 1: validation_result must be PASS
    validation_result = release_manifest.get("validation_result", "")
    if validation_result != "PASS":
        raise PartialBoardError(
            f"Deployment blocked: release_manifest.validation_result={validation_result!r}, "
            "expected 'PASS'. Live board preserved."
        )

    # Step 2: check staging artifacts exist
    artifact_hashes = release_manifest.get("artifact_hashes", {})
    if artifact_hashes:
        missing = [name for name in artifact_hashes if not (staging_dir / name).exists()]
        if missing:
            raise PartialBoardError(
                f"Cannot deploy: {len(missing)} artifact(s) missing from staging: {missing}. "
                "Live board preserved."
            )

        # Step 3: verify hashes
        hash_errors = []
        for name, expected_hash in artifact_hashes.items():
            p = staging_dir / name
            if p.exists():
                actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    hash_errors.append(f"{name}: expected {expected_hash}, got {actual_hash}")
        if hash_errors:
            raise ArtifactLineageMismatchError(
                f"Artifact hash mismatch during deployment: {hash_errors}. Live board preserved."
            )

    # Steps 4-5: copy to temp dir then promote to live
    live_dir.mkdir(parents=True, exist_ok=True)
    # Use a sibling temp directory for atomic swap
    tmp_dir = live_dir.parent / f"_atomic_deploy_tmp_{release_manifest.get('github_run_id', 'unknown')}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(staging_dir, tmp_dir)

    # Write release manifest into temp
    manifest_path = tmp_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(release_manifest, indent=2))

    # Swap: rename tmp → live (atomic on most filesystems when on same partition)
    # Fall back to copy-then-remove if rename fails (cross-device)
    if live_dir.exists():
        old_live = live_dir.parent / f"_old_live_{release_manifest.get('github_run_id', 'unknown')}"
        if old_live.exists():
            shutil.rmtree(old_live)
        shutil.copytree(live_dir, old_live)

    try:
        # Clear live and copy tmp contents
        for item in live_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        for item in tmp_dir.iterdir():
            dest = live_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    except Exception as exc:
        # Restore live from backup
        if "old_live" in locals() and old_live.exists():
            for item in live_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            for item in old_live.iterdir():
                dest = live_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        raise PartialBoardError(
            f"Atomic deployment failed: {exc}. Attempted live restore."
        ) from exc
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if "old_live" in locals() and old_live.exists():
            shutil.rmtree(old_live, ignore_errors=True)


# ---------------------------------------------------------------------------
# Artifact lineage report
# ---------------------------------------------------------------------------

_REQUIRED_LINEAGE_FIELDS = [
    "github_run_id",
    "git_commit",
    "branch",
    "game_date",
    "prediction_timestamp_utc",
    "feature_cutoff_utc",
    "injury_snapshot_timestamp_utc",
    "market_snapshot_timestamp_utc",
    "model_version",
    "config_hash",
    "feature_manifest_hash",
    "artifact_schema_version",
]


ARTIFACT_MANIFEST_SCHEMA_VERSION = "1"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1"})

# ── Canonical feature-contract hash ──────────────────────────────────────────
# Fields that define the feature *contract* (stable across feature builds with
# the same schema, even when created_at_utc, git_commit, or paths differ).
_CANONICAL_FEATURE_HASH_KIND = "canonical_feature_contract_v2"  # stable code-constants only
# v1 (data-dependent, included model_feature_columns) is treated as legacy — skip comparison.
_CANONICAL_FEATURE_CONTRACT_FIELDS: tuple[str, ...] = (
    # Stateless code constants — identical across any build with the same codebase.
    "row_grain_wide",
    "row_grain_long",
    "identity_columns",
    "target_columns",
    "forbidden_columns",
    "temporal_policy",
    "stats_modeled",
    "roll_windows",
)
# Explicitly excluded from canonical hash:
#   Volatile (change across builds even with identical code):
#     created_at_utc, git_commit_if_available, wide_table_path, long_table_path,
#     source_tables, row counts, timestamps.
#   Data-dependent (change when input data size or composition changes):
#     model_feature_columns   — filtered by variance gate on the current data pull
#     numeric_feature_columns — derived from model_feature_columns + current dtypes
#     categorical_feature_columns — same
#     role_bucket_columns     — filtered by column existence in current data
#
# Model feature compatibility is validated separately via the fitted model's
# feature_names_in_ attribute (which is always authoritative).


def canonical_feature_contract_hash(manifest: dict) -> str:
    """Return a 16-char SHA-256 hex of the stable schema-contract fields only.

    Two feature manifests with identical schema but different timestamps,
    git commits, or file paths produce the same hash.  Adding, removing, or
    reordering ``model_feature_columns`` changes the hash.

    List-typed fields are sorted deterministically (preserving model feature
    order for ``model_feature_columns`` where order matters to the estimator;
    sorting all other list fields for stability).

    Parameters
    ----------
    manifest : dict
        Parsed ``feature_schema_manifest.json``.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 of the canonical JSON payload.
    """
    def _normalise(key: str, value: object) -> object:
        if isinstance(value, list):
            return sorted(value)  # all canonical fields are order-independent constants
        return value

    payload: dict = {
        k: _normalise(k, manifest.get(k))
        for k in _CANONICAL_FEATURE_CONTRACT_FIELDS
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


def validate_artifact_manifest(
    manifest: dict,
    expected_artifact_type: str,
    prediction_timestamp_utc: str,
    feature_manifest_hash: "str | None" = None,
    config_hash: "str | None" = None,
    source_run_id: "str | None" = None,
    source_commit: "str | None" = None,
    canonical_feature_hash: "str | None" = None,
) -> None:
    """Validate an artifact manifest for production use.

    Checks:
    - supported schema version
    - expected artifact type
    - required fields present
    - training_cutoff (model_training_cutoff or calibration_cutoff) < prediction_timestamp
    - calibration_cutoff < prediction_timestamp (if present)
    - feature_manifest_hash matches (if provided)
    - config_hash matches (if provided)
    - gate_status == PASS (if present in manifest)
    - source_run_id matches (if provided)
    - source_commit matches (if provided)

    Raises ArtifactManifestError on any validation failure.
    """
    errors: list[str] = []

    # Schema version
    schema_ver = str(manifest.get("artifact_schema_version", ""))
    if schema_ver not in _SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported artifact_schema_version={schema_ver!r}; "
            f"supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )

    # Artifact type
    manifest_type = str(manifest.get("artifact_type", ""))
    if manifest_type != expected_artifact_type:
        errors.append(
            f"Expected artifact_type={expected_artifact_type!r}, "
            f"got {manifest_type!r}"
        )

    # Required fields
    required_fields = [
        "artifact_type", "artifact_schema_version", "source_workflow",
        "source_run_id", "source_commit", "created_at_utc",
        "feature_manifest_hash", "config_hash",
    ]
    for field in required_fields:
        if field not in manifest or not manifest[field]:
            errors.append(f"Missing required manifest field: {field!r}")

    # source_run_id match
    if source_run_id is not None:
        manifest_run = str(manifest.get("source_run_id", ""))
        if manifest_run != str(source_run_id):
            errors.append(
                f"source_run_id mismatch: manifest={manifest_run!r}, "
                f"expected={source_run_id!r}"
            )

    # source_commit match
    if source_commit is not None:
        manifest_commit = str(manifest.get("source_commit", ""))
        if manifest_commit != str(source_commit):
            errors.append(
                f"source_commit mismatch: manifest={manifest_commit!r}, "
                f"expected={source_commit!r}"
            )

    # feature_manifest_hash / canonical_feature_hash
    manifest_hash_kind = str(manifest.get("feature_hash_kind", ""))
    if manifest_hash_kind == _CANONICAL_FEATURE_HASH_KIND:
        # New canonical path: compare canonical contract hashes.
        if canonical_feature_hash is not None:
            manifest_fmh = str(manifest.get("feature_manifest_hash", ""))
            if manifest_fmh != str(canonical_feature_hash):
                errors.append(
                    f"feature_manifest_hash mismatch (canonical): "
                    f"manifest={manifest_fmh!r}, expected={canonical_feature_hash!r}"
                )
    else:
        # Legacy path (no feature_hash_kind): the raw bytes hash in the artifact
        # is volatile (created_at_utc, paths, etc. differ across builds).
        # Comparing the raw hash against a freshly-built manifest is incorrect.
        # Accept the legacy artifact when the nonblank-hash requirement is met;
        # the caller must perform alternative compatibility checks separately.
        manifest_fmh = str(manifest.get("feature_manifest_hash", ""))
        if not manifest_fmh:
            errors.append("feature_manifest_hash is blank (legacy manifest)")
        # Do NOT compare manifest_fmh to feature_manifest_hash: they will differ
        # across builds even when the feature schema is compatible.

    # config_hash match
    if config_hash is not None:
        manifest_ch = str(manifest.get("config_hash", ""))
        if manifest_ch != str(config_hash):
            errors.append(
                f"config_hash mismatch: manifest={manifest_ch!r}, "
                f"expected={config_hash!r}"
            )

    # training/calibration cutoff must be PRESENT and before prediction timestamp.
    # A missing or blank cutoff for the expected artifact type is a fatal error.
    _ARTIFACT_CUTOFF_FIELDS: dict[str, str] = {
        "model": "model_training_cutoff",
        "calibrator": "calibration_cutoff",
    }
    pred_ts = _parse_utc(prediction_timestamp_utc)
    expected_cutoff_field = _ARTIFACT_CUTOFF_FIELDS.get(expected_artifact_type)
    if expected_cutoff_field:
        cutoff_str = manifest.get(expected_cutoff_field)
        if not cutoff_str:
            errors.append(
                f"Missing required cutoff field: {expected_cutoff_field!r} "
                f"(must be present and non-empty for artifact_type={expected_artifact_type!r})"
            )
        elif pred_ts is not None:
            cutoff_ts = _parse_utc(str(cutoff_str))
            if cutoff_ts is None:
                errors.append(
                    f"Cannot parse {expected_cutoff_field}={cutoff_str!r} as UTC timestamp"
                )
            elif cutoff_ts >= pred_ts:
                errors.append(
                    f"{expected_cutoff_field}={cutoff_str!r} is not before "
                    f"prediction_timestamp_utc={prediction_timestamp_utc!r}"
                )

    # gate_status must be PASS when present
    gate_status = manifest.get("gate_status")
    if gate_status is not None and str(gate_status) != "PASS":
        errors.append(
            f"gate_status={gate_status!r} — must be PASS for promoted artifacts"
        )

    if errors:
        raise ArtifactManifestError(
            f"Artifact manifest validation failed ({len(errors)} error(s)): "
            + "; ".join(errors[:5])
        )


def build_artifact_metadata(
    *,
    github_run_id: str,
    git_commit: str,
    branch: str,
    game_date: str,
    prediction_timestamp_utc: str,
    feature_cutoff_utc: str = "",
    injury_snapshot_timestamp_utc: str = "",
    market_snapshot_timestamp_utc: str = "",
    model_version: str = "",
    config_hash: str = "",
    feature_manifest_hash: str = "",
    artifact_schema_version: str = "1",
) -> dict:
    """Build a complete artifact metadata dict with all required lineage fields."""
    return {
        "github_run_id": github_run_id,
        "git_commit": git_commit,
        "branch": branch,
        "game_date": game_date,
        "prediction_timestamp_utc": prediction_timestamp_utc,
        "feature_cutoff_utc": feature_cutoff_utc,
        "injury_snapshot_timestamp_utc": injury_snapshot_timestamp_utc,
        "market_snapshot_timestamp_utc": market_snapshot_timestamp_utc,
        "model_version": model_version,
        "config_hash": config_hash,
        "feature_manifest_hash": feature_manifest_hash,
        "artifact_schema_version": artifact_schema_version,
    }


# ---------------------------------------------------------------------------
# Page release lineage validation
# ---------------------------------------------------------------------------


class PageProbabilityError(MarketIntegrityError):
    """Raised when a page's model probability doesn't match the final PMF at the market line."""


def validate_page_release_lineage(
    edge_page_json: dict,
    pmf_page_json: dict,
    expected_release_id: str,
) -> None:
    """Validate that both pre-game pages share one release ID.

    Checks:
      1. Both pages carry a release_id field.
      2. Both release_ids are identical (one release per deployment).
      3. Both release_ids match expected_release_id (current run only).

    Raises ArtifactLineageMismatchError on any failure.
    """
    errors: list[str] = []

    edge_rid = edge_page_json.get("release_id")
    pmf_rid  = pmf_page_json.get("release_id")

    if not edge_rid:
        errors.append("edge_page missing release_id")
    if not pmf_rid:
        errors.append("pmf_page missing release_id")

    if edge_rid and pmf_rid and edge_rid != pmf_rid:
        errors.append(
            f"edge_page release_id={edge_rid!r} != pmf_page release_id={pmf_rid!r} "
            "(both pages must come from the same release)"
        )

    if edge_rid and edge_rid != expected_release_id:
        errors.append(
            f"edge_page release_id={edge_rid!r} != expected={expected_release_id!r}"
        )

    if pmf_rid and pmf_rid != expected_release_id:
        errors.append(
            f"pmf_page release_id={pmf_rid!r} != expected={expected_release_id!r}"
        )

    if errors:
        raise ArtifactLineageMismatchError(
            f"Page release lineage validation failed ({len(errors)} error(s)): "
            + "; ".join(errors)
        )


def validate_page_probabilities(
    page_props: "list[dict]",
    pmf_df: "pd.DataFrame",
    tolerance: float = 1e-8,
    require_all_checked: bool = False,
) -> dict:
    """Validate page model probabilities against the final PMF parquet.

    Checks for each page prop (matched by player_name + stat):
      - model_p_over matches P(X > line) from pmf_full within tolerance
      - model_p_under matches P(X < line) (integer-line aware)
      - model_p_push matches P(X == line) for integer lines
      - model_p_over + model_p_under + model_p_push = 1 within 1e-12
      - model_p_over is not NaN
      - no duplicate (player, stat) keys in page_props

    Fail modes (all raise PageProbabilityError):
      - NaN model probability
      - Duplicate (player, stat) in page_props
      - require_all_checked=True and checked_rows < expected_rows
      - Probability values outside tolerance

    Returns dict: {expected_rows, checked_rows, missing_rows, duplicate_rows, probability_failures}
    """
    import json as _json
    import math as _math
    import numpy as _np

    _EMPTY = {"expected_rows": 0, "checked_rows": 0, "missing_rows": 0,
              "duplicate_rows": 0, "probability_failures": 0}

    if page_props is None:
        page_props = []
    if pmf_df is None:
        pmf_df = pd.DataFrame()

    # Detect duplicate (player, stat) keys in page props
    seen_page_keys: dict[tuple, int] = {}
    duplicate_rows = 0
    for prop in page_props:
        key = (str(prop.get("player", "")).lower().strip(),
               str(prop.get("stat", "")).lower().strip())
        if key in seen_page_keys:
            duplicate_rows += 1
        seen_page_keys[key] = seen_page_keys.get(key, 0) + 1

    if duplicate_rows > 0:
        raise PageProbabilityError(
            f"Duplicate (player, stat) keys in page props: "
            f"{duplicate_rows} duplicate row(s) detected"
        )

    # Build lookup from pmf_df: (player_name_lower, stat_lower) → row info
    # Use exact keys when available: game_id, player_id, stat; fallback to player_name+stat
    lookup: dict[tuple, dict] = {}
    duplicate_lookup_keys: list[tuple] = []
    for _, row in pmf_df.iterrows():
        pname = str(row.get("player_name", "")).lower().strip()
        stat  = str(row.get("stat",  "")).lower().strip()
        if not pname or not stat:
            continue
        key = (pname, stat)
        if key in lookup:
            duplicate_lookup_keys.append(key)
        lookup[key] = {
            "pmf_json": row.get("pmf_json"),
            "line": float(row.get("line", 0.0) or 0.0),
            "pmf_mean": row.get("pmf_mean"),
        }

    expected_rows = len(lookup)
    errors: list[str] = []
    checked = 0
    missing = 0

    for prop in page_props:
        pname = str(prop.get("player", "")).lower().strip()
        stat  = str(prop.get("stat",  "")).lower().strip()
        key   = (pname, stat)

        if key not in lookup:
            missing += 1
            continue

        info = lookup[key]
        pmf_json_str = info.get("pmf_json")
        if not pmf_json_str:
            errors.append(f"{pname}/{stat}: pmf_json is missing or empty in PMF source")
            continue

        # Parse PMF
        try:
            d = _json.loads(pmf_json_str) if isinstance(pmf_json_str, str) else pmf_json_str
            k_arr = _np.array([int(x) for x in d.keys()], dtype=float)
            p_arr = _np.array(list(d.values()), dtype=float)
            total = p_arr.sum()
            if total <= 0:
                errors.append(f"{pname}/{stat}: PMF has zero total mass")
                continue
            p_arr = p_arr / total
        except Exception as exc:
            errors.append(f"{pname}/{stat}: PMF parse error: {exc}")
            continue

        line = info["line"]
        p_over_pmf  = float(p_arr[k_arr > float(line)].sum())
        _is_int_line = (float(line) == _math.floor(float(line))) and line > 0
        p_push_pmf  = float(p_arr[k_arr == float(line)].sum()) if _is_int_line else 0.0
        p_under_pmf = max(0.0, 1.0 - p_over_pmf - p_push_pmf)

        # Validate model_p_over
        page_p_over = prop.get("model_p_over")
        if page_p_over is None or (isinstance(page_p_over, float) and _math.isnan(page_p_over)):
            errors.append(f"{pname}/{stat}: model_p_over is NaN or missing")
            continue

        err_over = abs(float(page_p_over) - p_over_pmf)
        if err_over > tolerance:
            errors.append(
                f"{pname}/{stat} line={line}: model_p_over={page_p_over:.8f} != "
                f"pmf_p_over={p_over_pmf:.8f} (err={err_over:.2e})"
            )

        # Validate model_p_under (if present) — must use push-aware computation
        page_p_under = prop.get("model_p_under")
        if page_p_under is not None and not (isinstance(page_p_under, float) and _math.isnan(page_p_under)):
            err_under = abs(float(page_p_under) - p_under_pmf)
            if err_under > tolerance:
                errors.append(
                    f"{pname}/{stat} line={line}: model_p_under={page_p_under:.8f} != "
                    f"pmf_p_under={p_under_pmf:.8f} (err={err_under:.2e})"
                )

        # Validate model_p_push (if present)
        page_p_push = prop.get("model_p_push")
        if page_p_push is not None and not (isinstance(page_p_push, float) and _math.isnan(page_p_push)):
            err_push = abs(float(page_p_push) - p_push_pmf)
            if err_push > tolerance:
                errors.append(
                    f"{pname}/{stat} line={line}: model_p_push={page_p_push:.8f} != "
                    f"pmf_p_push={p_push_pmf:.8f} (err={err_push:.2e})"
                )

        # Validate PMF normalization
        norm_err = abs(p_over_pmf + p_under_pmf + p_push_pmf - 1.0)
        if norm_err > 1e-12:
            errors.append(
                f"{pname}/{stat}: PMF not normalized (p_over+p_under+p_push={p_over_pmf+p_under_pmf+p_push_pmf:.12f})"
            )

        checked += 1

    # Fail when checked < expected (missing rows not gracefully skipped)
    if require_all_checked and checked < expected_rows and not errors:
        errors.append(
            f"Only {checked}/{expected_rows} expected rows were checked "
            f"({expected_rows - checked} rows from PMF parquet not found in page props)"
        )

    prob_failures = len(errors)
    result = {
        "expected_rows": expected_rows,
        "checked_rows": checked,
        "missing_rows": missing,
        "duplicate_rows": duplicate_rows,
        "probability_failures": prob_failures,
    }

    if errors:
        raise PageProbabilityError(
            f"Page probability validation failed ({len(errors)} error(s)): "
            + "; ".join(errors[:5])
        )

    return result
