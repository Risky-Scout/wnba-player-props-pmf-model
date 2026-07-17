"""Join HISTORICAL sportsbook lines onto out-of-fold (OOF) predictions.

This path is for **historical-line validation of OOF distributions only**. It is
distinct from the live/daily opening-line scoring done in
``scripts/score_daily_predictions.py`` (which scores that day's shipped
predictions against the day's opening market and the closing market). Do not
conflate the two: raw OOF is distribution-only and immutable; this function
produces a *market-enriched* OOF evaluation table used to fit line-dependent
calibrators.

Fail-closed contract (P0): the join MUST NOT silently succeed with zero or
trivial coverage. It raises OOFLineJoinError when:
  * required canonical keys are missing from either input,
  * the props table has duplicate rows per (game_id, player_id, stat)
    (ambiguous quotes must be reduced to one consensus row upstream),
  * the left-merge multiplies OOF rows (many-to-many),
  * eligible line coverage is below ``min_coverage``.

The previous implementation picked whatever keys happened to be present, did a
silent left-merge against a LIVE/upcoming props feed (no historical overlap),
printed coverage, and always wrote a file — which is why calibration ran on zero
real lines. That is now impossible.
"""
from __future__ import annotations

import pandas as pd

CANONICAL_KEYS = ["game_id", "player_id", "stat"]
REQUIRED_PROPS_COLS = ["game_id", "player_id", "stat", "line"]


class OOFLineJoinError(RuntimeError):
    """Raised when the historical-line join cannot be trusted."""


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce canonical key columns to a stable string dtype for joining."""
    out = df.copy()
    for k in CANONICAL_KEYS:
        if k in out.columns:
            out[k] = out[k].astype("string").str.strip()
    return out


def join_historical_props_to_oof(
    oof_parquet: str,
    props_parquet: str,
    output_parquet: str,
    *,
    min_coverage: float = 0.10,
) -> pd.DataFrame:
    """Join historical prop lines onto OOF predictions, fail-closed.

    Parameters
    ----------
    oof_parquet:
        Raw OOF distribution parquet (must carry canonical keys).
    props_parquet:
        HISTORICAL props table — one consensus row per (game_id, player_id,
        stat). A live/upcoming feed will not overlap OOF games and will trip the
        coverage gate.
    output_parquet:
        Destination for the market-enriched OOF table.
    min_coverage:
        Minimum fraction of OOF rows that must receive a real line. Below this
        the join is rejected (default 10%).

    Raises
    ------
    OOFLineJoinError
        On missing keys, duplicate quotes, row multiplication, or low coverage.
    """
    oof = _normalize_keys(pd.read_parquet(oof_parquet))
    props = _normalize_keys(pd.read_parquet(props_parquet))

    missing_oof = [k for k in CANONICAL_KEYS if k not in oof.columns]
    if missing_oof:
        raise OOFLineJoinError(
            f"OOF parquet missing canonical keys {missing_oof}; cannot join lines."
        )
    missing_props = [c for c in REQUIRED_PROPS_COLS if c not in props.columns]
    if missing_props:
        raise OOFLineJoinError(
            f"Historical props parquet missing required columns {missing_props}. "
            "This path requires a HISTORICAL props table with canonical "
            "game_id/player_id/stat and line — not a live/upcoming feed."
        )

    # Reject ambiguous quotes: exactly one consensus row per canonical key.
    dup_mask = props.duplicated(subset=CANONICAL_KEYS, keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        raise OOFLineJoinError(
            f"Historical props have {n_dup} duplicate rows per {CANONICAL_KEYS}. "
            "Reduce to one consensus row per (game_id, player_id, stat) before "
            "joining (quote selection must be explicit, not implicit)."
        )

    keep_cols = CANONICAL_KEYS + [
        c for c in ["line", "over_odds", "under_odds",
                    "market_prob_over_no_vig", "vendor", "book"]
        if c in props.columns
    ]
    n_before = len(oof)
    merged = oof.merge(props[keep_cols], on=CANONICAL_KEYS, how="left")

    # A left-join against a 1:1 props table must not change the row count.
    if len(merged) != n_before:
        raise OOFLineJoinError(
            f"Join changed row count {n_before} -> {len(merged)} — the props "
            "table is not unique on the canonical keys (many-to-many)."
        )

    n_with_lines = int(merged["line"].notna().sum()) if "line" in merged.columns else 0
    coverage = (n_with_lines / n_before) if n_before else 0.0
    print(f"[OOF Line Join] {n_with_lines}/{n_before} rows have real lines "
          f"({coverage*100:.1f}% coverage; required >= {min_coverage*100:.1f}%)")

    if coverage < min_coverage:
        raise OOFLineJoinError(
            f"Historical line coverage {coverage*100:.1f}% is below the required "
            f"{min_coverage*100:.1f}%. Refusing to write a market-enriched OOF "
            "table that cannot support line-dependent calibration. Check that "
            "the props source is a HISTORICAL archive overlapping OOF games."
        )

    merged.to_parquet(output_parquet, index=False)
    return merged
