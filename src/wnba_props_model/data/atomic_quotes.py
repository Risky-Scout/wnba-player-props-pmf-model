"""W0.7 - atomic (same-book, point-in-time) quote + settlement store.

Promotion evidence requires EXACT quotes: same sportsbook, event, player, prop, line, and
snapshot. This module defines the append-only store schema and helpers. Books are NEVER
averaged - each row is one book's quote at one snapshot. If exact historical decision-time
quotes are unavailable for a game, its market-line metrics are marked BLOCKED_EXACT_QUOTES
and no promotion claim may be made from them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# One row = one book's over/under quote for one (event, player, prop, line) at one snapshot.
ATOMIC_QUOTE_COLUMNS = [
    "quote_id",                 # stable id (see atomic_quote_id)
    "sportsbook",               # single book (NEVER a consensus/aggregate)
    "event_id",                 # provider event id
    "game_id",                  # canonical game id (when resolved)
    "player_id",                # canonical player id (when resolved)
    "player_name",              # provider player name
    "prop",                     # canonical stat
    "line",                     # exact line
    "side",                     # 'over' | 'under'
    "american_odds",            # exact price
    "snapshot_label",           # 'decision' | 'closing'
    "snapshot_time",            # ISO UTC of the quote snapshot
    "decision_timestamp",       # ISO UTC decision cutoff (tip - lead)
    "scheduled_tip_utc",        # ISO UTC scheduled tip
    "prediction_timestamp",     # ISO UTC when the model prediction was made
    "model_prob_over_final",    # delivered probability (lineage output)
    "probability_lineage_version",
    "model_hash", "calibrator_hash", "feature_schema_hash", "quote_policy_hash",
    "settlement_status",        # 'settled' | 'push' | 'void' | 'pending'
    "actual_outcome",           # realized stat value (when settled)
    "exact_quote_status",       # 'EXACT' | 'BLOCKED_EXACT_QUOTES'
    "source",                   # provenance
]

BLOCKED_EXACT_QUOTES = "BLOCKED_EXACT_QUOTES"
EXACT = "EXACT"


def atomic_quote_id(sportsbook, event_id, player_id_or_name, prop, line, side, snapshot_time) -> str:
    """Deterministic id for one atomic quote (same book/event/player/prop/line/side/snapshot)."""
    payload = "|".join(str(x) for x in (
        sportsbook, event_id, player_id_or_name, prop, line, side, snapshot_time))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def assert_no_book_averaging(df: pd.DataFrame) -> None:
    """Fail if any row lacks a single named sportsbook or is flagged as an aggregate/consensus
    (averaging books across the +/-100 boundary destroys the sharp line and is forbidden)."""
    if df.empty:
        return
    if "sportsbook" not in df.columns:
        raise ValueError("atomic quotes must carry a 'sportsbook' column (no consensus rows)")
    bad = df["sportsbook"].isna() | df["sportsbook"].astype(str).str.strip().eq("")
    banned = df["sportsbook"].astype(str).str.lower().isin(
        {"consensus", "average", "mean", "median", "aggregate", "all"})
    if bool(bad.any()) or bool(banned.any()):
        raise ValueError(
            f"{int(bad.sum() + banned.sum())} atomic-quote row(s) are missing a book or are "
            "aggregates; atomic quotes must be single-book (never averaged).")


def append_atomic_quotes(store_path, new_rows: pd.DataFrame) -> dict:
    """APPEND-ONLY: add rows whose quote_id is not already in the store. Never mutates or
    overwrites existing quotes (a captured quote is immutable evidence). Returns a summary."""
    assert_no_book_averaging(new_rows)
    if "quote_id" not in new_rows.columns:
        raise ValueError("new_rows must include quote_id")
    p = Path(store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        existing = pd.read_parquet(p)
        have = set(existing["quote_id"].astype(str)) if "quote_id" in existing.columns else set()
        add = new_rows[~new_rows["quote_id"].astype(str).isin(have)].copy()
        combined = pd.concat([existing, add], ignore_index=True)
    else:
        existing = pd.DataFrame(columns=ATOMIC_QUOTE_COLUMNS)
        add = new_rows.drop_duplicates(subset=["quote_id"]).copy()
        combined = add
    # Immutability guard: never drop or rewrite existing quote_ids.
    combined.to_parquet(p, index=False)
    return {"existing": int(len(existing)), "added": int(len(add)), "total": int(len(combined))}
