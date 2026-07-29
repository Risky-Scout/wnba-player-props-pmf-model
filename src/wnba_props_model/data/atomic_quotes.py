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
    "snapshot_time",            # ISO UTC of the ACTUAL quote (provider/market) timestamp
    "decision_timestamp",       # ISO UTC decision cutoff (tip - lead)  [legacy name]
    "scheduled_tip_utc",        # ISO UTC scheduled tip
    # --- timestamp provenance (H): never mislabel the requested date as the quote time ---
    "requested_snapshot_time",  # the historical snapshot time WE requested
    "provider_snapshot_time",   # response wrapper 'timestamp' (the snapshot the API returned)
    "previous_timestamp",       # response wrapper 'previous_timestamp'
    "next_timestamp",           # response wrapper 'next_timestamp'
    "market_last_update",       # per-market 'last_update' (the true quote provenance time)
    "collection_timestamp",     # when WE collected the row
    "decision_cutoff_utc",      # ISO UTC decision cutoff (tip - lead)  [canonical name]
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


# Columns the pairing builder (quote_pairs.build_quote_pairs) requires.
_RAW_SIDE_COLUMNS = [
    "provider", "sportsbook", "event_id", "player_id", "prop", "line", "side",
    "snapshot_timestamp", "american_odds", "scheduled_tip_utc", "decision_timestamp_utc",
]


def to_raw_side_snapshots(atomic: pd.DataFrame, *, provider_default: str = "odds_api") -> pd.DataFrame:
    """Explicit, validated adapter: map atomic-store rows to the raw-side schema that
    ``quote_pairs.build_quote_pairs`` consumes, resolving the historical naming drift
    (source->provider, snapshot_time->snapshot_timestamp, decision_timestamp->decision_timestamp_utc)
    WITHOUT silent renaming elsewhere.

    The pairing/quote timestamp is the ACTUAL quote time — market_last_update if present,
    else provider_snapshot_time, else snapshot_time — never the requested snapshot date.
    """
    if atomic is None or atomic.empty:
        return pd.DataFrame(columns=_RAW_SIDE_COLUMNS)
    df = atomic.copy()

    def _first_present(cols: list[str]) -> pd.Series:
        out = pd.Series([None] * len(df), index=df.index, dtype=object)
        for c in cols:
            if c in df.columns:
                out = out.where(out.notna(), df[c])
        return out

    provider = df["provider"] if "provider" in df.columns else pd.Series([None] * len(df), index=df.index)
    if "source" in df.columns:
        provider = provider.where(provider.notna(), df["source"].apply(
            lambda s: "odds_api" if isinstance(s, str) and s.startswith("odds_api") else s))
    provider = provider.where(provider.notna(), provider_default)

    quote_ts = _first_present(["market_last_update", "provider_snapshot_time", "snapshot_time"])
    decision = _first_present(["decision_cutoff_utc", "decision_timestamp"])

    raw = pd.DataFrame({
        "provider": provider,
        "sportsbook": df.get("sportsbook"),
        "event_id": df.get("event_id"),
        "player_id": df.get("player_id"),
        "prop": df.get("prop"),
        "line": df.get("line"),
        "side": df.get("side"),
        "snapshot_timestamp": quote_ts,
        "american_odds": df.get("american_odds"),
        "scheduled_tip_utc": df.get("scheduled_tip_utc"),
        "decision_timestamp_utc": decision,
    })
    if "snapshot_label" in df.columns:
        raw["snapshot_label"] = df["snapshot_label"]
    return raw


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
