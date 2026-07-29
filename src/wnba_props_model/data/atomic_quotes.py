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
    "snapshot_label",           # legacy alias of snapshot_role
    # --- CANONICAL immutable timestamp provenance (Section 3/4) ---------------------
    "snapshot_role",            # 'decision' | 'closing'
    "requested_snapshot_utc",   # the historical snapshot time WE requested (NOT a quote time)
    "provider_snapshot_utc",    # response wrapper 'timestamp' (snapshot the API returned)
    "previous_timestamp",       # response wrapper 'previous_timestamp'
    "next_timestamp",           # response wrapper 'next_timestamp'
    "bookmaker_last_update_utc", # per-bookmaker 'last_update'
    "market_last_update_utc",   # per-market 'last_update' (the true quote provenance time)
    "quote_timestamp_utc",      # == market_last_update_utc; the ACTUAL quote time (pairing/id)
    "quote_timestamp_source",   # 'market_last_update' | 'BLOCKED_NO_MARKET_TIMESTAMP'
    "scheduled_tip_utc",        # ISO UTC scheduled tip
    "decision_cutoff_utc",      # tip - 12h
    "closing_cutoff_utc",       # tip - 5m
    "role_cutoff_utc",          # decision_cutoff_utc for decision; closing_cutoff_utc for closing
    "collection_timestamp_utc", # when WE collected the row
    # legacy names kept for back-compat with existing rows/tests (never used for pairing):
    "snapshot_time", "decision_timestamp", "requested_snapshot_time",
    "provider_snapshot_time", "market_last_update", "collection_timestamp",
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
    """Explicit, validated adapter: map canonical atomic-store rows to the raw-side schema
    ``quote_pairs.build_quote_pairs`` consumes, WITHOUT mutating any timestamp.

    * The pairing/quote timestamp is strictly ``quote_timestamp_utc`` == market_last_update.
      Rows with NO market timestamp are DROPPED here (blocked) rather than fabricating a time
      from the requested snapshot.
    * The cutoff passed to the pair builder is the ROLE cutoff:
        - snapshot_role == 'decision'  -> role_cutoff_utc = decision_cutoff_utc (tip - 12h)
        - snapshot_role == 'closing'   -> role_cutoff_utc = closing_cutoff_utc  (tip - 5m)
      We NEVER null the cutoff or substitute the tip; each role carries its own cutoff.
    """
    if atomic is None or atomic.empty:
        return pd.DataFrame(columns=_RAW_SIDE_COLUMNS)
    df = atomic.copy()

    def _first(cols: list[str]) -> pd.Series:
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

    role = _first(["snapshot_role", "snapshot_label"])
    # ACTUAL quote time only — no fallback to provider snapshot or requested date.
    quote_ts = _first(["quote_timestamp_utc", "market_last_update_utc", "market_last_update"])
    role_cut = _first(["role_cutoff_utc"])
    decision_cut = _first(["decision_cutoff_utc", "decision_timestamp"])
    closing_cut = _first(["closing_cutoff_utc"])

    # If role_cutoff_utc is not explicitly stored, derive it from the role's own cutoff
    # (never a null; never the tip).
    def _role_cut(r):
        if pd.notna(r["role_cut"]):
            return r["role_cut"]
        return r["closing_cut"] if str(r["role"]).lower() == "closing" else r["decision_cut"]

    helper = pd.DataFrame({"role": role, "role_cut": role_cut,
                           "decision_cut": decision_cut, "closing_cut": closing_cut})
    effective_cut = helper.apply(_role_cut, axis=1)

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
        "decision_timestamp_utc": effective_cut,     # ROLE cutoff (build_quote_pairs enforces)
        "snapshot_label": role,
    })
    # Block rows with no actual market quote timestamp (never fabricate).
    raw = raw[raw["snapshot_timestamp"].notna()].reset_index(drop=True)
    return raw


# Counterpart-rejection statuses (Section 5) — a diagnostic SEPARATE from build_quote_pairs.
ONE_SIDED = "ONE_SIDED"
CROSS_BOOK_COUNTERPART_ONLY = "CROSS_BOOK_COUNTERPART_ONLY"
CROSS_LINE_COUNTERPART_ONLY = "CROSS_LINE_COUNTERPART_ONLY"
DUPLICATE_SIDE = "DUPLICATE_SIDE"
AMBIGUOUS_PLAYER = "AMBIGUOUS_PLAYER"
HAS_EXACT_COUNTERPART = "HAS_EXACT_COUNTERPART"


def counterpart_rejection_audit(atomic: pd.DataFrame) -> pd.DataFrame:
    """Explain WHY a raw side did not form an exact same-book/same-line pair.

    The pair builder groups by (book, line), so it structurally cannot emit CROSS_BOOK or
    CROSS_LINE. This audit inspects the raw sides directly and classifies each side:

      * unresolved player                                 -> AMBIGUOUS_PLAYER
      * >1 same-key side (same book/line/role/side)        -> DUPLICATE_SIDE
      * exact opposite side at SAME book & line & role     -> HAS_EXACT_COUNTERPART
      * opposite side exists only at ANOTHER book          -> CROSS_BOOK_COUNTERPART_ONLY
      * opposite side exists only at ANOTHER line same book -> CROSS_LINE_COUNTERPART_ONLY
      * no opposite side anywhere                           -> ONE_SIDED

    Returns the input rows with a 'counterpart_status' column. A cross-line/-book status is
    NOT a pairing error — it explains why no *exact* counterpart existed.
    """
    cols = ["event_id", "sportsbook", "player_id", "prop", "line", "side"]
    if atomic is None or atomic.empty:
        return pd.DataFrame(columns=cols + ["counterpart_status"])
    df = atomic.copy()
    df["_side"] = df["side"].astype(str).str.lower()
    role_col = "snapshot_role" if "snapshot_role" in df.columns else (
        "snapshot_label" if "snapshot_label" in df.columns else None)
    df["_role"] = df[role_col].astype(str) if role_col else "decision"

    def _classify(row) -> str:
        pid = row["player_id"]
        if pid is None or (isinstance(pid, float) and pd.isna(pid)) or str(pid).strip() in ("", "nan", "None"):
            return AMBIGUOUS_PLAYER
        opp = "under" if row["_side"] == "over" else "over"
        same_ev_pl_prop_role = df[(df["event_id"] == row["event_id"]) & (df["player_id"] == pid) &
                                  (df["prop"] == row["prop"]) & (df["_role"] == row["_role"])]
        # duplicate same-key side?
        same_key_same_side = same_ev_pl_prop_role[
            (same_ev_pl_prop_role["sportsbook"] == row["sportsbook"]) &
            (same_ev_pl_prop_role["line"] == row["line"]) &
            (same_ev_pl_prop_role["_side"] == row["_side"])]
        if len(same_key_same_side) > 1:
            return DUPLICATE_SIDE
        opp_rows = same_ev_pl_prop_role[same_ev_pl_prop_role["_side"] == opp]
        if len(opp_rows) == 0:
            return ONE_SIDED
        same_book_line = opp_rows[(opp_rows["sportsbook"] == row["sportsbook"]) &
                                  (opp_rows["line"] == row["line"])]
        if len(same_book_line) >= 1:
            return HAS_EXACT_COUNTERPART
        same_book = opp_rows[opp_rows["sportsbook"] == row["sportsbook"]]
        if len(same_book) >= 1:
            return CROSS_LINE_COUNTERPART_ONLY   # opposite only at another line, same book
        return CROSS_BOOK_COUNTERPART_ONLY       # opposite only at another book

    df["counterpart_status"] = df.apply(_classify, axis=1)
    return df[cols + ["_role", "counterpart_status"]].rename(columns={"_role": "snapshot_role"})


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
