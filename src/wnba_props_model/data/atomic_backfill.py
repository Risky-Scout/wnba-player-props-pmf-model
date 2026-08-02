"""Shared durable normalization + persistence for the historical quote backfill.

One canonical parser turns an Odds API historical event-odds response into atomic side rows
(canonical schema, immutable timestamps, role cutoffs). Rows are persisted incrementally to
season/game_date/event_id/snapshot_role partitions with temp-file+atomic-rename semantics and
consolidated into the append-only store, deduped by a deterministic primary key that includes
the ACTUAL market quote timestamp. Used by both the offline raw-cache recovery and the live
resumable backfill so they cannot drift.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.constants import MODEL_PROP_MARKETS
from wnba_props_model.data.atomic_quotes import (
    ATOMIC_QUOTE_COLUMNS,
    BLOCKED_EXACT_QUOTES,
    EXACT,
    atomic_quote_id,
)
from wnba_props_model.data import identity_resolution
from wnba_props_model.evaluation import historical_market as hm  # noqa: F401 (kept for API compat)

DECISION_LEAD_HOURS = 12
CLOSING_MINUTES = 5

# checkpoint states
NOT_STARTED = "NOT_STARTED"
RAW_SAVED = "RAW_SAVED"
NORMALIZED = "NORMALIZED"
VALIDATED = "VALIDATED"
NO_DATA = "NO_DATA"
HTTP_404 = "HTTP_404"
BLOCKED = "BLOCKED"
COMPLETE = "COMPLETE"


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_tip(commence: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def cutoffs_for(tip: datetime) -> tuple[str, str]:
    """(decision_cutoff_utc, closing_cutoff_utc)."""
    return iso(tip - timedelta(hours=DECISION_LEAD_HOURS)), iso(tip - timedelta(minutes=CLOSING_MINUTES))


def parse_event_odds(odds: dict, *, role: str, tip: datetime, event_id: str, gid,
                     roster_df: pd.DataFrame, collection_ts: str,
                     requested_snapshot_utc: str, aliases: dict | None = None) -> list[dict]:
    """Canonical parser: response -> atomic side rows with immutable timestamps + role cutoff.
    quote_timestamp_utc == market_last_update_utc; if absent, the row is BLOCKED (never
    fabricated). Both Over and Under of one market object inherit that object's last_update.
    """
    wrapper_ts = (odds or {}).get("timestamp")
    prev_ts = (odds or {}).get("previous_timestamp")
    next_ts = (odds or {}).get("next_timestamp")
    decision_cut, closing_cut = cutoffs_for(tip)
    role_cut = closing_cut if role == "closing" else decision_cut
    tip_iso = iso(tip)
    # Build this game's roster index ONCE (perf: avoids rebuilding maps per outcome row).
    game_index = (identity_resolution.build_game_index(roster_df, gid)
                  if (gid is not None and roster_df is not None and not roster_df.empty) else None)
    rows: list[dict] = []
    for book in (odds or {}).get("data", {}).get("bookmakers", []):
        bkey = book.get("key", "")
        book_last = book.get("last_update")
        for m in book.get("markets", []):
            stat = MODEL_PROP_MARKETS.get(m.get("key", ""))
            if not stat:
                continue
            mkt_last = m.get("last_update")
            if mkt_last:
                quote_ts, quote_src = mkt_last, "market_last_update"
            else:
                quote_ts, quote_src = None, "BLOCKED_MISSING_MARKET_TIMESTAMP"
            id_ts = mkt_last or f"BLOCKED::{requested_snapshot_utc}"
            # TIMING status with MUTUALLY EXCLUSIVE precedence (does not depend on identity):
            #   INVALID_SCHEDULED_TIP > MISSING_MARKET_TIMESTAMP > AT_OR_AFTER_TIP >
            #   AFTER_ROLE_CUTOFF > INVALID_ROLE_CUTOFF > ELIGIBLE.
            # We never mutate a timestamp to make a row pass.
            _rc = parse_tip(role_cut)
            if tip is None:
                timing_status = "INVALID_SCHEDULED_TIP"
            elif mkt_last is None:
                timing_status = "MISSING_MARKET_TIMESTAMP"
            else:
                _q = parse_tip(mkt_last)
                if _rc is None:
                    timing_status = "INVALID_ROLE_CUTOFF"
                elif _q is None:
                    timing_status = "MISSING_MARKET_TIMESTAMP"
                elif _q >= tip:
                    timing_status = "AT_OR_AFTER_TIP"
                elif _q > _rc:
                    timing_status = "AFTER_ROLE_CUTOFF"
                else:
                    timing_status = "ELIGIBLE"
            for oc in m.get("outcomes", []):
                name = oc.get("description", "")
                pid, identity_method = identity_resolution.resolve_with_index(name, game_index, aliases)
                side = str(oc.get("name", "")).lower()
                line = oc.get("point")
                # eligibility requires BOTH valid timing AND resolved identity.
                identity_reason = ("AMBIGUOUS_GAME" if gid is None
                                   else ("AMBIGUOUS_PLAYER" if pid is None else None))
                # blocking_reason precedence: timing first (per contract), then identity.
                if timing_status != "ELIGIBLE":
                    blocking_reason = timing_status
                elif identity_reason is not None:
                    blocking_reason = identity_reason
                else:
                    blocking_reason = None
                eligible = blocking_reason is None
                eligibility_status = "ELIGIBLE" if eligible else "BLOCKED"
                usable_for_pairing = eligible
                block_reason = blocking_reason
                exact_ok = eligible
                rows.append({
                    "quote_id": atomic_quote_id(bkey, event_id, pid or name, stat, line, side, id_ts),
                    "sportsbook": bkey, "event_id": event_id, "game_id": gid, "player_id": pid,
                    "player_name": name, "prop": stat, "line": line, "side": side,
                    "american_odds": oc.get("price"),
                    "snapshot_role": role, "snapshot_label": role,
                    "requested_snapshot_utc": requested_snapshot_utc, "provider_snapshot_utc": wrapper_ts,
                    "previous_timestamp": prev_ts, "next_timestamp": next_ts,
                    "bookmaker_last_update_utc": book_last, "market_last_update_utc": mkt_last,
                    "quote_timestamp_utc": quote_ts, "quote_timestamp_source": quote_src,
                    "scheduled_tip_utc": tip_iso, "decision_cutoff_utc": decision_cut,
                    "closing_cutoff_utc": closing_cut, "role_cutoff_utc": role_cut,
                    "collection_timestamp_utc": collection_ts,
                    "snapshot_time": quote_ts, "decision_timestamp": decision_cut,
                    "requested_snapshot_time": requested_snapshot_utc, "provider_snapshot_time": wrapper_ts,
                    "market_last_update": mkt_last, "collection_timestamp": collection_ts,
                    "prediction_timestamp": None, "model_prob_over_final": None,
                    "probability_lineage_version": None, "model_hash": None,
                    "calibrator_hash": None, "feature_schema_hash": None, "quote_policy_hash": None,
                    "settlement_status": "pending", "actual_outcome": None,
                    "exact_quote_status": EXACT if exact_ok else BLOCKED_EXACT_QUOTES,
                    "exact_block_reason": block_reason,
                    "timing_status": timing_status,
                    "eligibility_status": eligibility_status,
                    "blocking_reason": blocking_reason,
                    "usable_for_pairing": bool(usable_for_pairing),
                    "usable_for_decision_analysis": bool(usable_for_pairing and role == "decision"),
                    "usable_for_closing_analysis": bool(usable_for_pairing and role == "closing"),
                    "identity_method": identity_method,
                    "source": "odds_api_v4_historical",
                })
    return rows


def side_partition_path(base: Path, season, game_date: str, event_id: str, role: str) -> Path:
    return (Path(base) / f"season={season}" / f"game_date={game_date}" /
            f"event={event_id}" / f"role={role}" / "part.parquet")


def write_rows_atomic(rows: list[dict], path: Path) -> int:
    """Write rows to `path` via a temp file + atomic rename (durable, no partial file)."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=ATOMIC_QUOTE_COLUMNS)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.parquet")
    os.close(fd)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)   # atomic on POSIX
    return len(df)


def consolidate_partitions(base: Path) -> pd.DataFrame:
    """Read all side partitions and dedup by quote_id (deterministic PK incl. actual quote ts)."""
    parts = sorted(Path(base).rglob("part.parquet"))
    if not parts:
        return pd.DataFrame(columns=ATOMIC_QUOTE_COLUMNS)
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df.drop_duplicates(subset=["quote_id"]).reset_index(drop=True)


def validate_timing(df: pd.DataFrame) -> dict:
    """Timing invariants for the ELIGIBLE (usable) rows: quote_ts <= role_cutoff < tip.
    Blocked rows are retained in `df` but must never be usable, so eligible rows must have
    ZERO post-cutoff / post-tip."""
    elig = df[df.get("eligibility_status") == "ELIGIBLE"].copy() if "eligibility_status" in df.columns \
        else df[df["exact_quote_status"] == EXACT].copy()
    if elig.empty:
        return {"n_eligible": 0, "post_cutoff": 0, "post_tip": 0, "ok": True}
    q = pd.to_datetime(elig["quote_timestamp_utc"], utc=True, errors="coerce")
    rc = pd.to_datetime(elig["role_cutoff_utc"], utc=True, errors="coerce")
    tp = pd.to_datetime(elig["scheduled_tip_utc"], utc=True, errors="coerce")
    post_cutoff = int((q > rc).sum())
    post_tip = int((q >= tp).sum())
    return {"n_eligible": int(len(elig)), "post_cutoff": post_cutoff, "post_tip": post_tip,
            "ok": post_cutoff == 0 and post_tip == 0}


# ---- checkpoint -------------------------------------------------------------------
def load_state(path: Path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_state(path: Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, p)


def state_key(event_id: str, role: str) -> str:
    return f"{event_id}::{role}"


# ---- durable per-event/snapshot processing (testable) -----------------------------
def raw_odds_path(raw_dir: Path, event_id: str, role: str) -> Path:
    return Path(raw_dir) / f"{event_id}_{role}.json"


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def save_json_atomic(obj, path: Path) -> str:
    """Write JSON via temp file + fsync + atomic rename; return the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.json")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def process_snapshot(
    client,
    *,
    event_id: str,
    role: str,
    tip: datetime,
    gid,
    season,
    game_date: str,
    roster_df: pd.DataFrame,
    raw_dir: Path,
    part_dir: Path,
    state: dict,
    state_path: Path,
    collection_ts: str,
    fault_after: str | None = None,
    no_fetch: bool = False,
) -> dict:
    """Durably process one event/snapshot with the mandated ordering (raw->close/fsync->
    normalize->partition->validate->checkpoint). Idempotent + resumable: COMPLETE/NO_DATA/
    HTTP_404 are skipped with no API call; a valid RAW_SAVED cache is reused (no API call).

    ``fault_after`` (tests only) raises RuntimeError immediately after the named step
    ('raw_save' | 'normalize') to simulate interruption.
    Returns a result dict {status, from_cache, n_rows, api_call, raw_sha, error}.
    """
    key = state_key(event_id, role)
    cur = state.get(key)
    res = {"event_id": event_id, "role": role, "status": cur or NOT_STARTED,
           "from_cache": False, "n_rows": 0, "api_call": False, "raw_sha": None, "error": None}
    if cur in (COMPLETE, NO_DATA, HTTP_404):
        res["status"] = cur
        res["skipped"] = True
        return res

    raw_path = raw_odds_path(raw_dir, event_id, role)
    requested_snap = cutoffs_for(tip)[1] if role == "closing" else cutoffs_for(tip)[0]
    payload = None

    # Reuse a valid cached raw response (no API spend).
    if raw_path.exists() and cur in (RAW_SAVED, NORMALIZED, VALIDATED):
        try:
            payload = json.loads(raw_path.read_text())
            res["from_cache"] = True
            res["raw_sha"] = sha256_file(raw_path)
        except Exception:  # noqa: BLE001
            payload = None
    if payload is None and raw_path.exists():
        try:
            payload = json.loads(raw_path.read_text())
            res["from_cache"] = True
            res["raw_sha"] = sha256_file(raw_path)
        except Exception:  # noqa: BLE001
            payload = None

    if payload is None and no_fetch:
        # cache-only warm-up: don't spend a request; leave for a later fetch pass.
        res["status"] = "NEEDS_FETCH"
        res["needs_fetch"] = True
        return res

    if payload is None:
        # Fetch (may raise OddsAPIError: budget -> propagate; 404 -> tombstone).
        from wnba_props_model.constants import MODEL_PROP_MARKET_KEYS
        try:
            payload = client.get_historical_event_odds(event_id, requested_snap,
                                                       markets=list(MODEL_PROP_MARKET_KEYS))
            res["api_call"] = True
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "budget reached" in msg:
                raise
            if "404" in msg:
                state[key] = HTTP_404
                save_state(state_path, state)
                res["status"] = HTTP_404
                res["error"] = msg[:200]
                return res
            res["error"] = f"{type(exc).__name__}: {msg}"[:200]
            return res
        res["raw_sha"] = save_json_atomic(payload, raw_path)   # 1-2. raw save + fsync + rename
        state[key] = RAW_SAVED
        save_state(state_path, state)                          # durable checkpoint
        if fault_after == "raw_save":
            raise RuntimeError("fault_after=raw_save")

    # 3. normalize
    rows = parse_event_odds(payload, role=role, tip=tip, event_id=event_id, gid=gid,
                            roster_df=roster_df, collection_ts=collection_ts,
                            requested_snapshot_utc=requested_snap)
    res["n_rows"] = len(rows)
    if not rows:
        state[key] = NO_DATA
        save_state(state_path, state)
        res["status"] = NO_DATA
        return res
    if fault_after == "normalize":
        raise RuntimeError("fault_after=normalize")

    # 4. write partition (temp + atomic rename)
    path = side_partition_path(part_dir, season, game_date, event_id, role)
    write_rows_atomic(rows, path)
    state[key] = NORMALIZED
    save_state(state_path, state)

    # 5. validate timing on eligible rows
    timing = validate_timing(pd.DataFrame(rows))
    if not timing["ok"]:
        state[key] = BLOCKED
        save_state(state_path, state)
        res["status"] = BLOCKED
        res["error"] = f"timing_invariant_failed: {timing}"
        return res

    # 6. checkpoint COMPLETE
    state[key] = COMPLETE
    save_state(state_path, state)
    res["status"] = COMPLETE
    return res
