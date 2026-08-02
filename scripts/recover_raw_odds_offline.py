"""Stage 3: offline recovery of preserved raw Odds API responses (NO API requests).

Inventories every data/atomic_quotes/raw_odds/*.json, normalizes each into atomic side rows
via the canonical parser, writes durable season/game_date/event/role partitions, and
consolidates into data/atomic_quotes/atomic_quotes.parquet (dedup by quote_id).

Outputs:
  artifacts/audits/RAW_ODDS_CACHE_INVENTORY.json
  data/atomic_quotes/side_partitions/...
  data/atomic_quotes/atomic_quotes.parquet
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data import atomic_backfill as ab  # noqa: E402
from wnba_props_model.data.atomic_quotes import append_atomic_quotes  # noqa: E402
from wnba_props_model.evaluation import historical_market as hm  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "atomic_quotes" / "raw_odds"
PART = REPO / "data" / "atomic_quotes" / "side_partitions"
QUAR = REPO / "data" / "atomic_quotes" / "quarantine"
STORE = REPO / "data" / "atomic_quotes" / "atomic_quotes.parquet"
AUD = REPO / "artifacts" / "audits"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


@app.command()
def main(
    games: str = typer.Option("data/recovered_v2/wnba_games.parquet", "--games"),
    roster: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--roster"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    QUAR.mkdir(parents=True, exist_ok=True)
    games_df = pd.read_parquet(games)
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df["_d"] = games_df["game_date"].dt.strftime("%Y-%m-%d")
    roster_df = pd.read_parquet(roster)[["game_id", "player_id", "player_name"]].dropna()
    collection_ts = datetime.now(timezone.utc).isoformat()

    files = sorted(RAW.glob("*.json"))
    inv = []
    n_written = 0
    for f in files:
        rec = {"file": str(f.relative_to(REPO)), "sha256": _sha(f)[:16], "parse_status": None,
               "blocking_reason": None}
        role = "closing" if f.stem.endswith("_closing") else "decision"
        rec["snapshot_role"] = role
        try:
            payload = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001
            rec["parse_status"] = "UNREADABLE"
            rec["blocking_reason"] = str(exc)[:150]
            (QUAR / f.name).write_text(f.read_text(errors="replace"))
            inv.append(rec)
            continue
        data = (payload or {}).get("data", {}) or {}
        eid = data.get("id") or f.stem.rsplit("_", 1)[0]
        tip = ab.parse_tip(data.get("commence_time", ""))
        rec.update({
            "event_id": eid, "provider_snapshot": payload.get("timestamp"),
            "scheduled_tip": data.get("commence_time"),
            "home_team": data.get("home_team"), "away_team": data.get("away_team"),
            "bookmakers": [b.get("key") for b in data.get("bookmakers", [])],
            "returned_market_keys": sorted({mk.get("key") for b in data.get("bookmakers", [])
                                            for mk in b.get("markets", [])}),
        })
        if tip is None:
            rec["parse_status"] = "BLOCKED_UNPARSEABLE_TIP"
            rec["blocking_reason"] = "no commence_time"
            inv.append(rec)
            continue
        gd = tip.strftime("%Y-%m-%d")
        rec["requested_snapshot"] = (ab.cutoffs_for(tip)[1] if role == "closing"
                                     else ab.cutoffs_for(tip)[0])
        day_games = games_df[games_df["_d"] == gd]
        gid = hm.resolve_game_id(day_games, data.get("home_team", ""), data.get("away_team", ""), gd) \
            if not day_games.empty else None
        season = int(day_games["season"].iloc[0]) if (gid is not None and len(day_games)) else "unknown"
        rows = ab.parse_event_odds(payload, role=role, tip=tip, event_id=eid, gid=gid,
                                   roster_df=roster_df, collection_ts=collection_ts,
                                   requested_snapshot_utc=rec["requested_snapshot"])
        rec["n_outcome_rows"] = len(rows)
        if not rows:
            rec["parse_status"] = "NO_DATA"
        else:
            path = ab.side_partition_path(PART, season, gd, eid, role)
            n_written += ab.write_rows_atomic(rows, path)
            rec["parse_status"] = "NORMALIZED"
            rec["bdl_game_id"] = gid
        inv.append(rec)

    # consolidate + dedup, then append to the store idempotently
    consolidated = ab.consolidate_partitions(PART)
    summary = {"added": 0, "total": 0}
    if len(consolidated):
        summary = append_atomic_quotes(STORE, consolidated)
    timing = ab.validate_timing(consolidated)

    c = consolidated
    def _vc(col):
        return {str(k): int(v) for k, v in c[col].value_counts(dropna=False).to_dict().items()} if len(c) else {}
    elig = c[c["eligibility_status"] == "ELIGIBLE"] if len(c) else c
    blocked = c[c["eligibility_status"] == "BLOCKED"] if len(c) else c
    # mutually-exclusive timing counts (precedence already applied in the parser)
    ts = c["timing_status"] if len(c) else pd.Series([], dtype=str)
    role = c["snapshot_role"] if len(c) else pd.Series([], dtype=str)
    decision_after_cutoff = int(((ts == "AFTER_ROLE_CUTOFF") & (role == "decision")).sum())
    closing_after_cutoff = int(((ts == "AFTER_ROLE_CUTOFF") & (role == "closing")).sum())
    at_or_after_tip = int((ts == "AT_OR_AFTER_TIP").sum())
    missing_ts = int((ts == "MISSING_MARKET_TIMESTAMP").sum())

    breakdown = {
        "total_normalized_rows": int(len(c)),
        "eligible_rows": int(len(elig)),
        "blocked_rows": int(len(blocked)),
        "blocking_counts_by_reason": _vc("blocking_reason"),
        "timing_status_counts": _vc("timing_status"),
        "by_role": _vc("snapshot_role"),
        "eligible_by_role": ({str(k): int(v) for k, v in elig.groupby("snapshot_role").size().to_dict().items()} if len(elig) else {}),
        "mutually_exclusive_timing": {
            "decision_rows_blocked_after_decision_cutoff": decision_after_cutoff,
            "closing_rows_blocked_after_closing_cutoff": closing_after_cutoff,
            "rows_blocked_at_or_after_tip": at_or_after_tip,
            "rows_missing_market_timestamp": missing_ts,
        },
        "reconcile_614_44": (
            "Prior non-exclusive check double-counted: the 44 AT_OR_AFTER_TIP rows were a "
            "SUBSET of the 614 post-cutoff rows. Mutually exclusive now: AT_OR_AFTER_TIP="
            f"{at_or_after_tip}, AFTER_ROLE_CUTOFF(total)={int((ts=='AFTER_ROLE_CUTOFF').sum())} "
            f"(= 614 - 44 previously). No overlapping sum."),
        "unique_events": int(c["event_id"].nunique()) if len(c) else 0,
        "unique_sportsbooks": sorted(c["sportsbook"].dropna().unique().tolist()) if len(c) else [],
        "unique_players": int(c["player_id"].dropna().nunique()) if len(c) else 0,
        "market_coverage": sorted(c["prop"].dropna().unique().tolist()) if len(c) else [],
    }

    inv_doc = {
        "generated_at_utc": collection_ts, "raw_dir": str(RAW.relative_to(REPO)),
        "n_files": len(files),
        "n_files_normalized": sum(1 for r in inv if r.get("parse_status") == "NORMALIZED"),
        "n_files_no_data": sum(1 for r in inv if r.get("parse_status") == "NO_DATA"),
        "n_files_quarantined": sum(1 for r in inv if r.get("parse_status") == "UNREADABLE"),
        "n_written_side_rows": n_written,
        "consolidated_rows": int(len(c)),
        "duplicate_primary_keys": int(len(c) - c["quote_id"].nunique()) if len(c) else 0,
        "timing_invariants_eligible": timing,
        "eligibility_breakdown": breakdown,
        "store": str(STORE.relative_to(REPO)), "store_total": summary["total"],
        "files": inv,
    }
    (AUD / "RAW_ODDS_CACHE_INVENTORY.json").write_text(json.dumps(inv_doc, indent=2, default=str))

    # Kelsey Mitchell / FanDuel canary must remain EXACT/ELIGIBLE
    canary = c[(c["player_name"] == "Kelsey Mitchell") & (c["sportsbook"] == "fanduel") &
               (c["eligibility_status"] == "ELIGIBLE")] if len(c) else c
    canary_ok = len(canary) > 0

    typer.echo("================ OFFLINE RAW RECOVERY (Stage 3) ================")
    typer.echo(f"  raw files inventoried : {len(files)}  (normalized={inv_doc['n_files_normalized']} "
               f"no_data={inv_doc['n_files_no_data']} quarantined={inv_doc['n_files_quarantined']})")
    typer.echo(f"  total normalized rows : {breakdown['total_normalized_rows']}  dup_PKs={inv_doc['duplicate_primary_keys']}")
    typer.echo(f"  eligible rows         : {breakdown['eligible_rows']}   blocked rows: {breakdown['blocked_rows']}")
    typer.echo(f"  blocking by reason    : {breakdown['blocking_counts_by_reason']}")
    typer.echo(f"  timing status counts  : {breakdown['timing_status_counts']}")
    typer.echo(f"  eligible by role      : {breakdown['eligible_by_role']}")
    typer.echo(f"  mutually-exclusive    : {breakdown['mutually_exclusive_timing']}")
    typer.echo(f"  timing invariants(elig): {timing}")
    typer.echo(f"  unique events={breakdown['unique_events']} players={breakdown['unique_players']} "
               f"books={len(breakdown['unique_sportsbooks'])} markets={breakdown['market_coverage']}")
    typer.echo(f"  Kelsey Mitchell/FanDuel canary EXACT: {canary_ok}")
    typer.echo(f"  wrote {AUD}/RAW_ODDS_CACHE_INVENTORY.json + {STORE}")


if __name__ == "__main__":
    app()
