"""Resumable, budgeted historical WNBA player-prop backfill (Sections D/E/F/H).

For each completed BDL game date: discover Odds API historical events, match to canonical
BDL game_ids by exact normalized team pair + date, and for each matched event pull the 12
model-supported markets at the frozen decision (tip-12h) and closing (tip-5m) snapshots.
Rows are written to the append-only atomic store with FULL timestamp provenance (requested
vs provider vs market_last_update); the quote_id and pairing use the ACTUAL quote time.

Resumable: raw responses are cached per (event, snapshot); re-runs skip cached events and
the append-only store dedups by quote_id. Fail-closed on the ODDS_API_MAX_CREDITS budget.

Canary: --max-events 1 --only-decision runs a single event/snapshot for verification.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.constants import (  # noqa: E402
    MODEL_PROP_MARKET_KEYS as ODDS_API_MODEL_MARKET_KEYS,
)
from wnba_props_model.constants import (  # noqa: E402
    MODEL_PROP_MARKETS as ODDS_API_MODEL_MARKETS,
)
from wnba_props_model.data.atomic_quotes import (  # noqa: E402
    ATOMIC_QUOTE_COLUMNS,
    BLOCKED_EXACT_QUOTES,
    EXACT,
    append_atomic_quotes,
    atomic_quote_id,
)
from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError  # noqa: E402
from wnba_props_model.evaluation import historical_market as hm  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"
RAW_CACHE = REPO / "data" / "atomic_quotes" / "raw_odds"

DECISION_LEAD_HOURS = 12
CLOSING_MINUTES = 5


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tip(commence: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def _cached_or_fetch(client: OddsAPIClient, event_id: str, snap: str, label: str,
                     markets: list[str], use_cache: bool) -> dict | None:
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_CACHE / f"{event_id}_{label}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:  # noqa: BLE001
            pass
    odds = client.get_historical_event_odds(event_id, snap, markets=markets)
    try:
        cache_file.write_text(json.dumps(odds))
    except OSError:
        pass
    return odds


def _parse_event_odds(odds: dict, *, requested_snap: str, role: str, decision_cut: str,
                      closing_cut: str, tip_iso: str, event_id: str, gid, roster_df: pd.DataFrame,
                      collection_ts: str) -> list[dict]:
    wrapper_ts = (odds or {}).get("timestamp")
    prev_ts = (odds or {}).get("previous_timestamp")
    next_ts = (odds or {}).get("next_timestamp")
    role_cut = closing_cut if role == "closing" else decision_cut
    rows: list[dict] = []
    for book in (odds or {}).get("data", {}).get("bookmakers", []):
        bkey = book.get("key", "")
        book_last = book.get("last_update")
        for m in book.get("markets", []):
            stat = ODDS_API_MODEL_MARKETS.get(m.get("key", ""))
            if not stat:
                continue
            mkt_last = m.get("last_update")
            # ACTUAL quote timestamp is strictly market_last_update. If absent, BLOCK exact
            # timing rather than fabricating it from the requested snapshot date.
            if mkt_last:
                quote_ts, quote_ts_src = mkt_last, "market_last_update"
            else:
                quote_ts, quote_ts_src = None, "BLOCKED_NO_MARKET_TIMESTAMP"
            id_ts = mkt_last or f"BLOCKED::{requested_snap}"
            for oc in m.get("outcomes", []):
                name = oc.get("description", "")
                pid, _method = (hm.resolve_player_id(name, gid, roster_df)
                                if (gid is not None and not roster_df.empty) else (None, "unmatched"))
                side = str(oc.get("name", "")).lower()
                line = oc.get("point")
                exact_ok = gid is not None and pid is not None and mkt_last is not None
                status = EXACT if exact_ok else BLOCKED_EXACT_QUOTES
                # BOTH sides of the same market object inherit that object's market_last_update.
                rows.append({
                    "quote_id": atomic_quote_id(bkey, event_id, pid or name, stat, line, side, id_ts),
                    "sportsbook": bkey, "event_id": event_id, "game_id": gid, "player_id": pid,
                    "player_name": name, "prop": stat, "line": line, "side": side,
                    "american_odds": oc.get("price"),
                    # canonical immutable timestamp provenance
                    "snapshot_role": role, "snapshot_label": role,
                    "requested_snapshot_utc": requested_snap, "provider_snapshot_utc": wrapper_ts,
                    "previous_timestamp": prev_ts, "next_timestamp": next_ts,
                    "bookmaker_last_update_utc": book_last, "market_last_update_utc": mkt_last,
                    "quote_timestamp_utc": quote_ts, "quote_timestamp_source": quote_ts_src,
                    "scheduled_tip_utc": tip_iso, "decision_cutoff_utc": decision_cut,
                    "closing_cutoff_utc": closing_cut, "role_cutoff_utc": role_cut,
                    "collection_timestamp_utc": collection_ts,
                    # legacy mirrors (not used for pairing)
                    "snapshot_time": quote_ts, "decision_timestamp": decision_cut,
                    "requested_snapshot_time": requested_snap, "provider_snapshot_time": wrapper_ts,
                    "market_last_update": mkt_last, "collection_timestamp": collection_ts,
                    "prediction_timestamp": None, "model_prob_over_final": None,
                    "probability_lineage_version": None, "model_hash": None,
                    "calibrator_hash": None, "feature_schema_hash": None, "quote_policy_hash": None,
                    "settlement_status": "pending", "actual_outcome": None,
                    "exact_quote_status": status, "source": "odds_api_v4_historical",
                })
    return rows


@app.command()
def main(
    games: str = typer.Option("data/recovered_v2/wnba_games.parquet", "--games"),
    roster: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--roster"),
    start_date: str = typer.Option("2023-05-03", "--start-date"),
    end_date: str = typer.Option("2026-12-31", "--end-date"),
    store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store"),
    max_credits: int = typer.Option(300000, "--max-credits", help="fail-closed budget"),
    sleep_s: float = typer.Option(0.15, "--sleep"),
    max_events: int = typer.Option(0, "--max-events", help="0 = all (use small for canary)"),
    only_decision: bool = typer.Option(False, "--only-decision", help="canary: decision snapshot only"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    request_audit = str(AUD / "ODDS_API_REQUEST_AUDIT.jsonl")
    client = OddsAPIClient(region="us", max_credits=max_credits, request_audit_path=request_audit,
                           enforce_model_markets=True)

    g = pd.read_parquet(games)
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["_d"] = g["game_date"].dt.strftime("%Y-%m-%d")
    final = g[(g.get("status_normalized") == "final") &
              (g["_d"] >= start_date) & (g["_d"] <= end_date)].copy()
    roster_df = (pd.read_parquet(roster)[["game_id", "player_id", "player_name"]].dropna()
                 if Path(roster).exists() else pd.DataFrame())

    labels = ["decision"] if only_decision else ["decision", "closing"]
    dates = sorted(final["_d"].unique())
    collection_ts = datetime.now(timezone.utc).isoformat()

    mapping_path = AUD / "EVENT_ID_MAPPING_AUDIT.csv"
    map_rows: list[dict] = []
    all_rows: list[dict] = []
    n_events_done = 0
    stopped_budget = False

    for gd in dates:
        gd_next = (datetime.fromisoformat(f"{gd}T00:00:00+00:00") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            ev = client.list_historical_events(
                f"{gd}T12:00:00Z", commence_time_from=f"{gd}T00:00:00Z",
                commence_time_to=f"{gd_next}T12:00:00Z")
        except OddsAPIError as exc:
            if "budget reached" in str(exc):
                stopped_budget = True
                break
            typer.echo(f"[backfill] events fetch failed {gd}: {exc}", err=True)
            continue
        events = (ev or {}).get("data", []) or []
        day_games = final[final["_d"] == gd]
        for e in events:
            if max_events and n_events_done >= max_events:
                break
            eid = e.get("id", "")
            tip = _tip(e.get("commence_time", ""))
            gid = (hm.resolve_game_id(day_games, e.get("home_team", ""), e.get("away_team", ""), gd)
                   if not day_games.empty else None)
            matched = gid is not None and tip is not None
            map_rows.append({
                "game_date": gd, "odds_event_id": eid, "home_team": e.get("home_team"),
                "away_team": e.get("away_team"), "commence_time": e.get("commence_time"),
                "bdl_game_id": gid, "matched": matched,
                "reason": ("ok" if matched else ("unparseable_tip" if tip is None else "no_bdl_match")),
            })
            if not matched:
                continue
            decision_iso = _iso(tip - timedelta(hours=DECISION_LEAD_HOURS))
            closing_iso = _iso(tip - timedelta(minutes=CLOSING_MINUTES))
            snaps = {"decision": decision_iso, "closing": closing_iso}
            for label in labels:
                snap = snaps[label]
                try:
                    odds = _cached_or_fetch(client, eid, snap, label,
                                            list(ODDS_API_MODEL_MARKET_KEYS), use_cache=not no_cache)
                except OddsAPIError as exc:
                    if "budget reached" in str(exc):
                        stopped_budget = True
                        break
                    continue
                all_rows.extend(_parse_event_odds(
                    odds, requested_snap=snap, role=label, decision_cut=decision_iso,
                    closing_cut=closing_iso, tip_iso=_iso(tip), event_id=eid, gid=gid,
                    roster_df=roster_df, collection_ts=collection_ts))
                if sleep_s:
                    time.sleep(sleep_s)
            n_events_done += 1
            if stopped_budget:
                break
        typer.echo(f"[backfill] {gd}: events={len(events)} rows={len(all_rows)} "
                   f"matched_events={n_events_done} spent={client.credits_spent_session} "
                   f"remaining={client.quota_remaining}")
        if stopped_budget or (max_events and n_events_done >= max_events):
            break

    # write event-id mapping audit (append/refresh)
    if map_rows:
        with open(mapping_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(map_rows[0].keys()))
            w.writeheader()
            w.writerows(map_rows)

    added = total = 0
    if all_rows:
        new_df = pd.DataFrame(all_rows).reindex(columns=ATOMIC_QUOTE_COLUMNS)
        summary = append_atomic_quotes(store, new_df)
        added, total = summary["added"], summary["total"]
        n_exact = int((new_df["exact_quote_status"] == EXACT).sum())
        by_prop = new_df.groupby("prop").size().to_dict()
    else:
        n_exact, by_prop = 0, {}

    typer.echo("")
    typer.echo("================ BACKFILL SUMMARY ================")
    typer.echo(f"  matched events processed : {n_events_done}")
    typer.echo(f"  raw side quotes           : {len(all_rows)}  appended={added} total={total}")
    typer.echo(f"  EXACT (id-resolved)       : {n_exact}")
    typer.echo(f"  by prop (raw sides)       : {by_prop}")
    typer.echo(f"  credits spent (session)   : {client.credits_spent_session}")
    typer.echo(f"  quota remaining           : {client.quota_remaining}")
    typer.echo(f"  stopped_on_budget         : {stopped_budget}")
    typer.echo(f"  event-id mapping          : {mapping_path}")
    typer.echo("==================================================")


if __name__ == "__main__":
    app()
