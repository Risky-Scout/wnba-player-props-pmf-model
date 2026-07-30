"""Durable, resumable historical WNBA player-prop backfill (Stages 4/5/6).

Per event/snapshot the mandated order is: save raw -> fsync/close -> normalize -> write
partition -> validate timestamps+PKs -> durable checkpoint -> next. Restart verifies cached
raw hashes, skips COMPLETE work, rebuilds NORMALIZED from RAW_SAVED, never repeats a valid
cached API request, and never needs a final in-memory append. Fail-closed on the credit
budget and on the exact 12-market scope; deterministic HTTP 404 tombstones.

Only completed WNBA games, on/after historical prop coverage, are eligible; discovery and
event-odds responses are cached deterministically. Never substitutes a closing quote for a
decision quote, another book/line/player, a later quote, or a requested timestamp.
"""
from __future__ import annotations

import csv
import json
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.constants import MODEL_PROP_MARKET_KEYS  # noqa: E402
from wnba_props_model.data import atomic_backfill as ab  # noqa: E402
from wnba_props_model.data.atomic_quotes import append_atomic_quotes  # noqa: E402
from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError  # noqa: E402
from wnba_props_model.evaluation import historical_market as hm  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"
AQ = REPO / "data" / "atomic_quotes"
RAW = AQ / "raw_odds"
RAW_EVENTS = AQ / "raw_events"
PART = AQ / "side_partitions"
STATE = AQ / "backfill_state.json"
STORE = AQ / "atomic_quotes.parquet"
COVERAGE_START = "2023-05-23"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _discover(client: OddsAPIClient, gd: str, use_cache: bool) -> list[dict]:
    RAW_EVENTS.mkdir(parents=True, exist_ok=True)
    cache = RAW_EVENTS / f"{gd}.json"
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text()).get("data", []) or []
        except Exception:  # noqa: BLE001
            pass
    gd_next = (datetime.fromisoformat(f"{gd}T00:00:00+00:00") + timedelta(days=1)).strftime("%Y-%m-%d")
    payload = client.list_historical_events(f"{gd}T12:00:00Z", commence_time_from=f"{gd}T00:00:00Z",
                                            commence_time_to=f"{gd_next}T12:00:00Z")
    ab.save_json_atomic(payload, cache)
    return (payload or {}).get("data", []) or []


def _audit_line(path: Path, rec: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


@app.command()
def main(
    games: str = typer.Option("data/recovered_v2/wnba_games.parquet", "--games"),
    roster: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--roster"),
    start_date: str = typer.Option(COVERAGE_START, "--start-date"),
    end_date: str = typer.Option("2026-12-31", "--end-date"),
    max_credits: int = typer.Option(260000, "--max-credits", help="fail-closed budget (session)"),
    sleep_s: float = typer.Option(0.1, "--sleep"),
    max_new_events: int = typer.Option(0, "--max-new-events",
                                       help="stop after N events that required a NEW api fetch (0=all)"),
    only_decision: bool = typer.Option(False, "--only-decision"),
    heartbeat: int = typer.Option(25, "--heartbeat"),
    consolidate: bool = typer.Option(True, "--consolidate/--no-consolidate"),
    no_fetch: bool = typer.Option(False, "--no-fetch", help="cache-only warm-up: never spend a request"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    request_audit = str(AUD / "ODDS_API_REQUEST_AUDIT.jsonl")
    backfill_audit = AUD / "BACKFILL_REQUEST_AUDIT.jsonl"
    client = OddsAPIClient(region="us", max_credits=max_credits, request_audit_path=request_audit,
                           enforce_model_markets=True)
    used_at_start = None

    g = pd.read_parquet(games)
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["_d"] = g["game_date"].dt.strftime("%Y-%m-%d")
    now_utc = datetime.now(timezone.utc)
    final = g[(g.get("status_normalized") == "final") & (g["_d"] >= start_date) &
              (g["_d"] <= end_date) & (g["game_date"] <= now_utc.tz_localize(None) if g["game_date"].dt.tz is None
                                       else g["game_date"] <= now_utc)].copy()
    roster_df = pd.read_parquet(roster)[["game_id", "player_id", "player_name"]].dropna()
    labels = ["decision"] if only_decision else ["decision", "closing"]
    dates = sorted(final["_d"].unique())
    collection_ts = now_utc.isoformat()

    state = ab.load_state(STATE)
    map_rows: list[dict] = []
    n_new_events = 0
    n_completed = 0
    stopped = None

    interrupted = {"flag": False}
    def _sigint(_sig, _frm):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    typer.echo(f"[backfill] dates={len(dates)} labels={labels} budget={max_credits} "
               f"cached_events={sum(1 for v in state.values() if v=='COMPLETE')}")
    try:
        for gd in dates:
            if interrupted["flag"] or stopped:
                break
            events = _discover(client, gd, use_cache=True)
            if used_at_start is None:
                used_at_start = client.quota_used
            day_games = final[final["_d"] == gd]
            for e in events:
                if interrupted["flag"] or stopped:
                    break
                eid = e.get("id", "")
                tip = ab.parse_tip(e.get("commence_time", ""))
                gid = (hm.resolve_game_id(day_games, e.get("home_team", ""), e.get("away_team", ""), gd)
                       if not day_games.empty else None)
                matched = gid is not None and tip is not None
                season = int(day_games["season"].iloc[0]) if (matched and len(day_games)) else "unknown"
                map_rows.append({"game_date": gd, "odds_event_id": eid, "home_team": e.get("home_team"),
                                 "away_team": e.get("away_team"), "commence_time": e.get("commence_time"),
                                 "bdl_game_id": gid, "matched": matched})
                if not matched:
                    continue
                event_had_new = False
                for role in labels:
                    if max_new_events and n_new_events >= max_new_events:
                        stopped = "max_new_events"
                        break
                    try:
                        r = ab.process_snapshot(
                            client, event_id=eid, role=role, tip=tip, gid=gid, season=season,
                            game_date=gd, roster_df=roster_df, raw_dir=RAW, part_dir=PART,
                            state=state, state_path=STATE, collection_ts=collection_ts,
                            no_fetch=no_fetch)
                    except OddsAPIError as exc:
                        if "budget reached" in str(exc):
                            stopped = "budget"
                            break
                        raise
                    if r.get("api_call"):
                        event_had_new = True
                    _audit_line(backfill_audit, {
                        "ts_utc": datetime.now(timezone.utc).isoformat(), "event_id": eid, "role": role,
                        "requested_markets": list(MODEL_PROP_MARKET_KEYS), "n_markets": 12,
                        "status": r["status"], "api_call": r["api_call"], "from_cache": r["from_cache"],
                        "n_rows": r["n_rows"], "raw_cache_sha256": r["raw_sha"],
                        "x_requests_used": client.quota_used, "x_requests_remaining": client.quota_remaining,
                        "x_requests_last": client.quota_last, "error": r.get("error")})
                    if r["status"] == ab.COMPLETE:
                        n_completed += 1
                        if n_completed % heartbeat == 0:
                            typer.echo(f"[heartbeat] completed={n_completed} new_events={n_new_events} "
                                       f"spent={client.credits_spent_session} remaining={client.quota_remaining}")
                    if r.get("api_call") and sleep_s:
                        time.sleep(sleep_s)
                if event_had_new:
                    n_new_events += 1
    except KeyboardInterrupt:
        interrupted["flag"] = True

    ab.save_state(STATE, state)
    # event-id mapping (append/refresh)
    if map_rows:
        with open(AUD / "EVENT_ID_MAPPING_AUDIT.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(map_rows[0].keys()))
            w.writeheader(); w.writerows(map_rows)

    store_total = 0
    if consolidate:
        consolidated = ab.consolidate_partitions(PART)
        if len(consolidated):
            store_total = append_atomic_quotes(STORE, consolidated)["total"]

    typer.echo("================ BACKFILL SUMMARY ================")
    typer.echo(f"  interrupted={interrupted['flag']} stopped={stopped}")
    typer.echo(f"  completed event/snapshots : {n_completed}   new-fetch events: {n_new_events}")
    typer.echo(f"  credits spent (session)   : {client.credits_spent_session}")
    typer.echo(f"  x-requests used/remaining : {client.quota_used} / {client.quota_remaining}  last={client.quota_last}")
    typer.echo(f"  store total rows          : {store_total}")
    typer.echo(f"  checkpoint                : {STATE}")


if __name__ == "__main__":
    app()
