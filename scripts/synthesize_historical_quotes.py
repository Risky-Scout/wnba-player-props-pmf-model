"""Synthesize the historical atomic player-prop quote store from The Odds API.

Fills the exact-quote evidence base that `collect_atomic_quotes.py` normally accrues live,
using The Odds API **historical** endpoints (player props back to May 2023, 5-minute
snapshots). It writes into the canonical append-only atomic quote store
(`data/atomic_quotes/atomic_quotes.parquet`) using the EXACT schema
`atomic_quotes.ATOMIC_QUOTE_COLUMNS` — the same rows `collect_atomic_quotes.py` produces —
NOT into the derived `EXACT_QUOTE_READINESS.json` summary (that file is regenerated from the
store by `scripts/audit_exact_quote_readiness.py`; hand-appending raw quotes into it would
corrupt a computed artifact and misstate readiness).

Blueprint (per the directive):

  Step 1  Fetch historical WNBA events per date  -> eventId + commence_time (tip).
  Step 2  Decision time = tip - lead_hours (ISO8601 UTC) == snapshot_date.
  Step 3  GET historical event odds at snapshot_date for the requested prop markets.
  Step 4  Parse outcomes -> canonical atomic-quote rows (single book, one side).
  Step 5  Append to the atomic store; optionally build validated pairs and regenerate
          the readiness report; print how many EXACT pairs were synthesized.

Identity (game_id / player_id) is resolved against the BDL canonical tables when present
(`--games`, `--roster`); otherwise rows are recorded honestly as BLOCKED_EXACT_QUOTES and
will NOT become EXACT pairs until the canonical tables exist (build_canonical_tables.py).

Example:
    python3 scripts/synthesize_historical_quotes.py \
        --start-date 2023-05-01 --end-date 2023-09-30 \
        --markets player_steals,player_blocks,player_turnovers \
        --lead-hours 1 --build-pairs --build-readiness
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data.atomic_quotes import (
    ATOMIC_QUOTE_COLUMNS,
    BLOCKED_EXACT_QUOTES,
    EXACT,
    append_atomic_quotes,
    atomic_quote_id,
)
from wnba_props_model.data.odds_api_client import OddsAPIClient, OddsAPIError
from wnba_props_model.evaluation import historical_market as hm

app = typer.Typer(add_completion=False)

# Empirical coverage note (verified 2024-2025, us + us2 regions, tip-1h snapshots):
#   player_points / player_rebounds / player_assists / player_threes  -> 7 US books offer these.
#   player_steals / player_blocks / player_turnovers                  -> ZERO books offer these.
# US sportsbooks do not post WNBA steals/blocks/turnovers player props, so those markets cannot
# be synthesized from historical data (they return empty bookmakers). The default therefore
# targets the four market-evaluable props that actually unblock the market-superiority proof;
# STL/BLK/TOV can still be requested via --markets but will return zero rows.
DEFAULT_MARKETS = "player_points,player_rebounds,player_assists,player_threes"


def _daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def _tip(commence: str) -> datetime | None:
    """Parse the scheduled tip; NEVER substitute an assumed tip (an unparseable commence
    time means the event is skipped so no fabricated tip leaks into proof evidence)."""
    try:
        return datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snapshots(tip: datetime, lead_hours: float, closing_minutes: int) -> dict[str, str]:
    """Decision snapshot at tip - lead_hours (Step 2) plus a closing snapshot a few minutes
    pre-tip for robustness. Both are strictly pre-tip, so the resulting pairs are valid."""
    return {
        "decision": _iso(tip - timedelta(hours=lead_hours)),
        "closing": _iso(tip - timedelta(minutes=closing_minutes)),
    }


def _load_canonical(games: str, roster: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_df = pd.read_parquet(games) if Path(games).exists() else pd.DataFrame()
    if not games_df.empty and "game_date" in games_df.columns:
        games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.strftime("%Y-%m-%d")
    roster_df = (
        pd.read_parquet(roster)[["game_id", "player_id", "player_name"]].dropna()
        if Path(roster).exists() else pd.DataFrame()
    )
    return games_df, roster_df


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date", help="YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., "--end-date", help="YYYY-MM-DD (inclusive)"),
    markets: str = typer.Option(DEFAULT_MARKETS, "--markets", help="comma-separated Odds API market keys"),
    lead_hours: float = typer.Option(1.0, "--lead-hours", help="decision buffer before tip (Step 2)"),
    closing_minutes: int = typer.Option(5, "--closing-minutes", help="closing snapshot minutes before tip"),
    regions: str = typer.Option("us", "--regions"),
    sleep_s: float = typer.Option(0.25, "--sleep", help="rate-limit sleep between event-odds calls"),
    store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store"),
    games: str = typer.Option("data/processed/wnba_games.parquet", "--games"),
    roster: str = typer.Option("data/processed/wnba_player_game_stats.parquet", "--roster"),
    max_events: int = typer.Option(0, "--max-events", help="0 = no cap (smoke-test with a small cap)"),
    build_pairs: bool = typer.Option(False, "--build-pairs", help="build validated pairs after collection"),
    build_readiness: bool = typer.Option(False, "--build-readiness", help="regenerate EXACT_QUOTE_READINESS.json"),
) -> None:
    market_list = [m.strip() for m in markets.split(",") if m.strip()]
    unknown = [m for m in market_list if m not in hm.MARKET_TO_STAT]
    if unknown:
        typer.echo(f"[synth] WARNING: markets not in MARKET_TO_STAT (will be dropped on parse): {unknown}", err=True)

    client = OddsAPIClient(region=regions)
    games_df, roster_df = _load_canonical(games, roster)
    if roster_df.empty:
        typer.echo("[synth] NOTE: no canonical roster/games parquet found -> player/game ids "
                   "cannot be resolved; rows recorded as BLOCKED_EXACT_QUOTES (not EXACT pairs) "
                   "until build_canonical_tables.py has run.", err=True)

    rows: list[dict] = []
    n_events = 0
    n_event_odds_calls = 0
    for gd in _daterange(start_date, end_date):
        gd_next = (datetime.fromisoformat(f"{gd}T00:00:00+00:00") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            ev = client.list_historical_events(
                f"{gd}T12:00:00Z",
                commence_time_from=f"{gd}T00:00:00Z",
                commence_time_to=f"{gd_next}T12:00:00Z",
            )
        except OddsAPIError as exc:
            typer.echo(f"[synth] events fetch failed {gd}: {exc}", err=True)
            continue
        events = (ev or {}).get("data", []) or []
        day_games = games_df[games_df.get("game_date") == gd] if ("game_date" in games_df.columns) else games_df
        for e in events:
            if max_events and n_events >= max_events:
                break
            eid = e.get("id", "")
            tip = _tip(e.get("commence_time", ""))
            if tip is None:
                typer.echo(f"[synth] {gd} event {eid}: unparseable commence_time -> SKIPPED", err=True)
                continue
            n_events += 1
            gid = (hm.resolve_game_id(day_games, e.get("home_team", ""), e.get("away_team", ""), gd)
                   if not day_games.empty else None)
            snaps = _snapshots(tip, lead_hours, closing_minutes)
            decision_iso = snaps["decision"]
            for label, snap in snaps.items():
                try:
                    odds = client.get_historical_event_odds(eid, snap, markets=market_list)
                    n_event_odds_calls += 1
                except OddsAPIError as exc:
                    typer.echo(f"[synth] {gd} event {eid} @ {snap}: odds fetch failed: {exc}", err=True)
                    continue
                for book in (odds or {}).get("data", {}).get("bookmakers", []):
                    bkey = book.get("key", "")
                    for m in book.get("markets", []):
                        stat = hm.MARKET_TO_STAT.get(m.get("key", ""))
                        if not stat:
                            continue
                        for oc in m.get("outcomes", []):
                            name = oc.get("description", "")
                            pid, _method = (
                                hm.resolve_player_id(name, gid, roster_df)
                                if (gid is not None and not roster_df.empty) else (None, "unmatched")
                            )
                            side = str(oc.get("name", "")).lower()
                            line = oc.get("point")
                            status = EXACT if (gid is not None and pid is not None) else BLOCKED_EXACT_QUOTES
                            rows.append({
                                "quote_id": atomic_quote_id(bkey, eid, pid or name, stat, line, side, snap),
                                "sportsbook": bkey, "event_id": eid, "game_id": gid, "player_id": pid,
                                "player_name": name, "prop": stat, "line": line, "side": side,
                                "american_odds": oc.get("price"), "snapshot_label": label,
                                "snapshot_time": snap, "decision_timestamp": decision_iso,
                                "scheduled_tip_utc": _iso(tip), "prediction_timestamp": None,
                                "model_prob_over_final": None, "probability_lineage_version": None,
                                "model_hash": None, "calibrator_hash": None,
                                "feature_schema_hash": None, "quote_policy_hash": None,
                                "settlement_status": "pending", "actual_outcome": None,
                                "exact_quote_status": status, "source": "odds_api_v4_historical_synth",
                            })
                if sleep_s:
                    time.sleep(sleep_s)
        typer.echo(f"[synth] {gd}: events={len(events)} rows_so_far={len(rows)} "
                   f"quota_remaining={client.quota_remaining} used={client.quota_used}")
        if max_events and n_events >= max_events:
            break

    if not rows:
        typer.echo("[synth] no quotes synthesized (check date range / historical plan access).", err=True)
        raise typer.Exit(1)

    new_df = pd.DataFrame(rows)[ATOMIC_QUOTE_COLUMNS]
    summary = append_atomic_quotes(store, new_df)
    n_exact = int((new_df["exact_quote_status"] == EXACT).sum())
    n_blocked = int((new_df["exact_quote_status"] == BLOCKED_EXACT_QUOTES).sum())
    typer.echo("")
    typer.echo("================ SYNTHESIS SUMMARY ================")
    typer.echo(f"  event-odds snapshots requested : {n_event_odds_calls}")
    typer.echo(f"  raw single-side quotes         : {len(new_df)}")
    typer.echo(f"  appended (new quote_ids)       : {summary['added']}")
    typer.echo(f"  store total rows               : {summary['total']}")
    typer.echo(f"  EXACT (id-resolved)            : {n_exact}")
    typer.echo(f"  BLOCKED_EXACT_QUOTES           : {n_blocked}")
    by_prop = new_df.groupby("prop").size().to_dict()
    typer.echo(f"  by prop (raw sides)            : {by_prop}")
    requested_stats = {hm.MARKET_TO_STAT[m] for m in market_list if m in hm.MARKET_TO_STAT}
    zero_cov = sorted(requested_stats - set(by_prop))
    if zero_cov:
        typer.echo(f"  ZERO-COVERAGE requested props  : {zero_cov} (no US book offers these; "
                   "not synthesizable from historical data)")
    typer.echo(f"  quota used / remaining         : {client.quota_used} / {client.quota_remaining}")
    typer.echo("===================================================")

    if build_pairs:
        _build_pairs_and_report(store)
    if build_readiness:
        _regenerate_readiness()


def _build_pairs_and_report(store: str) -> None:
    """Transform the flat atomic store into the raw-side schema and build validated pairs,
    writing data/atomic_quotes/pairs/synth/pairs.parquet, then report EXACT_PAIR counts."""
    from wnba_props_model.data.quote_pairs import EXACT_PAIR, build_quote_pairs
    flat = pd.read_parquet(store)
    raw = flat.rename(columns={
        "snapshot_time": "snapshot_timestamp",
        "decision_timestamp": "decision_timestamp_utc",
    }).copy()
    raw["provider"] = "odds_api"
    pairs_frames = []
    for label in sorted(raw["snapshot_label"].dropna().unique()):
        sub = raw[raw["snapshot_label"] == label]
        pairs_frames.append(build_quote_pairs(sub, snapshot_label=str(label)))
    pairs = pd.concat(pairs_frames, ignore_index=True) if pairs_frames else pd.DataFrame()
    out = Path("data/atomic_quotes/pairs/synth/pairs.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False)
    exact = pairs[pairs["quote_pair_status"] == EXACT_PAIR] if len(pairs) else pairs
    typer.echo(f"[synth] wrote {len(pairs)} pairs -> {out}")
    if len(pairs):
        typer.echo(f"[synth] pair status counts: {pairs['quote_pair_status'].value_counts().to_dict()}")
        typer.echo(f"[synth] EXACT_PAIR by prop: {exact.groupby('prop').size().to_dict()}")
        typer.echo("[synth] NOTE: EXACT pairs still need settlement (BDL actual outcomes) and, for the "
                   "prospective readiness level, a delivered model_prob_over_final captured at "
                   "prediction time. Historical-replay OOF supplies the model probability at proof time.")


def _regenerate_readiness() -> None:
    import subprocess
    typer.echo("[synth] regenerating EXACT_QUOTE_READINESS.json ...")
    subprocess.run([sys.executable, "scripts/audit_exact_quote_readiness.py"], check=False)


if __name__ == "__main__":
    app()
