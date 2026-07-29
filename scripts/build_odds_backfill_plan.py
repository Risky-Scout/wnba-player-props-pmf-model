"""Section G: Odds API historical backfill credit plan (printed & verified BEFORE spending).

Writes artifacts/audits/ODDS_API_BACKFILL_PLAN.json from the canonical games table + the
canonical market constant, and verifies the expected spend against the live remaining quota
and the ODDS_API_MAX_CREDITS budget. One cheap historical-events call is made to read the
current quota headers (never prints the key).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.constants import ODDS_API_MODEL_MARKET_KEYS  # noqa: E402
from wnba_props_model.data.odds_api_client import OddsAPIClient  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"
HISTORICAL_MARKET_MULTIPLIER = 10   # historical event-odds cost = 10 x markets x regions


@app.command()
def main(
    games: str = typer.Option("data/recovered_v2/wnba_games.parquet", "--games"),
    start_date: str = typer.Option("2023-05-03", "--start-date"),
    coverage_start: str = typer.Option(
        "2023-05-23", "--coverage-start",
        help="first date with historical player-prop coverage; earlier games are excluded"),
    regions: str = typer.Option("us", "--regions"),
    snapshots_per_event: int = typer.Option(2, "--snapshots", help="decision + closing"),
    no_quota_probe: bool = typer.Option(False, "--no-quota-probe",
                                        help="do NOT call the API; read last quota from the request audit"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(games)
    g["game_date"] = pd.to_datetime(g["game_date"])
    start = pd.Timestamp(start_date, tz=g["game_date"].dt.tz) if g["game_date"].dt.tz else pd.Timestamp(start_date)
    cov = pd.Timestamp(coverage_start, tz=g["game_date"].dt.tz) if g["game_date"].dt.tz else pd.Timestamp(coverage_start)
    all_final = g[(g.get("status_normalized") == "final") & (g["game_date"] >= start)].copy()
    n_games_all = int(len(all_final))
    n_excluded_pre_coverage = int((all_final["game_date"] < cov).sum())
    # eligible = completed games ON/AFTER player-prop coverage begins
    final = all_final[all_final["game_date"] >= cov].copy()
    n_games = int(len(final))
    unique_dates = int(final["game_date"].dt.date.nunique())

    region_list = [r.strip() for r in regions.split(",") if r.strip()]
    n_regions = len(region_list)
    n_markets = len(ODDS_API_MODEL_MARKET_KEYS)

    max_per_event_per_snapshot = HISTORICAL_MARKET_MULTIPLIER * n_markets * n_regions   # 10*12*1=120
    max_per_event = max_per_event_per_snapshot * snapshots_per_event                    # 240
    upper_bound_event_odds_credits = n_games * max_per_event
    # Expected is lower: ~3/12 markets (stl/blk/tov) return no US book coverage, and combos
    # are sparse, so ~9 effective markets is a realistic mean; early-2023 games have no props.
    expected_effective_markets = 9
    expected_per_event = HISTORICAL_MARKET_MULTIPLIER * expected_effective_markets * n_regions * snapshots_per_event
    expected_event_odds_credits = int(n_games * expected_per_event * 0.85)  # 0.85 = props-coverage factor
    discovery_requests = unique_dates   # one historical-events snapshot per game date

    # quota: either the last observed value from the request audit (no API call) or a live probe.
    quota = {"x_requests_remaining": None, "x_requests_used": None, "x_requests_last": None,
             "source": None}
    audit_path = AUD / "ODDS_API_REQUEST_AUDIT.jsonl"
    if no_quota_probe:
        if audit_path.exists():
            lines = [json.loads(x) for x in audit_path.read_text().splitlines() if x.strip()]
            if lines:
                last = lines[-1]
                quota = {"x_requests_remaining": last.get("x_requests_remaining"),
                         "x_requests_used": last.get("x_requests_used"),
                         "x_requests_last": last.get("x_requests_last"),
                         "source": "request_audit_last_line (no API call)"}
    else:
        try:
            c = OddsAPIClient(region=region_list[0])
            c.list_historical_events(f"{start_date}T12:00:00Z",
                                     commence_time_from=f"{start_date}T00:00:00Z",
                                     commence_time_to=f"{start_date}T23:59:59Z")
            quota = {"x_requests_remaining": c.quota_remaining, "x_requests_used": c.quota_used,
                     "x_requests_last": c.quota_last, "source": "live_probe"}
        except Exception as exc:  # noqa: BLE001
            quota["error"] = str(exc)[:200]

    import os
    env_budget = os.environ.get("ODDS_API_MAX_CREDITS")
    # Hard budget: env override, else the backfill's fail-closed default (300k > 251,760 UB).
    budget = int(env_budget) if env_budget else 300000
    budget_source = "ODDS_API_MAX_CREDITS env" if env_budget else "default (backfill --max-credits)"

    plan = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "AUTHORIZATION": "NOT_APPROVED — awaiting user 'APPROVE BACKFILL'",
        "requested_start_date": start_date,
        "player_prop_coverage_start": coverage_start,
        "player_prop_coverage_note": (
            "Empirically, decision-snapshot (tip-12h) player-prop coverage begins ~2023-05-23; "
            f"{n_excluded_pre_coverage} completed games between {start_date} and {coverage_start} "
            "are EXCLUDED (no historical player-prop coverage)."),
        "game_date_range": {"first": coverage_start,
                            "last": str(final["game_date"].max().date()) if n_games else None},
        "n_completed_games_from_requested_start": n_games_all,
        "n_excluded_pre_coverage": n_excluded_pre_coverage,
        "n_bdl_games_final": n_games,
        "n_unique_game_dates": unique_dates,
        "expected_matched_odds_events": n_games,  # 1 Odds event per BDL game (upper bound; unmatched blocked)
        "market_list": list(ODDS_API_MODEL_MARKET_KEYS),
        "n_markets": n_markets,
        "region_list": region_list,
        "n_regions": n_regions,
        "snapshots_per_event": snapshots_per_event,
        "historical_event_discovery_requests": discovery_requests,
        "historical_market_multiplier": HISTORICAL_MARKET_MULTIPLIER,
        "max_per_event_per_snapshot_credits": max_per_event_per_snapshot,
        "max_per_event_credits": max_per_event,
        "upper_bound_event_odds_credits": upper_bound_event_odds_credits,
        "expected_event_odds_credits": expected_event_odds_credits,
        "note_empty_markets": "Empty market responses consume no event-odds market credits; "
                              "stl/blk/tov have no US book coverage so contribute ~0.",
        "configured_max_credit_budget": budget,
        "configured_max_credit_budget_source": budget_source,
        "available_credits_before_execution": quota.get("x_requests_remaining"),
        "quota": quota,
        "budget_covers_upper_bound": (budget is None or upper_bound_event_odds_credits <= budget),
        "quota_covers_upper_bound": (quota.get("x_requests_remaining") is None
                                     or upper_bound_event_odds_credits <= quota["x_requests_remaining"]),
        "resume_cache_policy": {
            "raw_response_cache": "data/atomic_quotes/raw_odds/<event_id>_<snapshot>.json",
            "idempotent_store": "append-only by quote_id (data/atomic_quotes/atomic_quotes.parquet)",
            "event_id_map_cache": "artifacts/audits/EVENT_ID_MAPPING_AUDIT.csv",
            "resume": "skip events whose raw cache + store rows already exist",
        },
    }
    (AUD / "ODDS_API_BACKFILL_PLAN.json").write_text(json.dumps(plan, indent=2, default=str))

    typer.echo("================ ODDS API BACKFILL PLAN (NOT APPROVED) ================")
    typer.echo(f"  completed games >= {start_date} : {n_games_all}  (excluded pre-coverage: {n_excluded_pre_coverage})")
    typer.echo(f"  ELIGIBLE games (>= {coverage_start}) : {n_games}  over {unique_dates} dates")
    typer.echo(f"  markets                        : {n_markets}  regions: {n_regions}  snapshots/event: {snapshots_per_event}")
    typer.echo(f"  discovery requests             : {discovery_requests}")
    typer.echo(f"  UPPER-BOUND event-odds credits : {upper_bound_event_odds_credits:,}")
    typer.echo(f"  EXPECTED event-odds credits    : {expected_event_odds_credits:,}")
    typer.echo(f"  configured HARD budget         : {budget:,}  ({budget_source})")
    typer.echo(f"  available credits (before exec): {quota.get('x_requests_remaining')}  [{quota.get('source')}]")
    typer.echo(f"  budget_covers_UB={plan['budget_covers_upper_bound']}  quota_covers_UB={plan['quota_covers_upper_bound']}")
    typer.echo("  AUTHORIZATION: NOT APPROVED — awaiting user 'APPROVE BACKFILL'")
    typer.echo(f"  wrote {AUD}/ODDS_API_BACKFILL_PLAN.json")


if __name__ == "__main__":
    app()
