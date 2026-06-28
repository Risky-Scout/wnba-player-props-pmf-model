"""Format pre-game PMF output into the blueprint §3 JSON schema for wizardofodds.com.

Reads deliveries/tonight/player_projections_{date}.parquet (or full_pmfs_wide.parquet)
and writes the exact JSON structure required by the dashboard contract:

Output files (blueprint §12.1):
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/{date}_initial.json
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/{date}_injury_update.json
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/{date}_final.json
  tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/latest.json  (copy of most recent)

Dashboard fetch URL (blueprint §13.1):
  https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/Pre-Game-Edge/latest.json

Usage:
    python scripts/format_pregame_output.py \\
        --game-date 2026-06-24 \\
        --pipeline-run pregame_initial \\
        --pmfs deliveries/tonight/full_pmfs_wide.parquet \\
        --out-dir tools/odds-scanner/predictions/WNBA/Pre-Game-Edge
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.pipeline.output_schema import build_pregame_envelope

app = typer.Typer(add_completion=False)

VALID_RUN_TYPES = ("pregame_initial", "pregame_injury_update", "pregame_final")


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="Target game date YYYY-MM-DD."),
    pipeline_run: str = typer.Option(
        "pregame_initial", "--pipeline-run",
        help=f"Run type: {VALID_RUN_TYPES}",
    ),
    pmfs: str = typer.Option(
        "", "--pmfs",
        help="PMF parquet path. Auto-detected from deliveries/ if not set.",
    ),
    out_dir: str = typer.Option(
        "tools/odds-scanner/predictions/WNBA/Pre-Game-Edge",
        "--out-dir",
        help="Output directory for JSON files.",
    ),
    market_props: str = typer.Option(
        "", "--market-props",
        help="Optional: Odds API props parquet for edge / deep links.",
    ),
    injuries: str = typer.Option(
        "", "--injuries",
        help="Optional: injuries JSON or parquet.",
    ),
    games_parquet: str = typer.Option(
        "data/processed/wnba_games.parquet",
        "--games",
        help="Games parquet for spread/total context.",
    ),
) -> None:
    """Format pre-game PMFs into the blueprint JSON schema."""
    if pipeline_run not in VALID_RUN_TYPES:
        typer.echo(f"[WARN] Unknown pipeline_run '{pipeline_run}'. Using pregame_initial.", err=True)
        pipeline_run = "pregame_initial"

    # ── Load PMFs ─────────────────────────────────────────────────────────
    pmf_path = _resolve_pmfs(pmfs, game_date)
    if pmf_path is None or not pmf_path.exists():
        typer.echo(
            f"[INFO] No PMF file found for {game_date} — no WNBA games scheduled for this date. "
            "Skipping JSON output.",
            err=True,
        )
        raise typer.Exit(0)

    pmfs_df = pd.read_parquet(pmf_path)
    typer.echo(f"Loaded {len(pmfs_df):,} PMF rows from {pmf_path}")

    # ── Load supporting data ───────────────────────────────────────────────
    games_df = _safe_read_parquet(games_parquet)
    market_df = _safe_read_parquet(market_props) if market_props else None
    injuries_df = _load_injuries(injuries)

    if market_df is not None and not market_df.empty:
        n_mkt_players = market_df["player_name"].nunique() if "player_name" in market_df.columns else 0
        typer.echo(f"Market data: {len(market_df):,} rows, {n_mkt_players} unique players from Odds API")
    else:
        typer.echo("[WARN] No market data available — edges and Kelly will not be computed")

    # Load UTM transfer log and GTD scenarios from injury_report_{date}.json
    utm_log_df, gtd_log_rows = _load_utm_and_gtd(game_date)

    # ── Build envelope ─────────────────────────────────────────────────────
    envelope = build_pregame_envelope(
        pmfs_df=pmfs_df,
        game_date=game_date,
        pipeline_run=pipeline_run,
        games_df=games_df,
        market_df=market_df,
        injuries_df=injuries_df,
        utm_log_df=utm_log_df,
        gtd_log_rows=gtd_log_rows,
    )

    n_games = len(envelope.get("games", []))
    n_players = sum(len(g.get("players", [])) for g in envelope.get("games", []))
    n_with_market = sum(
        1
        for g in envelope.get("games", [])
        for p in g.get("players", [])
        for sp in p.get("stat_projections", {}).values()
        if sp.get("calibrated_p_over") is not None
    )
    typer.echo(f"Built envelope: {n_games} games, {n_players} player records, "
               f"{n_with_market} stat-lines with market edges")

    # ── Part 5: Portfolio-aware Kelly sizing ────────────────────────────────
    try:
        from wnba_props_model.models.market import compute_portfolio_kelly  # noqa: PLC0415
        all_bets = []
        for g in envelope.get("games", []):
            for p in g.get("players", []):
                for stat_key, sp in p.get("stat_projections", {}).items():
                    cal = sp.get("calibrated_p_over")
                    if cal and cal.get("kelly_fraction") is not None:
                        all_bets.append({
                            "player_id": p.get("player_id"),
                            "stat": stat_key,
                            "game_id": g.get("game_id"),
                            "kelly_individual": cal["kelly_fraction"],
                        })
        if all_bets:
            portfolio_bets = compute_portfolio_kelly(all_bets, max_total_exposure=0.15)
            # Build lookup: (player_id, stat, game_id) → kelly_portfolio
            kelly_portfolio_map = {
                (str(b.get("player_id")), b.get("stat"), str(b.get("game_id"))): b.get("kelly_portfolio")
                for b in portfolio_bets
            }
            # Write kelly_portfolio back into envelope
            for g in envelope.get("games", []):
                gid = str(g.get("game_id"))
                for p in g.get("players", []):
                    pid = str(p.get("player_id"))
                    for stat_key, sp in p.get("stat_projections", {}).items():
                        cal = sp.get("calibrated_p_over")
                        if cal:
                            kp = kelly_portfolio_map.get((pid, stat_key, gid))
                            cal["kelly_portfolio"] = kp
            typer.echo(f"Portfolio Kelly: {len(all_bets)} bets sized (max_exposure=15%)")
    except Exception as _e:
        typer.echo(f"[WARN] Portfolio Kelly failed (non-fatal): {_e}")

    # ── Write files ────────────────────────────────────────────────────────
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_suffix = pipeline_run.replace("pregame_", "")
    dated_path = out / f"{game_date}_{run_suffix}.json"
    latest_path = out / "latest.json"

    payload = json.dumps(envelope, indent=2, default=_json_default)

    dated_path.write_text(payload)
    latest_path.write_text(payload)

    typer.echo(f"Wrote {dated_path}")
    typer.echo(f"Wrote {latest_path}")

    # Print sample
    if envelope.get("games"):
        g0 = envelope["games"][0]
        typer.echo(f"\nGame sample: {g0['home_team']['name']} vs {g0['away_team']['name']}")
        if g0.get("players"):
            p0 = g0["players"][0]
            sp = p0.get("stat_projections", {}).get("points", {})
            typer.echo(f"  {p0['player_name']}: pts_mean={sp.get('mean')} conformal_ci={sp.get('conformal_90_ci')}")


def _resolve_pmfs(pmfs_arg: str, game_date: str) -> Path | None:
    if pmfs_arg:
        return Path(pmfs_arg)
    for candidate in [
        f"deliveries/tonight/full_pmfs_wide.parquet",
        f"deliveries/next_game/full_pmfs_wide.parquet",
        f"deliveries/today/full_pmfs_wide.parquet",
        f"deliveries/tonight/player_projections_{game_date}.parquet",
    ]:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def _safe_read_parquet(path_str: str) -> pd.DataFrame | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _load_utm_and_gtd(game_date: str) -> tuple[pd.DataFrame | None, list[dict] | None]:
    """Load UTM transfer log and GTD dual-scenario records from injury_report_{date}.json.

    apply_injury_updates.py writes `deliveries/tonight/injury_report_{date}.json`
    with:
      - `adjustments`: list with UTM transfer details per affected player
      - `gtd_scenarios_detail`: list of GTD dual-scenario records (blueprint §5.3)

    Returns (utm_log_df, gtd_log_rows).
    """
    candidates = [
        Path(f"deliveries/tonight/injury_report_{game_date}.json"),
        Path(f"deliveries/next_game/injury_report_{game_date}.json"),
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text())

            # GTD scenarios
            gtd_log_rows = raw.get("gtd_scenarios_detail") or []

            # UTM transfer rows
            adjustments = raw.get("adjustments", [])
            rows = []
            for adj in adjustments:
                injured_pid = adj.get("player_id")
                injured_name = adj.get("player_name", "")
                status = adj.get("status", "")
                utm_transfers = adj.get("utm_transfers") or {}
                transfers = utm_transfers.get("transfers", []) if isinstance(utm_transfers, dict) else []
                if not transfers:
                    continue
                for t in transfers:
                    rows.append({
                        "injured_player_id": injured_pid,
                        "injured_player_name": injured_name,
                        "status": status,
                        "beneficiary_player_id": t.get("player_id"),
                        "beneficiary_player_name": t.get("player_name", ""),
                        "minutes_transfer": float(t.get("extra_minutes", 0.0)),
                        "usage_transfer": float(t.get("extra_usage_pct", 0.0)),
                        "points_boost": 0.0,
                    })

            utm_df = pd.DataFrame(rows) if rows else None
            return utm_df, gtd_log_rows or None
        except Exception:
            pass
    return None, None


def _load_injuries(injuries_arg: str) -> pd.DataFrame | None:
    if not injuries_arg:
        return None
    p = Path(injuries_arg)
    if not p.exists():
        return None
    try:
        if p.suffix == ".json":
            raw = json.loads(p.read_text())
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            return pd.DataFrame([raw])
        return pd.read_parquet(p)
    except Exception:
        return None


def _json_default(obj):
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return str(obj)


if __name__ == "__main__":
    app()
