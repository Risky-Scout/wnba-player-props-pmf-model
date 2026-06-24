"""Live game tracker entry point.

Polls BDL PBP and player props endpoints for active games,
runs Gamma-Poisson Bayesian updates, computes live edges,
and writes results to artifacts/live/.

Designed to be called by the live_tracker.yml GitHub Actions workflow.
Runs for a specified duration (default 150 minutes = full WNBA game + buffer).

Usage:
    python scripts/run_live_tracker.py \\
        --game-ids 12345 67890 \\
        --duration 150 \\
        --poll-interval 15 \\
        --out-dir artifacts/live/
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

_DEFAULT_OUT_DIR = "artifacts/live"
_DEFAULT_PROJECTIONS = "deliveries/today/full_pmfs_wide.parquet"


@app.command()
def main(
    game_ids: list[int] = typer.Option(..., "--game-ids", help="BDL game IDs to track."),
    duration: int = typer.Option(150, "--duration", help="How long to track in minutes."),
    poll_interval: int = typer.Option(15, "--poll-interval", help="Seconds between PBP polls."),
    out_dir: str = typer.Option(_DEFAULT_OUT_DIR, "--out-dir"),
    projections_path: str = typer.Option(_DEFAULT_PROJECTIONS, "--projections", help="Pre-game PMF parquet."),
    players_path: str = typer.Option("data/processed/wnba_players.parquet", "--players"),
    games_path: str = typer.Option("data/processed/wnba_games.parquet", "--games"),
    min_edge: float = typer.Option(0.04, "--min-edge", help="Minimum edge pp to flag as bettable."),
) -> None:
    """Run the live WNBA player props tracker."""
    logging.basicConfig(level=logging.INFO)

    from wnba_props_model.data.bdl_client import BDLClient  # noqa: PLC0415
    from wnba_props_model.live.bayesian_updater import GammaPoissonLiveEngine  # noqa: PLC0415
    from wnba_props_model.live.live_edge import LiveEdgeCalculator  # noqa: PLC0415
    from wnba_props_model.live.orchestrator import LiveGameOrchestrator, build_roster_lookup  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load pre-game projections
    pre_game_projections: dict[int, dict] = {}
    if Path(projections_path).exists():
        pmfs_df = pd.read_parquet(projections_path)
        pre_game_projections = _build_projections_dict(pmfs_df)
        typer.echo(f"Loaded pre-game projections for {len(pre_game_projections)} players")
    else:
        typer.echo(f"[WARN] No pre-game projections at {projections_path} — using defaults")

    # Load players for roster lookup
    players_df = pd.DataFrame()
    if Path(players_path).exists():
        players_df = pd.read_parquet(players_path)

    # Load games for home/away team info
    games_df = pd.DataFrame()
    if Path(games_path).exists():
        games_df = pd.read_parquet(games_path)

    # Initialize components
    client = BDLClient()
    engine = GammaPoissonLiveEngine()
    edge_calc = LiveEdgeCalculator(min_edge=min_edge)
    orchestrator = LiveGameOrchestrator(
        client, engine, edge_calc,
        pbp_poll_interval=poll_interval,
        out_dir=out,
    )

    # Build roster lookups per game
    roster_lookups: dict[int, dict] = {}
    for gid in game_ids:
        if not games_df.empty and "game_id" in games_df.columns:
            game_row = games_df[games_df["game_id"] == gid]
            if not game_row.empty:
                home_tid = int(game_row.iloc[0].get("home_team_id") or 0)
                away_tid = int(game_row.iloc[0].get("visitor_team_id") or 0)
                if not players_df.empty:
                    roster_lookups[gid] = build_roster_lookup(players_df, home_tid, away_tid)
                    continue
        roster_lookups[gid] = {}

    typer.echo(f"Tracking {len(game_ids)} game(s): {game_ids}")
    typer.echo(f"Duration: {duration} minutes | Poll interval: {poll_interval}s")
    typer.echo(f"Output: {out}")

    start_time = time.time()
    deadline = start_time + duration * 60
    poll_count = 0

    summary_edges: list[dict] = []

    while time.time() < deadline:
        poll_count += 1
        ts = datetime.now(timezone.utc).isoformat()
        typer.echo(f"\n[{ts}] Poll #{poll_count}")

        all_edges: list[dict] = []
        game_states: dict[int, dict] = {}

        for gid in game_ids:
            roster = roster_lookups.get(gid, {})
            proj = pre_game_projections  # All player projections
            try:
                edges, game_state = orchestrator.run_game(gid, proj, roster)
                all_edges.extend(edges)
                game_states[gid] = game_state
                bettable = [e for e in edges if e.get("bettable")]
                typer.echo(
                    f"  game {gid}: {game_state.get('home_score', 0)}-"
                    f"{game_state.get('away_score', 0)} "
                    f"(Q{game_state.get('period', '?')} {game_state.get('clock', '?')}) | "
                    f"{len(bettable)}/{len(edges)} bettable edges"
                )
            except Exception as exc:
                typer.echo(f"  game {gid}: ERROR — {exc}", err=True)

        # Write current edges summary
        if all_edges:
            bettable_edges = [e for e in all_edges if e.get("bettable")]
            summary_path = out / "live_edges_latest.json"
            summary_path.write_text(json.dumps({
                "timestamp_utc": ts,
                "poll": poll_count,
                "game_ids": game_ids,
                "n_total_edges": len(all_edges),
                "n_bettable": len(bettable_edges),
                "game_states": game_states,
                "top_edges": sorted(all_edges, key=lambda x: abs(x.get("edge", 0)), reverse=True)[:20],
            }, indent=2, default=str))
            if bettable_edges:
                typer.echo(f"  TOP EDGES:")
                for e in bettable_edges[:5]:
                    typer.echo(
                        f"    player={e['player_id']} {e['stat']} "
                        f"line={e['line']} {e['direction'].upper()} "
                        f"edge={e['edge_pp']:+.1f}pp model={e['model_p_over']:.3f} market={e['market_p_over']:.3f}"
                    )
            summary_edges.extend(bettable_edges)

        # Sleep adaptively based on game state
        sleep_secs = poll_interval
        for gid, gs in game_states.items():
            adaptive = orchestrator.next_poll_interval(gs)
            sleep_secs = min(sleep_secs, adaptive)

        elapsed = time.time() - start_time
        remaining = max(0, deadline - time.time())
        typer.echo(f"  Sleeping {sleep_secs}s | elapsed={elapsed/60:.1f}min | remaining={remaining/60:.1f}min")

        if remaining <= 0:
            break
        time.sleep(min(sleep_secs, remaining))

    # Final summary
    typer.echo(f"\n=== Live Tracker Complete ===")
    typer.echo(f"Polls: {poll_count}")
    typer.echo(f"Total bettable edges found: {len(summary_edges)}")

    final_path = out / "live_session_summary.json"
    final_path.write_text(json.dumps({
        "session_start_utc": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "session_end_utc": datetime.now(timezone.utc).isoformat(),
        "game_ids": game_ids,
        "n_polls": poll_count,
        "n_bettable_edges_total": len(summary_edges),
        "bettable_edges": summary_edges,
    }, indent=2, default=str))
    typer.echo(f"Session summary → {final_path}")


def _build_projections_dict(pmfs_df: pd.DataFrame) -> dict[int, dict]:
    """Convert PMF parquet (wide OR long format) to {player_id: {stat: {mean, line}, projected_minutes}}.

    Wide format: columns pts_mean, reb_mean, ...
    Long format: columns player_id, stat, mean (one row per player-stat combination)
    """
    proj: dict[int, dict] = {}

    if "stat" in pmfs_df.columns and "mean" in pmfs_df.columns:
        # Long format: player_id | stat | mean | (minutes_mean)
        min_col_long = next((c for c in ["minutes_mean", "projected_minutes", "minutes_pred"] if c in pmfs_df.columns), None)
        for _, row in pmfs_df.iterrows():
            pid = int(row["player_id"])
            stat = str(row["stat"])
            mean_val = float(row["mean"]) if pd.notna(row.get("mean")) else 0.0
            proj.setdefault(pid, {})[stat] = {"mean": mean_val, "line": mean_val}
            if "projected_minutes" not in proj[pid] and min_col_long:
                mv = row.get(min_col_long)
                proj[pid]["projected_minutes"] = float(mv) if pd.notna(mv) else 28.0
        # Ensure all players have projected_minutes
        for pid in proj:
            proj[pid].setdefault("projected_minutes", 28.0)
    else:
        # Wide format: pts_mean, reb_mean, etc.
        stat_cols = {
            "pts_mean": "pts", "reb_mean": "reb", "ast_mean": "ast",
            "fg3m_mean": "fg3m", "stl_mean": "stl", "blk_mean": "blk",
            "turnover_mean": "turnover",
        }
        min_col = next((c for c in ["projected_minutes", "minutes_pred", "min_mean"] if c in pmfs_df.columns), None)
        for _, row in pmfs_df.drop_duplicates("player_id").iterrows():
            pid = int(row["player_id"])
            proj[pid] = {}
            for src_col, stat in stat_cols.items():
                if src_col in row.index and pd.notna(row[src_col]):
                    proj[pid][stat] = {"mean": float(row[src_col]), "line": float(row[src_col])}
            proj[pid]["projected_minutes"] = float(row.get(min_col, 28.0)) if min_col else 28.0
    return proj


if __name__ == "__main__":
    app()
