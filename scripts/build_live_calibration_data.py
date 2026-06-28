"""Part I: Build live calibration data from historical live prediction artifacts.

Scans artifacts/live/ for per-game live state files, extracts quarter-break
PMF snapshots, and joins them with actual final outcomes to produce a dataset
suitable for fitting separate live calibrators (fit_live_calibrators).

Output:
    data/processed/live_calibration_data.parquet
    Columns: player_id, game_id, game_date, stat, role_bucket, quarter,
             pmf_json, actual_outcome

Usage:
    python scripts/build_live_calibration_data.py
    python scripts/build_live_calibration_data.py --live-dir artifacts/live \
        --oof-pmfs data/oof/oof_player_stat_pmfs.parquet \
        --out data/processed/live_calibration_data.parquet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

# Quarter snapshots to extract (end of quarter k = 10 * k minutes elapsed)
QUARTER_BREAKS = [1, 2, 3, 4]
QUARTER_ELAPSED = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}


@app.command()
def main(
    live_dir: str = typer.Option("artifacts/live", "--live-dir"),
    oof_pmfs: str = typer.Option(
        "data/oof/oof_player_stat_pmfs.parquet",
        "--oof-pmfs",
        help="Pre-game OOF PMF parquet for role_bucket lookup",
    ),
    out: str = typer.Option(
        "data/processed/live_calibration_data.parquet",
        "--out",
    ),
) -> None:
    """Build live calibration data from historical live game state JSON files."""
    logging.basicConfig(level=logging.INFO)
    live_path = Path(live_dir)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all live_state_*.json files produced by the live tracker
    live_files = sorted(live_path.rglob("live_state_*.json"))
    if not live_files:
        typer.echo(f"[WARN] No live_state_*.json files found in {live_dir}")
        typer.echo("Live calibration data requires historical live tracking runs.")
        _write_empty(out_path)
        raise typer.Exit(0)

    typer.echo(f"Found {len(live_files)} live state files")

    # Load role_bucket lookup from OOF PMFs
    role_lookup: dict[tuple[int, str], str] = {}
    oof_path = Path(oof_pmfs)
    if oof_path.exists():
        try:
            oof_df = pd.read_parquet(oof_path, columns=["player_id", "stat", "role_bucket"])
            for _, row in oof_df.drop_duplicates(["player_id", "stat"]).iterrows():
                role_lookup[(int(row["player_id"]), str(row["stat"]))] = str(row.get("role_bucket", "rotation"))
        except Exception as exc:
            log.warning("Failed to load OOF PMFs for role lookup: %s", exc)

    rows = []
    for fpath in live_files:
        try:
            with open(fpath) as f:
                state = json.load(f)
            game_id = int(state.get("game_id", 0))
            game_date = str(state.get("game_date", ""))
            quarter = int(state.get("quarter", 0))
            elapsed = float(state.get("elapsed_minutes", 0.0))

            # Only collect at quarter breaks (Q1-Q4 boundary snapshots)
            if quarter not in QUARTER_BREAKS:
                continue
            expected_elapsed = QUARTER_ELAPSED.get(quarter, elapsed)
            if abs(elapsed - expected_elapsed) > 3.0:
                continue

            player_states = state.get("player_states", {})
            live_pmfs = state.get("live_pmfs", {})  # {player_id: {stat: {pmf: {...}}}}

            for pid_str, stat_pmfs in live_pmfs.items():
                pid = int(pid_str)
                ps = player_states.get(pid_str, {})
                for stat, pmf_data in stat_pmfs.items():
                    if stat not in STATS:
                        continue
                    pmf_dict = pmf_data.get("pmf")
                    actual = ps.get(f"final_{stat}")  # final actual outcome
                    if pmf_dict is None or actual is None:
                        continue
                    role = role_lookup.get((pid, stat), "rotation")
                    rows.append({
                        "player_id": pid,
                        "game_id": game_id,
                        "game_date": game_date,
                        "stat": stat,
                        "role_bucket": role,
                        "quarter": quarter,
                        "pmf_json": json.dumps({str(k): v for k, v in pmf_dict.items()}),
                        "actual_outcome": float(actual),
                    })
        except Exception as exc:
            log.warning("Failed to parse %s: %s", fpath, exc)

    if not rows:
        typer.echo("[WARN] No usable live calibration rows extracted")
        typer.echo(
            "Live state files must contain 'live_pmfs' with per-player PMFs "
            "and 'player_states' with 'final_{stat}' actual outcomes."
        )
        _write_empty(out_path)
        raise typer.Exit(0)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    typer.echo(f"Wrote {len(df):,} live calibration rows → {out_path}")
    typer.echo(f"Stats: {sorted(df['stat'].unique())}")
    typer.echo(f"Quarters: {sorted(df['quarter'].unique())}")
    typer.echo(f"Games: {df['game_id'].nunique()}")


def _write_empty(path: Path) -> None:
    pd.DataFrame(columns=[
        "player_id", "game_id", "game_date", "stat", "role_bucket",
        "quarter", "pmf_json", "actual_outcome",
    ]).to_parquet(path, index=False)


if __name__ == "__main__":
    app()
