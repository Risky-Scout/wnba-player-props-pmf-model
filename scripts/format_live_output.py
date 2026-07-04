"""Format live in-play tracker output into blueprint §4 JSON schema.

Reads artifacts/live/ outputs from run_live_tracker.py and writes
the per-game JSON files consumed by the wizardofodds.com live dashboard.

Output files (blueprint §12.2):
  tools/odds-scanner/predictions/WNBA/Inplay-Edge/game_{id}_latest.json
  tools/odds-scanner/predictions/WNBA/Inplay-Edge/game_{id}_pbp_log.json
  tools/odds-scanner/predictions/WNBA/Inplay-Edge/archive/game_{id}_final.json

Dashboard fetch URL (blueprint §13.2):
  https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/Inplay-Edge/game_{id}_latest.json

Usage:
    python scripts/format_live_output.py \\
        --live-dir artifacts/live \\
        --out-dir tools/odds-scanner/predictions/WNBA/Inplay-Edge \\
        --game-id 47821
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.pipeline.output_schema import build_live_envelope, SCHEMA_VERSION

_ODDSAPI_PROPS_DIRS = [
    "data/live/oddsapi_props",
    "data/processed",
]


def _norm_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _load_market_rows(player_states: list[dict]) -> list[dict]:
    """Load Odds API props parquet and add player_id via name-based lookup.

    Returns list of dicts ready to pass as live_market_rows to build_live_envelope.
    """
    # Build name → player_id map from player_states (BDL authoritative IDs)
    name_to_id: dict[str, int] = {}
    for ps in player_states:
        pid = int(ps.get("player_id", 0))
        name = str(ps.get("player_name", ""))
        if pid and name:
            name_to_id[_norm_name(name)] = pid

    # Find latest Odds API parquet
    parquet_path: Path | None = None
    for d in _ODDSAPI_PROPS_DIRS:
        p = Path(d)
        candidates = sorted(p.glob("wnba_player_props_oddsapi_latest.parquet"), reverse=True)
        if not candidates:
            candidates = sorted(p.glob("wnba_player_props_oddsapi_*.parquet"), reverse=True)
        if candidates:
            parquet_path = candidates[0]
            break

    if parquet_path is None or not parquet_path.exists():
        typer.echo("[INFO] No Odds API parquet found — edges will be model-only")
        return []

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        typer.echo(f"[WARN] Could not read Odds API parquet: {exc}", err=True)
        return []

    rows: list[dict] = []
    for rec in df.to_dict(orient="records"):
        pname = str(rec.get("player_name", ""))
        pid = name_to_id.get(_norm_name(pname), 0)
        rec["player_id"] = pid
        rows.append(rec)

    matched = sum(1 for r in rows if r["player_id"])
    typer.echo(f"[OddsAPI] Loaded {len(rows)} market rows, {matched} matched to player_id")
    return rows


app = typer.Typer(add_completion=False)


@app.command()
def main(
    live_dir: str = typer.Option("artifacts/live", "--live-dir"),
    out_dir: str = typer.Option(
        "tools/odds-scanner/predictions/WNBA/Inplay-Edge", "--out-dir"
    ),
    game_id: int = typer.Option(..., "--game-id", help="BDL game ID to format."),
    archive: bool = typer.Option(False, "--archive", help="Write to archive/ (post-game final)."),
) -> None:
    """Format live tracker outputs into the blueprint live JSON schema."""
    live = Path(live_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load live session summary
    summary_path = live / "live_session_summary.json"
    if not summary_path.exists():
        typer.echo(f"[WARN] No live_session_summary.json in {live_dir}", err=True)
        raise typer.Exit(0)

    summary = json.loads(summary_path.read_text())

    # Load player states if available
    states_path = live / f"game_{game_id}_player_states.json"
    player_states = []
    if states_path.exists():
        raw = json.loads(states_path.read_text())
        if isinstance(raw, list):
            player_states = raw
        elif isinstance(raw, dict):
            player_states = list(raw.values())

    # Load posterior PMFs if available
    pmfs_path = live / f"game_{game_id}_posterior_pmfs.json"
    posterior_pmfs: dict = {}
    if pmfs_path.exists():
        posterior_pmfs = json.loads(pmfs_path.read_text())

    # Build game state from summary
    game_state = {
        "game_status": "in_play",
        "current_period": summary.get("current_period", 1),
        "current_clock": summary.get("current_clock", "10:00"),
        "home_score": summary.get("home_score", 0),
        "away_score": summary.get("away_score", 0),
        "elapsed_possessions": summary.get("elapsed_possessions", 0),
        "remaining_possessions_est": summary.get("remaining_possessions_est", 60),
    }

    # Load live market rows so edge_vs_current_market is populated
    live_market_rows = _load_market_rows(player_states)

    envelope = build_live_envelope(
        game_id=game_id,
        game_state=game_state,
        player_states=player_states,
        posterior_pmfs=posterior_pmfs,
        live_market_rows=live_market_rows if live_market_rows else None,
    )

    payload = json.dumps(envelope, indent=2, default=_json_default)

    if archive:
        archive_dir = out / "archive"
        archive_dir.mkdir(exist_ok=True)
        out_path = archive_dir / f"game_{game_id}_final.json"
        envelope["game_status"] = "final"
    else:
        out_path = out / f"game_{game_id}_latest.json"

    out_path.write_text(payload)
    typer.echo(f"Wrote {out_path}")

    # Write PBP log if available
    pbp_log_src = live / f"game_{game_id}_pbp_log.json"
    if pbp_log_src.exists():
        pbp_dest = out / f"game_{game_id}_pbp_log.json"
        pbp_dest.write_text(pbp_log_src.read_text())
        typer.echo(f"Wrote {pbp_dest}")

    n_players = len(envelope.get("players", []))
    typer.echo(f"Live envelope: game {game_id} | {n_players} players | {game_state['current_period']}Q {game_state['current_clock']}")


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and (obj != obj):
        return None
    return str(obj)


if __name__ == "__main__":
    app()
