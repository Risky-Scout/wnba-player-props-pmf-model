"""Pull BDL closing-line prop snapshot for True CLV tracking.

BDL player prop lines are updated continuously throughout the day.  By pulling
props once in the morning (during prediction) and once close to game time, we
capture the *closing line* — the market's final efficient price.

True CLV = model_probability vs. CLOSING line.
Open-line CLV is useful but understates value when lines move in our favour.

This script is called by daily_pipeline.yml approximately 3 hours after the
morning prediction run (around noon ET for 7 PM games).  If the game has
already started or finished, the BDL endpoint returns the same props — so
calling it multiple times is idempotent.

Output:
    {out_dir}/closing_lines_{game_date}.parquet
        columns: game_id, player_id, stat, line, over_odds, under_odds,
                 market_prob_over_no_vig, shin_z, snapshot_type=closing,
                 pulled_at_utc

Usage:
    python scripts/pull_closing_lines.py \\
        --game-date 2026-06-16 \\
        --out-dir artifacts/audits \\
        --api-key $BDL_API_KEY
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    game_date: str = typer.Option(..., help="Game date (YYYY-MM-DD) to pull closing lines for."),
    out_dir: str = typer.Option("artifacts/audits", help="Output directory."),
    api_key: str = typer.Option("", envvar="BDL_API_KEY", help="BDL API key."),
) -> None:
    """Pull BDL closing-line props and save for true CLV computation."""
    key = api_key or os.environ.get("BDL_API_KEY", "")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"closing_lines_{game_date}.parquet"

    typer.echo(f"[closing_lines] Pulling closing props for {game_date}")

    if not key:
        typer.echo("[WARN] No BDL API key — cannot pull closing lines", err=True)
        _write_empty(out_path)
        return

    try:
        from wnba_props_model.data.bdl_client import BDLClient  # noqa: PLC0415
        from wnba_props_model.pipeline.deliver import normalize_player_props_snapshot
    except ImportError as exc:
        typer.echo(f"[ERROR] Import failed: {exc}", err=True)
        _write_empty(out_path)
        return

    try:
        client = BDLClient(api_key=key)
        # Pull props for the target date (BDL endpoint: /v1/wnba/player_props?date=...)
        props_raw = client.get_player_props(date=game_date)
    except Exception as exc:
        typer.echo(f"[ERROR] BDL pull failed: {exc}", err=True)
        _write_empty(out_path)
        return

    if props_raw is None or (hasattr(props_raw, "empty") and props_raw.empty):
        typer.echo(f"[WARN] No props returned for {game_date}")
        _write_empty(out_path)
        return

    if isinstance(props_raw, list):
        import pandas as _pd
        props_raw = _pd.DataFrame(props_raw)

    # Normalize using the same pipeline as morning props
    try:
        normalized = normalize_player_props_snapshot(props_raw)
    except Exception as exc:
        typer.echo(f"[ERROR] Normalize failed: {exc}", err=True)
        _write_empty(out_path)
        return

    if normalized.empty:
        typer.echo("[WARN] normalized closing props empty")
        _write_empty(out_path)
        return

    normalized["snapshot_type"] = "closing"
    normalized["pulled_at_utc"] = datetime.now(timezone.utc).isoformat()
    normalized.to_parquet(out_path, index=False)
    typer.echo(f"[closing_lines] Saved {len(normalized)} rows → {out_path}")


def _write_empty(path: Path) -> None:
    """Write empty parquet with correct schema."""
    pd.DataFrame(columns=[
        "game_id", "player_id", "stat", "line",
        "over_odds", "under_odds", "market_prob_over_no_vig", "shin_z",
        "snapshot_type", "pulled_at_utc",
    ]).to_parquet(path, index=False)


if __name__ == "__main__":
    app()
