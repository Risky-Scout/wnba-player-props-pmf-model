"""Track model-edge-at-entry for published edges.

Reads the edge parquet for a given date, computes model_edge_at_entry
(model_p_over - market_p_over_at_entry_time), and appends to a cumulative
tracking parquet.

This is NOT closing-line value (CLV).  True CLV requires a closing quote
and is only available after the game.  The 'has_closing_line' field will
be populated by post_game_scoring.py when an archived closing quote is available.

Usage:
    python scripts/track_clv.py \
        --date 2026-07-10 \
        --predictions deliveries/next_game/pmf_edges.parquet \
        --output data/clv_tracking.parquet
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    date: str = typer.Option(..., "--date", help="Game date (YYYY-MM-DD)."),
    predictions: str = typer.Option(..., "--predictions", help="Edge parquet path."),
    output: str = typer.Option("data/clv_tracking.parquet", "--output", help="CLV output parquet."),
    closing_lines: str = typer.Option(
        "artifacts/audits/closing_lines_*.parquet",
        "--closing-lines",
        help="Glob pattern for closing line parquets.",
    ),
) -> None:
    """Track model-edge-at-entry for published edges and append to cumulative tracking parquet."""
    pred_path = Path(predictions)
    out_path = Path(output)

    if not pred_path.exists():
        # Try fallback: publishable_edges.parquet in same dir
        fallback = pred_path.parent / "publishable_edges.parquet"
        if fallback.exists():
            pred_path = fallback
        else:
            typer.echo(f"[track_clv] No predictions file at {predictions} or fallback — skipping", err=True)
            raise typer.Exit(0)

    try:
        edges_df = pd.read_parquet(pred_path)
    except Exception as exc:
        typer.echo(f"[track_clv] Could not read predictions: {exc}", err=True)
        raise typer.Exit(0)

    if edges_df.empty:
        typer.echo(f"[track_clv] Empty predictions file — nothing to track")
        raise typer.Exit(0)

    # Build model-edge-at-entry tracking rows
    edge_rows = _compute_clv_rows(edges_df, date)

    if not edge_rows:
        typer.echo(f"[track_clv] No edge rows computed from {len(edges_df)} edges")
        raise typer.Exit(0)

    clv_df = pd.DataFrame(edge_rows)

    # Append to cumulative tracking parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            # De-duplicate: drop previous rows for same (game_date, player_id, stat, direction)
            dedup_keys = [c for c in ["game_date", "player_id", "stat", "direction"] if c in existing.columns]
            if dedup_keys:
                existing = existing[
                    ~(existing["game_date"] == date) if "game_date" in existing.columns
                    else existing
                ]
            combined = pd.concat([existing, clv_df], ignore_index=True)
        except Exception as exc:
            typer.echo(f"[track_clv] Could not read existing tracking file ({exc}), creating fresh", err=True)
            combined = clv_df
    else:
        combined = clv_df

    combined.to_parquet(out_path, index=False)
    typer.echo(
        f"[track_clv] Tracked {len(edge_rows)} model-edge-at-entry rows for {date} → {out_path} "
        f"(cumulative: {len(combined)} rows)"
    )


def _compute_clv_rows(edges_df: pd.DataFrame, game_date: str) -> list[dict]:
    """Build model-edge tracking rows from the edge DataFrame.

    model_edge_at_entry = model_p_over - market_p_over_at_entry_time.

    This is NOT CLV.  True CLV requires a closing quote (market_p_over_at_close).
    The 'has_closing_line' field will be updated by post_game_scoring.py when
    a closing quote is available.
    """
    rows = []
    now_utc = datetime.now(timezone.utc).isoformat()

    for _, row in edges_df.iterrows():
        try:
            player_id = row.get("player_id")
            player_name = row.get("player_name", "")
            stat = row.get("stat", "")
            direction = row.get("direction", "")
            edge_over = float(row.get("edge_over", 0.0) or 0.0)
            model_p_over = float(row.get("model_prob_over", 0.5) or 0.5)
            market_p_over = float(row.get("market_prob_over_no_vig", 0.5) or 0.5)
            line = float(row.get("line", 0.0) or 0.0)
            kelly_fraction = float(row.get("kelly_fraction", 0.0) or 0.0)

            # model_edge_at_entry: difference between model P(over) and entry-time
            # market no-vig P(over).  NOT a CLV measurement.
            model_edge_at_entry = model_p_over - market_p_over

            clv_row = {
                "game_date": game_date,
                "tracked_at": now_utc,
                "player_id": player_id,
                "player_name": player_name,
                "stat": stat,
                "direction": direction,
                "line": line,
                "model_p_over": model_p_over,
                "market_p_over_open": market_p_over,
                "edge_over": edge_over,
                "kelly_fraction": kelly_fraction,
                "model_edge_at_entry": round(float(model_edge_at_entry), 4),
                "has_closing_line": False,  # will be updated by post_game_scoring
            }
            rows.append(clv_row)
        except Exception:
            continue

    return rows


if __name__ == "__main__":
    app()
