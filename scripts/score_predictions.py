#!/usr/bin/env python3
"""Score model predictions against actuals for CLV drift tracking.

Reads the latest delivery's market_comparison.parquet (or full_pmfs_wide),
joins actual outcomes where available (post-game), and appends scored rows
to a rolling drift_window.parquet for longitudinal CLV monitoring.

Usage:
    python scripts/score_predictions.py \
        --delivery-dir deliveries/ \
        --out data/clv_tracking/drift_window.parquet
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)

_KEY_COLS = ["player_id", "game_id", "stat"]


def _find_latest_delivery(delivery_dir: Path) -> Path | None:
    """Return the most recently modified market_comparison.parquet in delivery_dir."""
    candidates = sorted(delivery_dir.rglob("market_comparison.parquet"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        # Fall back to publishable_edges.parquet
        candidates = sorted(delivery_dir.rglob("publishable_edges.parquet"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _score_rows(mc: pd.DataFrame) -> pd.DataFrame:
    """Add model_correct column when both actual_outcome and line are present."""
    if "actual_outcome" in mc.columns and "line" in mc.columns:
        mc = mc.copy()
        mc["model_direction"] = np.sign(mc.get("edge_over", pd.Series(dtype=float)))
        mc["outcome_direction"] = np.sign(
            mc["actual_outcome"].astype(float) - mc["line"].astype(float)
        )
        mc["model_correct"] = (mc["model_direction"] == mc["outcome_direction"]).astype(float)
        mc["scored"] = True
    else:
        mc = mc.copy()
        mc["scored"] = False
        mc["model_correct"] = float("nan")
    return mc


def score_delivery(delivery_dir: str, out_path: str) -> None:
    delivery = Path(delivery_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mc_file = _find_latest_delivery(delivery)
    if mc_file is None:
        typer.echo(f"No delivery file found under {delivery_dir}")
        return

    mc = pd.read_parquet(mc_file)
    typer.echo(f"Loaded: {mc_file} ({len(mc)} rows)")

    mc = _score_rows(mc)
    mc["scored_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Append to drift window, deduplicating on (player_id, game_id, stat)
    available_key_cols = [c for c in _KEY_COLS if c in mc.columns]
    if out.exists():
        existing = pd.read_parquet(out)
        if available_key_cols:
            existing_keys = set(map(tuple, existing[available_key_cols].values.tolist()))
            new_rows = mc[~mc[available_key_cols].apply(tuple, axis=1).isin(existing_keys)]
        else:
            new_rows = mc
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = mc

    combined.to_parquet(out, index=False)
    scored_count = int(combined["scored"].sum()) if "scored" in combined.columns else 0
    typer.echo(f"Drift window: {len(combined)} total rows, {scored_count} scored")
    typer.echo(f"Saved: {out}")


@app.command()
def run(
    delivery_dir: str = typer.Option("deliveries/", help="Directory containing dated delivery sub-folders."),
    out: str = typer.Option("data/clv_tracking/drift_window.parquet", help="Append-only drift window output path."),
) -> None:
    score_delivery(delivery_dir, out)


if __name__ == "__main__":
    app()
