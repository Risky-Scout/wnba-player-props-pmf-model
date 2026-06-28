#!/usr/bin/env python3
"""Build CLV training labels for each (player, stat, game) in the OOF dataset.

For every OOF row that has a corresponding closing line, computes:
  closing_p_over  — no-vig closing line probability (OVER) via Shin method
  clv_label       — log(model_p_over / closing_p_over): positive = beat closing line

These labels feed the CLV secondary head in train_baseline_pmfs.py which trains
a classifier that directly optimizes for beating the closing line, rather than
merely predicting the outcome.

Usage:
    python scripts/build_clv_training_labels.py \\
        --oof-pmfs data/oof/oof_player_stat_pmfs.parquet \\
        --closing-lines artifacts/audits/closing_lines_*.parquet \\
        --out data/oof/oof_clv_labels.parquet
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.models.market import shin_no_vig_two_way_with_z
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf

app = typer.Typer(add_completion=False)


def _p_over_from_pmf(pmf_json: str, line: float) -> float | None:
    """Compute P(Y > line) from a PMF JSON string."""
    try:
        arr = normalize_pmf(json_to_pmf(pmf_json))
        k_int = int(np.floor(line))
        return float(arr[k_int + 1:].sum()) if k_int + 1 < len(arr) else 0.0
    except Exception:
        return None


@app.command()
def main(
    oof_pmfs: Path = typer.Option(
        Path("data/oof/oof_player_stat_pmfs.parquet"), "--oof-pmfs"
    ),
    closing_lines_glob: str = typer.Option(
        "artifacts/audits/closing_lines_*.parquet", "--closing-lines"
    ),
    out: Path = typer.Option(
        Path("data/oof/oof_clv_labels.parquet"), "--out"
    ),
) -> None:
    print("=" * 70)
    print("CLV Training Label Builder")
    print("=" * 70)

    if not oof_pmfs.exists():
        typer.echo(f"ERROR: OOF PMFs not found at {oof_pmfs}", err=True)
        raise typer.Exit(1)

    # Load OOF PMFs
    print(f"Loading OOF PMFs: {oof_pmfs}")
    oof = pd.read_parquet(oof_pmfs)
    oof["game_date"] = pd.to_datetime(oof["game_date"])
    print(f"  {len(oof):,} total OOF rows")

    # Load all closing line parquets
    cl_files = sorted(glob.glob(closing_lines_glob))
    if not cl_files:
        typer.echo(f"[WARN] No closing line files found matching: {closing_lines_glob}")
        typer.echo("CLV labels require historical closing line data. Exiting.")
        raise typer.Exit(0)

    print(f"Loading {len(cl_files)} closing line files...")
    cl_dfs = []
    for f in cl_files:
        try:
            df = pd.read_parquet(f)
            cl_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to load {f}: {e}")

    if not cl_dfs:
        typer.echo("[WARN] No valid closing line data loaded.")
        raise typer.Exit(0)

    closing = pd.concat(cl_dfs, ignore_index=True)
    closing["game_date"] = pd.to_datetime(closing["game_date"])
    print(f"  {len(closing):,} closing line rows across {closing['game_date'].nunique()} dates")

    # Normalize stat names (closing lines may use different naming)
    _stat_map = {
        "points": "pts", "rebounds": "reb", "assists": "ast",
        "threes": "fg3m", "fg3m": "fg3m", "steals": "stl",
        "blocks": "blk", "turnovers": "turnover",
    }
    if "stat" in closing.columns:
        closing["stat"] = closing["stat"].map(lambda s: _stat_map.get(str(s).lower(), s))

    # Join OOF PMFs with closing lines
    join_keys = ["player_id", "game_date", "stat"]
    # Check which keys are available
    available_keys = [k for k in join_keys if k in oof.columns and k in closing.columns]
    if len(available_keys) < 2:
        typer.echo(f"[WARN] Insufficient join keys. OOF cols: {list(oof.columns)[:10]}, "
                   f"Closing cols: {list(closing.columns)[:10]}")
        raise typer.Exit(0)

    merged = oof.merge(
        closing[available_keys + [c for c in ["line", "over_odds", "under_odds", "closing_p_over"]
                                  if c in closing.columns]],
        on=available_keys,
        how="inner",
    )
    print(f"  Matched {len(merged):,} OOF rows with closing lines "
          f"({100 * len(merged) / len(oof):.1f}% coverage)")

    if len(merged) == 0:
        typer.echo("[WARN] No OOF rows matched closing lines.")
        raise typer.Exit(0)

    # Compute closing_p_over via Shin if not already present
    if "closing_p_over" not in merged.columns and "over_odds" in merged.columns:
        print("Computing no-vig closing probabilities via Shin method...")

        def _shin_p_over(row: pd.Series) -> float | None:
            p_over, _, _ = shin_no_vig_two_way_with_z(
                row.get("over_odds"), row.get("under_odds")
            )
            return p_over

        merged["closing_p_over"] = merged.apply(_shin_p_over, axis=1)

    # Drop rows where closing_p_over is missing
    merged = merged.dropna(subset=["closing_p_over"])
    merged = merged[(merged["closing_p_over"] > 0.01) & (merged["closing_p_over"] < 0.99)]
    print(f"  {len(merged):,} rows with valid closing probabilities")

    # Compute model P(over) from PMF at the closing line
    if "pmf_json" in merged.columns and "line" in merged.columns:
        print("Computing model P(over) from PMF...")
        merged["model_p_over"] = merged.apply(
            lambda r: _p_over_from_pmf(r["pmf_json"], float(r["line"]))
            if pd.notna(r.get("line")) else None,
            axis=1,
        )
        merged = merged.dropna(subset=["model_p_over"])
        merged = merged[(merged["model_p_over"] > 0.01) & (merged["model_p_over"] < 0.99)]
        print(f"  {len(merged):,} rows with valid model P(over)")

        # CLV label: log(model_p_over / closing_p_over)
        # Positive = model beats closing line (has edge)
        # Negative = model is behind the closing line
        merged["clv_label"] = np.log(
            merged["model_p_over"] / merged["closing_p_over"].clip(lower=0.01)
        )
        mean_clv = merged["clv_label"].mean()
        print(f"  Mean CLV log-ratio: {mean_clv:+.4f} "
              f"({'model leads closing' if mean_clv > 0 else 'model trails closing'})")

        # Binary beat-closing label for classifier head
        merged["beat_closing"] = (merged["actual_outcome"] > merged["line"]).astype(int)

    # Write output
    out.parent.mkdir(parents=True, exist_ok=True)
    cols_to_keep = [c for c in [
        "player_id", "game_id", "game_date", "stat", "role_bucket",
        "line", "closing_p_over", "model_p_over", "clv_label", "beat_closing",
        "actual_outcome", "pmf_mean",
    ] if c in merged.columns]
    merged[cols_to_keep].to_parquet(out, index=False)
    print(f"\nWrote {len(merged):,} CLV training rows → {out}")
    print(f"Columns: {cols_to_keep}")


if __name__ == "__main__":
    app()
