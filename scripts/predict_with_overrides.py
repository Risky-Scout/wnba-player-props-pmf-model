"""Generate PMF projections with injury/lineup overrides.

Supports three NoVig use cases:
  1. Mark players as DNP — redistributes minutes to teammates
  2. Override projected minutes for specific players
  3. Remove player and see full cascade impact on teammates

This is the primary script for day-of production adjustments.

Usage:
    # Mark player 341 as DNP
    python scripts/predict_with_overrides.py \
        --game-date 2026-06-16 \
        --dnp 341

    # Multiple DNPs
    python scripts/predict_with_overrides.py \
        --game-date 2026-06-16 \
        --dnp 341,419

    # Override minutes (player 341 plays 32 min instead of projected 28)
    python scripts/predict_with_overrides.py \
        --game-date 2026-06-16 \
        --override-minutes "341:32"

    # Combine DNP + minutes overrides
    python scripts/predict_with_overrides.py \
        --game-date 2026-06-16 \
        --dnp 341 \
        --override-minutes "419:32,756:22" \
        --out-dir deliveries/overrides/2026-06-16
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.pipeline.overrides import apply_overrides, override_summary
from wnba_props_model.pipeline.predict import predict_player_pmfs
from wnba_props_model.pipeline.deliver import write_delivery
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf
import numpy as np

app = typer.Typer(add_completion=False)


def _parse_dnp(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_minutes(s: str) -> dict[int, float]:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            pid, mins = part.split(":", 1)
            out[int(pid.strip())] = float(mins.strip())
    return out


def _add_novig_ladder(pmfs: pd.DataFrame) -> pd.DataFrame:
    """Add P(stat >= k) ladder columns required by NoVig output contract."""
    out = pmfs.copy()
    thresholds = [1, 3, 5, 10, 15, 20]
    for k in thresholds:
        col = f"p_ge_{k}"
        vals = []
        for pmf_json in out["pmf_json"]:
            pmf = normalize_pmf(json_to_pmf(pmf_json))
            vals.append(float(pmf[k:].sum()))
        out[col] = vals
    return out


@app.command()
def main(
    game_date: str | None = typer.Option(None, help="Target game date YYYY-MM-DD (default: tomorrow)."),
    slate_dir: str = typer.Option("deliveries/next_game", help="Directory with slate_{date}.parquet."),
    model_dir: str = typer.Option("artifacts/models/stage4_baseline"),
    config: str = typer.Option("config/model/stage4_baseline.yaml"),
    cal_dir: str | None = typer.Option("artifacts/models/calibration"),
    dnp: str | None = typer.Option(None, help="Comma-separated player_ids to mark DNP."),
    override_minutes: str | None = typer.Option(None, "--override-minutes", help="player_id:minutes pairs, e.g. '341:32,419:24'"),
    raw_props: str | None = typer.Option(None, help="BDL player props for edge calculation."),
    out_dir: str = typer.Option("deliveries/overrides"),
) -> None:
    """Run PMF projections with manual injury/lineup overrides."""
    today = game_date or (date.today()).isoformat()
    from datetime import timedelta
    target = game_date or (date.today() + timedelta(days=1)).isoformat()

    # Load slate (next-game features)
    slate_path = Path(slate_dir) / f"slate_{target}.parquet"
    if not slate_path.exists():
        typer.echo(f"[ERROR] Slate not found: {slate_path}")
        typer.echo(f"Run first: python scripts/build_next_game_slate.py --game-date {target}")
        raise typer.Exit(1)

    slate = pd.read_parquet(slate_path)
    typer.echo(f"Loaded slate: {len(slate)} players for {target}")

    # Parse overrides
    dnp_ids = _parse_dnp(dnp) if dnp else []
    min_overrides = _parse_minutes(override_minutes) if override_minutes else {}

    if not dnp_ids and not min_overrides:
        typer.echo("[INFO] No overrides specified — running baseline predictions")

    # Apply overrides to feature slate
    overridden = apply_overrides(slate, dnp_player_ids=dnp_ids, minutes_overrides=min_overrides)
    summary = override_summary(slate, overridden)

    typer.echo(f"\nOverride summary: {summary['n_players_changed']} players affected")
    for change in summary["changes"]:
        direction = "↑" if change["delta_minutes"] > 0 else "↓"
        typer.echo(
            f"  {change['player_name']} ({change['team_abbreviation']}): "
            f"{change['original_minutes']} → {change['overridden_minutes']} min "
            f"({direction}{abs(change['delta_minutes']):.1f}) [{change['override_source']}]"
        )

    # Run inference on overridden slate
    typer.echo(f"\nGenerating PMFs for {len(overridden)} player-game rows...")
    pmfs = predict_player_pmfs(
        feature_df=overridden,
        model_dir=model_dir,
        config_path=config,
        cal_dir=cal_dir,
        apply_calibration=True,
    )

    # Add override metadata to PMF output
    override_meta = overridden[["player_id", "override_applied", "override_source"]].drop_duplicates("player_id")
    pmfs = pmfs.merge(override_meta, on="player_id", how="left")

    # Add NoVig ladder columns
    pmfs = _add_novig_ladder(pmfs)

    # Tag output
    pmfs["game_date_et"] = target
    pmfs["model_version"] = "wnba_pmf_v1.0_hgb_calibrated"

    # Write outputs
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pmf_path = out / f"player_projections_{target}_override.parquet"
    pmfs.to_parquet(pmf_path, index=False)
    typer.echo(f"\nProjections → {pmf_path}")

    # Write JSON for NoVig ingestion
    json_path = out / f"player_projections_{target}_override.json"
    pmfs.drop(columns=["pmf_json"], errors="ignore").to_json(json_path, orient="records", indent=2)
    typer.echo(f"JSON → {json_path}")

    # Edge report if props available
    props_df = pd.read_parquet(raw_props) if raw_props else None
    paths = write_delivery(pmfs, out, props_df)
    for k, v in paths.items():
        typer.echo(f"  {k}: {v}")

    # Write override audit
    audit = {
        "game_date": target,
        "dnp_player_ids": dnp_ids,
        "minutes_overrides": {str(k): v for k, v in min_overrides.items()},
        "override_summary": summary,
    }
    (out / f"override_audit_{target}.json").write_text(json.dumps(audit, indent=2))
    typer.echo(f"Override audit → {out}/override_audit_{target}.json")


if __name__ == "__main__":
    app()
