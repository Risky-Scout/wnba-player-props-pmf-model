"""Pull WNBA BDL history for one or more seasons.

Attempts all available WNBA endpoints and writes:
  - data/raw/bdl/<table>.parquet  (one file per endpoint)
  - artifacts/audits/endpoint_availability_audit.json

Usage:
    python3 scripts/pull_bdl_history.py \\
        --start-season 2024 --end-season 2026 \\
        --out-dir data/raw/bdl
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    start_season: int = typer.Option(..., help="First WNBA season to pull (e.g. 2024)."),
    end_season: int = typer.Option(..., help="Last WNBA season to pull (inclusive)."),
    out_dir: str = typer.Option("data/raw/bdl", help="Raw output directory."),
    audit_dir: str = typer.Option(
        "artifacts/audits", help="Directory for endpoint availability audit JSON."
    ),
) -> None:
    from wnba_props_model.data.ingest import pull_full_history

    seasons = list(range(start_season, end_season + 1))
    typer.echo(f"Pulling WNBA BDL data for seasons {seasons} → {out_dir}")

    result = pull_full_history(seasons, out_dir=out_dir)

    # Print per-endpoint summary
    typer.echo("\n=== Endpoint availability ===")
    for ep, rec in result["endpoints"].items():
        status = rec["status"]
        count = rec["row_count"]
        err = rec.get("error_message") or ""
        flag = " ⚠" if status not in ("success", "skipped") else ""
        typer.echo(f"  {ep:40s} {status:12s} rows={count:6d}{flag}  {err[:80]}")

    # Print paths written
    typer.echo("\n=== Files written ===")
    for k, v in result["paths"].items():
        typer.echo(f"  {k}: {v}")

    # Write endpoint availability audit
    audit_path = Path(audit_dir)
    audit_path.mkdir(parents=True, exist_ok=True)
    audit_file = audit_path / "endpoint_availability_audit.json"
    audit_payload = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pull_timestamp_utc": result["pull_timestamp_utc"],
        "seasons_attempted": result["seasons"],
        "out_dir": out_dir,
        "endpoints": result["endpoints"],
    }
    audit_file.write_text(json.dumps(audit_payload, indent=2, default=str))
    typer.echo(f"\nEndpoint audit → {audit_file}")


if __name__ == "__main__":
    app()
