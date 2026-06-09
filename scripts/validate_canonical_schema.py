"""Validate canonical tables against schema definitions.

Reads data/processed/*.parquet, runs schema validation, writes audit JSON.
Exits 1 if any required table fails validation.

Usage:
    python3 scripts/validate_canonical_schema.py \\
        --data-dir data/processed \\
        --audit-out artifacts/audits/canonical_schema_audit.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from wnba_props_model.data.schema import ALL_SCHEMAS, REQUIRED_TABLES, validate_table

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_dir: str = typer.Option("data/processed", help="Canonical parquet directory."),
    audit_out: str = typer.Option(
        "artifacts/audits/canonical_schema_audit.json",
        help="Output JSON path for schema audit.",
    ),
) -> None:
    processed = Path(data_dir)
    results: list[dict] = []
    hard_fails: list[str] = []

    for schema in ALL_SCHEMAS.values():
        p = processed / f"{schema.name}.parquet"
        if not p.exists():
            status = "fail" if schema.name in REQUIRED_TABLES else "skipped"
            results.append({
                "table": schema.name,
                "path": str(p),
                "status": status,
                "errors": [f"File not found: {p}"] if status == "fail" else [],
                "warnings": [],
                "stats": {"rows": 0},
            })
            if status == "fail":
                hard_fails.append(f"{schema.name}: file not found")
            continue

        df = pd.read_parquet(p)
        result = validate_table(df, schema, str(p))
        results.append(result)

        status_symbol = {"pass": "✓", "warn": "~", "fail": "✗"}.get(result["status"], "?")
        typer.echo(
            f"  [{status_symbol}] {schema.name:45s} "
            f"rows={result['stats'].get('rows', 0):6,d}  "
            f"{result['status']}"
        )
        if result["errors"]:
            for e in result["errors"]:
                typer.echo(f"      ERROR: {e}", err=True)
        if result["warnings"]:
            for w in result["warnings"]:
                typer.echo(f"      WARN:  {w}")

        if result["status"] == "fail" and schema.name in REQUIRED_TABLES:
            hard_fails.append(f"{schema.name}: {result['errors']}")

    # Write audit
    audit = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir,
        "tables": results,
        "hard_failures": hard_fails,
        "status": "FAIL" if hard_fails else "PASS",
    }
    out_path = Path(audit_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, default=str))
    typer.echo(f"\nSchema audit → {out_path}")

    if hard_fails:
        typer.echo(f"\n[FAIL] {len(hard_fails)} required-table failure(s):", err=True)
        for f in hard_fails:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(code=1)

    warn_count = sum(1 for r in results if r["status"] == "warn")
    typer.echo(f"[PASS] Schema validation complete. Warnings: {warn_count}")


if __name__ == "__main__":
    app()
