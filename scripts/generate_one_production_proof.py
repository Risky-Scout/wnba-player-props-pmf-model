#!/usr/bin/env python3
"""Generate ONE_PRODUCTION_MODEL_PROOF.json from repository facts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.release import write_proof

app = typer.Typer(add_completion=False)


@app.command()
def main(
    bundle_dir: str = typer.Option(
        "artifacts/releases/wnba-pmf-production-v1", "--bundle-dir",
    ),
    out: str = typer.Option(
        "artifacts/sharp_v6/ONE_PRODUCTION_MODEL_PROOF.json", "--out",
    ),
) -> None:
    proof = write_proof(out, bundle_dir=bundle_dir)
    typer.echo(json.dumps({
        "out": out,
        "facts_consistent": proof.get("facts_consistent"),
        "origin_main": proof.get("origin_main"),
        "bundle_integrity_ok": proof.get("bundle_integrity_ok"),
    }, indent=2))
    if not proof.get("facts_consistent"):
        raise SystemExit("PROOF_INCONSISTENT")


if __name__ == "__main__":
    app()
