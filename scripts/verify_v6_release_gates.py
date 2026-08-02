#!/usr/bin/env python3
"""Evaluate V6 release gates and write the unified release matrix.

Any validation exception becomes an explicit FAILED gate — never swallowed.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.release import evaluate_release_matrix, gate_sample_size

app = typer.Typer(add_completion=False)


@app.command()
def main(
    bundle_dir: str = typer.Option(
        "artifacts/releases/candidates/wnba-pmf-production-v1.1-harden", "--bundle-dir",
    ),
    out: str = typer.Option(
        "artifacts/sharp_v6/hardening/RELEASE_MATRIX.json", "--out",
    ),
    n_stat_obs: int = typer.Option(0, "--n-stat-obs"),
    n_games: int = typer.Option(0, "--n-games"),
    n_players: int = typer.Option(0, "--n-players"),
    train_serve_parity: bool = typer.Option(False, "--train-serve-parity"),
    reproducibility_ok: bool = typer.Option(False, "--reproducibility-ok"),
    ci_ok: bool = typer.Option(False, "--ci-ok"),
    smoke_ok: bool = typer.Option(False, "--smoke-ok"),
    deployment_ok: bool = typer.Option(False, "--deployment-ok"),
    require_production_ready: bool = typer.Option(False, "--require-production-ready"),
) -> None:
    errors = []
    try:
        matrix = evaluate_release_matrix(
            bundle_dir=bundle_dir,
            n_stat_obs=n_stat_obs if n_stat_obs > 0 else None,
            n_games=n_games if n_games > 0 else None,
            n_players=n_players if n_players > 0 else None,
            train_serve_parity=train_serve_parity,
            reproducibility_ok=reproducibility_ok,
            ci_ok=ci_ok,
            smoke_ok=smoke_ok,
            deployment_ok=deployment_ok,
            market_validated=False,
        )
    except Exception as e:  # noqa: BLE001
        errors.append({
            "gate": "evaluate_release_matrix",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        # Explicit failed status — do not continue as though the check had no findings
        payload = {
            "artifact": "V6_RELEASE_MATRIX",
            "summary_label": "FAILED_VALIDATION_EXCEPTION",
            "production_ready": False,
            "exceptions": errors,
            "gates": [{
                "name": "validation_exception",
                "status": "FAIL",
                "detail": errors[0],
            }],
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        raise SystemExit("RELEASE_GATES_FAILED: validation exception") from e

    # Demonstrate vacuous-pass prevention when callers pass zeros explicitly
    if n_stat_obs == 0 and "--n-stat-obs" in sys.argv:
        vac = gate_sample_size(0, min_obs=1, name="forced_empty_eval")
        assert vac.status == "NOT_EVALUABLE"

    payload = matrix.to_dict()
    if errors:
        payload["exceptions"] = errors
        payload["production_ready"] = False
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps({
        "out": out,
        "summary_label": payload["summary_label"],
        "production_ready": payload["production_ready"],
        "market_superiority": payload["market_superiority"],
    }, indent=2))
    if require_production_ready and not payload["production_ready"]:
        raise SystemExit("RELEASE_GATES_FAILED")
    failed = [g for g in payload["gates"] if g["status"] == "FAIL"]
    if failed and require_production_ready:
        raise SystemExit(f"RELEASE_GATES_FAILED: {[g['name'] for g in failed]}")


if __name__ == "__main__":
    app()
