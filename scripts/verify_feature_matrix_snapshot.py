"""Build and verify a lightweight feature-matrix snapshot (drift detector).

The wide feature parquet lives under data/processed/ which is git-ignored (large, rebuilt
by the pipeline). Instead of committing another copy, we commit a small JSON snapshot that
pins the parquet's identity and shape. A changed matrix must either regenerate this snapshot
(with an explained source change) or create a new versioned snapshot; silent replacement is
not allowed.

Commands:
  build   Regenerate data/processed/feature_matrix_snapshot_v1.json from the present parquet.
  verify  Compare the present parquet (when available) against the committed snapshot.
          When the parquet is absent (e.g. a clean CI checkout), verify the snapshot's own
          internal consistency and report the parquet drift check as DEFERRED (explicit, not
          a silent skip). Use --require-parquet to hard-fail when the parquet is missing.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.features.feature_contract import MODEL_FEATURES  # noqa: E402

app = typer.Typer(add_completion=False)

DEFAULT_PARQUET = "data/processed/wnba_player_game_features_wide.parquet"
DEFAULT_SNAPSHOT = "data/processed/feature_matrix_snapshot_v1.json"
CANONICAL_KEY = ["player_id", "game_id"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract_hash() -> str:
    return hashlib.sha256(json.dumps(sorted(MODEL_FEATURES)).encode()).hexdigest()


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _schema_hash(ordered_cols_types: list[list[str]]) -> str:
    return hashlib.sha256(json.dumps(ordered_cols_types).encode()).hexdigest()


def compute_snapshot(parquet: Path) -> dict:
    df = pd.read_parquet(parquet)
    ordered = [[str(c), str(df[c].dtype)] for c in df.columns]
    numeric = df.select_dtypes(include="number")
    inf_count = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
    total_cells = int(df.shape[0] * df.shape[1])
    total_null = int(df.isna().sum().sum())
    col_null_frac = df.isna().mean()
    dup_keys = 0
    if all(k in df.columns for k in CANONICAL_KEY):
        dup_keys = int(df.duplicated(subset=CANONICAL_KEY).sum())
    gd = pd.to_datetime(df["game_date"]) if "game_date" in df.columns else None
    return {
        "schema_version": 1,
        "snapshot_version": "v1",
        "parquet_path": str(parquet).replace("\\", "/"),
        "parquet_sha256": _sha256_file(parquet),
        "file_size_bytes": int(parquet.stat().st_size),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "ordered_schema": ordered,
        "schema_hash": _schema_hash(ordered),
        "game_date_min": str(gd.min().date()) if gd is not None else None,
        "game_date_max": str(gd.max().date()) if gd is not None else None,
        "canonical_key": CANONICAL_KEY,
        "duplicate_canonical_key_count": dup_keys,
        "missingness": {
            "total_cells": total_cells,
            "total_null": total_null,
            "overall_null_fraction": round(total_null / total_cells, 6) if total_cells else 0.0,
            "columns_all_null": int((col_null_frac >= 0.999999).sum()),
            "columns_over_50pct_null": int((col_null_frac > 0.5).sum()),
        },
        "infinite_value_count": inf_count,
        "feature_contract_hash": _contract_hash(),
        "builder_source_commit": _source_commit(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


@app.command()
def build(parquet: str = typer.Option(DEFAULT_PARQUET, "--parquet"),
          out: str = typer.Option(DEFAULT_SNAPSHOT, "--out")) -> None:
    p = Path(parquet)
    if not p.exists():
        typer.echo(f"[FATAL] parquet not found: {parquet}", err=True)
        raise typer.Exit(1)
    snap = compute_snapshot(p)
    Path(out).write_text(json.dumps(snap, indent=2))
    typer.echo(f"[snapshot] wrote {out}: rows={snap['row_count']} cols={snap['column_count']} "
               f"sha={snap['parquet_sha256'][:16]} dates={snap['game_date_min']}..{snap['game_date_max']}")


@app.command()
def verify(parquet: str = typer.Option(DEFAULT_PARQUET, "--parquet"),
           snapshot: str = typer.Option(DEFAULT_SNAPSHOT, "--snapshot"),
           require_parquet: bool = typer.Option(
               False, "--require-parquet",
               help="Hard-fail if the parquet is not present (use in pipeline, not clean CI).")) -> None:
    snap_p = Path(snapshot)
    if not snap_p.exists():
        typer.echo(f"[SNAPSHOT FAIL] committed snapshot missing: {snapshot}", err=True)
        raise typer.Exit(1)
    snap = json.loads(snap_p.read_text())

    # Always-on locked check: the snapshot's recorded schema hash must match its own
    # ordered schema (self-consistency), and the pinned feature-contract hash must match
    # the live contract.
    if _schema_hash(snap["ordered_schema"]) != snap["schema_hash"]:
        typer.echo("[SNAPSHOT FAIL] schema_hash does not match ordered_schema", err=True)
        raise typer.Exit(1)
    if snap.get("feature_contract_hash") != _contract_hash():
        typer.echo("[SNAPSHOT FAIL] feature_contract_hash drift: the feature contract changed "
                   "without regenerating the snapshot.", err=True)
        raise typer.Exit(1)

    p = Path(parquet)
    if not p.exists():
        msg = (f"[SNAPSHOT] parquet not present in this checkout ({parquet}); "
               "parquet drift comparison DEFERRED. Snapshot self-consistency PASSED.")
        if require_parquet:
            typer.echo("[SNAPSHOT FAIL] parquet required but missing: " + parquet, err=True)
            raise typer.Exit(1)
        typer.echo(msg)
        raise typer.Exit(0)

    # Full drift comparison against the live parquet.
    live = compute_snapshot(p)
    drift = []
    for k in ("parquet_sha256", "row_count", "column_count", "schema_hash",
              "game_date_min", "game_date_max", "duplicate_canonical_key_count"):
        if live[k] != snap.get(k):
            drift.append(f"{k}: snapshot={snap.get(k)} live={live[k]}")
    if drift:
        typer.echo("[SNAPSHOT FAIL] feature matrix drift detected:", err=True)
        for d in drift:
            typer.echo(f"  - {d}", err=True)
        typer.echo("Regenerate the snapshot with an explained source change, or create a new "
                   "versioned snapshot. Silent replacement is not allowed.", err=True)
        raise typer.Exit(1)
    typer.echo(f"[SNAPSHOT PASS] feature matrix matches snapshot "
               f"(rows={snap['row_count']} cols={snap['column_count']}).")


if __name__ == "__main__":
    app()
