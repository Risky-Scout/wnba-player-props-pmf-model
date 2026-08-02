#!/usr/bin/env python3
"""Rewrap an existing V6 pickle into a candidate bundle with correct integrity hashes.

Does NOT overwrite the immutable baseline production bundle. Use after code/contract
hardening to produce a versioned candidate, then promote the pointer only after gates.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.bundle import save_bundle, verify_bundle_integrity

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source_dir: str = typer.Option(
        "artifacts/releases/wnba-pmf-production-v1", "--source-dir",
    ),
    out_dir: str = typer.Option(
        "artifacts/releases/candidates/wnba-pmf-production-v1.1-harden", "--out-dir",
    ),
    preserve_baseline: bool = typer.Option(True, "--preserve-baseline/--overwrite"),
) -> None:
    src = Path(source_dir)
    out = Path(out_dir)
    if preserve_baseline and src.resolve() == out.resolve():
        raise SystemExit("refusing to overwrite baseline; choose a candidate --out-dir")
    if out.exists() and any(out.iterdir()):
        # candidate rebuild is allowed to replace prior candidate
        for p in out.iterdir():
            if p.is_file():
                p.unlink()

    pkl = src / "model_bundle.pkl"
    if not pkl.exists():
        raise SystemExit(f"missing source pickle: {pkl}")
    bundle = pickle.loads(pkl.read_bytes())
    old_man = {}
    if (src / "MANIFEST.json").exists():
        old_man = json.loads((src / "MANIFEST.json").read_text())

    meta = {
        **(bundle.meta or {}),
        "training_cutoff": old_man.get("training_cutoff") or (bundle.meta or {}).get("training_cutoff"),
        "data_hashes": old_man.get("data_hashes") or (bundle.meta or {}).get("data_hashes", {}),
        "rollback_bundle": str(src),
        "bundle_id": out.name,
        "rewrap_from": str(src),
        "rewrap_from_claimed_sha": old_man.get("model_sha256"),
    }
    # Ensure selected families frozen
    if not bundle.selected_family and (src / "SELECTED_FAMILIES.json").exists():
        bundle.selected_family = json.loads((src / "SELECTED_FAMILIES.json").read_text())

    man = save_bundle(bundle, out, meta=meta)
    info = verify_bundle_integrity(out)
    # Copy baseline marker
    (out / "BASELINE_SOURCE.json").write_text(json.dumps({
        "baseline_dir": str(src),
        "baseline_claimed_model_sha256": old_man.get("model_sha256"),
        "candidate_model_sha256": info["model_sha256"],
        "note": "Baseline preserved immutable; candidate rebuilt with fail-closed integrity.",
    }, indent=2))
    typer.echo(json.dumps({
        "out_dir": str(out),
        "model_sha256": info["model_sha256"],
        "feature_contract_hash": man.get("feature_contract_hash"),
        "verified": True,
    }, indent=2))


if __name__ == "__main__":
    app()
