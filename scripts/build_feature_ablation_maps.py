"""Resolve the per-prop feature-ablation plan into concrete, validated candidate maps.

Reads config/feature_ablation_plan_v1.json (G0 baseline + S1-S5 transforms over the
candidate map) and config/prop_feature_map_candidate_v1.json, and writes
config/feature_ablation_maps_v1.json: for every candidate x prop, the exact feature list
the trainer will use via training.stat_feature_subset.

This is the deterministic prerequisite for the CI per-prop ablation (the actual retrain +
proper-score/CLV selection is CI-scale; this step is pure and testable locally). Every
resolved feature is validated against the feature contract; unknown/forbidden columns are a
hard error so a bad map can never reach training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.features.feature_contract import (  # noqa: E402
    MODEL_FEATURES, FORBIDDEN_MODEL_FEATURES)

app = typer.Typer(add_completion=False)
DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def resolve_candidate(prop_map: dict, spec: dict) -> dict:
    """Return {prop: [features]} for one ablation candidate."""
    add = spec.get("add", [])
    remove = set(spec.get("remove", []))
    out = {}
    for prop, base in prop_map.items():
        feats = [f for f in base if f not in remove]
        for a in add:
            if a not in feats:
                feats.append(a)
        out[prop] = feats
    return out


@app.command()
def main(plan: str = typer.Option("config/feature_ablation_plan_v1.json", "--plan"),
         candidate_map: str = typer.Option("config/prop_feature_map_candidate_v1.json", "--candidate-map"),
         out: str = typer.Option("config/feature_ablation_maps_v1.json", "--out")) -> None:
    plan_d = json.loads(Path(plan).read_text())
    prop_map = json.loads(Path(candidate_map).read_text())
    valid, forb = set(MODEL_FEATURES), set(FORBIDDEN_MODEL_FEATURES)

    resolved = {"schema_version": 1,
                "baseline_candidate": plan_d.get("baseline_candidate", "G0_current_global_feature_contract"),
                "candidates": {}}
    # G0 = full global contract for every prop (the current champion behavior).
    resolved["candidates"]["G0"] = {p: list(MODEL_FEATURES) for p in DIRECT_PROPS}

    errors = []
    for name, spec in plan_d.get("candidate_maps", {}).items():
        cand = resolve_candidate(prop_map, spec)
        for prop, feats in cand.items():
            missing = [f for f in feats if f not in valid]
            forbidden = [f for f in feats if f in forb]
            if missing:
                errors.append(f"{name}/{prop}: non-contract features {missing}")
            if forbidden:
                errors.append(f"{name}/{prop}: forbidden features {forbidden}")
            if len(feats) != len(set(feats)):
                errors.append(f"{name}/{prop}: duplicate features")
        resolved["candidates"][name] = cand

    if errors:
        for e in errors:
            typer.echo(f"[FATAL] {e}", err=True)
        raise typer.Exit(1)

    Path(out).write_text(json.dumps(resolved, indent=2))
    counts = {c: {p: len(f) for p, f in m.items()} for c, m in resolved["candidates"].items()}
    typer.echo(f"[ablation] resolved {len(resolved['candidates'])} candidates -> {out}")
    for c, pc in counts.items():
        typer.echo(f"  {c}: " + ", ".join(f"{p}={n}" for p, n in list(pc.items())[:7]))


if __name__ == "__main__":
    app()
