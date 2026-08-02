#!/usr/bin/env python3
# ruff: noqa: B008
"""Build current-slate participation/minutes context using the fail-closed collector."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.availability_policy import decide_availability

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]


@app.command()
def main(
    date: str = typer.Option(""),
    challenger: Path = typer.Option(REPO / "artifacts/releases/wnba-pmf-production-v1.2-rc1"),
    control: Path = typer.Option(REPO / "artifacts/releases/wnba-pmf-production-v1.1"),
    snapshot_manifest: Path = typer.Option(
        REPO / "artifacts/phase2_repair/ROSTER_INJURY_SNAPSHOT_MANIFEST.json"
    ),
    out_dir: Path = typer.Option(REPO / "artifacts/sharp_v6_phase3"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Collect fresh snapshots (fail-closed).
    import subprocess

    rc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/collect_roster_injury_snapshots.py"),
            "--date",
            date,
            "--manifest-out",
            str(snapshot_manifest),
        ],
        check=False,
    )
    manifest = json.loads(snapshot_manifest.read_text()) if snapshot_manifest.exists() else {}
    audit = {
        "date": date,
        "collector_exit_code": rc.returncode,
        "manifest_status": manifest.get("status"),
        "source_health": manifest.get("source_health"),
        "availability_gate": manifest.get("availability_gate"),
        "workflow_gate": manifest.get("workflow_gate"),
        "scheduled_games_exist": manifest.get("scheduled_games_exist"),
        "sources": {k: v.get("status") for k, v in (manifest.get("sources") or {}).items()},
    }
    if rc.returncode != 0:
        audit["result"] = "FAIL_CLOSED_NO_HEALTHY_DEFAULT"
        (out_dir / "CURRENT_SLATE_CONTEXT_AUDIT.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        typer.echo(json.dumps(audit, indent=2))
        raise SystemExit(rc.returncode)

    # Load roster/injury payloads from captured paths.
    sources = manifest.get("sources") or {}
    roster_path = sources.get("players_active", {}).get("path")
    injury_path = sources.get("player_injuries", {}).get("path")
    cross_path = sources.get("scheduled_team_player_crosscheck", {}).get("path")
    roster = json.loads(Path(roster_path).read_text())["payload"] if roster_path else []
    injuries = json.loads(Path(injury_path).read_text())["payload"] if injury_path else []
    cross = json.loads(Path(cross_path).read_text())["payload"] if cross_path else {}
    games = cross.get("games") or []
    team_players = cross.get("scheduled_team_players") or roster

    inj_by_pid = {}
    for row in injuries:
        player = row.get("player") or {}
        pid = player.get("id")
        if pid is None:
            continue
        status = (row.get("status") or row.get("description") or "").upper()
        inj_by_pid[int(pid)] = status

    # Build slate rows for scheduled-game teams.
    team_ids = set(cross.get("team_ids") or [])
    rows = []
    for p in team_players:
        pid = int(p.get("id"))
        tid = int((p.get("team") or {}).get("id") or p.get("team_id") or -1)
        if team_ids and tid not in team_ids:
            continue
        status_text = inj_by_pid.get(pid)
        explicitly_not_listed = pid not in inj_by_pid
        decision = decide_availability(
            status_text,
            snapshot_success=True,
            injury_model_in_domain=bool(status_text)
            and status_text
            in {"DOUBTFUL", "QUESTIONABLE", "PROBABLE", "DOUBT", "QUESTION", "PROB"},
            explicitly_not_listed=explicitly_not_listed,
        )
        rows.append(
            {
                "game_date": date,
                "player_id": pid,
                "team_id": tid,
                "player_name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                "availability_status": decision.status.value,
                "availability_action": decision.action,
                "availability_reason": decision.reason,
                "historically_calibrated": decision.historically_calibrated,
                "p_active_policy": decision.p_active,
                "dnp_mass": decision.dnp_mass,
            }
        )
    slate = pd.DataFrame(rows)
    # Minutes from challenger if present.
    minutes_path = challenger / "MINUTES_MODEL"
    if minutes_path.exists() and not slate.empty:
        import pickle

        with minutes_path.open("rb") as f:
            model = pickle.load(f)
        # Hierarchical prior features: fill missing with NaN; model must tolerate.
        for c in model.feature_cols:
            if c not in slate.columns:
                slate[c] = np.nan
        try:
            pmf = model.pmf(slate)
            grid = np.arange(pmf.shape[1])
            slate["expected_minutes"] = pmf @ grid
            slate["minutes_var"] = (
                pmf * (grid[None, :] - slate["expected_minutes"].to_numpy()[:, None]) ** 2
            ).sum(axis=1)
            for q, name in [(0.1, "p10"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (0.9, "p90")]:
                cdf = pmf.cumsum(axis=1)
                slate[f"minutes_{name}"] = np.argmax(cdf >= q, axis=1)
            slate["minutes_model_family"] = model.family
            slate["ood_status"] = (
                slate[model.feature_cols]
                .isna()
                .all(axis=1)
                .map({True: "OOD_HIERARCHICAL_PRIOR", False: "IN_DOMAIN"})
            )
            slate["applicability_status"] = np.where(
                slate["availability_action"].eq("ABSTAIN"),
                "ABSTAIN",
                "CONDITIONAL_ACTIVE_MINUTES",
            )
        except Exception as exc:  # noqa: BLE001
            audit["minutes_error"] = str(exc)[:300]
            slate["applicability_status"] = "MINUTES_INFERENCE_FAILED"
    else:
        slate["applicability_status"] = "NO_CHALLENGER_MINUTES_MODEL"

    # Do not price OUT players.
    slate.loc[slate["availability_action"].eq("ABSTAIN"), "expected_minutes"] = np.nan

    slate.to_csv(out_dir / "CURRENT_SLATE_PARTICIPATION_MINUTES.csv", index=False)
    audit.update(
        {
            "result": "OK",
            "n_games": len(games),
            "n_roster_players": len(slate),
            "availability_coverage": slate["availability_status"]
            .value_counts(dropna=False)
            .to_dict()
            if not slate.empty
            else {},
            "minutes_coverage": int(slate["expected_minutes"].notna().sum())
            if "expected_minutes" in slate.columns
            else 0,
            "control_bundle_loadable": (control / "model_bundle.pkl").exists(),
            "challenger_present": challenger.exists(),
            "historical_last_team_fallback": False,
        }
    )
    (out_dir / "CURRENT_SLATE_CONTEXT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    typer.echo(json.dumps({k: audit[k] for k in audit if k != "sources"}, indent=2))


if __name__ == "__main__":
    app()
