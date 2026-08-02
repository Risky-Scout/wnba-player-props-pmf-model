"""Thin production wrapper around ``sharp_v6.predict_slate``.

Default path loads the authoritative PRODUCTION_POINTER bundle and delegates to
``scripts/run_wnba_pmf.py`` (same inference graph as OOF / live / prospective).

Pass ``--legacy-stage4`` only for LEGACY_CONTROL / RESEARCH_ONLY Stage-4 rollback.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]
POINTER = REPO / "artifacts" / "releases" / "PRODUCTION_POINTER.json"


def _resolve_bundle_dir(bundle_dir: str | None) -> str:
    if bundle_dir:
        return bundle_dir
    if POINTER.exists():
        ptr = json.loads(POINTER.read_text())
        return str(ptr["production_bundle"])
    return "artifacts/releases/wnba-pmf-production-v1.1"


def _atoms_to_legacy_wide(atoms: pd.DataFrame) -> pd.DataFrame:
    """Convert V6 atom PMFs into the legacy full_pmfs_wide consumer schema."""
    if atoms.empty:
        return pd.DataFrame(
            columns=[
                "game_id", "player_id", "stat", "player_name",
                "active_pmf_json", "pmf_json", "p_active", "pmf_mean", "source_track",
            ]
        )
    rows: list[dict] = []
    keys = ["game_id", "player_id", "stat"]
    for (gid, pid, stat), g in atoms.groupby(keys, sort=False):
        g = g.sort_values("atom_value")
        pmf = {
            str(int(r.atom_value)): float(r.atom_probability)
            for r in g.itertuples()
            if float(r.atom_probability) > 0.0
        }
        ovf = float(g["overflow_probability"].iloc[0]) if "overflow_probability" in g.columns else 0.0
        if ovf > 1e-12:
            k_max = int(g["atom_value"].max()) + 1
            pmf[str(k_max)] = pmf.get(str(k_max), 0.0) + ovf
        mean = float(g["predictive_mean"].iloc[0]) if "predictive_mean" in g.columns else float("nan")
        pa = float(g["p_active"].iloc[0]) if "p_active" in g.columns else 1.0
        pname = str(g["player_name"].iloc[0]) if "player_name" in g.columns else ""
        pmf_json = json.dumps(pmf)
        rows.append({
            "game_id": int(gid),
            "player_id": int(pid),
            "stat": str(stat),
            "player_name": pname,
            "active_pmf_json": pmf_json,
            "pmf_json": pmf_json,
            "p_active": pa,
            "pmf_mean": mean,
            "mean": mean,
            "source_track": "CALIBRATED_V6_PMF",
            "model_version": "wnba-sharp-pmf-v6",
        })
    return pd.DataFrame(rows)


def _write_legacy_compat(v6_out: Path, dest: Path, game_date: str | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    atoms_path = v6_out / "active_atom_pmfs.parquet"
    if not atoms_path.exists():
        raise SystemExit(f"FAIL_CLOSED: missing V6 atoms at {atoms_path}")
    atoms = pd.read_parquet(atoms_path)
    wide = _atoms_to_legacy_wide(atoms)
    wide.to_parquet(dest / "full_pmfs_wide.parquet", index=False)
    date_tag = game_date or "latest"
    proj_cols = [
        c for c in [
            "game_id", "player_id", "player_name", "stat", "p_active",
            "pmf_mean", "mean", "active_pmf_json", "pmf_json", "source_track",
            "model_version",
        ] if c in wide.columns
    ]
    proj = wide[proj_cols].copy()
    proj.to_parquet(dest / f"player_projections_{date_tag}.parquet", index=False)
    proj.drop(columns=["pmf_json", "active_pmf_json"], errors="ignore").to_json(
        dest / f"player_projections_{date_tag}.json", orient="records", indent=2
    )
    # Fair board from V6 prices when present
    prices = v6_out / "fair_prices.parquet"
    if prices.exists():
        pdf = pd.read_parquet(prices)
        pdf.to_parquet(dest / "fair_odds_board.parquet", index=False)
    man = {
        "source": "sharp_v6.predict_slate",
        "wrapper": "scripts/predict_today.py",
        "v6_out": str(v6_out),
        "game_date": game_date,
        "n_pmfs": int(len(wide)),
        "legacy_stage4": False,
    }
    (dest / "PREDICT_TODAY_V6_WRAPPER.json").write_text(json.dumps(man, indent=2))


def _run_v6(
    *,
    game_date: str | None,
    out_dir: str,
    bundle_dir: str,
    features: str | None,
    mode: str,
) -> None:
    bundle = _resolve_bundle_dir(bundle_dir)
    v6_out = Path(out_dir)
    # Keep V6 native layout under sharp_v6 when caller uses tonight/next_game dirs
    native = REPO / "deliveries" / "sharp_v6" / (game_date or "live") / "T-live"
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_wnba_pmf.py"),
        "--bundle-dir", bundle,
        "--mode", mode,
        "--out-dir", str(native),
    ]
    if game_date:
        cmd.extend(["--date", game_date])
    if features:
        cmd.extend(["--features", features])
    typer.echo(f"[V6] Delegating to sharp_v6.predict_slate via run_wnba_pmf.py (bundle={bundle})")
    proc = subprocess.run(cmd, cwd=str(REPO), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    _write_legacy_compat(native, v6_out, game_date)
    typer.echo(f"[V6] Wrote legacy-compatible delivery → {v6_out}")


@app.command()
def main(
    features_wide: str = typer.Option(
        None, help="Optional features parquet (V6 rebuilds live features; used as --features hint).",
    ),
    model_dir: str = typer.Option(
        "artifacts/models/stage4_baseline",
        help="LEGACY_CONTROL only (ignored unless --legacy-stage4).",
    ),
    config: str = typer.Option(
        "config/model/stage4_baseline.yaml",
        help="LEGACY_CONTROL only (ignored unless --legacy-stage4).",
    ),
    cal_dir: str | None = typer.Option(
        "artifacts/models/calibration",
        help="LEGACY_CONTROL only (ignored unless --legacy-stage4).",
    ),
    no_calibration: bool = typer.Option(False, "--no-calibration"),
    raw_props: str | None = typer.Option(None, help="Optional props parquet for legacy edge board."),
    out_dir: str = typer.Option("deliveries/today", help="Delivery output directory."),
    game_date: str | None = typer.Option(None, help="ISO date filter (YYYY-MM-DD)."),
    overrides: str | None = typer.Option(None, "--overrides", help="Ignored on V6 path (availability snapshot)."),
    export_grids_json: bool = typer.Option(False, "--export-grids-json"),
    use_v6: bool = typer.Option(True, "--use-v6/--no-use-v6", help="Default: authoritative V6."),
    bundle_dir: str | None = typer.Option(None, "--bundle-dir", help="Override production bundle path."),
    legacy_stage4: bool = typer.Option(
        False, "--legacy-stage4", help="LEGACY_CONTROL Stage-4 path (research/rollback only).",
    ),
    mode: str = typer.Option("production", "--mode"),
) -> None:
    """Predict today's WNBA player stat PMFs (authoritative V6 by default)."""
    if use_v6 and not legacy_stage4:
        feat_hint = None
        if features_wide and Path(features_wide).exists():
            # Prefer recovered_v2 modeling features when the caller passes a slate parquet
            recovered = REPO / "data/recovered_v2/modeling/wnba_pregame_features_t12.parquet"
            feat_hint = str(recovered) if recovered.exists() else features_wide
        _run_v6(
            game_date=game_date,
            out_dir=out_dir,
            bundle_dir=bundle_dir or "",
            features=feat_hint,
            mode=mode,
        )
        return

    # ── LEGACY_CONTROL Stage-4 path (explicit opt-in only) ─────────────────
    from wnba_props_model.models.pmf_grid import pmfs_df_to_grids
    from wnba_props_model.pipeline.deliver import write_delivery
    from wnba_props_model.pipeline.overrides import apply_overrides, override_summary
    from wnba_props_model.pipeline.predict import predict_player_pmfs

    typer.echo("[LEGACY_CONTROL] Stage-4 predict_today path — not authoritative production")
    if not features_wide:
        raise SystemExit("FAIL_CLOSED: --features-wide required for --legacy-stage4")
    features_df = pd.read_parquet(features_wide)

    if game_date:
        if "game_date" in features_df.columns:
            filtered = features_df[features_df["game_date"].astype(str) == game_date].copy()
            typer.echo(f"Filtered to game_date={game_date}: {len(filtered):,} rows")
            if not filtered.empty:
                features_df = filtered
            else:
                _unique_input_dates = features_df["game_date"].astype(str).unique()
                if len(_unique_input_dates) > 1:
                    typer.echo(
                        f"[INFO] 0 rows for game_date={game_date} in historical feature table "
                        f"({len(_unique_input_dates)} dates, last={sorted(_unique_input_dates)[-1]}). "
                        "No WNBA games scheduled for this date. Exiting cleanly."
                    )
                    raise typer.Exit(0)
                else:
                    typer.echo(
                        f"[WARN] 0 rows for game_date={game_date} in slate. "
                        "Using all rows from single-date slate input."
                    )

    if features_df.empty:
        typer.echo(f"[WARN] No player rows to predict — no games on {game_date}. Exiting.")
        raise typer.Exit(0)

    if overrides:
        features_df = _apply_json_overrides(features_df, overrides, game_date, out_dir)

    typer.echo(f"Generating Stage-4 PMFs for {len(features_df):,} player-game rows...")
    apply_cal = not no_calibration
    effective_cal_dir = cal_dir if apply_cal else None
    pmfs = predict_player_pmfs(
        feature_df=features_df,
        model_dir=model_dir,
        config_path=config,
        cal_dir=effective_cal_dir,
        apply_calibration=apply_cal,
    )
    typer.echo(f"Generated {len(pmfs):,} PMF rows (stats × players × games)")
    props_df = pd.read_parquet(raw_props) if raw_props else None
    paths = write_delivery(pmfs, out_dir, props_df, game_date=game_date)
    for k, v in paths.items():
        typer.echo(f"  {k}: {v}")

    if export_grids_json:
        ctx_cols = ["game_id", "game_date", "team_id", "opponent_team_id", "is_home"]
        grids = pmfs_df_to_grids(pmfs, game_context_cols=ctx_cols)
        out_path = Path(out_dir) / f"pmf_grids_{game_date or 'latest'}.json"
        with open(out_path, "w") as f:
            json.dump([g.to_dict() for g in grids], f, default=str, indent=2)
        typer.echo(f"  pmf_grids_json: {out_path} ({len(grids)} grids)")


def _apply_json_overrides(
    features_df: pd.DataFrame,
    overrides_path: str,
    game_date: str | None,
    out_dir: str,
) -> pd.DataFrame:
    from wnba_props_model.pipeline.overrides import apply_overrides, override_summary

    p = Path(overrides_path)
    if not p.exists():
        typer.echo(f"[OVERRIDES] File not found: {overrides_path} — skipping")
        return features_df

    try:
        payload = json.loads(p.read_text())
    except Exception as exc:
        typer.echo(f"[OVERRIDES] Could not parse {overrides_path}: {exc} — skipping")
        return features_df

    entries = payload if isinstance(payload, list) else payload.get("overrides", [])
    if not entries:
        return features_df

    seen: dict[int, dict] = {}
    for entry in entries:
        entry_date = str(entry.get("game_date") or "")
        if game_date and entry_date and entry_date != game_date:
            continue
        pid = int(entry.get("player_id", 0))
        if pid in seen:
            typer.echo(f"[OVERRIDES] Duplicate for player_id={pid} on {entry_date} — last-write-wins")
        seen[pid] = entry

    if not seen:
        typer.echo("[OVERRIDES] No active overrides for this game date")
        return features_df

    dnp_ids: list[int] = []
    minutes_map: dict[int, float] = {}
    for pid, entry in seen.items():
        override_mins = entry.get("override_minutes")
        if override_mins is not None:
            mins_val = float(override_mins)
            if mins_val < 1.0:
                dnp_ids.append(pid)
            else:
                minutes_map[pid] = mins_val
        else:
            dnp_ids.append(pid)

    original = features_df.copy()
    features_df = apply_overrides(
        features_df, dnp_player_ids=dnp_ids or None, minutes_overrides=minutes_map or None,
    )
    summary = override_summary(original, features_df)
    typer.echo(f"[OVERRIDES] Applied {summary['n_players_changed']} player override(s)")

    log_path = Path(out_dir) / "override_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import datetime as _dt
    log_payload = {
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_date": game_date,
        "overrides_file": overrides_path,
        "changes": summary.get("changes", []),
        "entries_applied": [dict(e) for e in seen.values()],
    }
    log_path.write_text(json.dumps(log_payload, indent=2, default=str))
    typer.echo(f"[OVERRIDES] Log written → {log_path}")
    return features_df


if __name__ == "__main__":
    app()
