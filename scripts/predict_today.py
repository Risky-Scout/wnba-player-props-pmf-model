"""Generate calibrated PMF predictions for today's WNBA slate.

Uses the Stage 4 HGB engine + Stage 6 isotonic calibrators (if available).

Usage:
    python scripts/predict_today.py \\
        --features-wide data/processed/wnba_player_game_features_wide.parquet \\
        --model-dir artifacts/models/stage4_baseline \\
        --cal-dir artifacts/models/calibration \\
        --overrides config/player_overrides.json \\
        --raw-props data/processed/player_props.parquet \\
        --out-dir deliveries/today
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.pmf_grid import pmfs_df_to_grids
from wnba_props_model.pipeline.deliver import write_delivery
from wnba_props_model.pipeline.overrides import apply_overrides, override_summary
from wnba_props_model.pipeline.predict import predict_player_pmfs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    features_wide: str = typer.Option(..., help="Wide feature parquet from build_features.py."),
    model_dir: str = typer.Option("artifacts/models/stage4_baseline", help="Stage 4 HGB artifact directory."),
    config: str = typer.Option("config/model/stage4_baseline.yaml", help="Stage 4 YAML config."),
    cal_dir: str | None = typer.Option("artifacts/models/calibration", help="Calibrator directory; None to skip."),
    no_calibration: bool = typer.Option(False, "--no-calibration", help="Skip calibration application."),
    raw_props: str | None = typer.Option(None, help="BDL player props parquet for edge calculation."),
    out_dir: str = typer.Option("deliveries/today", help="Delivery output directory."),
    game_date: str | None = typer.Option(None, help="ISO date filter (YYYY-MM-DD); predicts only this date."),
    overrides: str | None = typer.Option(
        None, "--overrides",
        help="Path to player_overrides.json (blueprint §6.1). "
             "Reads active overrides for game_date and applies UTM redistribution.",
    ),
    export_grids_json: bool = typer.Option(False, "--export-grids-json",
        help="Also write a full WNBAPMFGrid JSON sidecar with all markets at 0.5-step lines."),
) -> None:
    """Predict today's WNBA player stat PMFs and compute market edges."""
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
                    # Multi-date historical table: this date has no scheduled games.
                    # Do NOT fall back to all historical rows — that produces nonsense output.
                    typer.echo(
                        f"[INFO] 0 rows for game_date={game_date} in historical feature table "
                        f"({len(_unique_input_dates)} dates, last={sorted(_unique_input_dates)[-1]}). "
                        "No WNBA games scheduled for this date. Exiting cleanly."
                    )
                    raise typer.Exit(0)
                else:
                    # Single-date slate: all rows belong to the target date already.
                    typer.echo(
                        f"[WARN] 0 rows for game_date={game_date} in slate. "
                        "Using all rows from single-date slate input."
                    )

    if features_df.empty:
        typer.echo(f"[WARN] No player rows to predict — no games on {game_date}. Exiting.")
        raise typer.Exit(0)

    # ── Apply manual overrides from config/player_overrides.json (blueprint §6.1) ──
    if overrides:
        features_df = _apply_json_overrides(features_df, overrides, game_date, out_dir)

    typer.echo(f"Generating PMFs for {len(features_df):,} player-game rows...")

    apply_cal = not no_calibration
    effective_cal_dir = cal_dir if apply_cal else None

    # #region agent log
    import json as _jpd, time as _tpd, sys as _spd
    _feat_src = features_wide
    _feat_dates = sorted(features_df["game_date"].astype(str).unique()) if "game_date" in features_df.columns else []
    _feat_players = sorted(features_df["player_name"].unique().tolist()) if "player_name" in features_df.columns else []
    print(f"[predict_today] Feature source: {_feat_src}")
    print(f"[predict_today] Feature rows: {len(features_df):,}, dates: {_feat_dates[-3:] if _feat_dates else 'N/A'}")
    print(f"[predict_today] Players ({len(_feat_players)}): {_feat_players[:8]}")
    try:
        with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
            _f.write(_jpd.dumps({"sessionId": "94807e", "hypothesisId": "H3", "location": "predict_today.py:features", "message": "feature_source_info", "data": {"source": _feat_src, "rows": len(features_df), "dates": _feat_dates[-3:], "n_players": len(_feat_players), "sample_players": _feat_players[:8]}, "timestamp": int(_tpd.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion agent log

    # Run WITHOUT calibration first to get raw means
    # #region agent log
    _raw_pmfs = predict_player_pmfs(feature_df=features_df, model_dir=model_dir, config_path=config, cal_dir=None, apply_calibration=False)
    _raw_pts = _raw_pmfs[_raw_pmfs["stat"] == "pts"][["player_name", "pmf_mean", "role_bucket"]].copy() if not _raw_pmfs.empty else pd.DataFrame()
    if not _raw_pts.empty:
        _raw_pts = _raw_pts.sort_values("pmf_mean", ascending=False)
        print(f"[predict_today] RAW pts predictions (top 8):")
        for _, _r in _raw_pts.head(8).iterrows():
            print(f"  {_r['player_name'][:22]:22s} raw_mean={_r['pmf_mean']:.2f} role={_r['role_bucket']}")
        try:
            with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
                _f.write(_jpd.dumps({"sessionId": "94807e", "hypothesisId": "H4", "location": "predict_today.py:raw_preds", "message": "raw_pts_predictions", "data": {"players": [{"name": r["player_name"], "raw_mean": round(r["pmf_mean"], 2), "role": r["role_bucket"]} for _, r in _raw_pts.iterrows()]}, "timestamp": int(_tpd.time() * 1000)}) + "\n")
        except Exception:
            pass
    # #endregion agent log

    pmfs = predict_player_pmfs(
        feature_df=features_df,
        model_dir=model_dir,
        config_path=config,
        cal_dir=effective_cal_dir,
        apply_calibration=apply_cal,
    )
    typer.echo(f"Generated {len(pmfs):,} PMF rows (stats × players × games)")
    n_cal = pmfs["is_calibrated"].sum() if "is_calibrated" in pmfs.columns else 0
    typer.echo(f"Calibrated: {n_cal:,}/{len(pmfs):,} rows")

    # #region agent log
    if apply_cal and not pmfs.empty:
        _cal_pts = pmfs[pmfs["stat"] == "pts"][["player_name", "pmf_mean", "role_bucket"]].copy()
        if not _cal_pts.empty and not _raw_pts.empty:
            _merged = _raw_pts.merge(_cal_pts.rename(columns={"pmf_mean": "cal_mean"}), on=["player_name", "role_bucket"])
            _merged["ratio"] = _merged["cal_mean"] / _merged["pmf_mean"].clip(lower=0.01)
            _merged = _merged.sort_values("pmf_mean", ascending=False)
            print(f"[predict_today] RAW vs CALIBRATED pts (top 8):")
            for _, _r in _merged.head(8).iterrows():
                print(f"  {_r['player_name'][:22]:22s} raw={_r['pmf_mean']:.2f} cal={_r['cal_mean']:.2f} ratio={_r['ratio']:.3f}")
            try:
                with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _f:
                    _f.write(_jpd.dumps({"sessionId": "94807e", "hypothesisId": "H4", "location": "predict_today.py:cal_preds", "message": "raw_vs_calibrated_pts", "data": {"players": [{"name": r["player_name"], "raw": round(r["pmf_mean"], 2), "cal": round(r["cal_mean"], 2), "ratio": round(r["ratio"], 3)} for _, r in _merged.iterrows()]}, "timestamp": int(_tpd.time() * 1000)}) + "\n")
            except Exception:
                pass
    # #endregion agent log

    props_df = pd.read_parquet(raw_props) if raw_props else None

    # Game Total Anchoring (Item 6) — ensure player projections are coherent
    # with the efficiently-priced game total market before delivering.
    try:
        from wnba_props_model.models.game_total_anchor import GameTotalAnchoring  # noqa: PLC0415
        odds_df: pd.DataFrame | None = None
        odds_candidates = [
            Path("data/processed/wnba_odds.parquet"),
            Path("data/raw/bdl/wnba_odds.parquet"),
        ]
        for cand in odds_candidates:
            if cand.exists():
                odds_df = pd.read_parquet(cand)
                break
        if odds_df is not None and not pmfs.empty:
            anchoring = GameTotalAnchoring(threshold=3.0, max_scale=1.15)
            # Build per-game player projection list for anchoring
            for game_id, g_rows in pmfs.groupby("game_id"):
                pts_rows = g_rows[g_rows["stat"] == "pts"]
                if pts_rows.empty:
                    continue
                game_odds = odds_df[odds_df["game_id"] == game_id] if "game_id" in odds_df.columns else pd.DataFrame()
                market_total = anchoring.get_market_total(game_odds)
                if market_total is None:
                    continue
                projs = pts_rows[["player_id", "team_id", "mean"]].copy()
                projs = projs.rename(columns={"mean": "pts_mean"})
                projs["team"] = projs.apply(
                    lambda r: "home" if r.get("is_home", True) else "away", axis=1
                )
                anchored = anchoring.anchor(projs.to_dict("records"), market_total)
                scale_map = {
                    r["player_id"]: r.get("anchor_scale_factor", 1.0) for r in anchored
                }
                if "anchor_scale_factor" not in pmfs.columns:
                    pmfs["anchor_scale_factor"] = 1.0
                pmfs.loc[pmfs["game_id"] == game_id, "anchor_scale_factor"] = (
                    pmfs.loc[pmfs["game_id"] == game_id, "player_id"].map(scale_map).fillna(1.0)
                )
            typer.echo("Game Total Anchoring applied.")
    except Exception as exc:
        typer.echo(f"[WARN] Game Total Anchoring failed (non-fatal): {exc}")

    # Beta Calibration of P(over) (Item 5A) — apply Beta calibrators fitted by
    # fit_calibrators.py. Beta calibrators recalibrate the P(over) scalar, which
    # is more powerful than PMF-level isotonic calibration for binary market edges.
    try:
        import joblib as _jl  # noqa: PLC0415
        from wnba_props_model.pipeline.calibrate import apply_beta_calibrators  # noqa: PLC0415
        _beta_cal_dir = Path(effective_cal_dir or "artifacts/models/calibration")
        if not pmfs.empty and "stat" in pmfs.columns:
            _beta_applied = 0
            for _stat, _stat_rows in pmfs.groupby("stat"):
                _cal_path = _beta_cal_dir / f"beta_cal_{_stat}.pkl"
                if not _cal_path.exists():
                    continue
                _cal = _jl.load(_cal_path)
                # Compute P(over) from PMF at the line column (use market line if present)
                _line_col = "line" if "line" in _stat_rows.columns else None
                _mean_col = "mean" if "mean" in _stat_rows.columns else None
                if _mean_col is None:
                    continue
                # We only recalibrate the scalar p_over, not the full PMF shape.
                # store into p_over_beta column for downstream edge calc.
                _raw_p = _stat_rows.get("p_over", _stat_rows[_mean_col] / (_stat_rows[_mean_col] + 1.0))
                _vals = _raw_p.fillna(0.5).clip(1e-6, 1 - 1e-6).values.reshape(-1, 1)
                try:
                    _cal_p = _cal.predict(_vals)
                    pmfs.loc[_stat_rows.index, "p_over_beta"] = _cal_p
                    _beta_applied += len(_stat_rows)
                except Exception:
                    pass
            if _beta_applied:
                typer.echo(f"Beta calibration applied to {_beta_applied:,} rows.")
    except Exception as _bexc:
        typer.echo(f"[WARN] Beta calibrators skipped (non-fatal): {_bexc}")

    # Conformal Prediction Intervals (Item 5D) — flag props where model uncertainty
    # is too high to have a meaningful edge (line inside conformal interval → no edge).
    try:
        import pickle as _pkl  # noqa: PLC0415
        _conformal_path = Path(effective_cal_dir or "artifacts/models/calibration") / "conformal_predictor.pkl"
        if _conformal_path.exists() and not pmfs.empty and "mean" in pmfs.columns:
            with open(_conformal_path, "rb") as _f:
                _conformal = _pkl.load(_f)
            _stat_col = pmfs["stat"] if "stat" in pmfs.columns else pd.Series(["pts"] * len(pmfs))
            _role_col = pmfs["role_bucket"] if "role_bucket" in pmfs.columns else pd.Series(["all"] * len(pmfs))
            _means = pmfs["mean"].to_numpy(dtype=float)
            _lows, _highs = np.empty(len(pmfs)), np.empty(len(pmfs))
            for i, (stat, role, mu) in enumerate(zip(_stat_col, _role_col, _means)):
                _lows[i], _highs[i] = _conformal.predict_interval(mu, stat=str(stat), role=str(role))
            pmfs = pmfs.copy()
            pmfs["conformal_lower"] = _lows
            pmfs["conformal_upper"] = _highs
            typer.echo(f"Conformal prediction intervals applied ({len(_conformal.quantiles)} buckets).")
    except Exception as _exc:
        typer.echo(f"[WARN] Conformal intervals skipped (non-fatal): {_exc}")

    paths = write_delivery(pmfs, out_dir, props_df, game_date=game_date)
    for k, v in paths.items():
        typer.echo(f"  {k}: {v}")

    if export_grids_json:
        import json as _json
        ctx_cols = ["game_id", "game_date", "team_id", "opponent_team_id", "is_home"]
        grids = pmfs_df_to_grids(pmfs, game_context_cols=ctx_cols)
        out_path = Path(out_dir) / f"pmf_grids_{game_date or 'latest'}.json"
        with open(out_path, "w") as f:
            _json.dump([g.to_dict() for g in grids], f, default=str, indent=2)
        typer.echo(f"  pmf_grids_json: {out_path} ({len(grids)} grids)")


def _apply_json_overrides(
    features_df: pd.DataFrame,
    overrides_path: str,
    game_date: str | None,
    out_dir: str,
) -> pd.DataFrame:
    """Read config/player_overrides.json, apply active overrides, log changes.

    Blueprint §6.1: overrides expire after game_date; multiple overrides for
    the same player on the same date are rejected (last-write-wins with a warning).
    """
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

    # Filter to active entries for this game_date; de-duplicate (last wins)
    seen: dict[int, dict] = {}
    for entry in entries:
        entry_date = str(entry.get("game_date") or "")
        if game_date and entry_date and entry_date != game_date:
            continue  # expired or different day
        pid = int(entry.get("player_id", 0))
        if pid in seen:
            typer.echo(f"[OVERRIDES] Duplicate for player_id={pid} on {entry_date} — last-write-wins")
        seen[pid] = entry

    if not seen:
        typer.echo("[OVERRIDES] No active overrides for this game date")
        return features_df

    # Separate DNP (minutes=0) vs minutes overrides
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
    features_df = apply_overrides(features_df, dnp_player_ids=dnp_ids or None, minutes_overrides=minutes_map or None)
    summary = override_summary(original, features_df)

    typer.echo(f"[OVERRIDES] Applied {summary['n_players_changed']} player override(s)")
    for ch in summary.get("changes", []):
        reason = seen.get(ch["player_id"], {}).get("reason", "")
        typer.echo(
            f"  player_id={ch['player_id']} {ch.get('player_name','')} "
            f"{ch['original_minutes']:.1f}→{ch['overridden_minutes']:.1f} min "
            f"({reason})"
        )

    # Write override log next to delivery outputs
    log_path = Path(out_dir) / "override_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import datetime as _dt
    log_payload = {
        "generated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_date": game_date,
        "overrides_file": overrides_path,
        "changes": summary.get("changes", []),
        "entries_applied": [
            {k: v for k, v in e.items()}
            for e in seen.values()
        ],
    }
    log_path.write_text(json.dumps(log_payload, indent=2, default=str))
    typer.echo(f"[OVERRIDES] Log written → {log_path}")
    return features_df


if __name__ == "__main__":
    app()
