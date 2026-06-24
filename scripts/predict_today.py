"""Generate calibrated PMF predictions for today's WNBA slate.

Uses the Stage 4 HGB engine + Stage 6 isotonic calibrators (if available).

Usage:
    python scripts/predict_today.py \\
        --features-wide data/processed/wnba_player_game_features_wide.parquet \\
        --model-dir artifacts/models/stage4_baseline \\
        --cal-dir artifacts/models/calibration \\
        --raw-props data/processed/player_props.parquet \\
        --out-dir deliveries/today
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.pmf_grid import pmfs_df_to_grids
from wnba_props_model.pipeline.deliver import write_delivery
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
                typer.echo(
                    f"[WARN] 0 rows for game_date={game_date}. "
                    "Using all rows from input (slate forward-dated features)."
                )
                # Slate files have game_date already set to the target date so no filter needed

    if features_df.empty:
        typer.echo(f"[WARN] No player rows to predict — no games on {game_date}. Exiting.")
        raise typer.Exit(0)

    typer.echo(f"Generating PMFs for {len(features_df):,} player-game rows...")

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
    n_cal = pmfs["is_calibrated"].sum() if "is_calibrated" in pmfs.columns else 0
    typer.echo(f"Calibrated: {n_cal:,}/{len(pmfs):,} rows")

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


if __name__ == "__main__":
    app()
