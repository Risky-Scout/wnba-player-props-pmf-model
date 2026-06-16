"""Post-game CLV and accuracy scorer.

Runs after games complete. Pulls actual BDL player stats, joins against
pre-game model predictions, and computes:
  - NLL (Negative Log-Likelihood) per PMF
  - RPS (Ranked Probability Score)
  - Binary Log Loss (Ignorance Score) for over/under market lines
  - CLV (Closing Line Value) = model_prob - market_prob_at_prediction_time

Appends scored rows to data/clv_tracking/results.parquet for longitudinal tracking.

Usage:
    python scripts/score_daily_predictions.py \\
        --predictions deliveries/today/full_pmfs_wide.parquet \\
        --market-comparison deliveries/today/market_comparison.parquet \\
        --actuals data/processed/player_game_stats.parquet \\
        --game-date 2026-06-15 \\
        --out-dir data/clv_tracking
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.evaluation.diagnostics import pmf_nll, rps
from wnba_props_model.models.market import binary_logloss, ignorance_score_binary, prob_over_from_pmf
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf

app = typer.Typer(add_completion=False)

_STAT_ALIASES = {
    "tov": "turnover",
    "to": "turnover",
    "3pm": "fg3m",
    "3p": "fg3m",
}


def _normalize_stat(s: str) -> str:
    return _STAT_ALIASES.get(s.lower(), s.lower())


@app.command()
def main(
    predictions: str = typer.Option(..., help="Pre-game PMF parquet (full_pmfs_wide.parquet)."),
    market_comparison: str | None = typer.Option(None, help="Market comparison parquet (for CLV)."),
    actuals: str = typer.Option(..., help="Post-game player_game_stats parquet."),
    game_date: str | None = typer.Option(None, help="ISO date of games scored (YYYY-MM-DD)."),
    out_dir: str = typer.Option("data/clv_tracking", help="CLV tracking output directory."),
    results_file: str = typer.Option("data/clv_tracking/results.parquet", help="Cumulative results file."),
    closing_lines: str | None = typer.Option(None, help="Closing-line props parquet (from pull_closing_lines.py)."),
    predictions_dir: str | None = typer.Option(None, help="Delivery dir to scan for full_pmfs_wide.parquet."),
    features_wide: str | None = typer.Option(None, help="Wide feature table (fallback actuals source)."),
) -> None:
    """Score post-game predictions and compute CLV."""
    today = game_date or date.today().isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve predictions path (may be a directory scan or direct path)
    pred_path = Path(predictions)
    if pred_path.is_dir() or not pred_path.exists():
        # Scan delivery dir
        scan_dir = Path(predictions_dir or predictions)
        candidates = list(scan_dir.rglob("full_pmfs_wide.parquet"))
        if not candidates:
            typer.echo(f"[WARN] No full_pmfs_wide.parquet found in {scan_dir}")
            return
        pred_path = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        typer.echo(f"[score] Using predictions from {pred_path}")

    pmfs_df = pd.read_parquet(pred_path)

    # Resolve actuals (may be wide feature table or direct long format)
    actuals_path = Path(actuals) if actuals else (Path(features_wide) if features_wide else None)
    if actuals_path is None or not actuals_path.exists():
        if features_wide and Path(features_wide).exists():
            actuals_path = Path(features_wide)
        else:
            typer.echo(f"[WARN] No actuals source found")
            return
    actuals_df = pd.read_parquet(actuals_path)

    # Normalize stat names in actuals
    if "stat" not in actuals_df.columns:
        melted = []
        for stat in ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "tov"):
            if stat in actuals_df.columns:
                tmp = actuals_df[["game_id", "player_id", stat]].copy()
                tmp.rename(columns={stat: "actual_outcome"}, inplace=True)
                tmp["stat"] = _normalize_stat(stat)
                melted.append(tmp)
        actuals_df = pd.concat(melted, ignore_index=True) if melted else pd.DataFrame()

    actuals_df["stat"] = actuals_df["stat"].map(_normalize_stat)

    joined = pmfs_df.merge(
        actuals_df[["game_id", "player_id", "stat", "actual_outcome"]],
        on=["game_id", "player_id", "stat"],
        how="inner",
    )
    if joined.empty:
        typer.echo(f"[WARN] No overlap between predictions and actuals for {today}")
        return
    typer.echo(f"Joined {len(joined):,} prediction/actual pairs")

    # Score PMFs
    pmf_arrays = [normalize_pmf(json_to_pmf(j)) for j in joined["pmf_json"]]
    outcomes = joined["actual_outcome"].astype(int).tolist()

    joined["pmf_nll"] = [pmf_nll(p, y) for p, y in zip(pmf_arrays, outcomes)]
    joined["pmf_rps"] = [rps(p, y) for p, y in zip(pmf_arrays, outcomes)]

    # Compute CLV if market comparison is available
    if market_comparison and Path(market_comparison).exists():
        mkt = pd.read_parquet(market_comparison)
        if not mkt.empty and "market_prob_over_no_vig" in mkt.columns:
            mkt_sub = mkt[["game_id", "player_id", "stat", "line",
                            "market_prob_over_no_vig", "model_prob_over"]].copy()
            joined = joined.merge(mkt_sub, on=["game_id", "player_id", "stat"], how="left")
            hit = (joined["actual_outcome"].astype(float) > joined["line"].astype(float))
            push = (joined["actual_outcome"].astype(float) == joined["line"].astype(float))
            valid = joined["market_prob_over_no_vig"].notna() & ~push

            joined.loc[valid, "hit_result"] = hit[valid].astype(int)
            joined.loc[valid, "model_bin_logloss"] = [
                binary_logloss(p, y)
                for p, y in zip(
                    joined.loc[valid, "model_prob_over"],
                    joined.loc[valid, "hit_result"],
                )
            ]
            joined.loc[valid, "market_bin_logloss"] = [
                binary_logloss(p, y)
                for p, y in zip(
                    joined.loc[valid, "market_prob_over_no_vig"],
                    joined.loc[valid, "hit_result"],
                )
            ]
            joined.loc[valid, "model_ignorance_score"] = [
                ignorance_score_binary(p, y)
                for p, y in zip(
                    joined.loc[valid, "model_prob_over"],
                    joined.loc[valid, "hit_result"],
                )
            ]
            joined.loc[valid, "market_ignorance_score"] = [
                ignorance_score_binary(p, y)
                for p, y in zip(
                    joined.loc[valid, "market_prob_over_no_vig"],
                    joined.loc[valid, "hit_result"],
                )
            ]
            # CLV (open-line): positive = model edge was correct direction vs. OPEN market
            joined.loc[valid, "clv"] = (
                joined.loc[valid, "model_prob_over"] - joined.loc[valid, "market_prob_over_no_vig"]
            ) * (2 * joined.loc[valid, "hit_result"] - 1)
            joined.loc[valid, "clv_type"] = "open"

    # True CLV: compute vs. closing lines if available (the gold standard)
    if closing_lines and Path(closing_lines).exists():
        try:
            cl_df = pd.read_parquet(closing_lines)
            if not cl_df.empty and "market_prob_over_no_vig" in cl_df.columns:
                cl_df = cl_df.rename(columns={
                    "market_prob_over_no_vig": "closing_prob_over_no_vig",
                    "line": "closing_line",
                })
                cl_sub = cl_df[["game_id", "player_id", "stat",
                                "closing_prob_over_no_vig", "closing_line"]].dropna()
                joined = joined.merge(cl_sub, on=["game_id", "player_id", "stat"], how="left")

                cl_valid = (
                    joined["closing_prob_over_no_vig"].notna()
                    & joined["hit_result"].notna()
                )
                if cl_valid.any():
                    joined.loc[cl_valid, "true_clv"] = (
                        joined.loc[cl_valid, "model_prob_over"]
                        - joined.loc[cl_valid, "closing_prob_over_no_vig"]
                    ) * (2 * joined.loc[cl_valid, "hit_result"].astype(float) - 1)
                    typer.echo(
                        f"[score] True CLV computed for {cl_valid.sum()} rows "
                        f"(mean={joined.loc[cl_valid, 'true_clv'].mean():+.4f})"
                    )
        except Exception as exc:
            typer.echo(f"[WARN] Closing-line CLV computation failed: {exc}", err=True)

    joined["game_date"] = today
    joined["scored_at"] = datetime.now(timezone.utc).isoformat()

    # Save today's scored rows
    today_path = out / f"scored_{today}.parquet"
    drop_cols = [c for c in ("pmf_json",) if c in joined.columns]
    joined.drop(columns=drop_cols).to_parquet(today_path, index=False)
    typer.echo(f"Wrote today's scored rows → {today_path} ({len(joined):,} rows)")

    # Append to cumulative results
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        existing = pd.read_parquet(results_path)
        # De-duplicate: remove any previous rows for this game_date
        existing = existing[existing["game_date"] != today]
        combined = pd.concat([existing, joined.drop(columns=drop_cols)], ignore_index=True)
    else:
        combined = joined.drop(columns=drop_cols)
    combined.to_parquet(results_path, index=False)
    typer.echo(f"Updated cumulative results → {results_path} ({len(combined):,} total rows)")


if __name__ == "__main__":
    app()
