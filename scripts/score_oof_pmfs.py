#!/usr/bin/env python3
"""Stage 5 OOF PMF scoring script.

Computes uncalibrated PMF metrics (NLL, RPS, mean_error, variance_ratio)
for all stats and role breakdowns.  Writes scoring audit JSON and readable
Markdown summary.  Does NOT fit calibrators.

Usage:
    python3 scripts/score_oof_pmfs.py \\
      --pmfs data/oof/oof_player_stat_pmfs.parquet \\
      --audit-out artifacts/audits/stage5_oof_scoring_audit.json \\
      --summary-out artifacts/audits/stage5_oof_summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.evaluation.oof_scoring import score_oof_dataframe

PMF_SUPPORT_CAPS = {
    "pts": 60, "reb": 30, "ast": 25, "fg3m": 15,
    "stl": 10, "blk": 10, "turnover": 12,
}

app = typer.Typer(add_completion=False)


@app.command()
def score(
    pmfs: Path = typer.Option(
        Path("data/oof/oof_player_stat_pmfs.parquet"), "--pmfs"
    ),
    audit_out: Path = typer.Option(
        Path("artifacts/audits/stage5_oof_scoring_audit.json"),
        "--audit-out",
    ),
    summary_out: Path = typer.Option(
        Path("artifacts/audits/stage5_oof_summary.md"),
        "--summary-out",
    ),
    calibration_only: bool = typer.Option(
        True, "--calibration-only/--all-rows",
        help="Score only calibration_eligible=True rows (default)."
    ),
) -> None:
    print("=" * 70)
    print("Stage 5 — OOF PMF Scoring")
    print("=" * 70)

    if not pmfs.exists():
        typer.echo(f"ERROR: {pmfs} not found", err=True)
        raise typer.Exit(1)

    print(f"Loading: {pmfs}")
    oof = pd.read_parquet(pmfs)
    oof["game_date"] = pd.to_datetime(oof["game_date"])
    print(f"  {len(oof):,} total rows")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    print(f"\nScoring {'calibration_eligible only' if calibration_only else 'all rows'}...")

    # Try to merge in role columns from the long features table for breakdowns
    role_cols = ["projected_minutes_bucket", "role_status", "role_uncertainty_bucket"]
    for col in role_cols:
        if col not in oof.columns:
            # Try to load from feature table
            feat_path = Path("data/processed/wnba_player_game_features_long.parquet")
            if feat_path.exists():
                try:
                    feat = pd.read_parquet(feat_path, columns=["player_id", "game_id"] + role_cols[:1])
                    oof = oof.merge(feat.drop_duplicates(subset=["player_id","game_id"]),
                                    on=["player_id","game_id"], how="left", suffixes=("","_feat"))
                    break
                except Exception:
                    pass

    results = score_oof_dataframe(oof, PMF_SUPPORT_CAPS, calibration_only=calibration_only)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'Stat':<12} {'N':>7} {'NLL':>8} {'IS':>8} {'RPS':>8} {'Brier':>8} {'MeanErr':>9} {'VarRatio':>10}")
    print("-" * 90)
    for stat in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
        s = results.get("by_stat", {}).get(stat, {})
        if not s:
            continue
        print(f"{stat:<12} {s['n']:>7,} "
              f"{s['pmf_nll_mean']:>8.4f} "
              f"{s.get('ignorance_score_mean', 0):>8.4f} "
              f"{s['pmf_rps_mean']:>8.4f} "
              f"{s.get('brier_mean', 0):>8.4f} "
              f"{s['mean_error']:>+9.3f} "
              f"{s['variance_ratio'] or 0:>10.3f}")

    # ------------------------------------------------------------------
    # Write audit
    # ------------------------------------------------------------------
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved scoring audit: {audit_out}")

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    _write_summary_md(results, oof, summary_out, calibration_only)
    print(f"Saved summary: {summary_out}")

    print("\n" + "=" * 70)
    print("Scoring Complete (uncalibrated Stage 5 baseline)")
    print("NOTE: This model is NOT calibrated. Do not claim market superiority.")
    print("      Positive mean_error indicates upward bias; Stage 6 will calibrate.")
    print("=" * 70)


def _write_summary_md(
    results: dict,
    oof_df: pd.DataFrame,
    out_path: Path,
    cal_only: bool,
) -> None:
    from datetime import datetime

    lines = [
        "# Stage 5 OOF PMF Scoring Summary",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Rows scored:** {'calibration_eligible only' if cal_only else 'all rows'}",
        f"**Total OOF rows:** {results.get('n_total', 0):,}",
        f"**model_oof rows:** {results.get('n_model_oof', 0):,}",
        f"**prior_only rows:** {results.get('n_prior_only', 0):,}",
        f"**calibration_eligible:** {results.get('n_calibration_eligible', 0):,}",
        "",
        "## ⚠️ Status: Uncalibrated",
        "",
        "This is a Stage 5 baseline. These PMFs are **not calibrated**.",
        "- Do NOT claim market superiority.",
        "- Do NOT claim calibration.",
        "- Mean errors are expected and will be corrected in Stage 6.",
        "",
        "## Per-Stat Scoring",
        "",
        "| Stat | N | NLL | IS | RPS | Brier | Mean Error | Var Ratio |",
        "|------|---|-----|----|-----|-------|------------|-----------|",
    ]

    for stat in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
        s = results.get("by_stat", {}).get(stat)
        if not s:
            continue
        lines.append(
            f"| {stat} | {s['n']:,} "
            f"| {s['pmf_nll_mean']:.4f} "
            f"| {s.get('ignorance_score_mean', 0):.4f} "
            f"| {s['pmf_rps_mean']:.4f} "
            f"| {s.get('brier_mean', 0):.4f} "
            f"| {s['mean_error']:+.3f} "
            f"| {s['variance_ratio'] or 0:.3f} |"
        )

    # Worst cells by mean error
    role_results = results.get("by_stat_role_bucket", {})
    if role_results:
        sorted_cells = sorted(role_results.items(),
                              key=lambda x: abs(x[1].get("mean_error", 0)), reverse=True)
        lines += [
            "",
            "## Top 10 Cells by |Mean Error|",
            "",
            "| Cell | N | Mean Actual | Mean PMF | Mean Error | NLL |",
            "|------|---|-------------|----------|------------|-----|",
        ]
        for cell, stats_d in sorted_cells[:10]:
            lines.append(
                f"| {cell} | {stats_d['n']:,} | {stats_d['mean_actual']:.3f} "
                f"| {stats_d['mean_pmf']:.3f} | {stats_d['mean_error']:+.3f} "
                f"| {stats_d['pmf_nll_mean']:.4f} |"
            )

    lines += [
        "",
        "## Next Steps",
        "",
        "1. Stage 6: Fit PMF calibrators using OOF PMFs (calibration_eligible=True rows).",
        "2. Use PIT KS ≤ 0.075 and ECE ≤ 0.025 as calibration pass gates.",
        "3. Do not use market data for calibration.",
    ]

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    app()
