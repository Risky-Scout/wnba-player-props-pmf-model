"""Build historical review package for model evaluation.

Generates per-stat accuracy metrics, calibration curves, and hit rate tables
over the OOF holdout window.

Outputs:
    artifacts/historical_review/
    ├── summary_metrics.json          per-stat MAE, RMSE, Ignorance Score delta
    ├── calibration_curves.json       reliability diagram data per stat
    ├── hit_rate_tables.json          P(model prob) vs actual hit rate at various lines
    ├── historical_review.parquet     full row-level results for ad-hoc analysis
    └── docs/HISTORICAL_REVIEW.md     auto-generated human-readable report

Usage:
    python scripts/build_historical_review.py \
        --oof-scored artifacts/audits/oof_scored.parquet \
        --out-dir artifacts/historical_review
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.market import no_vig_two_way, ignorance_score_binary

app = typer.Typer(add_completion=False)

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
COMMON_LINES: dict[str, list[float]] = {
    "pts": [4.5, 7.5, 9.5, 12.5, 14.5, 17.5, 19.5, 24.5],
    "reb": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    "ast": [1.5, 2.5, 3.5, 4.5, 5.5],
    "fg3m": [0.5, 1.5, 2.5, 3.5],
    "stl": [0.5, 1.5],
    "blk": [0.5, 1.5],
    "turnover": [0.5, 1.5, 2.5],
}


def _prob_over(pmf: np.ndarray, line: float) -> float:
    """P(outcome > line) using PMF tail sum."""
    k = int(math.floor(line)) + 1
    if k >= len(pmf):
        return 0.0
    return float(pmf[k:].sum())


def _load_oof(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _safe_mean(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) > 0 else np.nan


def compute_stat_metrics(stat_df: pd.DataFrame, stat: str) -> dict:
    """Compute MAE, RMSE, NLL, and ignorance score for one stat."""
    if "pmf_mean" in stat_df.columns and f"actual_{stat}" in stat_df.columns:
        actual = stat_df[f"actual_{stat}"].dropna()
        pred = stat_df.loc[actual.index, "pmf_mean"].dropna()
        aligned = actual.align(pred, join="inner")
        actual, pred = aligned[0].values, aligned[1].values
        mae = float(np.abs(actual - pred).mean()) if len(actual) > 0 else np.nan
        rmse = float(np.sqrt(((actual - pred) ** 2).mean())) if len(actual) > 0 else np.nan
    else:
        mae, rmse = np.nan, np.nan

    nll = float(stat_df.get("pmf_nll", pd.Series([np.nan])).mean(skipna=True))
    ign = float(stat_df.get("binary_logloss", pd.Series([np.nan])).mean(skipna=True))
    mkt_ign = float(stat_df.get("market_logloss", pd.Series([np.nan])).mean(skipna=True))

    return {
        "stat": stat,
        "n_rows": int(len(stat_df)),
        "mae": round(mae, 4) if not np.isnan(mae) else None,
        "rmse": round(rmse, 4) if not np.isnan(rmse) else None,
        "mean_nll": round(nll, 4) if not np.isnan(nll) else None,
        "mean_ignorance_score": round(ign, 4) if not np.isnan(ign) else None,
        "market_ignorance_score": round(mkt_ign, 4) if not np.isnan(mkt_ign) else None,
        "ignorance_delta": round(ign - mkt_ign, 4) if not np.isnan(ign) and not np.isnan(mkt_ign) else None,
    }


def compute_calibration_curve(stat_df: pd.DataFrame, n_bins: int = 10) -> list[dict]:
    """Build reliability diagram data (model probability vs. empirical frequency)."""
    rows = []
    if "model_prob_over" not in stat_df.columns or "outcome_over" not in stat_df.columns:
        return rows
    df = stat_df[["model_prob_over", "outcome_over"]].dropna()
    if len(df) < 20:
        return rows
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]
    df["bin"] = pd.cut(df["model_prob_over"], bins=bins, labels=labels, include_lowest=True)
    for center, grp in df.groupby("bin", observed=True):
        rows.append({
            "bin_center": round(float(center), 2),
            "n": int(len(grp)),
            "model_prob": round(float(grp["model_prob_over"].mean()), 4),
            "empirical_freq": round(float(grp["outcome_over"].mean()), 4),
            "calibration_error": round(float(abs(grp["model_prob_over"].mean() - grp["outcome_over"].mean())), 4),
        })
    return rows


def compute_hit_rate_table(stat_df: pd.DataFrame, stat: str) -> list[dict]:
    """For each market line, compute model accuracy and calibration."""
    rows = []
    lines = COMMON_LINES.get(stat, [])
    if "pmf_json" not in stat_df.columns or f"actual_{stat}" not in stat_df.columns:
        return rows

    import json as _json
    for line in lines:
        subset = []
        for _, r in stat_df.iterrows():
            actual = r.get(f"actual_{stat}")
            pmf_json = r.get("pmf_json")
            if pd.isna(actual) or not pmf_json:
                continue
            try:
                pmf_dict = _json.loads(pmf_json)
                pmf = np.array([pmf_dict.get(str(k), 0.0) for k in range(max(int(k) for k in pmf_dict) + 1)])
                pmf = pmf / pmf.sum()
            except Exception:
                continue
            model_prob = _prob_over(pmf, line)
            outcome = int(actual > line)
            subset.append({"model_prob": model_prob, "outcome": outcome})

        if not subset:
            continue
        sub = pd.DataFrame(subset)
        n = len(sub)
        model_avg = float(sub["model_prob"].mean())
        actual_rate = float(sub["outcome"].mean())
        # Brier score
        brier = float(((sub["model_prob"] - sub["outcome"]) ** 2).mean())
        rows.append({
            "line": line,
            "n": n,
            "model_prob_avg": round(model_avg, 4),
            "actual_hit_rate": round(actual_rate, 4),
            "calibration_error": round(abs(model_avg - actual_rate), 4),
            "brier_score": round(brier, 4),
        })
    return rows


def build_markdown_report(metrics: list[dict], cal_curves: dict, hit_tables: dict, out_dir: Path) -> str:
    lines = [
        "# WNBA PMF Model — Historical Review Package",
        "",
        "> Generated by `scripts/build_historical_review.py`  ",
        f"> Based on walk-forward OOF (out-of-fold) validation  ",
        "",
        "## Overview",
        "",
        "This package documents model accuracy over the historical holdout window.",
        "All metrics are computed on **out-of-fold** predictions — the model never",
        "saw the test rows during training.",
        "",
        "## Per-Stat Summary Metrics",
        "",
        "| Stat | N | MAE | RMSE | Ignorance Score | Market Ign. | Delta |",
        "|------|---|-----|------|----------------|-------------|-------|",
    ]
    for m in metrics:
        mae = f"{m['mae']:.3f}" if m["mae"] is not None else "—"
        rmse = f"{m['rmse']:.3f}" if m["rmse"] is not None else "—"
        ign = f"{m['mean_ignorance_score']:.4f}" if m["mean_ignorance_score"] is not None else "—"
        mkt = f"{m['market_ignorance_score']:.4f}" if m["market_ignorance_score"] is not None else "—"
        delta = f"{m['ignorance_delta']:.4f}" if m["ignorance_delta"] is not None else "—"
        lines.append(f"| {m['stat']} | {m['n_rows']} | {mae} | {rmse} | {ign} | {mkt} | {delta} |")

    lines += [
        "",
        "**Ignorance Score** = log-loss in bits. Lower is better. Delta < 0 means model beats market.",
        "",
        "## Calibration Curves",
        "",
        "Reliability diagrams show model probability vs. actual frequency.",
        "A perfectly calibrated model lies on the diagonal.",
        "",
    ]
    for stat, curve in cal_curves.items():
        if not curve:
            continue
        lines.append(f"### {stat.upper()}")
        lines.append("")
        lines.append("| Bin | N | Model Prob | Empirical Freq | Cal. Error |")
        lines.append("|-----|---|-----------|----------------|------------|")
        for row in curve:
            lines.append(
                f"| {row['bin_center']:.2f} | {row['n']} | "
                f"{row['model_prob']:.3f} | {row['empirical_freq']:.3f} | {row['calibration_error']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Hit Rate Tables",
        "",
        "Model over-probability vs. actual hit rate at common market lines.",
        "",
    ]
    for stat, table in hit_tables.items():
        if not table:
            continue
        lines.append(f"### {stat.upper()}")
        lines.append("")
        lines.append("| Line | N | Model P(over) | Actual Hit Rate | Cal. Error | Brier |")
        lines.append("|------|---|--------------|----------------|------------|-------|")
        for row in table:
            lines.append(
                f"| {row['line']} | {row['n']} | {row['model_prob_avg']:.3f} | "
                f"{row['actual_hit_rate']:.3f} | {row['calibration_error']:.3f} | {row['brier_score']:.4f} |"
            )
        lines.append("")

    lines += [
        "## Methodology Notes",
        "",
        "- **OOF Protocol**: Strict chronological walk-forward validation with expanding window.",
        "  Minimum 100 games training data before first prediction.",
        "- **Calibration**: Role-aware isotonic regression on PIT (Probability Integral Transform) values.",
        "  ECE gate: < 0.03. PIT KS gate: < 0.075.",
        "- **Market Comparison**: Shin's method for implied probability extraction (no-vig).",
        "  Log-loss (ignorance score) used per PenaltyBlog recommendation.",
        "- **Minutes Model**: HistGradientBoosting with role-aware uncertainty.",
        "- **Stat Models**: HGB rate models + hurdle models for sparse stats (stl, blk).",
        "",
    ]
    return "\n".join(lines)


@app.command()
def main(
    oof_scored: str = typer.Argument(..., help="Path to scored OOF parquet."),
    out_dir: str = typer.Option("artifacts/historical_review"),
    n_calibration_bins: int = typer.Option(10),
    docs_path: str = typer.Option("docs/HISTORICAL_REVIEW.md"),
) -> None:
    """Build the historical review package."""
    typer.echo(f"Loading OOF scored data: {oof_scored}")
    oof = pd.read_parquet(oof_scored)
    typer.echo(f"  Rows: {len(oof)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    cal_curves: dict = {}
    hit_tables: dict = {}

    for stat in STATS:
        stat_df = oof[oof["stat"] == stat] if "stat" in oof.columns else oof
        typer.echo(f"Processing {stat}: {len(stat_df)} rows")

        metrics = compute_stat_metrics(stat_df, stat)
        all_metrics.append(metrics)

        cal_curves[stat] = compute_calibration_curve(stat_df, n_calibration_bins)
        hit_tables[stat] = compute_hit_rate_table(stat_df, stat)

    # Write JSON outputs
    (out / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (out / "calibration_curves.json").write_text(json.dumps(cal_curves, indent=2))
    (out / "hit_rate_tables.json").write_text(json.dumps(hit_tables, indent=2))

    # Write full parquet
    review_path = out / "historical_review.parquet"
    oof.to_parquet(review_path, index=False)

    # Build markdown report
    report_md = build_markdown_report(all_metrics, cal_curves, hit_tables, out)
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_path).write_text(report_md)

    typer.echo(f"\nHistorical review outputs:")
    typer.echo(f"  Summary metrics → {out}/summary_metrics.json")
    typer.echo(f"  Calibration curves → {out}/calibration_curves.json")
    typer.echo(f"  Hit rate tables → {out}/hit_rate_tables.json")
    typer.echo(f"  Full results → {review_path}")
    typer.echo(f"  Report → {docs_path}")

    # Print summary to console
    typer.echo("\n=== Per-Stat Summary ===")
    for m in all_metrics:
        ign_str = f"{m['mean_ignorance_score']:.4f}" if m["mean_ignorance_score"] is not None else "N/A"
        delta_str = f"{m['ignorance_delta']:.4f}" if m["ignorance_delta"] is not None else "N/A"
        typer.echo(f"  {m['stat']:10s} MAE={m['mae'] or 'N/A':>7}  Ign={ign_str}  Δ={delta_str}")


if __name__ == "__main__":
    app()
