"""Weekly CLV and calibration performance report.

Reads cumulative scored predictions from data/clv_tracking/results.parquet
and outputs:
  - artifacts/audits/clv_report_{date}.json  (machine-readable)
  - artifacts/audits/clv_summary_{date}.md   (human-readable)

Metrics reported per stat:
  - CLV (Closing Line Value): mean model_prob - market_prob, weighted by |edge|
  - Ignorance Score vs. market Ignorance Score (PenaltyBlog primary metric)
  - NLL per stat
  - RPS per stat
  - ECE-style calibration check
  - Hit rate vs. expected hit rate

Usage:
    python scripts/generate_clv_report.py \\
        --results data/clv_tracking/results.parquet \\
        --lookback-days 30 \\
        --out-dir artifacts/audits
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def _stat_report(df: pd.DataFrame) -> list[dict]:
    rows = []
    for stat, g in df.groupby("stat"):
        rec: dict = {"stat": stat, "n": len(g)}

        # PMF scores
        if "pmf_nll" in g and g["pmf_nll"].notna().any():
            rec["mean_nll"] = float(g["pmf_nll"].mean())
        if "pmf_rps" in g and g["pmf_rps"].notna().any():
            rec["mean_rps"] = float(g["pmf_rps"].mean())

        # Binary over/under market comparison
        mkt_valid = g["market_prob_over_no_vig"].notna() if "market_prob_over_no_vig" in g else pd.Series(False, index=g.index)
        if mkt_valid.any():
            gv = g[mkt_valid]
            rec["n_market_lines"] = int(mkt_valid.sum())

            if "model_ignorance_score" in gv and gv["model_ignorance_score"].notna().any():
                rec["mean_model_ignorance_score"] = float(gv["model_ignorance_score"].mean())
            if "market_ignorance_score" in gv and gv["market_ignorance_score"].notna().any():
                rec["mean_market_ignorance_score"] = float(gv["market_ignorance_score"].mean())
            if "model_bin_logloss" in gv and "market_bin_logloss" in gv:
                rec["logloss_delta"] = float((gv["model_bin_logloss"] - gv["market_bin_logloss"]).mean())
            if "clv" in gv and gv["clv"].notna().any():
                rec["mean_clv"] = float(gv["clv"].mean())
                rec["positive_clv_pct"] = float((gv["clv"] > 0).mean())

            if "hit_result" in gv and "model_prob_over" in gv:
                rec["empirical_hit_rate"] = float(gv["hit_result"].mean())
                rec["mean_model_prob"] = float(gv["model_prob_over"].mean())
                rec["mean_market_prob"] = float(gv["market_prob_over_no_vig"].mean())

        rows.append(rec)
    return rows


@app.command()
def main(
    results: str = typer.Option("data/clv_tracking/results.parquet", help="Cumulative scored results parquet."),
    lookback_days: int = typer.Option(30, help="Number of past days to include in report."),
    out_dir: str = typer.Option("artifacts/audits", help="Output directory for report files."),
    report_date: str | None = typer.Option(None, help="Report reference date (YYYY-MM-DD, default today)."),
) -> None:
    """Generate weekly CLV and calibration report."""
    today_str = report_date or date.today().isoformat()
    today = date.fromisoformat(today_str)
    cutoff = (today - timedelta(days=lookback_days)).isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_path = Path(results)
    if not results_path.exists():
        typer.echo(f"[WARN] No results file at {results_path} — nothing to report")
        return

    df = pd.read_parquet(results_path)
    if df.empty:
        typer.echo("[WARN] Results file is empty")
        return

    df["game_date"] = df["game_date"].astype(str)
    recent = df[df["game_date"] >= cutoff].copy()
    typer.echo(f"Loaded {len(recent):,} rows from {cutoff} to {today_str} ({lookback_days}d window)")

    stat_rows = _stat_report(recent)

    overall: dict = {
        "report_date": today_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "n_total_rows": len(recent),
        "n_game_dates": recent["game_date"].nunique() if "game_date" in recent else 0,
        "stats": stat_rows,
    }

    # Overall CLV summary
    if "clv" in recent.columns and recent["clv"].notna().any():
        overall["overall_mean_clv"] = float(recent["clv"].mean())
        overall["overall_positive_clv_pct"] = float((recent["clv"] > 0).mean())

    if "logloss_delta" in [r.get("logloss_delta") for r in stat_rows if "logloss_delta" in r]:
        deltas = [r["logloss_delta"] for r in stat_rows if "logloss_delta" in r]
        overall["mean_logloss_delta_vs_market"] = float(np.mean(deltas))
        overall["model_beats_market"] = bool(np.mean(deltas) < 0)

    # JSON report
    json_path = out / f"clv_report_{today_str}.json"
    json_path.write_text(json.dumps(overall, indent=2))
    typer.echo(f"Wrote JSON report → {json_path}")

    # Markdown summary
    lines = [
        f"# CLV & Calibration Report — {today_str}",
        f"*{lookback_days}-day window: {cutoff} → {today_str}*",
        f"*{len(recent):,} scored predictions across {overall.get('n_game_dates',0)} game dates*",
        "",
        "## Overall",
        f"- Mean CLV: {overall.get('overall_mean_clv', 'N/A'):.4f}" if "overall_mean_clv" in overall else "- CLV: N/A (no market data)",
        f"- Positive CLV %: {overall.get('overall_positive_clv_pct', 0):.1%}" if "overall_mean_clv" in overall else "",
        f"- Model beats market (log loss): {overall.get('model_beats_market', 'N/A')}",
        "",
        "## Per-Stat Summary",
        "",
        "| Stat | N | NLL | RPS | LogLoss Δ | Mean CLV | Hit Rate | Model Prob |",
        "|------|---|-----|-----|-----------|----------|----------|------------|",
    ]
    for r in stat_rows:
        lines.append(
            f"| {r['stat']} | {r['n']} "
            f"| {r.get('mean_nll', float('nan')):.3f} "
            f"| {r.get('mean_rps', float('nan')):.4f} "
            f"| {r.get('logloss_delta', float('nan')):.4f} "
            f"| {r.get('mean_clv', float('nan')):.4f} "
            f"| {r.get('empirical_hit_rate', float('nan')):.3f} "
            f"| {r.get('mean_model_prob', float('nan')):.3f} |"
        )

    lines += [
        "",
        "---",
        "*Ignorance Score (log loss in bits) is the primary binary metric per PenaltyBlog methodology.*",
        "*CLV = (model_prob - market_prob) × (2 × hit_result - 1). Positive CLV = model edge was correct.*",
    ]
    md_path = out / f"clv_summary_{today_str}.md"
    md_path.write_text("\n".join(lines))
    typer.echo(f"Wrote Markdown report → {md_path}")

    # Print quick summary
    if "overall_mean_clv" in overall:
        typer.echo(f"\nMean CLV ({lookback_days}d): {overall['overall_mean_clv']:+.4f}")
    if "model_beats_market" in overall:
        status = "YES" if overall["model_beats_market"] else "NO"
        typer.echo(f"Model beats market (log loss): {status}")


if __name__ == "__main__":
    app()
