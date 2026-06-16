"""Export daily betting sheets for NoVig, Kalshi, and Polymarket.

Reads publishable_edges.parquet (|edge| >= 4pp) and formats platform-specific
CSV outputs ready for copy-paste or programmatic submission.

Outputs:
  {out_dir}/betting_sheet_{date}.csv      — full NoVig-ready sheet
  {out_dir}/kalshi_sheet_{date}.csv       — Kalshi binary format
  {out_dir}/polymarket_sheet_{date}.csv   — Polymarket binary format
  {out_dir}/betting_summary_{date}.json   — machine-readable audit

Column schema (betting_sheet):
  player_name, stat, line, model_prob_over, market_prob_over_shin_novig,
  edge, edge_pct, fair_over_american, fair_under_american,
  recommendation, confidence, is_calibrated, role_bucket, game_date

Usage:
    python scripts/export_betting_sheet.py \\
        --edges deliveries/today/publishable_edges.parquet \\
        --out-dir deliveries/today \\
        --game-date 2026-06-15
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.models.market import fair_american

app = typer.Typer(add_completion=False)

# Confidence tiers based on |edge|
_CONFIDENCE_TIERS = [
    (0.10, "HIGH"),
    (0.07, "MEDIUM"),
    (0.04, "LOW"),
]


def _confidence(edge_abs: float) -> str:
    for threshold, label in _CONFIDENCE_TIERS:
        if edge_abs >= threshold:
            return label
    return "MARGINAL"


def _recommendation(edge: float) -> str:
    """OVER if model_prob > market_prob, UNDER otherwise."""
    return "OVER" if edge > 0 else "UNDER"


def _kalshi_contract(player_name: str, stat: str, line: float, direction: str, game_date: str) -> str:
    """Generate a Kalshi-style contract name."""
    line_str = str(line).replace(".5", ".5").replace(".0", "")
    return f"WNBA-{game_date}-{player_name.replace(' ', '_').upper()}-{stat.upper()}-{direction}-{line_str}"


@app.command()
def main(
    edges: str = typer.Option(..., help="publishable_edges.parquet from build_edge_report.py."),
    out_dir: str = typer.Option(..., help="Output directory for betting sheet CSVs."),
    game_date: str | None = typer.Option(None, help="ISO date (YYYY-MM-DD)."),
    min_edge: float = typer.Option(0.04, help="Minimum |edge| to include (4pp default)."),
    top_n: int | None = typer.Option(None, help="Limit to top N picks by |edge|. None = all."),
) -> None:
    """Format publishable edges into NoVig, Kalshi, and Polymarket betting sheets."""
    today = game_date or date.today().isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    edges_path = Path(edges)
    if not edges_path.exists() or edges_path.stat().st_size == 0:
        typer.echo(f"[WARN] No publishable edges found at {edges_path}")
        _write_empty(out, today)
        return

    df = pd.read_parquet(edges_path)
    if df.empty:
        typer.echo("[WARN] publishable_edges.parquet is empty — no bets today")
        _write_empty(out, today)
        return

    df = df[df["edge_over"].abs() >= min_edge].copy()
    if df.empty:
        typer.echo(f"[WARN] No edges meet minimum threshold of {min_edge:.2%}")
        _write_empty(out, today)
        return

    df["edge_abs"] = df["edge_over"].abs()
    df = df.sort_values("edge_abs", ascending=False)
    if top_n:
        df = df.head(top_n)

    # Build NoVig sheet
    sheet = pd.DataFrame({
        "player_name": df.get("player_name", df.get("player_id", "")),
        "stat": df["stat"],
        "line": df["line"],
        "model_prob_over": df["model_prob_over"].round(4),
        "market_prob_over_shin_novig": df["market_prob_over_no_vig"].round(4),
        "edge": df["edge_over"].round(4),
        "edge_pct": (df["edge_over"] * 100).round(2).astype(str) + "%",
        "fair_over_american": df["model_prob_over"].map(fair_american).round(0).astype(int),
        "fair_under_american": (1 - df["model_prob_over"]).map(fair_american).round(0).astype(int),
        "recommendation": df["edge_over"].map(_recommendation),
        "confidence": df["edge_abs"].map(_confidence),
        "is_calibrated": df.get("is_calibrated", False),
        "role_bucket": df.get("role_bucket", "unknown"),
        "game_date": today,
    })

    novig_path = out / f"betting_sheet_{today}.csv"
    sheet.to_csv(novig_path, index=False)
    typer.echo(f"Wrote NoVig betting sheet → {novig_path} ({len(sheet)} picks)")

    # Kalshi format
    kalshi = pd.DataFrame({
        "contract": [
            _kalshi_contract(
                row["player_name"] if "player_name" in row else str(row.get("player_id", "PLAYER")),
                row["stat"],
                row["line"],
                _recommendation(row["edge_over"]),
                today,
            )
            for _, row in df.iterrows()
        ],
        "direction": df["edge_over"].map(_recommendation).values,
        "line": df["line"].values,
        "model_yes_prob": df.apply(
            lambda r: float(r["model_prob_over"]) if r["edge_over"] > 0 else float(1 - r["model_prob_over"]),
            axis=1,
        ).round(4).values,
        "market_yes_prob": df.apply(
            lambda r: float(r["market_prob_over_no_vig"]) if r["edge_over"] > 0 else float(1 - r["market_prob_over_no_vig"]),
            axis=1,
        ).round(4).values,
        "edge": df["edge_over"].abs().round(4).values,
        "confidence": df["edge_abs"].map(_confidence).values,
        "game_date": today,
    })

    kalshi_path = out / f"kalshi_sheet_{today}.csv"
    kalshi.to_csv(kalshi_path, index=False)
    typer.echo(f"Wrote Kalshi sheet → {kalshi_path} ({len(kalshi)} picks)")

    # Polymarket format (same schema as Kalshi)
    poly = kalshi.copy()
    poly_path = out / f"polymarket_sheet_{today}.csv"
    poly.to_csv(poly_path, index=False)
    typer.echo(f"Wrote Polymarket sheet → {poly_path} ({len(poly)} picks)")

    # Summary JSON
    conf_counts = sheet["confidence"].value_counts().to_dict()
    rec_counts = sheet["recommendation"].value_counts().to_dict()
    stats_in = sorted(sheet["stat"].unique().tolist())

    summary = {
        "game_date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_picks": len(sheet),
        "confidence": conf_counts,
        "recommendation": rec_counts,
        "stats_covered": stats_in,
        "mean_edge": float(sheet["edge"].abs().mean()),
        "max_edge": float(sheet["edge"].abs().max()),
        "min_edge": float(sheet["edge"].abs().min()),
        "high_confidence_picks": int(conf_counts.get("HIGH", 0)),
        "is_calibrated_pct": float(sheet["is_calibrated"].mean()) if "is_calibrated" in sheet else None,
    }
    summary_path = out / f"betting_summary_{today}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"Wrote summary → {summary_path}")

    # Print readable output
    typer.echo(f"\n{'='*50}")
    typer.echo(f"  WNBA BETTING SHEET — {today}")
    typer.echo(f"{'='*50}")
    typer.echo(f"  Total picks: {len(sheet)}")
    for conf in ("HIGH", "MEDIUM", "LOW"):
        n = conf_counts.get(conf, 0)
        if n:
            typer.echo(f"  {conf}: {n}")
    typer.echo(f"  OVER: {rec_counts.get('OVER', 0)} | UNDER: {rec_counts.get('UNDER', 0)}")
    typer.echo(f"  Mean edge: {summary['mean_edge']:.2%}")
    typer.echo(f"{'='*50}\n")

    # Print top picks
    typer.echo("Top picks by |edge|:")
    for _, r in sheet.head(10).iterrows():
        typer.echo(
            f"  {r['recommendation']} {r.get('player_name','?')} {r['stat']} {r['line']} "
            f"(model={r['model_prob_over']:.3f}, mkt={r['market_prob_over_shin_novig']:.3f}, "
            f"edge={r['edge']:+.3f}, {r['confidence']})"
        )


def _write_empty(out: Path, today: str) -> None:
    pd.DataFrame().to_csv(out / f"betting_sheet_{today}.csv", index=False)
    pd.DataFrame().to_csv(out / f"kalshi_sheet_{today}.csv", index=False)
    pd.DataFrame().to_csv(out / f"polymarket_sheet_{today}.csv", index=False)
    summary = {
        "game_date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_picks": 0,
        "note": "no_publishable_edges",
    }
    (out / f"betting_summary_{today}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
