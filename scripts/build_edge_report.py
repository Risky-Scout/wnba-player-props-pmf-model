"""Stage 7 — Build edge report comparing model PMFs vs. market odds.

Data source priority (highest fidelity first):
  1. The Odds API v4  (ODDS_API_KEY set) — multi-book, deep links, Shin-z
  2. BDL player props (fallback)

Uses Shin's no-vig method to extract true market probabilities, then computes
model edge for each player prop line. Deep links to bookmaker betslips are
included when available via The Odds API.

Writes:
  {out_dir}/market_comparison.parquet  — full joined table
  {out_dir}/publishable_edges.parquet  — |edge| >= edge_threshold rows
  {out_dir}/edge_report_{date}.json    — summary audit

Usage:
    python scripts/build_edge_report.py \\
        --pmfs deliveries/today/full_pmfs_wide.parquet \\
        --raw-props data/processed/wnba_player_props.parquet \\
        --out-dir deliveries/today \\
        --edge-threshold 0.04 \\
        --game-date 2026-06-15 \\
        [--odds-api-props data/processed/wnba_player_props_oddsapi_latest.parquet]
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.pipeline.deliver import build_market_comparison, normalize_player_props_snapshot
from wnba_props_model.models.market import fair_american, prob_over_from_pmf
from wnba_props_model.models.simulation import json_to_pmf

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pmfs: str = typer.Option(..., help="Calibrated PMF parquet (full_pmfs_wide.parquet)."),
    raw_props: str = typer.Option(..., help="BDL player props parquet (fallback)."),
    out_dir: str = typer.Option(..., help="Output directory for edge report files."),
    edge_threshold: float = typer.Option(0.04, help="Minimum |edge| to publish (default 4pp)."),
    game_date: str | None = typer.Option(None, help="ISO date for audit (YYYY-MM-DD)."),
    min_market_prob: float = typer.Option(0.05, help="Skip lines where market no-vig prob < this."),
    max_shin_z: float = typer.Option(
        0.06,
        help=(
            "Shin-z soft filter threshold. Edges where shin_z > max_shin_z are flagged as "
            "'high_adversity' (sharp market, higher adverse-selection risk) but NOT removed. "
            "Lower z = softer market = better for retail bettor. Default 0.06."
        ),
    ),
    odds_api_props: str = typer.Option(
        "",
        "--odds-api-props",
        help=(
            "Path to Odds API props parquet (wnba_player_props_oddsapi_latest.parquet). "
            "When supplied and non-empty, Odds API data is PREFERRED over BDL; "
            "deep link columns are added to publishable_edges."
        ),
    ),
) -> None:
    """Compare model PMFs vs. market lines using Shin no-vig.

    Data source priority: Odds API (preferred) → BDL (fallback).
    Edges are tiered by Shin-z (market sharpness proxy):
    - shin_z <= max_shin_z  → confidence_tier = 'standard'   (softer market)
    - shin_z > max_shin_z   → confidence_tier = 'high_adversity' (sharp market — flagged)
    - shin_z is None/NaN    → confidence_tier = 'unknown'
    """
    today = game_date or date.today().isoformat()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Exit cleanly when no predictions were generated (no games scheduled).
    pmfs_path = Path(pmfs)
    if not pmfs_path.exists():
        typer.echo(f"[INFO] No PMF file found at {pmfs} — no games scheduled for {today}. Exiting cleanly.")
        raise typer.Exit(0)

    pmfs_df = pd.read_parquet(pmfs)
    typer.echo(f"Loaded {len(pmfs_df):,} PMF rows")

    # ── Data source selection ─────────────────────────────────────────────────
    props_df, props_source = _load_props(raw_props, odds_api_props, today)
    typer.echo(f"Loaded {len(props_df):,} market prop rows [source={props_source}]")

    if props_df.empty:
        typer.echo("[WARN] Props file is empty — no market lines to compare")
        _write_empty(out, today, edge_threshold)
        return

    comp = build_market_comparison(pmfs_df, props_df)

    if comp.empty:
        typer.echo("[WARN] No player/game overlap between PMFs and props")
        _write_empty(out, today, edge_threshold)
        return

    # Filter to sensible market lines (avoid very thin edge markets)
    comp = comp[comp["market_prob_over_no_vig"].notna()]
    comp = comp[comp["market_prob_over_no_vig"] >= min_market_prob]
    comp = comp[comp["market_prob_over_no_vig"] <= (1.0 - min_market_prob)]

    # Annotate with calibration status
    for col in ("is_calibrated", "cal_source", "model_version"):
        if col not in comp.columns and col in pmfs_df.columns:
            merge_col = pmfs_df[["player_id", "game_id", "stat", col]].drop_duplicates()
            comp = comp.merge(merge_col, on=["player_id", "game_id", "stat"], how="left")

    # ── Shin-z tiering ────────────────────────────────────────────────────────
    # Low shin_z = soft market (recreational / low-limit book).
    # High shin_z = sharp market (lots of informed money, higher adverse selection).
    # Edges in sharp markets are NOT removed but are flagged for lower-confidence
    # interpretation by downstream consumers.
    if "shin_z" in comp.columns:
        def _tier(z):
            if pd.isna(z):
                return "unknown"
            return "standard" if float(z) <= max_shin_z else "high_adversity"
        comp["confidence_tier"] = comp["shin_z"].map(_tier)
    else:
        comp["confidence_tier"] = "unknown"

    comp_path = out / "market_comparison.parquet"
    comp.to_parquet(comp_path, index=False)
    typer.echo(f"Wrote market_comparison → {comp_path} ({len(comp):,} rows)")

    # Carry Odds API deep links into the comparison table
    if props_source == "odds_api" and "deep_link" in props_df.columns:
        # link_cols must NOT include any column that is also in link_keys
        # (bookmaker is both a join key and was mistakenly in link_cols, causing
        # duplicate column names in Arrow/parquet).
        link_keys = [c for c in ["player_name", "stat", "line", "bookmaker"] if c in props_df.columns]
        link_cols = [c for c in ["deep_link", "event_link", "market_link",
                                  "outcome_link_over"] if c in props_df.columns]
        if link_keys and link_cols:
            all_link_cols = list(dict.fromkeys(link_keys + link_cols))  # dedup preserving order
            links_df = props_df[all_link_cols].drop_duplicates(subset=link_keys)
            merge_keys = [c for c in link_keys if c in comp.columns]
            if merge_keys:
                comp = comp.merge(links_df[merge_keys + link_cols], on=merge_keys, how="left")

    # ── Extreme model-vs-market disagreement guard ─────────────────────────
    # When the model's pmf_mean is < 35% OR > 300% of the market_implied_mean
    # the pick is almost certainly from a grossly mis-predicted player (e.g.,
    # a recent hot/cold streak not yet in rolling features).  These produce
    # artificial 50-cent "edges" that have zero predictive value and completely
    # overwhelm the betting sheet.  Flag them and exclude from publishable picks.
    _EXTREME_LOW_RATIO  = 0.35   # model_mean < 35% of market → suppressed
    _EXTREME_HIGH_RATIO = 3.00   # model_mean > 300% of market → suppressed
    if "pmf_mean" in comp.columns and "market_implied_mean" in comp.columns:
        _ratio = comp["pmf_mean"] / comp["market_implied_mean"].replace(0, np.nan)
        comp["model_market_ratio"] = _ratio.round(4)
        # #region agent log
        _n_before = len(comp)
        _extreme_mask = (
            (_ratio < _EXTREME_LOW_RATIO) | (_ratio > _EXTREME_HIGH_RATIO)
        ) & comp["market_implied_mean"].notna()
        _n_extreme = int(_extreme_mask.sum())
        typer.echo(
            f"[extreme-guard] {_n_extreme}/{_n_before} rows removed "
            f"(model/market ratio outside [{_EXTREME_LOW_RATIO:.0%}, {_EXTREME_HIGH_RATIO:.0%}])"
        )
        comp = comp[~_extreme_mask].copy()
        # #endregion
    else:
        comp["model_market_ratio"] = np.nan

    # Publishable edges: |edge| >= threshold on either side (all tiers included)
    edges = comp[comp["edge_over"].abs() >= edge_threshold].copy()
    edges = edges.sort_values("edge_over", key=np.abs, ascending=False)
    edges_path = out / "publishable_edges.parquet"
    edges.to_parquet(edges_path, index=False)

    standard_edges = int((edges.get("confidence_tier", pd.Series(dtype=str)) == "standard").sum()) if "confidence_tier" in edges.columns else len(edges)
    high_adv_edges = int((edges.get("confidence_tier", pd.Series(dtype=str)) == "high_adversity").sum()) if "confidence_tier" in edges.columns else 0

    typer.echo(
        f"Wrote publishable_edges → {edges_path} ({len(edges):,} rows at |edge| >= {edge_threshold:.2%}) "
        f"[{standard_edges} standard | {high_adv_edges} high_adversity (shin_z>{max_shin_z})]"
    )

    # Audit JSON
    deep_link_pct = (
        float((edges["deep_link"].notna() & (edges["deep_link"] != "")).mean())
        if "deep_link" in edges.columns and len(edges) else None
    )
    audit = {
        "game_date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "no_vig_method": "shin",
        "props_source": props_source,
        "edge_threshold": edge_threshold,
        "max_shin_z_threshold": max_shin_z,
        "total_market_rows": len(comp),
        "publishable_edge_rows": len(edges),
        "standard_edge_rows": standard_edges,
        "high_adversity_edge_rows": high_adv_edges,
        "stats_with_edges": sorted(edges["stat"].unique().tolist()) if len(edges) else [],
        "mean_abs_edge": float(edges["edge_over"].abs().mean()) if len(edges) else None,
        "max_edge": float(edges["edge_over"].abs().max()) if len(edges) else None,
        "over_edges": int((edges["edge_over"] > 0).sum()),
        "under_edges": int((edges["edge_over"] < 0).sum()),
        "deep_link_coverage_pct": deep_link_pct,
    }
    audit_path = out / f"edge_report_{today}.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    typer.echo(f"Wrote edge audit → {audit_path}")
    typer.echo(
        f"\nSummary: {len(edges)} publishable edges "
        f"({audit['over_edges']} OVER / {audit['under_edges']} UNDER)"
    )


def _load_props(
    raw_props_path: str,
    odds_api_props_path: str,
    today: str,
) -> tuple["pd.DataFrame", str]:
    """Load market props with source priority: Odds API > BDL.

    Returns (DataFrame, source_name) where source_name is 'odds_api' or 'bdl'.
    """
    # Try Odds API first
    if odds_api_props_path:
        p = Path(odds_api_props_path)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if not df.empty and "over_odds" in df.columns:
                    typer.echo(f"[EdgeReport] Using Odds API props: {p} ({len(df):,} rows)")
                    return df, "odds_api"
            except Exception as exc:
                typer.echo(f"[WARN] Odds API props unreadable ({exc}) — falling back to BDL", err=True)

    # Fall back to BDL
    p = Path(raw_props_path)
    if not p.exists():
        typer.echo(f"[WARN] No props at {raw_props_path}", err=True)
        return pd.DataFrame(), "none"
    try:
        df = pd.read_parquet(p)
        return df, "bdl"
    except Exception as exc:
        typer.echo(f"[WARN] BDL props unreadable: {exc}", err=True)
        return pd.DataFrame(), "none"


def _write_empty(out: Path, today: str, edge_threshold: float) -> None:
    empty = pd.DataFrame()
    empty.to_parquet(out / "market_comparison.parquet", index=False)
    empty.to_parquet(out / "publishable_edges.parquet", index=False)
    audit = {
        "game_date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "no_vig_method": "shin",
        "edge_threshold": edge_threshold,
        "total_market_rows": 0,
        "publishable_edge_rows": 0,
        "note": "no_props_data",
    }
    (out / f"edge_report_{today}.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    app()
