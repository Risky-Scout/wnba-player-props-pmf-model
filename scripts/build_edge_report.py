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


def _get_publishable_stats(cal_dir: str | Path | None) -> frozenset[str]:
    """Dynamically determine publishable stats based on calibration bias corrections.

    STL/BLK are re-enabled only when their multiplicative bias correction factor is >= 0.85,
    meaning the raw model over-bias is <= 15% — correctable by the existing calibration.
    """
    # Combo stats are always publishable: their PMFs are derived from calibrated
    # component distributions via convolution, not from raw model outputs.
    # Correlation adjustments (bivariate_pmf.py) are applied at prediction time.
    # Note: "stocks" (stl+blk) is a combo and is always-on; individual "stl" and
    # "blk" edges remain gated on their bias correction factor.
    _base = {
        "pts", "reb", "ast", "fg3m",
        "pts_reb", "pts_ast", "reb_ast", "pts_reb_ast",
        "stocks",
    }
    _conditional = {"stl": 0.85, "blk": 0.85}

    if cal_dir is None:
        return frozenset(_base)

    bc_path = Path(cal_dir) / "bias_corrections.json"
    if not bc_path.exists():
        return frozenset(_base)

    try:
        bc = json.loads(bc_path.read_text())
        for stat, min_factor in _conditional.items():
            factor = float(bc.get(stat, 0.0))
            if factor >= min_factor:
                _base.add(stat)
                print(f"[edge_report] {stat} re-enabled: bias_correction={factor:.3f} >= {min_factor}")
            else:
                print(f"[edge_report] {stat} suppressed: bias_correction={factor:.3f} < {min_factor}")
    except Exception as exc:
        print(f"[edge_report] Could not read bias_corrections.json: {exc}")

    return frozenset(_base)


@app.command()
def main(
    pmfs: str = typer.Option(..., help="Calibrated PMF parquet (full_pmfs_wide.parquet)."),
    raw_props: str = typer.Option(..., help="BDL player props parquet (fallback)."),
    out_dir: str = typer.Option(..., help="Output directory for edge report files."),
    edge_threshold: float = typer.Option(0.0, help="Minimum |edge| to publish (default 0 — show all props)."),
    game_date: str | None = typer.Option(None, help="ISO date for audit (YYYY-MM-DD)."),
    min_market_prob: float = typer.Option(0.05, help="Skip lines where market no-vig prob < this."),
    max_shin_z: float = typer.Option(
        0.15,
        help=(
            "Shin-z soft filter threshold. Edges where shin_z > max_shin_z are flagged as "
            "'high_adversity' (sharp market, higher adverse-selection risk) but NOT removed. "
            "Lower z = softer market = better for retail bettor. Default 0.15."
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
    cal_dir: str = typer.Option(
        "artifacts/models/calibration",
        "--cal-dir",
        help="Directory containing calibration artifacts (bias_corrections.json).",
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

    # Game_ID cross-join guard: fail loudly if projections and market props are from different slates
    if not pmfs_df.empty and not props_df.empty:
        if "game_id" in pmfs_df.columns and "game_id" in props_df.columns:
            _pmfs_game_ids = set(pmfs_df["game_id"].dropna().unique())
            _props_game_ids = set(props_df["game_id"].dropna().unique())
            _shared_game_ids = _pmfs_game_ids & _props_game_ids
            if not _shared_game_ids:
                typer.echo(f"[WARN] GAME_ID MISMATCH: PMF game_ids={_pmfs_game_ids}, market game_ids={_props_game_ids}. No shared games — edge report will be empty.", err=True)

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

    # Stats eligible for published edges — dynamically determined from calibration bias gate.
    # STL/BLK are re-enabled when bias_corrections.json shows factor >= 0.85 (< 15% raw over-bias).
    PUBLISHABLE_STATS = _get_publishable_stats(cal_dir)
    if "stat" in comp.columns:
        comp = comp[comp["stat"].isin(PUBLISHABLE_STATS)].copy()
        typer.echo(f"[filter] Filtered to publishable stats {PUBLISHABLE_STATS}: {len(comp):,} rows remain")

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
    # Position-blind inflated predictions (e.g. guards predicted to rebound
    # like centers) produce model_market_ratio >> 1.  The original thresholds
    # of [0.50, 2.50] were too permissive — ratios up to 2.50 were merely
    # "caution" flagged and still published.  Tightened per stat:
    #   REB/AST: suppress above 1.65 (guards rarely produce 65%+ more than line)
    #   PTS/FG3M/combos: suppress above 1.80 (higher natural variance)
    #   Low side: suppress below 0.55 (model < 55% of market is implausible)
    #
    # Sanity tiers:
    #   0 = clean  — ratio inside stat-specific tight band
    #   1 = caution — ratio in stat-specific medium band (show, warn)
    #   2 = suppressed — ratio outside stat-specific extreme band (drop)
    _STAT_EXTREME_HIGH = {"reb": 1.65, "ast": 1.65, "pts": 1.80, "fg3m": 1.80}
    _STAT_EXTREME_LOW  = {"reb": 0.55, "ast": 0.55, "pts": 0.55, "fg3m": 0.50}
    _DEFAULT_EXTREME_HIGH = 1.80
    _DEFAULT_EXTREME_LOW  = 0.55
    _CAUTION_HIGH = 1.40
    _CAUTION_LOW  = 0.65
    if "pmf_mean" in comp.columns and "market_implied_mean" in comp.columns:
        _ratio = comp["pmf_mean"] / comp["market_implied_mean"].replace(0, np.nan)
        comp["model_market_ratio"] = _ratio.round(4)
        # Per-stat extreme thresholds (fall back to default for combo stats)
        _stat_col = comp.get("stat", pd.Series([""] * len(comp), index=comp.index))
        _hi = _stat_col.map(lambda s: _STAT_EXTREME_HIGH.get(str(s), _DEFAULT_EXTREME_HIGH))
        _lo = _stat_col.map(lambda s: _STAT_EXTREME_LOW.get(str(s), _DEFAULT_EXTREME_LOW))
        _extreme_mask = (
            (_ratio < _lo) | (_ratio > _hi)
        ) & comp["market_implied_mean"].notna()
        _caution_mask = (
            ~_extreme_mask
            & ((_ratio < _CAUTION_LOW) | (_ratio > _CAUTION_HIGH))
            & comp["market_implied_mean"].notna()
        )
        comp["projection_sanity_flag"] = np.where(
            _extreme_mask, 2,
            np.where(_caution_mask, 1, 0)
        ).astype(int)
        _n_before = len(comp)
        _n_extreme = int(_extreme_mask.sum())
        _n_caution = int(_caution_mask.sum())
        typer.echo(
            f"[sanity-guard] {_n_extreme}/{_n_before} rows suppressed "
            f"(stat-specific ratio bounds); {_n_caution} rows flagged as caution"
        )
        comp = comp[~_extreme_mask].copy()
    else:
        comp["model_market_ratio"] = np.nan
        comp["projection_sanity_flag"] = 0

    # Publishable edges: |edge| >= threshold; exclude sanity_flag=2 (already gone),
    # keep sanity_flag=1 (caution) but sort them to the bottom.
    # Primary sort: by CLV-decay-adjusted edge (time-corrected signal);
    # fall back to raw edge if decay column is absent.
    # Uniform threshold across individual and combo stats — no markup.
    edges = comp[comp["edge_over"].abs() >= edge_threshold].copy()
    typer.echo(f"[filter] Edge threshold: {edge_threshold:.2%} → {len(edges)} props")
    _sort_col = "clv_decay_adjusted_edge" if "clv_decay_adjusted_edge" in edges.columns else "edge_over"

    # Combo props (pts_reb, pts_ast, etc.) are offered by fewer sportsbooks
    # than individual props — requiring 2 books would silently eliminate nearly
    # all combo edges. Apply a 1-book minimum for combos, 2-book for individuals.
    _COMBO_STATS = frozenset({"pts_reb", "pts_ast", "reb_ast", "pts_reb_ast", "stocks"})
    _MIN_BOOKS_INDIVIDUAL = 2
    _MIN_BOOKS_COMBO = 1
    if "number_of_books_offering" in edges.columns and "stat" in edges.columns:
        _pre_book_n = len(edges)
        _is_combo = edges["stat"].isin(_COMBO_STATS)
        _min_books_arr = _is_combo.map({True: _MIN_BOOKS_COMBO, False: _MIN_BOOKS_INDIVIDUAL})
        # When number_of_books_offering is null (Odds API doesn't always populate it),
        # treat it as "unknown" and pass through rather than defaulting to 1 book which
        # would incorrectly eliminate every individual prop from publication.
        _known_books = edges["number_of_books_offering"].notna()
        _book_mask = (
            ~_known_books  # null = unknown = pass through
            | (edges["number_of_books_offering"].fillna(1).astype(int) >= _min_books_arr)
        )
        edges = edges[_book_mask].copy()
        typer.echo(
            f"[filter] Book consensus (individual>={_MIN_BOOKS_INDIVIDUAL}, "
            f"combo>={_MIN_BOOKS_COMBO}): {_pre_book_n} → {len(edges)} edges"
        )

    # Direction-contradiction filter: suppress edges where model direction
    # contradicts significant line movement (steam)
    if "line_movement_direction" in edges.columns and "line_movement_magnitude" in edges.columns:
        _steam_mag_threshold = 0.5
        _contradiction_mask = (
            ((edges["edge_over"] > 0) & (edges["line_movement_direction"] == -1) &
             (edges["line_movement_magnitude"].fillna(0) > _steam_mag_threshold)) |
            ((edges["edge_over"] < 0) & (edges["line_movement_direction"] == 1) &
             (edges["line_movement_magnitude"].fillna(0) > _steam_mag_threshold))
        )
        _n_suppressed_steam = int(_contradiction_mask.sum())
        if _n_suppressed_steam > 0:
            edges = edges[~_contradiction_mask].copy()
            typer.echo(f"[filter] Suppressed {_n_suppressed_steam} edges contradicting line steam")
    if "projection_sanity_flag" in edges.columns:
        edges = edges.sort_values(
            ["projection_sanity_flag", _sort_col],
            key=lambda s: s.abs() if s.name == _sort_col else s,
            ascending=[True, False],
        )
    else:
        edges = edges.sort_values(_sort_col, key=np.abs, ascending=False)
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
    _caution_edges = int((edges.get("projection_sanity_flag", pd.Series(0, dtype=int)) == 1).sum()) if "projection_sanity_flag" in edges.columns else 0
    _clean_edges   = len(edges) - _caution_edges
    audit = {
        "game_date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "no_vig_method": "shin",
        "props_source": props_source,
        "edge_threshold": edge_threshold,
        "max_shin_z_threshold": max_shin_z,
        "total_market_rows": len(comp),
        "publishable_edge_rows": len(edges),
        "clean_edge_rows": _clean_edges,
        "caution_edge_rows": _caution_edges,
        "standard_edge_rows": standard_edges,
        "high_adversity_edge_rows": high_adv_edges,
        "stats_with_edges": sorted(edges["stat"].unique().tolist()) if len(edges) else [],
        "mean_abs_edge": float(edges["edge_over"].abs().mean()) if len(edges) else None,
        "max_edge": float(edges["edge_over"].abs().max()) if len(edges) else None,
        "over_edges": int((edges["edge_over"] > 0).sum()),
        "under_edges": int((edges["edge_over"] < 0).sum()),
        "deep_link_coverage_pct": deep_link_pct,
        "mean_kelly_fraction": float(edges["kelly_fraction"].mean()) if "kelly_fraction" in edges.columns and len(edges) else None,
        "max_kelly_fraction": float(edges["kelly_fraction"].max()) if "kelly_fraction" in edges.columns and len(edges) else None,
        "mean_clv_decay_edge": float(edges["clv_decay_adjusted_edge"].mean()) if "clv_decay_adjusted_edge" in edges.columns and len(edges) else None,
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
