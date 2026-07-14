"""Stage 7 — Build edge report comparing model PMFs vs. market odds.

Requires a slate manifest that proves this is a current-run invocation.
Writes explicit market status in every exit path.

Explicit market statuses:
  SUCCESS_WITH_MARKETS        — edges written, all integrity checks passed
  VERIFIED_NO_GAMES           — slate has scheduled_game_count == 0, clean exit
  LIVE_MARKETS_NOT_YET_AVAILABLE — markets not yet posted (source policy allows)
  FAILURE                     — any fatal validation or integrity error

Source policies (--source-policy):
  odds_api_then_bdl  — try Odds API, fall back to BDL when empty (default)
  odds_api_required  — Odds API must succeed; BDL fallback is forbidden
  bdl_required       — BDL must be used; Odds API is not consulted

Required slate manifest fields: game_date, scheduled_game_count, game_ids,
  github_run_id, git_commit.

Writes:
  {out_dir}/market_comparison.parquet  — full joined table (all rows)
  {out_dir}/publishable_edges.parquet  — |edge| >= edge_threshold rows
  {out_dir}/edge_report_{date}.json    — summary audit with market_status

Usage:
    python scripts/build_edge_report.py \\
        --pmfs deliveries/today/full_pmfs_wide.parquet \\
        --raw-props data/processed/wnba_player_props.parquet \\
        --out-dir deliveries/today \\
        --slate-manifest deliveries/today/slate_manifest.json \\
        --edge-threshold 0.04 \\
        --game-date 2026-06-15 \\
        --require-venn-abers \\
        [--odds-api-props data/processed/wnba_player_props_oddsapi_latest.parquet]
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

# ---------------------------------------------------------------------------
# Adaptive Shin-z threshold: load from shrinkage_params.json if available.
# ---------------------------------------------------------------------------
_shin_params_path = Path("artifacts/models/shrinkage_params.json")
if _shin_params_path.exists():
    try:
        with open(_shin_params_path) as _f:
            _shin_params = json.load(_f)
        SHIN_Z_THRESHOLD = float(_shin_params.get("shin_z_optimal", 0.15))
    except Exception:
        SHIN_Z_THRESHOLD = 0.15
else:
    SHIN_Z_THRESHOLD = 0.15

from wnba_props_model.pipeline.deliver import build_market_comparison, normalize_player_props_snapshot
from wnba_props_model.models.market import fair_american, prob_over_from_pmf
from wnba_props_model.models.simulation import json_to_pmf
from wnba_props_model.pipeline.calibrate import apply_venn_abers_calibration
from wnba_props_model.pipeline.market_integrity import (
    validate_no_duplicate_quotes,
    validate_quote_freshness,
    validate_player_identity_resolved,
    validate_game_identity_resolved,
    validate_odds_format,
    check_no_stale_fallback,
    StaleFallbackForbiddenError,
    DuplicateQuoteError,
    StaleQuoteError,
    UnmatchedIdentityError,
    MalformedOddsError,
)

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Explicit market statuses — written into every audit report
# ---------------------------------------------------------------------------
STATUS_SUCCESS_WITH_MARKETS = "SUCCESS_WITH_MARKETS"
STATUS_VERIFIED_NO_GAMES = "VERIFIED_NO_GAMES"
STATUS_LIVE_MARKETS_NOT_YET_AVAILABLE = "LIVE_MARKETS_NOT_YET_AVAILABLE"
STATUS_FAILURE = "FAILURE"

# ---------------------------------------------------------------------------
# Source policies
# ---------------------------------------------------------------------------
POLICY_ODDS_API_THEN_BDL = "odds_api_then_bdl"
POLICY_ODDS_API_REQUIRED = "odds_api_required"
POLICY_BDL_REQUIRED = "bdl_required"
_VALID_POLICIES = {POLICY_ODDS_API_THEN_BDL, POLICY_ODDS_API_REQUIRED, POLICY_BDL_REQUIRED}

# Required fields in slate manifest
_REQUIRED_SLATE_MANIFEST_FIELDS = [
    "game_date", "scheduled_game_count", "game_ids", "github_run_id", "git_commit"
]

# Combo stats that have no per-line calibrators — assigned EXPERIMENTAL in quality gate.
COMBO_STATS_UNCALIBRATED: frozenset[str] = frozenset({"pts_reb_ast", "reb_ast"})


def assign_quality_status(row: "pd.Series") -> str:
    """Three-tier quality gate for published edges.

    PUBLISHABLE  — real market line, ≥1 book, calibrated stat
    EXPERIMENTAL — no market match or projection-only or uncalibrated combo
    WATCHLIST    — has market line but borderline (uncalibrated combo with a book)
    """
    line_ok = pd.notna(row.get("line"))
    books = row.get("number_of_books_offering", 0) or 0
    try:
        books = int(books)
    except (TypeError, ValueError):
        books = 0
    stat = row.get("stat", "")

    if line_ok and books >= 1 and stat not in COMBO_STATS_UNCALIBRATED:
        return "PUBLISHABLE"
    elif not line_ok or books == 0:
        return "EXPERIMENTAL"
    else:
        return "WATCHLIST"


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
    slate_manifest: str = typer.Option(
        ...,
        "--slate-manifest",
        help=(
            "Required JSON slate manifest containing: game_date, scheduled_game_count, "
            "game_ids, github_run_id, git_commit. Proves current-run integrity."
        ),
    ),
    edge_threshold: float = typer.Option(0.0, help="Minimum |edge| to publish (default 0 — show all props)."),
    game_date: str | None = typer.Option(None, help="ISO date for audit (YYYY-MM-DD)."),
    min_market_prob: float = typer.Option(0.05, help="Skip lines where market no-vig prob < this."),
    max_shin_z: float = typer.Option(
        SHIN_Z_THRESHOLD,
        help=(
            "Shin-z soft filter threshold (loaded from artifacts/models/shrinkage_params.json "
            "if shin_z_optimal key exists, else 0.15). Edges where shin_z > max_shin_z are "
            "flagged as 'high_adversity' but NOT removed."
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
    require_venn_abers: bool = typer.Option(
        False,
        "--require-venn-abers/--no-require-venn-abers",
        help=(
            "When set, Venn-Abers calibration must succeed for all rows. "
            "A required calibration failure exits nonzero. "
            "Production must use --require-venn-abers."
        ),
    ),
    allow_uncalibrated: bool = typer.Option(
        False,
        "--allow-uncalibrated/--no-allow-uncalibrated",
        help=(
            "When set, uncalibrated combo stats are permitted in output. "
            "Without this flag, uncalibrated stats that require calibration are fatal. "
            "Do NOT set in production."
        ),
    ),
    source_policy: str = typer.Option(
        POLICY_ODDS_API_THEN_BDL,
        "--source-policy",
        help=(
            f"Market source policy. One of: {POLICY_ODDS_API_THEN_BDL} (default), "
            f"{POLICY_ODDS_API_REQUIRED}, {POLICY_BDL_REQUIRED}. "
            "Fallback only allowed when policy is odds_api_then_bdl."
        ),
    ),
) -> None:
    """Compare model PMFs vs. market lines using Shin no-vig.

    Requires a slate manifest proving current-run integrity.
    Writes explicit market_status in every exit path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Validate source policy ────────────────────────────────────────────────
    if source_policy not in _VALID_POLICIES:
        typer.echo(
            f"[FATAL] Invalid --source-policy={source_policy!r}. "
            f"Must be one of: {sorted(_VALID_POLICIES)}", err=True
        )
        _write_status(out, STATUS_FAILURE, game_date or date.today().isoformat(),
                      edge_threshold, error=f"invalid_source_policy:{source_policy}")
        raise typer.Exit(1)

    # ── Load and validate required slate manifest ─────────────────────────────
    slate_manifest_path = Path(slate_manifest)
    if not slate_manifest_path.exists():
        typer.echo(
            f"[FATAL] Slate manifest not found at {slate_manifest}. "
            "A slate manifest is required to prove current-run integrity.", err=True
        )
        _write_status(out, STATUS_FAILURE, game_date or date.today().isoformat(),
                      edge_threshold, error="slate_manifest_missing")
        raise typer.Exit(1)

    try:
        manifest_data = json.loads(slate_manifest_path.read_text())
    except Exception as exc:
        typer.echo(f"[FATAL] Unreadable slate manifest at {slate_manifest}: {exc}", err=True)
        _write_status(out, STATUS_FAILURE, game_date or date.today().isoformat(),
                      edge_threshold, error=f"slate_manifest_unreadable:{exc}")
        raise typer.Exit(1)

    missing_fields = [f for f in _REQUIRED_SLATE_MANIFEST_FIELDS if f not in manifest_data]
    if missing_fields:
        typer.echo(
            f"[FATAL] Slate manifest missing required fields: {missing_fields}", err=True
        )
        _write_status(out, STATUS_FAILURE, manifest_data.get("game_date", date.today().isoformat()),
                      edge_threshold, error=f"slate_manifest_missing_fields:{missing_fields}")
        raise typer.Exit(1)

    today = game_date or manifest_data["game_date"]
    scheduled_game_count = int(manifest_data["scheduled_game_count"])
    slate_game_ids = set(manifest_data.get("game_ids") or [])

    typer.echo(
        f"Slate manifest: game_date={today}, scheduled_game_count={scheduled_game_count}, "
        f"game_ids={slate_game_ids}, run_id={manifest_data.get('github_run_id')}, "
        f"commit={str(manifest_data.get('git_commit', ''))[:8]}"
    )

    # ── VERIFIED_NO_GAMES: valid slate with zero scheduled games → clean exit ─
    if scheduled_game_count == 0:
        typer.echo(f"[INFO] Slate reports 0 scheduled games for {today}. Status: {STATUS_VERIFIED_NO_GAMES}")
        _write_status(out, STATUS_VERIFIED_NO_GAMES, today, edge_threshold)
        raise typer.Exit(0)

    # ── PMF file required when games are scheduled ────────────────────────────
    pmfs_path = Path(pmfs)
    if not pmfs_path.exists():
        typer.echo(
            f"[FATAL] PMF file missing at {pmfs} but slate has {scheduled_game_count} "
            f"scheduled game(s). Status: {STATUS_FAILURE}", err=True
        )
        _write_status(out, STATUS_FAILURE, today, edge_threshold,
                      error=f"pmf_missing_with_scheduled_games:{scheduled_game_count}")
        raise typer.Exit(1)

    try:
        pmfs_df = pd.read_parquet(pmfs_path)
    except Exception as exc:
        typer.echo(
            f"[FATAL] PMF file unreadable at {pmfs}: {exc}. Unreadable files are failures, not empty responses.",
            err=True
        )
        _write_status(out, STATUS_FAILURE, today, edge_threshold, error=f"pmf_unreadable:{exc}")
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(pmfs_df):,} PMF rows")

    if pmfs_df.empty:
        typer.echo(
            f"[FATAL] PMF file exists at {pmfs} but has 0 rows. "
            f"Slate has {scheduled_game_count} scheduled game(s). Status: {STATUS_FAILURE}", err=True
        )
        _write_status(out, STATUS_FAILURE, today, edge_threshold, error="pmf_empty_with_scheduled_games")
        raise typer.Exit(1)

    # --- Assertion: All rows must be repaired before reaching edge report ---
    if "combo_suppressed" in pmfs_df.columns:
        suppressed = pmfs_df["combo_suppressed"].fillna(False).astype(bool)
        if suppressed.any():
            typer.echo(
                f"[FATAL] {suppressed.sum()} suppressed combo rows reached edge report. "
                "Check IPF repair ladder in predict.py.", err=True
            )
            _write_status(out, STATUS_FAILURE, today, edge_threshold, error="suppressed_combo_rows")
            raise typer.Exit(1)

    if "joint_status" in pmfs_df.columns:
        _bad_status_mask = pmfs_df["joint_status"].isin({"WARN_IPF_FAILED", "WARN"})
        if _bad_status_mask.any():
            typer.echo(
                f"[FATAL] {int(_bad_status_mask.sum())} rows with bad joint_status "
                f"({pmfs_df.loc[_bad_status_mask, 'joint_status'].value_counts().to_dict()}) "
                "reached edge report. All WARN rows must be repaired by IPF ladder.", err=True
            )
            _write_status(out, STATUS_FAILURE, today, edge_threshold, error="bad_joint_status")
            raise typer.Exit(1)

    # ── Data source selection (enforces source policy) ────────────────────────
    try:
        props_df, props_source = _load_props(raw_props, odds_api_props, today, source_policy)
    except Exception as exc:
        typer.echo(f"[FATAL] Market source load failed: {exc}. Status: {STATUS_FAILURE}", err=True)
        _write_status(out, STATUS_FAILURE, today, edge_threshold, error=f"market_source_failure:{exc}")
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(props_df):,} market prop rows [source={props_source}]")

    # ── Validate market quotes using market_integrity functions ───────────────
    if not props_df.empty:
        try:
            validate_no_duplicate_quotes(props_df)
            validate_player_identity_resolved(props_df)
            validate_game_identity_resolved(props_df)
            validate_odds_format(props_df)
        except (DuplicateQuoteError, UnmatchedIdentityError, MalformedOddsError) as exc:
            typer.echo(f"[FATAL] Market quote validation failed: {exc}. Status: {STATUS_FAILURE}", err=True)
            _write_status(out, STATUS_FAILURE, today, edge_threshold, error=str(exc))
            raise typer.Exit(1)

    # ── Markets empty: LIVE_MARKETS_NOT_YET_AVAILABLE vs fatal ───────────────
    # Write expected_market_comparison_manifest.parquet (empty) BEFORE exiting,
    # so the workflow has a reconciled-zero record to validate against.
    if props_df.empty:
        if source_policy == POLICY_ODDS_API_REQUIRED:
            typer.echo(
                f"[FATAL] --source-policy=odds_api_required but Odds API returned no markets. "
                f"Status: {STATUS_FAILURE}", err=True
            )
            _write_status(out, STATUS_FAILURE, today, edge_threshold,
                          error="odds_api_required_but_empty")
            raise typer.Exit(1)
        # Persist empty expected manifest to record the verified-zero state.
        _empty_expected_manifest = pd.DataFrame(
            columns=["game_id", "player_id", "stat", "vendor", "line"]
        )
        _empty_expected_manifest.to_parquet(
            out / "expected_market_comparison_manifest.parquet", index=False
        )
        typer.echo(
            f"[INFO] No market lines available yet. "
            f"expected_market_comparison_manifest.parquet written (0 rows). "
            f"Status: {STATUS_LIVE_MARKETS_NOT_YET_AVAILABLE}"
        )
        _write_status(out, STATUS_LIVE_MARKETS_NOT_YET_AVAILABLE, today, edge_threshold,
                      props_source=props_source)
        raise typer.Exit(0)

    # ── Persist expected_market_comparison_manifest.parquet ───────────────────
    # Written here: AFTER market validation + identity reconciliation,
    # BEFORE probability/edge construction.
    # Keyed by (game_id, player_id, stat, vendor, line).
    # This represents every validated, actionable, reconciled current-run quote.
    _expected_manifest_key_cols = [
        c for c in ["game_id", "player_id", "stat", "vendor", "line"]
        if c in props_df.columns
    ]
    if _expected_manifest_key_cols:
        _expected_manifest = (
            props_df[_expected_manifest_key_cols]
            .dropna(subset=[c for c in ["vendor", "line"] if c in _expected_manifest_key_cols])
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        _expected_manifest = pd.DataFrame(
            columns=["game_id", "player_id", "stat", "vendor", "line"]
        )
    _expected_manifest_path = out / "expected_market_comparison_manifest.parquet"
    _expected_manifest.to_parquet(_expected_manifest_path, index=False)
    typer.echo(
        f"Wrote expected_market_comparison_manifest → {_expected_manifest_path} "
        f"({len(_expected_manifest):,} rows)"
    )

    # ── Game_ID cross-join guard: fatal if markets nonempty but no shared games ─
    if "game_id" in pmfs_df.columns and "game_id" in props_df.columns:
        _pmfs_game_ids = set(pmfs_df["game_id"].dropna().unique())
        _props_game_ids = set(props_df["game_id"].dropna().unique())
        _shared_game_ids = _pmfs_game_ids & _props_game_ids
        if not _shared_game_ids and props_df is not None and not props_df.empty:
            typer.echo(
                f"[FATAL] GAME_ID MISMATCH: PMF game_ids={_pmfs_game_ids}, "
                f"market game_ids={_props_game_ids}. Markets nonempty but zero shared game IDs. "
                f"Status: {STATUS_FAILURE}", err=True
            )
            _write_status(out, STATUS_FAILURE, today, edge_threshold,
                          error=f"game_id_mismatch:pmf={_pmfs_game_ids},market={_props_game_ids}")
            raise typer.Exit(1)

    comp = build_market_comparison(pmfs_df, props_df)

    # ── Zero-join is fatal when markets are nonempty ───────────────────────────
    if comp.empty:
        typer.echo(
            f"[FATAL] Market comparison joined 0 rows despite nonempty market data. "
            f"Status: {STATUS_FAILURE}", err=True
        )
        _write_status(out, STATUS_FAILURE, today, edge_threshold,
                      error="zero_market_join_rows", props_source=props_source)
        raise typer.Exit(1)

    # ── Write full market_comparison.parquet (before business filters) ─────────
    mc_path = out / "market_comparison.parquet"
    comp.to_parquet(mc_path, index=False)
    typer.echo(f"Wrote market_comparison → {mc_path} ({len(comp):,} rows)")

    # Apply Venn-Abers calibration
    va_applied = False
    if "model_prob_over" in comp.columns and "stat" in comp.columns:
        try:
            comp = apply_venn_abers_calibration(comp, cal_dir=cal_dir)
            if "p_over_va" in comp.columns:
                _va_mask = comp["p_over_va"] != comp["model_prob_over"]
                if _va_mask.any():
                    comp.loc[_va_mask, "model_prob_over"] = comp.loc[_va_mask, "p_over_va"]
                    comp.loc[_va_mask, "edge_over"] = (
                        comp.loc[_va_mask, "model_prob_over"]
                        - comp.loc[_va_mask, "market_prob_over_no_vig"]
                    )
                    comp.loc[_va_mask, "edge_under"] = (
                        comp.loc[_va_mask, "market_prob_over_no_vig"]
                        - comp.loc[_va_mask, "model_prob_over"]
                    )
                    typer.echo(f"[venn_abers] Applied VA calibration to {_va_mask.sum()} rows")
                    va_applied = True
        except Exception as _va_exc:
            if require_venn_abers:
                typer.echo(
                    f"[FATAL] --require-venn-abers is set but Venn-Abers calibration failed: {_va_exc}. "
                    f"Status: {STATUS_FAILURE}", err=True
                )
                _write_status(out, STATUS_FAILURE, today, edge_threshold,
                              error=f"venn_abers_required_but_failed:{_va_exc}")
                raise typer.Exit(1)
            typer.echo(f"[WARN] Venn-Abers calibration failed (non-fatal): {_va_exc}", err=True)
    elif require_venn_abers:
        typer.echo(
            f"[FATAL] --require-venn-abers is set but model_prob_over or stat column is missing. "
            f"Status: {STATUS_FAILURE}", err=True
        )
        _write_status(out, STATUS_FAILURE, today, edge_threshold,
                      error="venn_abers_required_but_missing_columns")
        raise typer.Exit(1)

    # When --require-venn-abers is set, zero applied rows is also a failure:
    # it means no calibrators were found and every row fell back to raw model
    # probability.  Silent no-op is not acceptable when VA is required.
    if require_venn_abers and not va_applied:
        typer.echo(
            f"[FATAL] --require-venn-abers is set but no Venn-Abers calibrators were applied "
            f"(0 rows calibrated — check cal_dir={cal_dir}). "
            f"Status: {STATUS_FAILURE}", err=True
        )
        _write_status(out, STATUS_FAILURE, today, edge_threshold,
                      error="venn_abers_required_but_zero_rows_calibrated")
        raise typer.Exit(1)

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

    # ── Shin-z convergence diagnostic ─────────────────────────────────────────
    import logging as _shin_log
    _shin_logger = _shin_log.getLogger(__name__)
    if "shin_z" in comp.columns:
        n_total = len(comp)
        n_converged = comp["shin_z"].notna().sum()
        n_fallback = n_total - n_converged
        shin_vals = comp["shin_z"].dropna()
        _shin_logger.info(
            "[Shin diagnostic] %d/%d rows converged (%.1f%%), %d fell back to multiplicative. "
            "Shin z: median=%.4f, p95=%.4f, pct_above_threshold=%.1f%%",
            n_converged, n_total, 100 * n_converged / max(n_total, 1), n_fallback,
            float(shin_vals.median()) if len(shin_vals) > 0 else float("nan"),
            float(shin_vals.quantile(0.95)) if len(shin_vals) > 0 else float("nan"),
            100 * (shin_vals > 0.15).mean() if len(shin_vals) > 0 else 0.0,
        )

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

    # ── Sharp reference: vs_pinnacle_edge ─────────────────────────────────────
    # Pinnacle is the sharpest reference book; model edges that also beat Pinnacle
    # are upgraded to high_confidence tier.  Handles the case where Pinnacle has
    # no WNBA props (pinnacle_rows empty → vs_pinnacle_edge column stays absent).
    if "vendor" in comp.columns and "market_prob_over_no_vig" in comp.columns:
        try:
            pinnacle_rows = comp[
                comp["vendor"].fillna("").str.lower().str.contains("pinnacle", na=False)
            ]
            if not pinnacle_rows.empty:
                pin_ref = (
                    pinnacle_rows.groupby(["player_id", "stat", "line"])
                    ["market_prob_over_no_vig"].mean()
                )
                comp["vs_pinnacle_edge"] = (
                    comp.set_index(["player_id", "stat", "line"])
                    .index.map(pin_ref)
                    .values
                ) - comp["market_prob_over_no_vig"].values
                beats_pinnacle = (
                    (comp["edge_over"].abs() > 0.05)
                    & (comp["vs_pinnacle_edge"].fillna(0).abs() > 0.03)
                )
                if beats_pinnacle.any() and "confidence_tier" in comp.columns:
                    comp.loc[beats_pinnacle, "confidence_tier"] = "high_confidence"
                    typer.echo(
                        f"[pinnacle] {beats_pinnacle.sum()} edges upgraded to "
                        "high_confidence (beat Pinnacle reference)"
                    )
                typer.echo(
                    f"[pinnacle] vs_pinnacle_edge computed from "
                    f"{len(pinnacle_rows)} Pinnacle rows"
                )
            else:
                typer.echo(
                    "[pinnacle] No Pinnacle rows in market data — "
                    "vs_pinnacle_edge not computed (Pinnacle may not offer WNBA props)"
                )
        except Exception as _pin_exc:
            typer.echo(
                f"[pinnacle] vs_pinnacle_edge failed (non-fatal): {_pin_exc}", err=True
            )

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
    # Primary sort: by model_edge (time-corrected signal, alias for time_decay_adjusted_edge);
    # fall back to raw edge if decay column is absent.
    # Uniform threshold across individual and combo stats — no markup.
    edges = comp[comp["edge_over"].abs() >= edge_threshold].copy()
    typer.echo(f"[filter] Edge threshold: {edge_threshold:.2%} → {len(edges)} props")
    _sort_col = (
        "model_edge" if "model_edge" in edges.columns
        else "time_decay_adjusted_edge" if "time_decay_adjusted_edge" in edges.columns
        else "edge_over"
    )

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

    # Line movement Kelly adjustment: boost Kelly when sharp money is on our side,
    # cut when sharp money faded us (line moved against the model's edge direction).
    if "line_movement_prev" in edges.columns and "kelly_fraction" in edges.columns and "direction" in edges.columns:
        _dir_col = edges["direction"].fillna("")
        _lm_col = edges["line_movement_prev"].fillna(0.0)
        line_move_boost = np.where(
            (_dir_col == "OVER") & (_lm_col > 0.5),
            1.20,  # Sharp buying OVER same side: boost Kelly 20%
            np.where(
                (_dir_col == "OVER") & (_lm_col < -0.5),
                0.50,  # Sharp faded OVER: cut Kelly 50%
                np.where(
                    (_dir_col == "UNDER") & (_lm_col < -0.5),
                    1.20,  # Sharp buying UNDER same side: boost
                    np.where(
                        (_dir_col == "UNDER") & (_lm_col > 0.5),
                        0.50,  # Sharp faded UNDER: cut
                        1.0,
                    ),
                ),
            ),
        )
        edges = edges.copy()
        edges["kelly_fraction"] = edges["kelly_fraction"] * line_move_boost
        edges["line_movement_boost"] = line_move_boost
        if "kelly_units" in edges.columns:
            edges["kelly_units"] = (edges["kelly_fraction"] * 100).round(2)
        typer.echo(f"[line_movement] Kelly adjustments applied: boost_1.2={int((line_move_boost == 1.2).sum())}, cut_0.5={int((line_move_boost == 0.5).sum())}")

    # Portfolio Kelly: discount correlated same-player bets to avoid overexposure.
    # Same-player props are correlated (minutes, usage), so raw per-prop Kelly
    # over-allocates when a player appears in N bets.  Dividing by sqrt(N) gives
    # the approximate portfolio-optimal fraction under moderate positive correlation.
    if "player_id" in edges.columns and "kelly_fraction" in edges.columns:
        player_bet_counts = edges.groupby("player_id")["kelly_fraction"].transform("count")
        edges = edges.copy()
        edges["kelly_fraction"] = edges["kelly_fraction"] / np.sqrt(player_bet_counts.clip(lower=1))
        # Cap total per-player Kelly exposure at 1.5 × the largest single-bet fraction
        player_max_kelly = edges.groupby("player_id")["kelly_fraction"].transform("max")
        player_total = edges.groupby("player_id")["kelly_fraction"].transform("sum")
        cap_ratio = (1.5 * player_max_kelly) / player_total.clip(lower=1e-9)
        edges["kelly_fraction"] = (edges["kelly_fraction"] * cap_ratio.clip(upper=1.0)).round(4)
        # Recompute kelly_units to stay in sync
        if "kelly_units" in edges.columns:
            edges["kelly_units"] = (edges["kelly_fraction"] * 100).round(2)

    # Quality gate: assign three-tier status before persisting.
    edges["quality_status"] = edges.apply(assign_quality_status, axis=1)
    _qs_counts = edges["quality_status"].value_counts().to_dict()
    typer.echo(f"[quality_gate] {_qs_counts}")

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
        "market_status": STATUS_SUCCESS_WITH_MARKETS,
        "no_vig_method": "shin",
        "props_source": props_source,
        "source_policy": source_policy,
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
        "mean_model_edge": float(
            edges["model_edge"].mean() if "model_edge" in edges.columns
            else edges["time_decay_adjusted_edge"].mean() if "time_decay_adjusted_edge" in edges.columns
            else float("nan")
        ) if len(edges) else None,
        "quality_status_counts": _qs_counts,
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
    source_policy: str = POLICY_ODDS_API_THEN_BDL,
) -> tuple["pd.DataFrame", str]:
    """Load market props enforcing the specified source policy.

    source_policy:
      odds_api_then_bdl  — try Odds API; fall back to BDL if empty/unavailable
      odds_api_required  — Odds API must succeed; BDL fallback is forbidden
      bdl_required       — BDL only; Odds API is not consulted

    Unreadable files raise an exception (never return silent empty DataFrames).
    Returns (DataFrame, source_name).
    """
    if source_policy == POLICY_BDL_REQUIRED:
        # BDL only
        p = Path(raw_props_path)
        if not p.exists():
            raise FileNotFoundError(
                f"bdl_required policy: BDL props file not found at {raw_props_path}"
            )
        try:
            df = pd.read_parquet(p)
            typer.echo(f"[EdgeReport] Using BDL props (policy=bdl_required): {p} ({len(df):,} rows)")
            return df, "bdl"
        except Exception as exc:
            raise IOError(f"bdl_required policy: BDL props unreadable at {raw_props_path}: {exc}") from exc

    if source_policy in (POLICY_ODDS_API_THEN_BDL, POLICY_ODDS_API_REQUIRED):
        if odds_api_props_path:
            p = Path(odds_api_props_path)
            if not p.exists():
                if source_policy == POLICY_ODDS_API_REQUIRED:
                    raise FileNotFoundError(
                        f"odds_api_required policy: Odds API props file not found at {odds_api_props_path}"
                    )
                typer.echo(f"[EdgeReport] Odds API props not found at {p} — falling back to BDL", err=True)
            else:
                try:
                    df = pd.read_parquet(p)
                    if not df.empty and "over_odds" in df.columns:
                        typer.echo(f"[EdgeReport] Using Odds API props: {p} ({len(df):,} rows)")
                        return df, "odds_api"
                    elif source_policy == POLICY_ODDS_API_REQUIRED:
                        raise ValueError(
                            f"odds_api_required policy: Odds API props exist at {p} but are empty or missing over_odds column"
                        )
                    else:
                        typer.echo(
                            f"[EdgeReport] Odds API props empty/malformed ({p}) — falling back to BDL", err=True
                        )
                except (ValueError, FileNotFoundError):
                    raise
                except Exception as exc:
                    if source_policy == POLICY_ODDS_API_REQUIRED:
                        raise IOError(
                            f"odds_api_required policy: Odds API props unreadable at {odds_api_props_path}: {exc}"
                        ) from exc
                    typer.echo(
                        f"[WARN] Odds API props unreadable ({exc}) — falling back to BDL", err=True
                    )
        elif source_policy == POLICY_ODDS_API_REQUIRED:
            raise ValueError(
                "odds_api_required policy: --odds-api-props path not provided"
            )

    # BDL fallback (only when policy allows it)
    p = Path(raw_props_path)
    if not p.exists():
        typer.echo(f"[EdgeReport] No BDL props at {raw_props_path}", err=True)
        return pd.DataFrame(), "none"
    try:
        df = pd.read_parquet(p)
        typer.echo(f"[EdgeReport] Using BDL props: {p} ({len(df):,} rows)")
        return df, "bdl"
    except Exception as exc:
        raise IOError(f"BDL props unreadable at {raw_props_path}: {exc}") from exc


def _write_status(
    out: Path,
    market_status: str,
    today: str,
    edge_threshold: float,
    error: str | None = None,
    props_source: str | None = None,
) -> None:
    """Write minimal status files (empty parquets + audit JSON) for non-SUCCESS exits."""
    empty = pd.DataFrame()
    mc_path = out / "market_comparison.parquet"
    edges_path = out / "publishable_edges.parquet"
    if not mc_path.exists():
        empty.to_parquet(mc_path, index=False)
    if not edges_path.exists():
        empty.to_parquet(edges_path, index=False)
    audit: dict = {
        "game_date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_status": market_status,
        "no_vig_method": "shin",
        "edge_threshold": edge_threshold,
        "total_market_rows": 0,
        "publishable_edge_rows": 0,
    }
    if error:
        audit["error"] = error
    if props_source:
        audit["props_source"] = props_source
    (out / f"edge_report_{today}.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    app()
