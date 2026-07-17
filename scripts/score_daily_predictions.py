"""Post-game CLV and accuracy scorer.

Runs after games complete. Pulls actual BDL player stats, joins against
pre-game model predictions, and computes:
  - NLL (Negative Log-Likelihood) per PMF
  - RPS (Ranked Probability Score)
  - Binary Log Loss (Ignorance Score) for over/under market lines
  - model_edge_open = model_prob - OPEN market no-vig prob (OUTCOME-INDEPENDENT; NOT CLV)
  - model_close_edge = model PMF prob at CLOSING line (selected side) - closing no-vig prob
  - price_clv / line_clv = market movement for the selected side between open and close
  - hit_result / realized_profit = settled outcome (kept DISTINCT from the edge/CLV fields)

NOTE: A model-versus-market probability difference is a MODEL EDGE, not CLV. CLV is
market movement (price/line) and is known at close, independent of the game result.
None of the edge/CLV fields here are multiplied by the outcome.

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

_IS_DELTA_FLAG_THRESHOLD = 0.0  # IS delta > 0 → market is better → flag for improvement


def _p_over_cond(pmf, line: float) -> float:
    """Model P(over | non-push) at a discrete ``line``.

    Over = strictly greater than the line; the winning region starts at
    floor(line)+1 (Over 10 and Over 10.5 both mean 11+). For an INTEGER line the
    exact-line mass is a PUSH and is removed from the denominator, so the result
    is conditional on a non-push (matching the pushes-excluded binary target).
    """
    import math as _math
    arr = np.asarray(pmf, dtype=float)
    if arr.size == 0:
        return float("nan")
    cutoff = _math.floor(line) + 1
    p_over = float(arr[cutoff:].sum()) if cutoff < len(arr) else 0.0
    if float(line).is_integer():
        li = int(round(line))
        p_push = float(arr[li]) if 0 <= li < len(arr) else 0.0
    else:
        p_push = 0.0
    denom = 1.0 - p_push
    return float(min(max(p_over / denom, 0.0), 1.0)) if denom > 1e-9 else float(p_over)


def _print_rolling_is_summary(combined: pd.DataFrame, window_days: int = 30) -> None:
    """Print 30-day rolling Ignorance Score delta vs market (Shin and Power methods).

    IS delta = model_ignorance_score - market_ignorance_score
    Negative delta = model better (lower IS = sharper forecast).
    Positive delta = market better (flag for improvement).
    Reports which vig-removal method (Shin vs Power) produces better IS delta.
    """
    req_cols = {"game_date", "stat", "model_ignorance_score", "market_ignorance_score"}
    if not req_cols.issubset(combined.columns):
        return  # not enough data yet (market comparison not run)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
    recent = combined.copy()
    if "game_date" in recent.columns:
        try:
            recent["game_date_dt"] = pd.to_datetime(recent["game_date"], utc=True, errors="coerce")
            recent = recent[recent["game_date_dt"] >= cutoff]
        except Exception:
            pass

    valid = recent[recent["model_ignorance_score"].notna() & recent["market_ignorance_score"].notna()]
    if valid.empty:
        return

    typer.echo(f"\n{'='*65}")
    typer.echo(f"Model vs Market — Last {window_days} Days Ignorance Score Delta")
    typer.echo(f"{'='*65}")
    typer.echo(f"  (negative IS delta = model better; positive = market better)")
    typer.echo()

    stat_results = []
    for stat, grp in valid.groupby("stat"):
        model_is = float(grp["model_ignorance_score"].mean())
        market_is_shin = float(grp["market_ignorance_score"].mean())
        delta_shin = model_is - market_is_shin
        n = len(grp)
        flag = "  ← flag" if delta_shin > _IS_DELTA_FLAG_THRESHOLD else ""
        direction = "model better" if delta_shin < 0 else "market better"
        typer.echo(
            f"  {stat:<12} IS delta (Shin)  = {delta_shin:+.4f}  ({direction})   n={n}{flag}"
        )

        # Power method IS delta (if market_prob_over_power column exists)
        if "market_prob_over_power" in grp.columns:
            valid_p = grp[grp["market_prob_over_power"].notna() & grp["actual_outcome"].notna()]
            if len(valid_p) >= 5:
                from wnba_props_model.models.market import ignorance_score_binary  # noqa: PLC0415
                power_is = float(np.mean([
                    ignorance_score_binary(float(row["market_prob_over_power"]), int(row["actual_outcome"] > row["line"]))
                    for _, row in valid_p.iterrows()
                    if pd.notna(row.get("line"))
                ]))
                delta_power = model_is - power_is
                better_method = "Shin" if delta_shin < delta_power else "Power"
                typer.echo(
                    f"  {stat:<12} IS delta (Power) = {delta_power:+.4f}  (best method: {better_method})"
                )
                stat_results.append({"stat": stat, "delta_shin": delta_shin, "delta_power": delta_power})
            else:
                stat_results.append({"stat": stat, "delta_shin": delta_shin, "delta_power": None})
        else:
            stat_results.append({"stat": stat, "delta_shin": delta_shin, "delta_power": None})

    overall_delta = float(valid["model_ignorance_score"].mean()) - float(valid["market_ignorance_score"].mean())
    typer.echo()
    typer.echo(f"  Overall IS delta (Shin) = {overall_delta:+.4f}   n={len(valid):,}")

    # Summary: which method wins most stats
    has_power = [r for r in stat_results if r["delta_power"] is not None]
    if has_power:
        shin_wins = sum(1 for r in has_power if r["delta_shin"] <= r["delta_power"])
        power_wins = len(has_power) - shin_wins
        typer.echo(f"  Vig method wins: Shin={shin_wins}/{len(has_power)}, Power={power_wins}/{len(has_power)}")

    typer.echo(f"{'='*65}\n")

    # P4.1: IS delta segmentation by line movement bucket
    if "line_moved_toward_over" in valid.columns and "line_moved_toward_under" in valid.columns:
        typer.echo(f"{'='*65}")
        typer.echo("IS Delta by Line Movement (P4.1)")
        typer.echo(f"{'='*65}")
        typer.echo("  stale = line unchanged  |  steam_over = line moved toward over  |  steam_under = moved toward under")
        typer.echo()

        def _bucket_movement(row: pd.Series) -> str:
            toward_over  = row.get("line_moved_toward_over")
            toward_under = row.get("line_moved_toward_under")
            if toward_over is True:
                return "steam_over"
            if toward_under is True:
                return "steam_under"
            return "stale"

        valid["_line_bucket"] = valid.apply(_bucket_movement, axis=1)
        for bucket, bgrp in valid.groupby("_line_bucket"):
            b_model = float(bgrp["model_ignorance_score"].mean())
            b_market = float(bgrp["market_ignorance_score"].mean())
            b_delta = b_model - b_market
            direction = "model better" if b_delta < 0 else "market better"
            typer.echo(
                f"  {bucket:<14} IS delta = {b_delta:+.4f}  ({direction})   n={len(bgrp):,}"
            )
        typer.echo(f"{'='*65}\n")


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
            # Include vendor and shin_z so per-book CLV analysis works downstream.
            _optional_mkt_cols = ["vendor", "shin_z"]
            _mkt_cols = ["game_id", "player_id", "stat", "line",
                         "market_prob_over_no_vig", "model_prob_over"] + \
                        [c for c in _optional_mkt_cols if c in mkt.columns]
            mkt_sub = mkt[_mkt_cols].copy()
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
            # Model edge vs OPEN market (OUTCOME-INDEPENDENT; this is NOT CLV).
            # It is the model's probability minus the open no-vig market probability
            # for the OVER; it is never multiplied by the game result.
            joined.loc[valid, "model_edge_open"] = (
                joined.loc[valid, "model_prob_over"] - joined.loc[valid, "market_prob_over_no_vig"]
            )

    # Closing-line edge + CLV (OUTCOME-INDEPENDENT). Requires a CANONICAL closing
    # table (game_id, player_id, stat, market_prob_over_no_vig, line). All of these
    # are known at close and NONE are multiplied by the game result.
    if closing_lines and Path(closing_lines).exists():
        try:
            cl_df = pd.read_parquet(closing_lines)
            required_cl = {"game_id", "player_id", "stat", "market_prob_over_no_vig", "line"}
            if cl_df.empty or not required_cl.issubset(cl_df.columns):
                missing = required_cl - set(cl_df.columns)
                typer.echo(
                    f"[WARN] Closing table unusable (empty={cl_df.empty}, missing={missing}). "
                    "Skipping closing-line edge. It must be canonicalized to "
                    "game_id/player_id/stat/market_prob_over_no_vig/line before scoring.",
                    err=True,
                )
            else:
                cl_df = cl_df.rename(columns={
                    "market_prob_over_no_vig": "closing_prob_over_no_vig",
                    "line": "closing_line",
                })
                cl_sub = cl_df[["game_id", "player_id", "stat",
                                "closing_prob_over_no_vig", "closing_line"]].dropna(
                    subset=["game_id", "player_id", "stat"])
                # Fail-closed on many-to-many: closing table must be 1 row per key.
                if cl_sub.duplicated(subset=["game_id", "player_id", "stat"]).any():
                    raise ValueError(
                        "Closing lines are not unique per (game_id, player_id, stat) — "
                        "refusing many-to-many join (would inflate sample size)."
                    )
                for k in ["game_id", "player_id", "stat"]:
                    joined[k] = joined[k].astype("string")
                    cl_sub[k] = cl_sub[k].astype("string")
                joined = joined.merge(cl_sub, on=["game_id", "player_id", "stat"], how="left")

                cl_valid = joined["closing_prob_over_no_vig"].notna() & joined["closing_line"].notna()
                n_done = 0
                for i in joined.index[cl_valid]:
                    row = joined.loc[i]
                    try:
                        pmf = normalize_pmf(json_to_pmf(row["pmf_json"]))
                    except Exception:
                        continue
                    cl_line = float(row["closing_line"])
                    m_over_close = _p_over_cond(pmf, cl_line)
                    close_p_over = float(row["closing_prob_over_no_vig"])
                    # Selected side = the side the model recommends (from open edge; fall
                    # back to model P(over) vs 0.5). Determined WITHOUT the outcome.
                    me = row.get("model_edge_open")
                    if pd.notna(me):
                        side = "over" if float(me) > 0 else "under"
                    else:
                        side = "over" if float(row.get("model_prob_over", 0.5)) >= 0.5 else "under"
                    if side == "over":
                        m_side, close_side = m_over_close, close_p_over
                    else:
                        m_side, close_side = 1.0 - m_over_close, 1.0 - close_p_over
                    joined.loc[i, "selected_side"] = side
                    # model_close_edge: model PMF prob for the selected side at the
                    # closing line MINUS the closing no-vig prob for that side.
                    joined.loc[i, "model_close_edge"] = m_side - close_side
                    # price_clv: closing no-vig prob for the selected side minus the OPEN
                    # no-vig prob for that side (favourable price movement toward your side).
                    open_p_over = row.get("market_prob_over_no_vig")
                    if pd.notna(open_p_over):
                        open_side = float(open_p_over) if side == "over" else 1.0 - float(open_p_over)
                        joined.loc[i, "price_clv"] = close_side - open_side
                    # line_clv: quoted-line movement favourable to the selected side.
                    open_line = row.get("line")
                    if pd.notna(open_line):
                        ol = float(open_line)
                        joined.loc[i, "line_clv"] = (ol - cl_line) if side == "over" else (cl_line - ol)
                    n_done += 1
                if n_done:
                    _mce = joined.loc[cl_valid, "model_close_edge"].dropna()
                    typer.echo(
                        f"[score] Closing-line edge computed for {n_done} rows "
                        f"(mean model_close_edge={_mce.mean():+.4f}, outcome-independent)"
                    )
        except Exception as exc:
            typer.echo(f"[WARN] Closing-line edge computation failed: {exc}", err=True)

    joined["game_date"] = today
    joined["scored_at"] = datetime.now(timezone.utc).isoformat()

    # Save today's scored rows (named by game date for easy lookup)
    today_path = out / f"scored_{today}.parquet"
    drop_cols = [c for c in ("pmf_json",) if c in joined.columns]
    joined.drop(columns=drop_cols).to_parquet(today_path, index=False)
    typer.echo(f"Wrote today's scored rows → {today_path} ({len(joined):,} rows)")

    # Append to cumulative results (no pmf_json — keeps file small)
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

    # ── 30-day rolling Ignorance Score delta summary ───────────────────────
    _print_rolling_is_summary(combined, window_days=30)

    # Write rolling drift-window file WITH pmf_json so check_calibration_drift.py
    # can compute ECE from raw PMF arrays (not just aggregate scores).
    # Keeps last 300 rows per stat to bound file size.
    if "pmf_json" in joined.columns:
        _DRIFT_WINDOW = 300
        drift_frames = []
        for _stat, _grp in joined.groupby("stat"):
            drift_frames.append(_grp.tail(_DRIFT_WINDOW))
        drift_df = pd.concat(drift_frames, ignore_index=True) if drift_frames else joined
        drift_path = results_path.parent / "drift_window.parquet"
        # Merge with prior drift window to maintain rolling history
        if drift_path.exists():
            try:
                prior_drift = pd.read_parquet(drift_path)
                prior_drift = prior_drift[prior_drift["game_date"] != today]
                drift_df = pd.concat([prior_drift, drift_df], ignore_index=True)
                # Re-trim after merge
                trimmed = []
                for _stat, _grp in drift_df.groupby("stat"):
                    trimmed.append(_grp.tail(_DRIFT_WINDOW))
                drift_df = pd.concat(trimmed, ignore_index=True) if trimmed else drift_df
            except Exception:
                pass
        drift_df.to_parquet(drift_path, index=False)
        typer.echo(f"Updated drift window → {drift_path} ({len(drift_df):,} rows with pmf_json)")


# ---------------------------------------------------------------------------
# Item 11: CLV closing-line comparison utility functions
# ---------------------------------------------------------------------------

def compute_closing_line_value(
    predictions_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Closing Line Value (CLV) for each prediction.

    CLV = model_price - closing_price
    A positive CLV means the model identified value before the market adjusted.

    Targets:
      - CLV% (fraction of bets with positive CLV) > 60%
      - Mean CLV > 2pp (percentage points)

    Args:
        predictions_df: DataFrame with model_p_over, bet_direction, game_id, player_id, prop_type
        closing_odds_df: DataFrame from /wnba/v1/odds with latest odds (closing odds)

    Returns:
        DataFrame with added: closing_p_over, model_close_edge,
        model_close_edge_positive, model_close_edge_pp (all outcome-independent)
    """
    # Get closing odds (latest update before game start per player/prop)
    if "updated_at" in closing_odds_df.columns:
        closing = (
            closing_odds_df
            .sort_values("updated_at")
            .groupby(["game_id", "player_id", "prop_type"])
            .last()
            .reset_index()
        )
    else:
        closing = closing_odds_df.copy()

    # Merge closing odds onto predictions
    merge_cols = ["game_id", "player_id", "prop_type"]
    available_merge = [c for c in merge_cols if c in predictions_df.columns and c in closing.columns]
    if not available_merge:
        return predictions_df

    odds_cols = [c for c in ["over_odds", "under_odds"] if c in closing.columns]
    merged = predictions_df.merge(
        closing[available_merge + odds_cols],
        on=available_merge,
        how="left",
        suffixes=("", "_closing"),
    )

    # Compute closing implied probability (American odds → prob)
    def _american_to_prob(odds_series: pd.Series) -> pd.Series:
        def _convert(odds):
            try:
                o = int(float(odds))
                if o > 0:
                    return 100.0 / (100.0 + o)
                elif o < 0:
                    return abs(o) / (100.0 + abs(o))
            except (TypeError, ValueError):
                pass
            return float("nan")
        return odds_series.apply(_convert)

    over_col = "over_odds_closing" if "over_odds_closing" in merged.columns else "over_odds"
    under_col = "under_odds_closing" if "under_odds_closing" in merged.columns else "under_odds"

    if over_col in merged.columns and under_col in merged.columns:
        imp_over = _american_to_prob(merged[over_col])
        imp_under = _american_to_prob(merged[under_col])
        total = (imp_over + imp_under).clip(lower=1e-6)
        merged["closing_p_over"] = imp_over / total
    else:
        merged["closing_p_over"] = float("nan")

    # model_close_edge = model - closing no-vig prob (model edge vs the close).
    # This is a MODEL-VS-MARKET probability difference — it is NOT CLV, and it is
    # outcome-independent (never multiplied by the result).
    if "model_p_over" in merged.columns:
        merged["model_close_edge"] = merged["model_p_over"] - merged["closing_p_over"]
        merged["model_close_edge_positive"] = (merged["model_close_edge"] > 0).astype(int)
        merged["model_close_edge_pp"] = merged["model_close_edge"] * 100
    else:
        merged["model_close_edge"] = float("nan")
        merged["model_close_edge_positive"] = 0
        merged["model_close_edge_pp"] = float("nan")

    return merged


def generate_clv_report(results_parquet: str, lookback: int = 100) -> dict:
    """Generate a rolling MODEL-CLOSE-EDGE report (outcome-independent).

    Reports the model's edge vs. the closing market (model_close_edge). This is a
    model-vs-market probability difference, NOT CLV. It is not multiplied by the
    game result. Economic CLV (price_clv/line_clv) and realized P&L are reported
    separately (P3).
    """
    try:
        df = pd.read_parquet(results_parquet)
    except Exception as exc:
        return {"error": str(exc)}

    recent = df.tail(lookback)
    if recent.empty:
        return {"n_bets": 0, "model_close_edge_pct": 0.0, "mean_model_close_edge_pp": 0.0}

    edge_col = "model_close_edge" if "model_close_edge" in recent.columns else None
    if edge_col is None:
        return {"n_bets": len(recent), "model_close_edge_pct": float("nan"),
                "mean_model_close_edge_pp": float("nan")}

    edge_valid = recent[edge_col].dropna()
    report = {
        "n_bets": len(recent),
        "n_with_edge": len(edge_valid),
        "model_close_edge_pct": float((edge_valid > 0).mean() * 100),
        "mean_model_close_edge_pp": float(edge_valid.mean() * 100),
        "median_model_close_edge_pp": float(edge_valid.median() * 100),
    }
    if "stat" in recent.columns:
        report["model_close_edge_by_stat"] = (
            recent.groupby("stat")[edge_col].mean().multiply(100).round(2).to_dict()
        )
    if "role_bucket" in recent.columns:
        report["model_close_edge_by_role"] = (
            recent.groupby("role_bucket")[edge_col].mean().multiply(100).round(2).to_dict()
        )
    return report


if __name__ == "__main__":
    app()
