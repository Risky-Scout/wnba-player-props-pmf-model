"""Champion vs Challenger PMF evaluator.

Filters both OOF prediction sets to the untouched evaluation period,
matches on exact keys (game_id, player_id, stat), and computes
proper scoring rules per stat and role.

Writes promotion_decision.json to the output directory.

Usage:
    python scripts/compare_champion_challenger.py \\
        --champion-oof artifacts/models/calibration/oof_predictions.parquet \\
        --challenger-oof data/oof/challenger_v1/oof_player_stat_pmfs.parquet \\
        --eval-start 2026-06-26 \\
        --eval-end   2026-07-13 \\
        --out-dir    deliveries/staging/challenger_<run_id>/evaluation

Exit codes:
    0 — evaluation completed and promotion_decision.json written
    1 — required evidence absent or fatal error
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_champion_challenger")

app = typer.Typer(add_completion=False)

# ── Scoring helpers ────────────────────────────────────────────────────────

def _parse_pmf(pmf_json: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        d = json.loads(pmf_json)
        k = np.array([int(x) for x in d.keys()], dtype=float)
        p = np.array(list(d.values()), dtype=float)
        s = p.sum()
        if s > 0:
            p = p / s
        return k, p
    except Exception:
        return None


def _pmf_log_score(pmf_json: str, outcome: int) -> float:
    r = _parse_pmf(pmf_json)
    if r is None:
        return float("nan")
    k, p = r
    idx = np.where(k == float(outcome))[0]
    if len(idx) == 0:
        return float("nan")
    prob = float(p[idx[0]])
    return math.log(max(prob, 1e-12))


def _pmf_rps(pmf_json: str, outcome: int) -> float:
    """Ranked probability score (CRPS for discrete distributions)."""
    r = _parse_pmf(pmf_json)
    if r is None:
        return float("nan")
    k, p = r
    cdf = np.cumsum(p)
    heaviside = (k >= float(outcome)).astype(float)
    return float(np.sum((cdf - heaviside) ** 2))


def _pmf_brier(pmf_json: str, outcome: int, line: float) -> float | None:
    """Binary Brier score at the given line (P(over) vs 1_actual_over)."""
    r = _parse_pmf(pmf_json)
    if r is None or math.isnan(line) or line <= 0:
        return None
    k, p = r
    p_over = float(p[k > float(line)].sum())
    y = float(outcome > line)
    return (p_over - y) ** 2


def _calibration_slope_intercept(p_hat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Linear regression of empirical outcome on predicted probability."""
    if len(p_hat) < 5:
        return float("nan"), float("nan")
    x = np.column_stack([np.ones(len(p_hat)), p_hat])
    try:
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return float(coef[1]), float(coef[0])  # slope, intercept
    except Exception:
        return float("nan"), float("nan")


def _ece(p_hat: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p_hat)
    for i in range(n_bins):
        mask = (p_hat >= bins[i]) & (p_hat < bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(p_hat[mask].mean() - y[mask].mean())
    return float(ece)


# ── Main ───────────────────────────────────────────────────────────────────

@app.command()
def main(
    champion_oof: str = typer.Option(..., "--champion-oof",
                                      help="Champion OOF predictions parquet."),
    challenger_oof: str = typer.Option(..., "--challenger-oof",
                                        help="Challenger OOF predictions parquet."),
    eval_start: str = typer.Option("2026-06-26", "--eval-start",
                                    help="Untouched eval start (inclusive) YYYY-MM-DD."),
    eval_end: str = typer.Option("2026-07-13", "--eval-end",
                                  help="Untouched eval end (inclusive) YYYY-MM-DD."),
    out_dir: str = typer.Option(..., "--out-dir", help="Output directory for evaluation files."),
) -> None:
    """Paired champion/challenger PMF evaluation on untouched chronological data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load both OOF sets
    for label, path in [("champion", champion_oof), ("challenger", challenger_oof)]:
        if not Path(path).exists():
            typer.echo(f"[FATAL] {label} OOF not found: {path}", err=True)
            raise typer.Exit(1)

    champ_df = pd.read_parquet(champion_oof)
    chal_df  = pd.read_parquet(challenger_oof)

    # Normalise column names
    for df, name in [(champ_df, "champion"), (chal_df, "challenger")]:
        if "actual_outcome" in df.columns and "outcome" not in df.columns:
            df["outcome"] = df["actual_outcome"]
        if "pmf_json" not in df.columns:
            typer.echo(f"[FATAL] {name} OOF missing pmf_json column", err=True)
            raise typer.Exit(1)

    # Filter to untouched evaluation period
    for df, label in [(champ_df, "champion"), (chal_df, "challenger")]:
        if "game_date" in df.columns:
            gd = pd.to_datetime(df["game_date"], utc=True, errors="coerce").dt.date
            mask = (gd >= pd.to_datetime(eval_start).date()) & \
                   (gd <= pd.to_datetime(eval_end).date())
            df_filtered = df[mask].copy()
        else:
            df_filtered = pd.DataFrame()
        if label == "champion":
            champ_eval = df_filtered
        else:
            chal_eval = df_filtered

    typer.echo(f"[eval] Champion eval rows: {len(champ_eval)}")
    typer.echo(f"[eval] Challenger eval rows: {len(chal_eval)}")

    if champ_eval.empty:
        typer.echo("[FATAL] No champion evaluation rows in untouched period", err=True)
        raise typer.Exit(1)
    if chal_eval.empty:
        typer.echo("[FATAL] No challenger evaluation rows in untouched period", err=True)
        raise typer.Exit(1)

    # Match on exact keys
    KEY_COLS = ["game_id", "player_id", "stat"]
    champ_eval = champ_eval.set_index(KEY_COLS)
    chal_eval  = chal_eval.set_index(KEY_COLS)
    shared_keys = champ_eval.index.intersection(chal_eval.index)
    champ_unmatched = len(champ_eval) - len(shared_keys)
    chal_unmatched  = len(chal_eval)  - len(shared_keys)
    typer.echo(f"[eval] Matched keys: {len(shared_keys)} | champion unmatched: {champ_unmatched} | challenger unmatched: {chal_unmatched}")

    if len(shared_keys) == 0:
        typer.echo("[FATAL] Zero matched evaluation rows", err=True)
        raise typer.Exit(1)

    champ_matched = champ_eval.loc[shared_keys].reset_index()
    chal_matched  = chal_eval.loc[shared_keys].reset_index()
    n_games = champ_matched["game_id"].nunique() if "game_id" in champ_matched.columns else 0

    PRIMARY_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    ROLES = ["fringe", "bench", "rotation", "core", "starter"]

    results_by_stat: dict = {}
    promotion_signals: dict[str, str] = {}

    for stat in PRIMARY_STATS:
        cm = champ_matched[champ_matched["stat"] == stat]
        cl = chal_matched[chal_matched["stat"] == stat]
        if cm.empty or cl.empty:
            results_by_stat[stat] = {"status": "NO_DATA"}
            promotion_signals[stat] = "INSUFFICIENT_DATA"
            continue

        stat_results: dict = {"n": len(cm), "games": int(cm["game_id"].nunique())}

        for label, df in [("champion", cm), ("challenger", cl)]:
            log_scores, rps_scores, briers = [], [], []
            outcomes = df["outcome"].values if "outcome" in df.columns else df.get("actual_outcome", pd.Series()).values
            lines = df.get("line", pd.Series([float("nan")] * len(df))).values if "line" in df.columns else np.full(len(df), float("nan"))

            for i, row in df.iterrows():
                o = outcomes[i - df.index[0]] if hasattr(df.index, '__getitem__') else row.get("outcome", row.get("actual_outcome", float("nan")))
                pj = row.get("pmf_json")
                if pj is None or math.isnan(float(o)):
                    continue
                ls = _pmf_log_score(pj, int(o))
                if not math.isnan(ls):
                    log_scores.append(ls)
                rps = _pmf_rps(pj, int(o))
                if not math.isnan(rps):
                    rps_scores.append(rps)
                ln = float(lines[i - df.index[0]] if hasattr(df.index, '__getitem__') else row.get("line", float("nan")))
                b = _pmf_brier(pj, int(o), ln)
                if b is not None:
                    briers.append(b)

            # P(over) for calibration
            p_overs = []
            y_overs = []
            for i, row in df.iterrows():
                pj = row.get("pmf_json")
                ln_val = float(lines[i - df.index[0]] if hasattr(df.index, '__getitem__') else row.get("line", float("nan")))
                o = float(outcomes[i - df.index[0]] if hasattr(df.index, '__getitem__') else row.get("outcome", row.get("actual_outcome", float("nan"))))
                if pj is None or math.isnan(ln_val) or ln_val <= 0 or math.isnan(o):
                    continue
                r = _parse_pmf(pj)
                if r is None:
                    continue
                k, p = r
                p_overs.append(float(p[k > ln_val].sum()))
                y_overs.append(float(o > ln_val))

            p_hat = np.array(p_overs)
            y_hat = np.array(y_overs)
            slope, intercept = _calibration_slope_intercept(p_hat, y_hat)

            # Mean bias from PMF mean
            pmf_means = [float(np.dot(*_parse_pmf(r.get("pmf_json")))) if r.get("pmf_json") and _parse_pmf(r.get("pmf_json")) is not None else float("nan")
                         for _, r in df.iterrows()]
            actual_vals = outcomes[:len(pmf_means)]
            bias = float(np.nanmean(np.array(pmf_means) - actual_vals.astype(float)))

            stat_results[label] = {
                "log_score": round(float(np.mean(log_scores)), 5) if log_scores else None,
                "rps": round(float(np.mean(rps_scores)), 5) if rps_scores else None,
                "brier": round(float(np.mean(briers)), 5) if briers else None,
                "calibration_slope": round(slope, 4) if not math.isnan(slope) else None,
                "calibration_intercept": round(intercept, 4) if not math.isnan(intercept) else None,
                "ece": round(_ece(p_hat, y_hat), 5) if len(p_hat) > 0 else None,
                "mean_bias": round(bias, 4) if not math.isnan(bias) else None,
                "n_calibration": len(log_scores),
            }

        # Promotion decision per stat
        champ_ls = stat_results.get("champion", {}).get("log_score")
        chal_ls  = stat_results.get("challenger", {}).get("log_score")
        if champ_ls is None or chal_ls is None:
            promotion_signals[stat] = "INSUFFICIENT_DATA"
        elif chal_ls > champ_ls - 0.01:  # noninferiority margin 0.01
            promotion_signals[stat] = "PROMOTE_CHALLENGER"
        else:
            promotion_signals[stat] = "KEEP_COMPLETE_CHAMPION"

        results_by_stat[stat] = stat_results

    # Overall promotion decision
    primary_results = [v for k, v in promotion_signals.items() if k in PRIMARY_STATS]
    n_promotes = sum(1 for v in primary_results if v == "PROMOTE_CHALLENGER")
    n_regress  = sum(1 for v in primary_results if v == "KEEP_COMPLETE_CHAMPION")
    n_insuff   = sum(1 for v in primary_results if v == "INSUFFICIENT_DATA")

    if n_insuff > 3:
        overall = "INSUFFICIENT_DATA"
    elif n_regress > 0:
        overall = "KEEP_COMPLETE_CHAMPION"
    elif n_promotes >= len(PRIMARY_STATS) // 2:
        overall = "PROMOTE_COMPLETE_CHALLENGER"
    else:
        overall = "KEEP_COMPLETE_CHAMPION"

    promotion_doc = {
        "evaluation_dates": {"start": eval_start, "end": eval_end},
        "matched_rows": len(shared_keys),
        "independent_games": n_games,
        "champion_unmatched": champ_unmatched,
        "challenger_unmatched": chal_unmatched,
        "metrics_by_stat": results_by_stat,
        "promotion_signals_by_stat": promotion_signals,
        "overall_decision": overall,
        "primary_stats_evaluated": PRIMARY_STATS,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (out / "paired_metrics.json").write_text(json.dumps(promotion_doc, indent=2, default=str))
    (out / "promotion_decision.json").write_text(json.dumps({
        "decision": overall,
        "reason": f"{n_promotes}/{len(PRIMARY_STATS)} stats promote, {n_regress} regress, {n_insuff} insufficient",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    typer.echo(f"\n{'='*50}")
    typer.echo(f"PROMOTION DECISION: {overall}")
    typer.echo(f"  promotes={n_promotes} regresses={n_regress} insufficient={n_insuff}")
    for stat, dec in promotion_signals.items():
        typer.echo(f"  {stat}: {dec}")
    typer.echo(f"{'='*50}")
    typer.echo(f"Wrote evaluation → {out}/paired_metrics.json")
    typer.echo(f"Wrote decision    → {out}/promotion_decision.json")


if __name__ == "__main__":
    app()
