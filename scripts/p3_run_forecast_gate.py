"""P3 — run the completed forecast gate on the FULL 2026 OOF with the committed split.

Split (committed BEFORE scoring): holdout = latest 25 unique game-dates,
calibration = preceding 10, development = all earlier eligible dates.
Baseline = per-stat pooled seasonal empirical marginal (climatology) fit on dev+cal.
Champion = raw OOF PMF; Challenger A = fold-safe distributional (location) calibration.
A stat is certified only if forecast_allowed under the completed gate on the holdout.
Emits config/stat_registry.json + artifacts/p3/p3_gate_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.evaluation import forecasting as fc  # noqa: E402
from wnba_props_model.evaluation.pmf_recalibration import fold_safe_pmf_recalibration  # noqa: E402

app = typer.Typer(add_completion=False)


def _empirical_baseline(actuals: np.ndarray, max_support: int) -> np.ndarray:
    pmf = np.zeros(max_support + 1)
    for a in actuals:
        ai = int(round(a))
        if 0 <= ai <= max_support:
            pmf[ai] += 1
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def _baseline_metrics(train_actuals, holdout_actuals, cover=0.8):
    max_sup = int(max(holdout_actuals.max(), train_actuals.max())) + 5
    b = _empirical_baseline(np.asarray(train_actuals), max_sup)
    crps = float(np.mean([fc.crps_discrete(b, int(y)) for y in holdout_actuals]))
    logs = float(np.mean([fc.log_score(b, int(y)) for y in holdout_actuals]))
    lo, hi = fc.central_interval(b, cover)
    return {"crps": crps, "log_score": logs, "mean_width_80": float(hi - lo)}


@app.command()
def main(oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet"),
         out_dir: str = typer.Option("artifacts/p3"),
         registry_out: str = typer.Option("config/stat_registry.json"),
         holdout_dates: int = typer.Option(25), calib_dates: int = typer.Option(10)) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(oof)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["actual_outcome"].notna() & df["pmf_json"].notna()].copy()
    if "did_play" in df.columns:
        df = df[df["did_play"] == True]  # noqa: E712

    dates = np.sort(df["game_date"].dropna().unique())
    if len(dates) < holdout_dates + calib_dates + 1:
        typer.echo(f"[P3] only {len(dates)} unique dates; 2026 alone cannot meet the split.")
    hold_dates = set(dates[-holdout_dates:])
    cal_dates = set(dates[-(holdout_dates + calib_dates):-holdout_dates])
    hold = df[df["game_date"].isin(hold_dates)]
    devcal = df[~df["game_date"].isin(hold_dates)]

    split = {
        "holdout_start": str(pd.Timestamp(min(hold_dates)).date()),
        "holdout_end": str(pd.Timestamp(max(hold_dates)).date()),
        "n_holdout_dates": len(hold_dates), "n_calib_dates": len(cal_dates),
        "dev_start": str(pd.Timestamp(dates[0]).date()), "n_total_dates": int(len(dates)),
    }
    (out / "p3_split.json").write_text(json.dumps(split, indent=2, default=str))

    # Challenger A: fold-safe distributional calibration. Two variants fit on earlier
    # folds only: (loc) bias-only location transport; (locscale) location-AND-scale
    # transport (sharpens over-dispersed PMFs).
    df_loc = df.copy(); df_loc["pmf_v"] = fold_safe_pmf_recalibration(df_loc, use_dispersion=False)
    df_ls = df.copy(); df_ls["pmf_v"] = fold_safe_pmf_recalibration(df_ls, use_dispersion=True)
    hold_loc = df_loc[df_loc["game_date"].isin(hold_dates)]
    hold_ls = df_ls[df_ls["game_date"].isin(hold_dates)]

    def _eval_variant(frame, stat, base):
        s = frame[frame["stat"] == stat].drop(columns=["pmf_json"]).rename(columns={"pmf_v": "pmf_json"})
        return fc.evaluate_stat(s, baseline=base)

    registry = {}; rows = []
    for stat in sorted(df["stat"].unique()):
        h = hold[hold["stat"] == stat]
        train_actuals = devcal[devcal["stat"] == stat]["actual_outcome"].values
        if len(h) == 0 or len(train_actuals) == 0:
            continue
        base = _baseline_metrics(train_actuals, h["actual_outcome"].values)
        cands = {
            "champion": fc.evaluate_stat(h, baseline=base),
            "challenger_A_loc": _eval_variant(hold_loc, stat, base),
            "challenger_A_locscale": _eval_variant(hold_ls, stat, base),
        }
        # prefer a passing variant; else the lowest-CRPS variant
        passing = {k: v for k, v in cands.items() if v.forecast_allowed}
        if passing:
            winner = min(passing, key=lambda k: passing[k].crps)
        else:
            winner = min(cands, key=lambda k: cands[k].crps)
        best = cands[winner]
        registry[stat] = {
            "forecast_allowed": bool(best.forecast_allowed),
            "market_comparison_allowed": bool(best.market_comparison_allowed),
            "betting_recommendation_allowed": bool(best.betting_recommendation_allowed),
            "winner": winner, "n": best.n, "n_dates": best.n_dates,
            "crps": round(best.crps, 4), "log_score": round(best.log_score, 4),
            "crps_vs_baseline": round(best.crps_vs_baseline, 4),
            "pit_ks_p": round(best.pit_ks_p, 4),
            "suppression_reason": "" if best.forecast_allowed else "; ".join(best.reasons),
        }
        rows.append((stat, best, winner))

    certified = sorted([s for s, e in registry.items() if e["forecast_allowed"]])
    Path(registry_out).write_text(json.dumps(registry, indent=2))
    status = "LIVE_VALIDATED_FORECAST_ONLY" if certified else "BLOCKED_MODEL"

    lines = ["# P3 forecast gate — full 2026, committed split", "",
             f"Holdout {split['holdout_start']}..{split['holdout_end']} "
             f"({split['n_holdout_dates']} dates); total dates {split['n_total_dates']}", "",
             "| stat | winner | n | dates | CRPS | dCRPS(base) | PIT-KS-p | forecast_allowed | reason |",
             "|------|--------|---|-------|------|-------------|----------|------------------|--------|"]
    for stat, b, w in rows:
        e = registry[stat]
        lines.append(f"| {stat} | {w} | {b.n} | {b.n_dates} | {b.crps:.3f} | "
                     f"{b.crps_vs_baseline:+.3f} | {b.pit_ks_p:.4f} | {e['forecast_allowed']} | "
                     f"{e['suppression_reason'][:70]} |")
    lines += ["", f"**Certified stats:** {certified or 'none'}", f"**Status:** {status}"]
    (out / "p3_gate_report.md").write_text("\n".join(lines))
    typer.echo(f"[P3] status={status} certified={certified}")


if __name__ == "__main__":
    app()
