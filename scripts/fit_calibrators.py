from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.pipeline.calibrate import fit_calibrators as fit

app = typer.Typer(add_completion=False)


class NoMarketLinesError(RuntimeError):
    """Raised when per-line (line-dependent) calibration is attempted without real market lines.

    Line-dependent calibration must NEVER substitute the model's own pmf_mean for
    the sportsbook line (that collapses every observation to line_z=0 and calibrates
    against "did the outcome beat the model's own mean" rather than the market line).
    """


@app.command()
def main(
    oof_pmfs: str = typer.Option(...),
    out_dir: str = typer.Option("artifacts/models/calibration"),
    props_parquet: str = typer.Option("", help="Optional path to historical player props parquet for beta calibrator line joining."),
    oof_with_lines: str = typer.Option("", help="Validated market-enriched OOF parquet (oof_pmfs_with_lines) carrying REAL lines. Required for per-line calibration."),
    game_totals_mode: bool = typer.Option(False, "--game-totals-mode", help="No-op flag for game totals calibration (future use)."),
):
    paths = fit(oof_pmfs, out_dir, props_parquet_path=props_parquet if props_parquet else None)
    for stat, path in paths.items():
        typer.echo(f"{stat}: {path}")

    # OOF persistence: write OOF predictions to parquet for calibration validation.
    try:
        oof_raw = pd.read_parquet(oof_pmfs)
        oof_path = Path(out_dir) / "oof_predictions.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        # Persist all available columns (superset of required); downstream can select.
        oof_raw.to_parquet(oof_path, index=False)
        typer.echo(f"OOF predictions persisted: {oof_path} ({len(oof_raw):,} rows)")
    except Exception as _oof_exc:
        typer.echo(f"[WARN] OOF persistence failed (non-fatal): {_oof_exc}", err=True)

    # Per-line isotonic calibrators: P(over|line_bucket) correction.
    # These are LINE-DEPENDENT and are fit ONLY from the validated market-enriched
    # OOF table (oof_pmfs_with_lines), which carries REAL sportsbook lines. If that
    # table is not provided (or has no real lines), per-line calibration is SKIPPED
    # loudly — we do NOT fall back to raw OOF / pmf_mean.
    PER_LINE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

    if not oof_with_lines:
        typer.echo(
            "[P0][SKIP] Per-line (line-dependent) calibration SKIPPED: "
            "no --oof-with-lines provided. Per-line calibration requires the "
            "validated market-enriched OOF table with REAL lines; it must NOT be "
            "fit from raw OOF (which would substitute pmf_mean for the line).",
            err=True,
        )
    else:
        try:
            oof_df = pd.read_parquet(oof_with_lines)
            per_line_calibrators = _fit_per_line_calibrators(oof_df, PER_LINE_STATS)
            if per_line_calibrators:
                per_line_path = Path(out_dir) / "per_line_calibrators.pkl"
                per_line_path.parent.mkdir(parents=True, exist_ok=True)
                with open(per_line_path, "wb") as f:
                    pickle.dump(per_line_calibrators, f, protocol=5)
                typer.echo(
                    f"Per-line calibrators saved ({len(per_line_calibrators)} keys): "
                    f"{list(per_line_calibrators.keys())[:10]}"
                )
            else:
                typer.echo("Per-line calibrators: no eligible (stat, bucket) pairs found")
        except NoMarketLinesError as exc:
            # Fail-closed but scoped: do NOT produce a bogus per-line calibrator, and
            # do NOT crash the (valid) line-independent calibration above.
            typer.echo(
                f"[P0][SKIP] Per-line calibration SKIPPED — no real market lines: {exc}",
                err=True,
            )
        except Exception as exc:
            typer.echo(f"[WARN] Per-line calibrator fitting failed (non-fatal): {exc}", err=True)


def _p_over_conditional(pmf: np.ndarray, line: float) -> float:
    """P(over | non-push) for a discrete PMF at a sportsbook ``line``.

    Over means strictly greater than the line. For a discrete stat the winning
    "Over" region begins at ``floor(line) + 1`` (so Over 10 and Over 10.5 both
    mean 11+). For an INTEGER line, an outcome exactly equal to the line is a
    PUSH — its mass is removed from the denominator so the returned probability
    is conditional on a non-push, matching the pushes-excluded binary target.
    """
    cutoff = math.floor(line) + 1
    p_over = float(pmf[cutoff:].sum()) if cutoff < len(pmf) else 0.0
    if float(line).is_integer():
        li = int(round(line))
        p_push = float(pmf[li]) if 0 <= li < len(pmf) else 0.0
    else:
        p_push = 0.0
    denom = 1.0 - p_push
    return float(min(max(p_over / denom, 0.0), 1.0)) if denom > 1e-9 else float(p_over)


def _fit_per_line_calibrators(oof_df: pd.DataFrame, stats: list[str]) -> dict:
    """Fit per-line isotonic calibrators from market-enriched OOF data.

    For each (stat, line_z_bucket) pair, train an isotonic regressor mapping the
    model's conditional P(over | non-push) at the REAL line -> the realized
    over rate. Line z-score is computed relative to the predicted pmf_mean/std.

    Requires a real ``line`` column with meaningful coverage. If the ``line``
    column is absent or entirely null, raises NoMarketLinesError — it never
    substitutes pmf_mean for the line.
    """
    from sklearn.isotonic import IsotonicRegression
    from wnba_props_model.models.simulation import json_to_pmf

    if "line" not in oof_df.columns:
        raise NoMarketLinesError(
            "OOF table has no 'line' column; per-line calibration requires real "
            "market lines (the validated oof_pmfs_with_lines table)."
        )
    if int(oof_df["line"].notna().sum()) == 0:
        raise NoMarketLinesError(
            "OOF 'line' column is entirely null; per-line calibration requires "
            "real market lines, not a pmf_mean proxy."
        )
    if "pmf_json" not in oof_df.columns:
        raise NoMarketLinesError(
            "OOF table has no 'pmf_json'; cannot compute model P(over) at the line."
        )

    per_line_calibrators: dict = {}

    # Filter to calibration-eligible, played rows WITH a real line.
    oof = oof_df.copy()
    if "calibration_eligible" in oof.columns:
        oof = oof[oof["calibration_eligible"] == True].copy()  # noqa: E712
    if "did_play" in oof.columns:
        oof = oof[oof["did_play"] == True].copy()  # noqa: E712
    if "actual_minutes" in oof.columns:
        oof = oof[oof["actual_minutes"].fillna(0) >= 10].copy()
    oof = oof[oof["line"].notna()].copy()

    for stat in stats:
        stat_oof = oof[oof["stat"] == stat].copy() if "stat" in oof.columns else pd.DataFrame()
        if len(stat_oof) < 100:
            continue
        required = ["pmf_mean", "actual_outcome", "line", "pmf_json"]
        if not all(c in stat_oof.columns for c in required):
            continue

        if "pmf_variance" in stat_oof.columns:
            stat_oof["pmf_std"] = stat_oof["pmf_variance"].clip(lower=0).pow(0.5)
        else:
            stat_oof["pmf_std"] = (stat_oof["pmf_mean"].clip(lower=0.5) * 0.4)

        line_vals = stat_oof["line"].astype(float)
        stat_oof["line_z"] = (line_vals - stat_oof["pmf_mean"]) / stat_oof["pmf_std"].clip(lower=0.1)

        # Model conditional P(over | non-push) at the REAL line (floor(line)+1 cutoff).
        p_overs = []
        for _, row in stat_oof.iterrows():
            try:
                pmf = json_to_pmf(row["pmf_json"])
                p_overs.append(_p_over_conditional(pmf, float(row["line"])))
            except Exception:
                p_overs.append(np.nan)
        stat_oof["model_p_over_cond"] = p_overs

        actual = stat_oof["actual_outcome"].astype(float).values
        ln = line_vals.values
        # Exclude pushes: only integer lines can push (actual == line).
        push_mask = actual != ln
        # Binary over target conditioned on non-push.
        y_all = (actual > ln).astype(float)

        z_bins = pd.cut(
            stat_oof["line_z"],
            bins=[-np.inf, -1, -0.33, 0.33, 1, np.inf],
            labels=["very_low", "low", "mid", "high", "very_high"],
        )

        for bucket in ["very_low", "low", "mid", "high", "very_high"]:
            m = (z_bins == bucket).values & push_mask & np.isfinite(stat_oof["model_p_over_cond"].values)
            if m.sum() < 30:
                continue
            X = stat_oof["model_p_over_cond"].values[m]
            y = y_all[m]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(X, y)
            per_line_calibrators[f"{stat}|{bucket}"] = iso

    return per_line_calibrators


if __name__ == "__main__":
    app()
