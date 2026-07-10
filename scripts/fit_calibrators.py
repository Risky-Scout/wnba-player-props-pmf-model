from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from wnba_props_model.pipeline.calibrate import fit_calibrators as fit

app = typer.Typer(add_completion=False)


@app.command()
def main(
    oof_pmfs: str = typer.Option(...),
    out_dir: str = typer.Option("artifacts/models/calibration"),
    props_parquet: str = typer.Option("", help="Optional path to historical player props parquet for beta calibrator line joining."),
    game_totals_mode: bool = typer.Option(False, "--game-totals-mode", help="No-op flag for game totals calibration (future use)."),
):
    paths = fit(oof_pmfs, out_dir, props_parquet_path=props_parquet if props_parquet else None)
    for stat, path in paths.items():
        typer.echo(f"{stat}: {path}")

    # Per-line isotonic calibrators: P(over|line_bucket) correction
    # Line buckets relative to predicted mean z-score: very_low, low, mid, high, very_high
    PER_LINE_STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

    try:
        oof_df = pd.read_parquet(oof_pmfs)
        per_line_calibrators = _fit_per_line_calibrators(oof_df, PER_LINE_STATS)

        if per_line_calibrators:
            per_line_path = Path(out_dir) / "per_line_calibrators.pkl"
            per_line_path.parent.mkdir(parents=True, exist_ok=True)
            with open(per_line_path, "wb") as f:
                pickle.dump(per_line_calibrators, f, protocol=5)
            typer.echo(f"Per-line calibrators saved ({len(per_line_calibrators)} keys): {list(per_line_calibrators.keys())[:10]}")
        else:
            typer.echo("Per-line calibrators: no eligible (stat, bucket) pairs found")
    except Exception as exc:
        typer.echo(f"[WARN] Per-line calibrator fitting failed (non-fatal): {exc}", err=True)


def _fit_per_line_calibrators(oof_df: pd.DataFrame, stats: list[str]) -> dict:
    """Fit per-line isotonic calibrators from OOF data.

    For each (stat, line_z_bucket) pair, train an isotonic regressor mapping
    model_p_over -> empirical_p_over. Line z-score is computed relative to
    the predicted pmf_mean and pmf_std.
    """
    from sklearn.isotonic import IsotonicRegression

    per_line_calibrators = {}

    # Filter to calibration-eligible, played rows
    oof = oof_df.copy()
    if "calibration_eligible" in oof.columns:
        oof = oof[oof["calibration_eligible"] == True].copy()  # noqa: E712
    if "did_play" in oof.columns:
        oof = oof[oof["did_play"] == True].copy()  # noqa: E712
    if "actual_minutes" in oof.columns:
        oof = oof[oof["actual_minutes"].fillna(0) >= 10].copy()

    for stat in stats:
        stat_oof = oof[oof["stat"] == stat].copy() if "stat" in oof.columns else pd.DataFrame()

        if len(stat_oof) < 100:
            continue

        required = ["pmf_mean", "actual_outcome"]
        if not all(c in stat_oof.columns for c in required):
            continue

        # Derive pmf_std from pmf_variance if available, else estimate
        if "pmf_variance" in stat_oof.columns:
            stat_oof["pmf_std"] = stat_oof["pmf_variance"].clip(lower=0).pow(0.5)
        else:
            stat_oof["pmf_std"] = (stat_oof["pmf_mean"].clip(lower=0.5) * 0.4)

        # Compute line z-score; use market line if available else pmf_mean as proxy
        if "line" in stat_oof.columns:
            line_vals = stat_oof["line"]
        else:
            line_vals = stat_oof["pmf_mean"]  # proxy: z=0 (at-the-money)

        stat_oof["line_z"] = (line_vals - stat_oof["pmf_mean"]) / stat_oof["pmf_std"].clip(lower=0.1)

        # Compute p_over from pmf_json if available, else derive from pmf_mean/line
        if "p_over" in stat_oof.columns:
            model_p_over = stat_oof["p_over"].values
        elif "pmf_json" in stat_oof.columns:
            try:
                from wnba_props_model.models.simulation import json_to_pmf
                import math
                p_overs = []
                for _, row in stat_oof.iterrows():
                    try:
                        pmf = json_to_pmf(row["pmf_json"])
                        ln = float(line_vals.loc[row.name]) if hasattr(line_vals, "loc") else float(stat_oof["pmf_mean"].loc[row.name])
                        p_o = float(pmf[math.ceil(ln):].sum()) if math.ceil(ln) < len(pmf) else 0.0
                        p_overs.append(p_o)
                    except Exception:
                        p_overs.append(0.5)
                model_p_over = np.array(p_overs)
            except Exception:
                # Fallback: use normal approximation
                model_p_over = 1.0 - (0.5 * (1.0 + np.sign(stat_oof["line_z"].values) * 0.4))
        else:
            continue

        actual_outcome = stat_oof["actual_outcome"].values

        # Bin z-scores into 5 buckets
        z_bins = pd.cut(
            stat_oof["line_z"],
            bins=[-np.inf, -1, -0.33, 0.33, 1, np.inf],
            labels=["very_low", "low", "mid", "high", "very_high"],
        )

        for bucket in ["very_low", "low", "mid", "high", "very_high"]:
            mask = z_bins == bucket
            bucket_df_p_over = model_p_over[mask.values]
            bucket_line = line_vals.values[mask.values] if hasattr(line_vals, "values") else line_vals[mask.values]
            bucket_actual = actual_outcome[mask.values]

            if len(bucket_df_p_over) < 30:
                continue

            # y = 1 if actual > line (OVER), 0 otherwise
            y = (bucket_actual > bucket_line).astype(float)

            # Filter pushes (actual == line)
            push_mask = bucket_actual != bucket_line
            if push_mask.sum() < 20:
                continue

            X = bucket_df_p_over[push_mask]
            y = y[push_mask]

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(X, y)
            per_line_calibrators[f"{stat}|{bucket}"] = iso

    return per_line_calibrators


if __name__ == "__main__":
    app()
