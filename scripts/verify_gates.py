from __future__ import annotations

import sys

import pandas as pd
import typer
import yaml

from wnba_props_model.evaluation.diagnostics import calibration_report, market_superiority_report

app = typer.Typer(add_completion=False)

_DEFAULT_ECE = 0.03
_DEFAULT_KS = 0.075
_DEFAULT_MEAN_ERR = 0.15

# Pre-calibration sanity defaults (looser — catches broken models, not miscalibration)
_DEFAULT_PRE_ECE = 0.12
_DEFAULT_PRE_KS = 0.20
_DEFAULT_PRE_MEAN_ERR = 1.0


def _load_pmf_df(oof_scored: str) -> pd.DataFrame:
    """Load + normalize OOF parquet for calibration_report."""
    from wnba_props_model.models.simulation import json_to_pmf

    df = pd.read_parquet(oof_scored).copy()
    if "outcome" not in df.columns and "actual_outcome" in df.columns:
        df["outcome"] = df["actual_outcome"]
    if "role_bucket" not in df.columns:
        df["role_bucket"] = "all"
    if "pmf" not in df.columns and "pmf_json" in df.columns:
        df["pmf"] = df["pmf_json"].map(json_to_pmf)
    if "calibration_eligible" in df.columns:
        df = df[df["calibration_eligible"] == True].copy()  # noqa: E712
    return df


def _run_gates(
    rep: pd.DataFrame,
    ece_threshold: float,
    ks_threshold: float,
    mean_error_threshold: float,
    label: str = "",
) -> list[str]:
    """Print gate results, return list of failed gate names."""
    prefix = f"[{label}] " if label else ""
    failures: list[str] = []

    if "ece" in rep.columns:
        bad_ece = rep[rep["ece"] > ece_threshold][["stat", "role_bucket", "ece", "n"]]
        if len(bad_ece):
            typer.echo(f"\n{prefix}[FAIL] ECE gate breached ({len(bad_ece)} rows):")
            typer.echo(bad_ece.to_string(index=False))
            failures.append("ECE")

    bad_ks = rep[rep["pit_ks"] > ks_threshold][["stat", "role_bucket", "pit_ks", "n"]]
    if len(bad_ks):
        typer.echo(f"\n{prefix}[FAIL] PIT KS gate breached ({len(bad_ks)} rows):")
        typer.echo(bad_ks.to_string(index=False))
        failures.append("PIT_KS")

    bad_mean = rep[rep["mean_error"].abs() > mean_error_threshold][["stat", "role_bucket", "mean_error", "n"]]
    if len(bad_mean):
        typer.echo(f"\n{prefix}[FAIL] mean_error gate breached ({len(bad_mean)} rows):")
        typer.echo(bad_mean.to_string(index=False))
        failures.append("mean_error")

    return failures


@app.command()
def calibration(
    oof_scored: str = typer.Argument(..., help="Path to OOF scored parquet (requires stat, role_bucket, pmf, outcome cols)."),
    config: str | None = typer.Option(None, help="Path to stage6_calibration.yaml; overrides defaults."),
    ece_threshold: float = typer.Option(_DEFAULT_ECE, help="ECE gate per stat (PenaltyBlog: < 0.03)."),
    ks_threshold: float = typer.Option(_DEFAULT_KS, help="PIT KS gate per stat (< 0.075)."),
    mean_error_threshold: float = typer.Option(_DEFAULT_MEAN_ERR, help="|mean_error| gate per stat/role."),
    pre_cal: bool = typer.Option(False, "--pre-cal", help="Run loose pre-calibration sanity gates (raw OOF, before calibrators)."),
) -> None:
    """Verify calibration quality gates.

    By default (--no-pre-cal) checks CALIBRATED predictions against strict gates.
    Use --pre-cal to check raw OOF predictions against loose sanity gates.

    Strict post-cal gates (default): ECE<0.03, PIT KS<0.075, |mean_error|<0.15
    Loose pre-cal gates (--pre-cal):  ECE<0.12, PIT KS<0.20,  |mean_error|<1.0
    """
    cfg: dict = {}
    if config:
        cfg = yaml.safe_load(open(config))

    if pre_cal:
        ece_threshold = cfg.get("pre_cal_ece_threshold", _DEFAULT_PRE_ECE)
        ks_threshold = cfg.get("pre_cal_pit_ks_threshold", _DEFAULT_PRE_KS)
        mean_error_threshold = cfg.get("pre_cal_mean_error_threshold", _DEFAULT_PRE_MEAN_ERR)
        mode_label = "PRE-CAL SANITY"
    else:
        ece_threshold = cfg.get("ece_threshold", ece_threshold)
        ks_threshold = cfg.get("pit_ks_threshold", ks_threshold)
        mean_error_threshold = cfg.get("mean_error_threshold", mean_error_threshold)
        mode_label = "POST-CAL"

    df = _load_pmf_df(oof_scored)
    rep = calibration_report(df)

    typer.echo(f"\n=== Calibration Gate Report ({mode_label}) ===")
    typer.echo(rep.to_string(index=False))
    typer.echo(f"\nGates: ECE < {ece_threshold} | PIT KS < {ks_threshold} | |mean_error| < {mean_error_threshold}")

    failures = _run_gates(rep, ece_threshold, ks_threshold, mean_error_threshold, label=mode_label)

    if failures:
        typer.echo(f"\n[GATE FAIL] Failed gates: {', '.join(failures)}")
        raise typer.Exit(1)

    typer.echo(f"\n[GATE PASS] All {mode_label} calibration gates passed.")
    raise typer.Exit(0)


@app.command()
def market(
    loss_rows: str = typer.Argument(..., help="Path to event loss rows parquet."),
    min_rows: int = typer.Option(100, help="Minimum rows to consider a stat/role eligible."),
) -> None:
    """Verify market superiority gate (model must beat no-vig market on logloss + Brier)."""
    df = pd.read_parquet(loss_rows)
    rep = market_superiority_report(df, min_rows=min_rows)

    typer.echo("\n=== Market Superiority Report ===")
    typer.echo(rep.to_string(index=False))

    eligible = rep[rep["eligible"]]
    if len(eligible) and not eligible["certified_pass"].all():
        failed = eligible[~eligible["certified_pass"]][["stat", "role_bucket", "event_logloss_delta_ucb95", "brier_delta_ucb95"]]
        typer.echo(f"\n[GATE FAIL] Market superiority not certified for {len(failed)} stat/role(s):")
        typer.echo(failed.to_string(index=False))
        raise typer.Exit(1)

    typer.echo("\n[GATE PASS] Market superiority certified.")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
