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


def _load_pmf_df(oof_scored: str, skip_eligibility_filter: bool = False) -> pd.DataFrame:
    """Load + normalize OOF parquet for calibration_report.

    Parameters
    ----------
    skip_eligibility_filter:
        If True, include all rows regardless of calibration_eligible flag.
        Used for the pre-cal sanity gate so prior_only rows don't cause an
        empty DataFrame (expected on first run / limited-data seasons).
    """
    from wnba_props_model.models.simulation import json_to_pmf

    df = pd.read_parquet(oof_scored).copy()
    if "outcome" not in df.columns and "actual_outcome" in df.columns:
        df["outcome"] = df["actual_outcome"]
    if "role_bucket" not in df.columns:
        df["role_bucket"] = "all"
    if "pmf" not in df.columns and "pmf_json" in df.columns:
        df["pmf"] = df["pmf_json"].map(json_to_pmf)
    if not skip_eligibility_filter and "calibration_eligible" in df.columns:
        df = df[df["calibration_eligible"] == True].copy()  # noqa: E712
    return df


def _run_gates(
    rep: pd.DataFrame,
    ece_threshold: float,
    ks_threshold: float,
    mean_error_threshold: float,
    label: str = "",
    exclude_role_buckets: list[str] | None = None,
) -> list[str]:
    """Print gate results, return list of failed gate names.

    Returns [] if rep is empty (treated as pass-through — no data to evaluate).
    Rows whose role_bucket is in exclude_role_buckets are removed before gating
    (logged as info, not failures — e.g. inactive_risk uses global-only calibration).
    """
    prefix = f"[{label}] " if label else ""
    failures: list[str] = []

    if rep.empty:
        typer.echo(f"{prefix}[WARN] No rows in calibration report — gate treated as pass-through.")
        return failures

    # Exclude special role buckets from the strict gate
    if exclude_role_buckets and "role_bucket" in rep.columns:
        excluded = rep[rep["role_bucket"].isin(exclude_role_buckets)]
        rep = rep[~rep["role_bucket"].isin(exclude_role_buckets)].copy()
        if len(excluded):
            typer.echo(
                f"{prefix}[INFO] Excluding {len(excluded)} row(s) from gate "
                f"(role_buckets: {exclude_role_buckets}) — these use global-only calibration."
            )
        # #region agent log — H2 verification: inactive_risk exclusion
        import json as _json, time as _time
        try:
            with open("/Users/josephshackelford/SportsModels/wnba-player-props-pmf-model/.cursor/debug-94807e.log", "a") as _lf:
                _lf.write(_json.dumps({"sessionId": "94807e", "runId": "post-fix-gate", "hypothesisId": "H2", "location": "verify_gates.py:_run_gates", "message": "inactive_risk_exclusion", "data": {"exclude_role_buckets": exclude_role_buckets, "excluded_rows": int(len(excluded)), "remaining_rows": int(len(rep)), "excluded_stats": sorted(excluded["stat"].unique().tolist()) if len(excluded) else []}, "timestamp": int(_time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion

    if rep.empty:
        typer.echo(f"{prefix}[WARN] No rows remain after role exclusions — gate treated as pass-through.")
        return failures

    if "ece" in rep.columns:
        bad_ece = rep[rep["ece"] > ece_threshold][["stat", "role_bucket", "ece", "n"]]
        if len(bad_ece):
            typer.echo(f"\n{prefix}[FAIL] ECE gate breached ({len(bad_ece)} rows):")
            typer.echo(bad_ece.to_string(index=False))
            failures.append("ECE")

    if "pit_ks" in rep.columns:
        bad_ks = rep[rep["pit_ks"] > ks_threshold][["stat", "role_bucket", "pit_ks", "n"]]
        if len(bad_ks):
            typer.echo(f"\n{prefix}[FAIL] PIT KS gate breached ({len(bad_ks)} rows):")
            typer.echo(bad_ks.to_string(index=False))
            failures.append("PIT_KS")

    if "mean_error" in rep.columns:
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
        exclude_roles: list[str] = []  # no exclusions for loose pre-cal gate
    else:
        ece_threshold = cfg.get("ece_threshold", ece_threshold)
        ks_threshold = cfg.get("pit_ks_threshold", ks_threshold)
        mean_error_threshold = cfg.get("mean_error_threshold", mean_error_threshold)
        mode_label = "POST-CAL"
        exclude_roles = cfg.get("gate_exclude_role_buckets", [])

    # For pre-cal sanity: include all OOF rows (not just calibration_eligible)
    # so first-run / limited-data seasons with all-prior_only folds don't crash.
    skip_elig = pre_cal
    df = _load_pmf_df(oof_scored, skip_eligibility_filter=skip_elig)

    if df.empty:
        typer.echo(
            f"\n[{mode_label}] OOF data has 0 scoreable rows after loading — "
            "no predictions to gate. This is expected on first pipeline run "
            "before any real model folds complete.\n"
            f"[{mode_label}] Treating as PASS (no data to evaluate)."
        )
        raise typer.Exit(0)

    rep = calibration_report(df)

    typer.echo(f"\n=== Calibration Gate Report ({mode_label}) ===")
    typer.echo(rep.to_string(index=False) if not rep.empty else "(empty)")
    typer.echo(f"\nGates: ECE < {ece_threshold} | PIT KS < {ks_threshold} | |mean_error| < {mean_error_threshold}")

    failures = _run_gates(
        rep, ece_threshold, ks_threshold, mean_error_threshold,
        label=mode_label,
        exclude_role_buckets=exclude_roles if not pre_cal else [],
    )

    if failures:
        typer.echo(f"\n[GATE FAIL] Failed gates: {', '.join(failures)}")
        raise typer.Exit(1)

    typer.echo(f"\n[GATE PASS] All {mode_label} calibration gates passed.")
    raise typer.Exit(0)


def _prepare_market_loss_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise scored results.parquet into the format expected by market_superiority_report.

    Accepts either:
    - Already-prepared data with event_logloss_delta / brier_delta columns.
    - Raw scored data with model_bin_logloss / market_bin_logloss columns
      (output of score_daily_predictions.py).

    Brier score delta is approximated from the binary hit_result and model/market
    probabilities: brier = (p - outcome)^2, delta = model_brier - market_brier.
    """
    out = df.copy()

    # Compute deltas if not already present
    if "event_logloss_delta" not in out.columns:
        if "model_bin_logloss" in out.columns and "market_bin_logloss" in out.columns:
            out["event_logloss_delta"] = out["model_bin_logloss"] - out["market_bin_logloss"]
        else:
            out["event_logloss_delta"] = float("nan")

    if "brier_delta" not in out.columns:
        if all(c in out.columns for c in ("model_prob_over", "market_prob_over_no_vig", "hit_result")):
            model_brier = (out["model_prob_over"] - out["hit_result"].astype(float)) ** 2
            market_brier = (out["market_prob_over_no_vig"] - out["hit_result"].astype(float)) ** 2
            out["brier_delta"] = model_brier - market_brier
        else:
            out["brier_delta"] = float("nan")

    if "ignorance_score_delta" not in out.columns:
        if "model_ignorance_score" in out.columns and "market_ignorance_score" in out.columns:
            out["ignorance_score_delta"] = out["model_ignorance_score"] - out["market_ignorance_score"]

    if "role_bucket" not in out.columns:
        out["role_bucket"] = "all"

    # Only rows with valid market comparisons
    out = out.dropna(subset=["event_logloss_delta", "brier_delta"])
    return out


@app.command()
def market(
    loss_rows: str = typer.Argument(..., help="Path to scored results parquet (results.parquet or pre-computed loss rows)."),
    min_rows: int = typer.Option(100, help="Minimum rows to consider a stat/role eligible (default 100; use 300 for hard gate)."),
    informational: bool = typer.Option(
        True,
        "--informational/--hard",
        help="Informational mode: report gate status but exit 0 even on failure (default). "
             "Use --hard to make gate failures exit 1 once 300+ samples exist per stat.",
    ),
) -> None:
    """Verify market superiority gate (model must beat no-vig market on logloss + Brier).

    Informational mode (default): prints the gate report but always exits 0.
    Hard mode (--hard): exits 1 if any eligible stat fails the certified_pass test.

    Certified pass requires (per PenaltyBlog / UCB95 bootstrap):
    - event_logloss_delta UCB95 < -0.0025  (model log-loss reliably below market)
    - brier_delta UCB95 < -0.0010          (model Brier score reliably below market)
    with min_rows qualifying predictions for that stat/role.
    """
    df = pd.read_parquet(loss_rows)
    df = _prepare_market_loss_rows(df)

    if df.empty:
        typer.echo("[MARKET GATE] No valid market comparison rows — insufficient data. Skipping gate.")
        raise typer.Exit(0)

    rep = market_superiority_report(df, min_rows=min_rows)

    typer.echo("\n=== Market Superiority Gate Report ===")
    typer.echo(rep.to_string(index=False))

    eligible = rep[rep["eligible"]]
    total_rows = int(rep["n"].sum())

    if len(eligible) == 0:
        typer.echo(
            f"\n[MARKET GATE] Informational — only {total_rows:,} total scored rows "
            f"({min_rows} per stat required). Accumulating data."
        )
        raise typer.Exit(0)

    failing = eligible[~eligible["certified_pass"]]

    if informational:
        # Always exit 0 in informational mode — annotation only
        if len(failing):
            typer.echo(
                f"\n[MARKET GATE] ⚠ Informational ({len(failing)} stat(s) not yet certified):"
            )
            typer.echo(failing[["stat", "role_bucket", "n", "event_logloss_delta_mean", "event_logloss_delta_ucb95"]].to_string(index=False))
            typer.echo("\nPromote to --hard gate once 300+ samples exist per stat and first full season completes.")
        else:
            typer.echo("\n[MARKET GATE] ✓ All eligible stats certified (informational).")
        raise typer.Exit(0)
    else:
        if len(failing):
            typer.echo(f"\n[GATE FAIL] Market superiority not certified for {len(failing)} stat/role(s):")
            typer.echo(failing[["stat", "role_bucket", "event_logloss_delta_ucb95", "brier_delta_ucb95"]].to_string(index=False))
            raise typer.Exit(1)
        typer.echo("\n[GATE PASS] Market superiority certified.")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
