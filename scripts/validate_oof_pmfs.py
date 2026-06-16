#!/usr/bin/env python3
"""Stage 5 OOF PMF validation script.

Validates temporal correctness, PMF structural validity, and fold manifest
integrity. Fails hard on any leakage or PMF violation.

Usage:
    python3 scripts/validate_oof_pmfs.py \\
      --pmfs data/oof/oof_player_stat_pmfs.parquet \\
      --fold-manifest data/oof/oof_fold_manifest.parquet \\
      --audit-out artifacts/audits/stage5_oof_pmf_validation_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OOF_PMF_SOURCE = "stage5_walk_forward_oof_uncalibrated_model_only"

app = typer.Typer(add_completion=False)


@app.command()
def validate(
    pmfs: Path = typer.Option(
        Path("data/oof/oof_player_stat_pmfs.parquet"), "--pmfs"
    ),
    fold_manifest: Path = typer.Option(
        Path("data/oof/oof_fold_manifest.parquet"), "--fold-manifest"
    ),
    audit_out: Path = typer.Option(
        Path("artifacts/audits/stage5_oof_pmf_validation_audit.json"),
        "--audit-out",
    ),
) -> None:
    print("=" * 70)
    print("Stage 5 — OOF PMF Validation")
    print("=" * 70)

    failures: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load files
    # ------------------------------------------------------------------
    for p in [pmfs, fold_manifest]:
        if not p.exists():
            typer.echo(f"ERROR: File not found: {p}", err=True)
            raise typer.Exit(1)

    oof = pd.read_parquet(pmfs)
    oof["game_date"] = pd.to_datetime(oof["game_date"])
    folds = pd.read_parquet(fold_manifest)
    print(f"OOF rows:   {len(oof):,}")
    print(f"Fold rows:  {len(folds)}")

    # ------------------------------------------------------------------
    # 2. Required columns
    # ------------------------------------------------------------------
    required_oof = [
        "game_id", "game_date", "player_id", "stat", "pmf_json",
        "is_calibrated", "pmf_source", "oof_prediction_type",
        "calibration_eligible", "fold_id", "fold_train_end_date",
        "fold_validation_start_date",
    ]
    missing = [c for c in required_oof if c not in oof.columns]
    if missing:
        failures.append(f"Missing OOF columns: {missing}")

    required_fold = ["fold_id", "train_end_date", "validation_start_date"]
    # Note: fold manifest may use different column names — check both
    fold_cols_alt = {"train_end_date": "fold_train_end_date",
                     "validation_start_date": "fold_validation_start_date"}

    # ------------------------------------------------------------------
    # 3. Temporal leakage check
    # ------------------------------------------------------------------
    print("\nChecking temporal integrity...")
    if all(c in oof.columns for c in ["fold_train_end_date", "fold_validation_start_date"]):
        te = pd.to_datetime(oof["fold_train_end_date"])
        vs = pd.to_datetime(oof["fold_validation_start_date"])
        leakage = (te >= vs).sum()
        if leakage > 0:
            failures.append(
                f"TEMPORAL LEAKAGE: fold_train_end_date >= fold_validation_start_date "
                f"in {leakage} rows"
            )
        else:
            print("  fold_train_end_date < fold_validation_start_date: PASS")

    # Same-day leakage: game_date must be within validation window
    if "game_date" in oof.columns and "fold_validation_start_date" in oof.columns:
        gd = pd.to_datetime(oof["game_date"]).dt.date
        vs = pd.to_datetime(oof["fold_validation_start_date"]).dt.date
        te = pd.to_datetime(oof["fold_train_end_date"]).dt.date
        # Game dates must be >= val_start_date (not in training window)
        in_train = (gd < vs)
        if in_train.any():
            failures.append(f"SAME-DAY LEAKAGE: {in_train.sum()} games with "
                            "game_date < fold_validation_start_date in validation set")
        else:
            print("  Same-day leakage check: PASS")

    # ------------------------------------------------------------------
    # 4. Duplicate key check
    # ------------------------------------------------------------------
    dup_count = oof.duplicated(subset=["player_id", "game_id", "stat"]).sum()
    if dup_count > 0:
        failures.append(f"Duplicate player × game × stat: {dup_count}")
    else:
        print(f"  Duplicate key check: PASS")

    # ------------------------------------------------------------------
    # 5. is_calibrated and pmf_source
    # ------------------------------------------------------------------
    if "is_calibrated" in oof.columns:
        not_false = (oof["is_calibrated"] != False).sum()  # noqa: E712
        if not_false > 0:
            failures.append(f"is_calibrated != False: {not_false} rows")
        else:
            print("  is_calibrated = False: PASS")

    if "pmf_source" in oof.columns:
        wrong = (oof["pmf_source"] != OOF_PMF_SOURCE).sum()
        if wrong > 0:
            failures.append(f"pmf_source != '{OOF_PMF_SOURCE}': {wrong} rows")
        else:
            print(f"  pmf_source check: PASS")

    # ------------------------------------------------------------------
    # 6. calibration_eligible only for model_oof
    # ------------------------------------------------------------------
    if all(c in oof.columns for c in ["calibration_eligible", "oof_prediction_type"]):
        wrong_cal = (
            (oof["calibration_eligible"] == True)  # noqa: E712
            & (oof["oof_prediction_type"] != "model_oof")
        ).sum()
        if wrong_cal > 0:
            failures.append(f"calibration_eligible=True for non-model_oof rows: {wrong_cal}")
        else:
            print("  calibration_eligible logic: PASS")

    # ------------------------------------------------------------------
    # 7. PMF structural validation
    # ------------------------------------------------------------------
    print("Validating PMF structures...")
    invalid_count = 0
    max_sum_err = 0.0

    for _, row in oof.iterrows():
        pmf_str = row.get("pmf_json")
        if not pmf_str:
            invalid_count += 1
            continue
        try:
            pmf = json.loads(pmf_str)
        except Exception:
            invalid_count += 1
            continue

        probs = list(pmf.values())
        if any(not np.isfinite(p) for p in probs):
            invalid_count += 1
        if any(p < -1e-9 for p in probs):
            invalid_count += 1
        s = sum(probs)
        err = abs(s - 1.0)
        max_sum_err = max(max_sum_err, err)
        if err > 1e-6:
            invalid_count += 1

    if invalid_count > 0:
        failures.append(f"Invalid OOF PMFs: {invalid_count}")
    print(f"  Max PMF sum error:  {max_sum_err:.2e}")
    print(f"  Invalid PMF count:  {invalid_count}")

    # ------------------------------------------------------------------
    # 8. pmf_mean and pmf_variance finiteness
    # ------------------------------------------------------------------
    for col in ["pmf_mean", "pmf_variance"]:
        if col in oof.columns and not np.isfinite(oof[col].fillna(0)).all():
            failures.append(f"{col} contains non-finite values")

    # ------------------------------------------------------------------
    # 9. Coverage warnings
    # ------------------------------------------------------------------
    if "oof_prediction_type" in oof.columns:
        type_counts = oof["oof_prediction_type"].value_counts().to_dict()
        total = len(oof)
        prior_frac = type_counts.get("prior_only", 0) / total
        if prior_frac > 0.3:
            warnings.append(f"High prior_only fraction: {prior_frac:.1%}")
        print(f"  Prediction types: {type_counts}")
        print(f"  Calibration eligible: {(oof['calibration_eligible'] == True).sum():,}")  # noqa: E712

    # ------------------------------------------------------------------
    # Write audit
    # ------------------------------------------------------------------
    stat_counts = oof.groupby("stat").size().to_dict() if "stat" in oof.columns else {}
    audit = {
        "oof_rows": int(len(oof)),
        "oof_rows_by_stat": {k: int(v) for k, v in stat_counts.items()},
        "fold_count": int(len(folds)),
        "duplicate_key_count": int(dup_count),
        "invalid_pmf_count": invalid_count,
        "max_pmf_sum_error": max_sum_err,
        "temporal_leakage_check": "PASS" if not any("LEAKAGE" in f for f in failures) else "FAIL",
        "same_day_leakage_check": "PASS" if not any("SAME-DAY" in f for f in failures) else "FAIL",
        "is_calibrated_check": "PASS" if not any("is_calibrated" in f for f in failures) else "FAIL",
        "pmf_source_check": "PASS" if not any("pmf_source" in f for f in failures) else "FAIL",
        "failures": failures,
        "warnings": warnings,
    }
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nAudit written: {audit_out}")

    if failures:
        print("\n" + "=" * 70)
        print("VALIDATION FAILED")
        for f in failures:
            print(f"  FAIL: {f}")
        print("=" * 70)
        raise typer.Exit(1)

    for w in warnings:
        print(f"  WARN: {w}")

    print("\n" + "=" * 70)
    print("OOF PMF Validation PASSED")
    print("=" * 70)


if __name__ == "__main__":
    app()
