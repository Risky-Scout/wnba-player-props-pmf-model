#!/usr/bin/env python3
"""Stage 4 PMF validation script.

Validates structural properties of generated atom PMFs.  Fails hard on any
violation.  Writes an audit JSON with per-stat summary statistics.

Usage:
    python3 scripts/validate_pmfs.py \\
      --pmfs data/model_outputs/stage4_baseline/player_stat_pmfs.parquet \\
      --audit-out artifacts/audits/stage4_pmf_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(add_completion=False)

REQUIRED_PMF_SOURCE = "stage4_baseline_uncalibrated_model_only"


def _load_pmf_json(s: str) -> dict[int, float]:
    raw = json.loads(s)
    return {int(k): float(v) for k, v in raw.items()}


@app.command()
def validate(
    pmfs: Path = typer.Option(
        Path("data/model_outputs/stage4_baseline/player_stat_pmfs.parquet"),
        "--pmfs",
    ),
    audit_out: Path = typer.Option(
        Path("artifacts/audits/stage4_pmf_audit.json"),
        "--audit-out",
    ),
) -> None:
    print("=" * 70)
    print("Stage 4 — PMF Validation")
    print("=" * 70)

    if not pmfs.exists():
        typer.echo(f"ERROR: PMF file not found: {pmfs}", err=True)
        raise typer.Exit(1)

    print(f"Loading: {pmfs}")
    df = pd.read_parquet(pmfs)
    print(f"  {len(df):,} rows")

    failures: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Required columns
    # ------------------------------------------------------------------
    required_cols = [
        "game_id", "player_id", "stat", "pmf_json",
        "pmf_mean", "pmf_variance", "p0", "is_calibrated", "pmf_source",
        "actual_outcome", "actual_minutes",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")

    # ------------------------------------------------------------------
    # 2. Duplicate player × game × stat check
    # ------------------------------------------------------------------
    dup_count = df.duplicated(subset=["player_id", "game_id", "stat"]).sum()
    if dup_count > 0:
        failures.append(f"Duplicate player_id × game_id × stat rows: {dup_count}")
    else:
        print(f"  Duplicate key check: PASS (0 duplicates)")

    # ------------------------------------------------------------------
    # 3. is_calibrated must be False for Stage 4
    # ------------------------------------------------------------------
    if "is_calibrated" in df.columns:
        not_false = (df["is_calibrated"] != False).sum()  # noqa: E712
        if not_false > 0:
            failures.append(f"is_calibrated != False for {not_false} rows")
        else:
            print(f"  is_calibrated = False: PASS")

    # ------------------------------------------------------------------
    # 4. pmf_source check
    # ------------------------------------------------------------------
    if "pmf_source" in df.columns:
        wrong_source = (df["pmf_source"] != REQUIRED_PMF_SOURCE).sum()
        if wrong_source > 0:
            failures.append(
                f"pmf_source != '{REQUIRED_PMF_SOURCE}' for {wrong_source} rows"
            )
        else:
            print(f"  pmf_source check: PASS")

    # ------------------------------------------------------------------
    # 5. Per-PMF structural validation (batch)
    # ------------------------------------------------------------------
    print("Validating PMF structures...")
    invalid_pmf_count = 0
    max_sum_error = 0.0
    min_prob_global = 1.0
    max_prob_global = 0.0
    neg_prob_count = 0
    nonfinite_count = 0
    zero_support_count = 0
    empty_pmf_count = 0

    for i, row in enumerate(df.itertuples(index=False)):
        pmf_str = getattr(row, "pmf_json", None)
        if pmf_str is None:
            empty_pmf_count += 1
            invalid_pmf_count += 1
            continue

        try:
            pmf = _load_pmf_json(pmf_str)
        except Exception as e:
            invalid_pmf_count += 1
            failures.append(f"Row {i}: PMF JSON parse error: {e}")
            continue

        if not pmf:
            empty_pmf_count += 1
            invalid_pmf_count += 1
            continue

        # Support starts at 0
        if min(pmf.keys()) < 0:
            zero_support_count += 1
            failures.append(f"Row {i}: PMF support below 0")

        probs = list(pmf.values())

        # Non-finite
        if any(not np.isfinite(p) for p in probs):
            nonfinite_count += 1
            invalid_pmf_count += 1
            continue

        # Non-negative
        if any(p < -1e-9 for p in probs):
            neg_prob_count += 1
            invalid_pmf_count += 1

        # Sum to 1
        total = sum(probs)
        err = abs(total - 1.0)
        max_sum_error = max(max_sum_error, err)
        if err > 1e-6:
            invalid_pmf_count += 1

        min_prob_global = min(min_prob_global, min(probs))
        max_prob_global = max(max_prob_global, max(probs))

    if invalid_pmf_count > 0:
        failures.append(f"Invalid PMF count: {invalid_pmf_count}")
    if neg_prob_count > 0:
        failures.append(f"PMFs with negative probabilities: {neg_prob_count}")
    if nonfinite_count > 0:
        failures.append(f"PMFs with non-finite probabilities: {nonfinite_count}")

    print(f"  Max PMF sum error: {max_sum_error:.2e}")
    print(f"  Invalid PMF count: {invalid_pmf_count}")
    print(f"  Min probability: {min_prob_global:.6f}")

    # ------------------------------------------------------------------
    # 6. Per-stat summaries
    # ------------------------------------------------------------------
    stat_summaries: dict[str, dict] = {}
    for stat in sorted(df["stat"].unique()):
        sub = df[df["stat"] == stat]
        actual_mean = float(sub["actual_outcome"].mean()) if "actual_outcome" in sub.columns else None
        pmf_mean_avg = float(sub["pmf_mean"].mean()) if "pmf_mean" in sub.columns else None
        p0_avg = float(sub["p0"].mean()) if "p0" in sub.columns else None
        empirical_zero = float((sub["actual_outcome"] == 0).mean()) if "actual_outcome" in sub.columns else None
        stat_summaries[stat] = {
            "n_rows": int(len(sub)),
            "pmf_mean_avg": pmf_mean_avg,
            "actual_mean": actual_mean,
            "mean_delta": round(pmf_mean_avg - actual_mean, 4) if (pmf_mean_avg is not None and actual_mean is not None) else None,
            "p0_avg": p0_avg,
            "empirical_zero_rate": empirical_zero,
        }
        print(f"  {stat}: n={len(sub):,}  pmf_mean={pmf_mean_avg:.3f}  "
              f"actual_mean={actual_mean:.3f}  delta={pmf_mean_avg - actual_mean:.3f}  "
              f"p0={p0_avg:.3f}")

    # ------------------------------------------------------------------
    # 7. pmf_mean / pmf_variance finiteness
    # ------------------------------------------------------------------
    if "pmf_mean" in df.columns and not np.isfinite(df["pmf_mean"].fillna(0)).all():
        failures.append("pmf_mean contains non-finite values")
    if "pmf_variance" in df.columns and not np.isfinite(df["pmf_variance"].fillna(0)).all():
        failures.append("pmf_variance contains non-finite values")

    # ------------------------------------------------------------------
    # 8. Row count completeness
    # ------------------------------------------------------------------
    stat_counts = df.groupby("stat").size().to_dict()

    # ------------------------------------------------------------------
    # Write audit
    # ------------------------------------------------------------------
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "pmf_row_count": int(len(df)),
        "pmf_rows_by_stat": {k: int(v) for k, v in stat_counts.items()},
        "duplicate_key_count": int(dup_count),
        "invalid_pmf_count": invalid_pmf_count,
        "empty_pmf_count": empty_pmf_count,
        "neg_prob_count": neg_prob_count,
        "nonfinite_count": nonfinite_count,
        "max_pmf_sum_error": max_sum_error,
        "min_pmf_probability": float(min_prob_global),
        "max_pmf_probability": float(max_prob_global),
        "is_calibrated_check": "PASS" if "is_calibrated" in df.columns and (df["is_calibrated"] != False).sum() == 0 else "FAIL",  # noqa: E712
        "model_only_source_check": "PASS" if "pmf_source" in df.columns and (df["pmf_source"] != REQUIRED_PMF_SOURCE).sum() == 0 else "FAIL",
        "stat_summaries": stat_summaries,
        "support_cap_by_stat": df.groupby("stat")["pmf_support_max"].first().to_dict() if "pmf_support_max" in df.columns else {},
        "failures": failures,
        "warnings": warnings,
    }
    audit_out.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nAudit written: {audit_out}")

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------
    if failures:
        print("\n" + "=" * 70)
        print("VALIDATION FAILED")
        for f in failures:
            print(f"  FAIL: {f}")
        print("=" * 70)
        raise typer.Exit(1)

    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")

    print("\n" + "=" * 70)
    print("PMF Validation PASSED")
    print(f"  {len(df):,} PMF rows across {len(stat_counts)} stats")
    print(f"  Max sum error: {max_sum_error:.2e}")
    print(f"  All PMFs: is_calibrated=False, source='{REQUIRED_PMF_SOURCE}'")
    print("=" * 70)


if __name__ == "__main__":
    app()
