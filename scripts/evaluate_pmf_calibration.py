#!/usr/bin/env python3
"""PMF Calibration Diagnostic Script — WNBA Player Props Model.

Evaluates how well the model computes P(stat = i) for every i in the
distribution.  Metrics: PIT (KS), ECE, Brier, LogLoss — broken down
by stat, role, and PMF step (if multiple PMF columns exist).

Usage:
    python3 scripts/evaluate_pmf_calibration.py
    python3 scripts/evaluate_pmf_calibration.py --stat pts
    python3 scripts/evaluate_pmf_calibration.py --oof-dir artifacts/oof
    python3 scripts/evaluate_pmf_calibration.py --out artifacts/audits/cal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Representative betting lines per stat
# ---------------------------------------------------------------------------
STAT_LINES: dict[str, list[float]] = {
    "pts":      [5.5, 8.5, 10.5, 12.5, 15.5, 20.5],
    "reb":      [2.5, 4.5, 6.5, 8.5],
    "ast":      [1.5, 2.5, 3.5, 5.5],
    "fg3m":     [0.5, 1.5, 2.5, 3.5],
    "stl":      [0.5, 1.5],
    "blk":      [0.5, 1.5],
    "turnover": [0.5, 1.5, 2.5],
}

ECE_BINS = 10
PIT_BINS = 20

# ---------------------------------------------------------------------------
# PMF utilities
# ---------------------------------------------------------------------------

def parse_pmf(s) -> np.ndarray:
    """Parse pmf_json column → normalised probability array (index = count)."""
    if isinstance(s, (list, np.ndarray)):
        arr = np.array(s, dtype=float)
    else:
        d = json.loads(s)
        if isinstance(d, list):
            arr = np.array(d, dtype=float)
        else:
            max_k = max(int(k) for k in d.keys())
            arr = np.zeros(max_k + 1)
            for k, v in d.items():
                arr[int(k)] = float(v)
    total = arr.sum()
    if total > 0:
        arr = arr / total
    return arr


def compute_pit(pmf: np.ndarray, actual: int) -> float:
    """Discrete PIT (midpoint convention).  Uniform → perfect calibration."""
    k = int(round(actual))
    cdf = np.cumsum(pmf)
    if k <= 0:
        return float(pmf[0]) / 2.0
    k = min(k, len(pmf) - 1)
    lo = float(cdf[k - 1]) if k > 0 else 0.0
    hi = float(cdf[k])
    return (lo + hi) / 2.0


def p_over_line(pmf: np.ndarray, line: float) -> float:
    """P(stat > line) from PMF array."""
    threshold = int(np.floor(line)) + 1  # first integer strictly above line
    if threshold >= len(pmf):
        return 0.0
    return float(pmf[threshold:].sum())


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------

def compute_pit_metrics(pits: np.ndarray) -> dict:
    """KS test vs Uniform[0,1], mean, std."""
    if len(pits) < 10:
        return {"ks_stat": np.nan, "ks_pvalue": np.nan, "mean": np.nan, "std": np.nan}
    ks_stat, ks_p = scipy_stats.kstest(pits, "uniform")
    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "mean": float(np.mean(pits)),
        "std": float(np.std(pits)),
    }


def compute_ece(model_ps: np.ndarray, actuals: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error for binary over/under."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(model_ps)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (model_ps >= lo) & (model_ps < hi)
        if mask.sum() == 0:
            continue
        bin_p = model_ps[mask].mean()
        bin_a = actuals[mask].mean()
        ece += (mask.sum() / n) * abs(bin_p - bin_a)
    return float(ece)


def compute_p_over_accuracy_table(
    model_ps: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Bucket rows by predicted P(over line) and measure actual hit rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (model_ps >= lo) & (model_ps < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred_mean = float(model_ps[mask].mean())
        actual_freq = float(actuals[mask].mean())
        diff = actual_freq - pred_mean
        cal_label = "✓ calibrated" if abs(diff) < 0.05 else "✗ model over-confident" if diff < 0 else "✗ model under-confident"
        rows.append({
            "bucket": f"{lo*100:.0f}-{hi*100:.0f}%",
            "pred_mean": pred_mean,
            "actual_freq": actual_freq,
            "n": n,
            "diff": diff,
            "label": cal_label,
        })
    return rows


def stat_level_metrics(df: pd.DataFrame, pmf_col: str = "pmf_json") -> pd.DataFrame:
    """Compute PIT, ECE, Brier, LogLoss for each stat."""
    records = []
    for stat, grp in df.groupby("stat"):
        pmfs = grp[pmf_col].apply(parse_pmf)
        actuals = grp["actual_outcome"].values
        n = len(grp)

        # PIT
        pits = np.array([compute_pit(p, a) for p, a in zip(pmfs, actuals)])
        pit_m = compute_pit_metrics(pits)

        # Brier + LogLoss + ECE — aggregate over all lines for this stat
        lines = STAT_LINES.get(stat, [])
        briers, loglosses, eces = [], [], []
        for line in lines:
            model_ps = pmfs.apply(lambda p, l=line: p_over_line(p, l)).values
            act_over = (actuals > line).astype(float)
            briers.append(float(np.mean((model_ps - act_over) ** 2)))
            eps = 1e-7
            loglosses.append(float(-np.mean(
                act_over * np.log(np.clip(model_ps, eps, 1 - eps)) +
                (1 - act_over) * np.log(np.clip(1 - model_ps, eps, 1 - eps))
            )))
            eces.append(compute_ece(model_ps, act_over))

        records.append({
            "stat": stat,
            "n": n,
            "pit_mean": pit_m["mean"],
            "pit_ks": pit_m["ks_stat"],
            "pit_p": pit_m["ks_pvalue"],
            "ece": np.mean(eces) if eces else np.nan,
            "brier": np.mean(briers) if briers else np.nan,
            "logloss": np.mean(loglosses) if loglosses else np.nan,
        })
    return pd.DataFrame(records)


def role_breakdown(df: pd.DataFrame, stat: str, role_col: str, pmf_col: str = "pmf_json") -> pd.DataFrame:
    """Per-role PIT breakdown for a given stat."""
    sub = df[df["stat"] == stat].copy()
    records = []
    for role, grp in sub.groupby(role_col):
        pmfs = grp[pmf_col].apply(parse_pmf)
        actuals = grp["actual_outcome"].values
        pits = np.array([compute_pit(p, a) for p, a in zip(pmfs, actuals)])
        pit_m = compute_pit_metrics(pits)
        pit_mean = pit_m["mean"]
        direction = "well calibrated"
        if not np.isnan(pit_mean):
            diff_pp = (pit_mean - 0.5) * 100
            if diff_pp > 2:
                direction = f"model UNDER-predicts by {diff_pp:.1f}pp"
            elif diff_pp < -2:
                direction = f"model OVER-predicts by {abs(diff_pp):.1f}pp"
        records.append({
            "role": role,
            "n": len(grp),
            "pit_mean": pit_mean,
            "pit_ks": pit_m["ks_stat"],
            "pit_p": pit_m["ks_pvalue"],
            "direction": direction,
        })
    return pd.DataFrame(records)


def p_over_table_for_stat(df: pd.DataFrame, stat: str, pmf_col: str = "pmf_json") -> dict[float, list[dict]]:
    """P(over line) accuracy tables for all representative lines of a stat."""
    sub = df[df["stat"] == stat]
    pmfs = sub[pmf_col].apply(parse_pmf)
    actuals = sub["actual_outcome"].values
    result = {}
    for line in STAT_LINES.get(stat, []):
        model_ps = pmfs.apply(lambda p, l=line: p_over_line(p, l)).values
        act_over = (actuals > line).astype(float)
        result[line] = compute_p_over_accuracy_table(model_ps, act_over)
    return result


def step_decomposition(df: pd.DataFrame, pmf_cols: list[str]) -> pd.DataFrame:
    """If multiple PMF columns exist, compare PIT and ECE across steps."""
    records = []
    for col in pmf_cols:
        for stat, grp in df.groupby("stat"):
            pmfs = grp[col].apply(parse_pmf)
            actuals = grp["actual_outcome"].values
            pits = np.array([compute_pit(p, a) for p, a in zip(pmfs, actuals)])
            pit_m = compute_pit_metrics(pits)
            lines = STAT_LINES.get(stat, [])
            eces = []
            for line in lines:
                model_ps = pmfs.apply(lambda p, l=line: p_over_line(p, l)).values
                act_over = (actuals > line).astype(float)
                eces.append(compute_ece(model_ps, act_over))
            records.append({
                "pmf_col": col,
                "stat": stat,
                "pit_mean": pit_m["mean"],
                "pit_ks": pit_m["ks_stat"],
                "ece": np.mean(eces) if eces else np.nan,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# OOF data loading
# ---------------------------------------------------------------------------

OOF_SEARCH_DIRS = ["data/oof", "artifacts/oof"]
OOF_FILENAME_PATTERNS = [
    "oof_player_stat_pmfs.parquet",
    "oof_predictions*.parquet",
    "full_oof*.parquet",
    "oof_pmfs*.parquet",
]


def find_oof_file(oof_dir: str | None = None) -> Path | None:
    search_dirs = [oof_dir] if oof_dir else OOF_SEARCH_DIRS
    for d in search_dirs:
        p = REPO_ROOT / d
        if not p.exists():
            continue
        for pat in OOF_FILENAME_PATTERNS:
            matches = sorted(p.glob(pat))
            if matches:
                return matches[-1]  # most recent
    return None


def load_oof(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Canonicalize outcome column
    for col in ["actual_outcome", "actual_value", "y_actual", "stat_value"]:
        if col in df.columns:
            df["actual_outcome"] = df[col]
            break

    # Drop rows with missing actuals
    before = len(df)
    df = df.dropna(subset=["actual_outcome"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with NaN actual_outcome")

    # Keep only played rows for PMF calibration
    if "did_play" in df.columns:
        played = df["did_play"].sum()
        df = df[df["did_play"] == True].copy()
        print(f"  Keeping {len(df):,} played rows (dropped {before - played:,} did_play=False)")

    return df


def try_merge_role(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Try to merge a role column from the features table."""
    feat_path = REPO_ROOT / "data/processed/wnba_player_game_features_long.parquet"
    if not feat_path.exists():
        return df, None

    role_candidates = ["role_status", "projected_minutes_bucket", "rotation_minutes_role"]
    try:
        feat = pd.read_parquet(feat_path, columns=["player_id", "game_id"] + role_candidates)
        merge_cols = [c for c in role_candidates if c in feat.columns]
        if not merge_cols:
            return df, None
        role_col = merge_cols[0]
        feat = feat[["player_id", "game_id", role_col]].drop_duplicates(
            subset=["player_id", "game_id"]
        )
        merged = df.merge(feat, on=["player_id", "game_id"], how="left")
        n_matched = merged[role_col].notna().sum()
        print(f"  Merged role column '{role_col}': {n_matched:,}/{len(merged):,} rows matched")
        return merged, role_col
    except Exception as e:
        print(f"  Warning: could not merge role column: {e}")
        return df, None


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _grade(ks_p: float | None) -> str:
    if ks_p is None or np.isnan(ks_p):
        return "N/A"
    if ks_p >= 0.10:
        return "GOOD"
    if ks_p >= 0.05:
        return "MARGINAL"
    return "POOR"


def print_report(
    df: pd.DataFrame,
    stat_metrics: pd.DataFrame,
    role_col: str | None,
    pmf_col: str = "pmf_json",
    stat_filter: str | None = None,
    role_filter: str | None = None,
) -> dict:
    """Print the human-readable calibration report and return a JSON-serialisable summary."""
    n_rows = len(df)
    today = date.today().isoformat()

    print()
    print("=" * 68)
    print("  WNBA PMF Calibration Report")
    print(f"  Generated: {today}  |  N rows: {n_rows:,}")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # Stat-level summary
    # -----------------------------------------------------------------------
    print()
    print("STAT-LEVEL SUMMARY")
    hdr = f"{'stat':<12}{'N':>8}  {'PIT_mean':>8}  {'PIT_KS':>7}  {'PIT_p':>7}  {'ECE':>7}  {'Brier':>7}  {'LogLoss':>8}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in stat_metrics.iterrows():
        pit_dir = ""
        if not np.isnan(row["pit_mean"]):
            if row["pit_mean"] > 0.52:
                pit_dir = " ↑"
            elif row["pit_mean"] < 0.48:
                pit_dir = " ↓"
        print(
            f"{row['stat']:<12}{int(row['n']):>8}  "
            f"{row['pit_mean']:>8.3f}{pit_dir:<3}"
            f"{row['pit_ks']:>7.4f}  "
            f"{row['pit_p']:>7.4f}  "
            f"{row['ece']*100:>6.1f}%  "
            f"{row['brier']:>7.4f}  "
            f"{row['logloss']:>8.4f}"
        )

    # -----------------------------------------------------------------------
    # Role breakdown
    # -----------------------------------------------------------------------
    summary_roles: dict[str, list] = {}
    stats_to_show = [stat_filter] if stat_filter else stat_metrics["stat"].tolist()

    if role_col:
        for stat in stats_to_show[:3]:  # keep output manageable
            rb = role_breakdown(df, stat, role_col, pmf_col)
            if rb.empty:
                continue
            print()
            print(f"ROLE BREAKDOWN  ({stat}  |  role col: {role_col})")
            hdr2 = f"  {'role':<28}{'N':>7}  {'PIT_mean':>8}  {'direction'}"
            print(hdr2)
            print("  " + "-" * (len(hdr2) - 2))
            for _, r in rb.iterrows():
                print(f"  {str(r['role']):<28}{int(r['n']):>7}  {r['pit_mean']:>8.3f}  {r['direction']}")
            summary_roles[stat] = rb.to_dict(orient="records")

    # -----------------------------------------------------------------------
    # P(over line) accuracy tables
    # -----------------------------------------------------------------------
    summary_pover: dict[str, dict] = {}
    for stat in stats_to_show:
        if role_filter:
            if role_col and role_col in df.columns:
                sub = df[df[role_col] == role_filter]
            else:
                sub = df
        else:
            sub = df
        tables = p_over_table_for_stat(sub, stat, pmf_col)
        if not tables:
            continue
        print()
        lines_list = STAT_LINES.get(stat, [])
        display_lines = lines_list[:3]
        print(f"P(over line) ACCURACY — {stat}  (lines: {', '.join(str(l) for l in display_lines)})")
        summary_pover[stat] = {}
        for line in display_lines:
            tbl = tables.get(line, [])
            if not tbl:
                continue
            print(f"  line = {line}")
            print(f"    {'bucket':<12}  {'pred%':>7}  {'actual%':>8}  {'n':>6}  label")
            print("    " + "-" * 50)
            for r in tbl:
                print(
                    f"    {r['bucket']:<12}  {r['pred_mean']*100:>6.1f}%  "
                    f"{r['actual_freq']*100:>7.1f}%  {r['n']:>6}  {r['label']}"
                )
            summary_pover[stat][str(line)] = tbl

    # -----------------------------------------------------------------------
    # Step decomposition (if multiple PMF cols)
    # -----------------------------------------------------------------------
    _step_known = {"pmf_raw", "pmf_calibrated", "pmf_shrunk", "pmf_uncalibrated"}
    pmf_cols = [c for c in df.columns if c.endswith("_json") or c in _step_known]
    if len(pmf_cols) > 1:
        print()
        print("STEP DECOMPOSITION  (PIT and ECE across PMF pipeline stages)")
        step_df = step_decomposition(df, pmf_cols)
        for col in pmf_cols:
            sub = step_df[step_df["pmf_col"] == col]
            avg_ece = sub["ece"].mean() * 100
            avg_pit = sub["pit_mean"].mean()
            print(f"  {col:<30}  avg_pit_mean={avg_pit:.3f}  avg_ece={avg_ece:.2f}%")

    # -----------------------------------------------------------------------
    # Overall grade
    # -----------------------------------------------------------------------
    print()
    print("OVERALL GRADE")
    avg_ks_p = stat_metrics["pit_p"].mean()
    avg_ece = stat_metrics["ece"].mean() * 100
    avg_brier = stat_metrics["brier"].mean()
    pit_grade = _grade(avg_ks_p)
    print(f"  PIT uniformity (avg KS p-value = {avg_ks_p:.4f}): {pit_grade}")
    print(f"  ECE (avg across stats):   {avg_ece:.2f}%")
    print(f"  Brier (avg across stats): {avg_brier:.4f}")
    print()

    # -----------------------------------------------------------------------
    # Build JSON summary
    # -----------------------------------------------------------------------
    summary = {
        "generated": today,
        "n_rows": n_rows,
        "pmf_col": pmf_col,
        "stat_filter": stat_filter,
        "role_filter": role_filter,
        "stat_metrics": stat_metrics.replace({np.nan: None}).to_dict(orient="records"),
        "role_breakdown": summary_roles,
        "p_over_accuracy": summary_pover,
        "overall": {
            "avg_ks_pvalue": float(avg_ks_p) if not np.isnan(avg_ks_p) else None,
            "pit_grade": pit_grade,
            "avg_ece_pct": float(avg_ece),
            "avg_brier": float(avg_brier),
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(
    df: pd.DataFrame,
    stat_filter: str | None = None,
    role_filter: str | None = None,
    pmf_col: str = "pmf_json",
    out: str | None = None,
) -> dict:
    """Run full calibration diagnostics.  Returns JSON-serialisable summary dict."""
    if stat_filter:
        df = df[df["stat"] == stat_filter].copy()
        if df.empty:
            print(f"ERROR: no rows for stat='{stat_filter}'")
            return {}

    if role_filter:
        # Defer to print_report for role filtering (needs role_col)
        pass

    df, role_col = try_merge_role(df)

    # Detect all PMF-like JSON columns (must end with _json or be known step names)
    _known_step_cols = {"pmf_raw", "pmf_calibrated", "pmf_shrunk", "pmf_uncalibrated"}
    pmf_candidates = [
        c for c in df.columns
        if c.endswith("_json") or c in _known_step_cols
    ]
    if pmf_col not in df.columns and pmf_candidates:
        pmf_col = pmf_candidates[0]
        print(f"  Using PMF column: {pmf_col}")

    print(f"  Computing metrics for {len(df):,} rows …")
    stat_metrics = stat_level_metrics(df, pmf_col)

    summary = print_report(
        df=df,
        stat_metrics=stat_metrics,
        role_col=role_col,
        pmf_col=pmf_col,
        stat_filter=stat_filter,
        role_filter=role_filter,
    )

    # Save JSON
    if out:
        out_path = Path(out)
    else:
        audits_dir = REPO_ROOT / "artifacts" / "audits"
        audits_dir.mkdir(parents=True, exist_ok=True)
        out_path = audits_dir / f"calibration_report_{date.today().isoformat()}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"JSON report saved → {out_path.relative_to(REPO_ROOT)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WNBA PMF calibration (PIT, ECE, Brier, LogLoss)."
    )
    parser.add_argument(
        "--oof-dir",
        default=None,
        help="Directory to search for OOF parquet (default: data/oof then artifacts/oof)",
    )
    parser.add_argument(
        "--oof-file",
        default=None,
        help="Direct path to OOF parquet file (overrides --oof-dir search).",
    )
    parser.add_argument("--stat", default=None, help="Filter to one stat (e.g. pts)")
    parser.add_argument("--role", default=None, help="Filter to one role value")
    parser.add_argument(
        "--pmf-col",
        default="pmf_json",
        help="Name of the PMF JSON column (default: pmf_json)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Include calibration_eligible=False rows (default: only eligible rows)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save JSON report (default: artifacts/audits/calibration_report_<date>.json)",
    )
    args = parser.parse_args()

    # Locate OOF file
    if args.oof_file:
        oof_path = Path(args.oof_file)
    else:
        oof_path = find_oof_file(args.oof_dir)

    if oof_path is None or not oof_path.exists():
        print("ERROR: No OOF parquet file found.")
        print("  Searched directories:", OOF_SEARCH_DIRS)
        print("  Use --oof-dir or --oof-file to specify a path.")
        sys.exit(1)

    print(f"Loading OOF data: {oof_path}")
    df = load_oof(oof_path)
    print(f"  {len(df):,} rows loaded  |  stats: {sorted(df['stat'].unique())}")

    # Optionally restrict to calibration-eligible rows
    if not args.all_rows and "calibration_eligible" in df.columns:
        before = len(df)
        df = df[df["calibration_eligible"] == True].copy()
        print(f"  Filtered to calibration_eligible=True: {len(df):,} rows (dropped {before - len(df):,})")

    run_diagnostics(
        df=df,
        stat_filter=args.stat,
        role_filter=args.role,
        pmf_col=args.pmf_col,
        out=args.out,
    )


if __name__ == "__main__":
    main()
