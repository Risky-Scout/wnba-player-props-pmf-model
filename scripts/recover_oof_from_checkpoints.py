#!/usr/bin/env python3
"""Reuse the completed OOF fold compute from run 30236023013 (5h51m) — no refit.

The 12 fold checkpoints are VALID model compute; the run only failed at the post-aggregation
active-PMF lineage validator (a pts serialization false positive). This aggregates the checkpoints
into ``oof_player_stat_pmfs.parquet`` and applies the AST/turnover construction correction proven
in RUN_30236023013_CORRECTED_INTEGRITY_AUDIT.json:

  * pts/reb/fg3m/stl/blk : checkpoint active_pmf_json is already correct
    (production _blend_with_dnp(active, p_dnp) reproduces the mixture to <1e-7) — kept as-is.
  * ast/turnover         : checkpoint stored the offset-rebuilt pmf_json at the CONDITIONAL target
    without folding DNP and left active_pmf_json stale. The stored pmf_json is therefore the
    conditional (active) distribution at the correct minutes-adjusted mean. We adopt it as the
    active PMF and fold DNP to form the corrected mixture — exactly the corrected construction
    order now implemented in pmf_utils.apply_minutes_offset_rebuild. This needs no feature/minutes
    data because the conditional target was already applied in the stored pmf_json.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import typer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.simulation import json_to_pmf, pmf_to_json  # noqa: E402

app = typer.Typer(add_completion=False)
OFFSET_STATS = ("ast", "turnover")


def _blend(active: np.ndarray, d: float) -> np.ndarray:
    d = float(min(max(d, 0.0), 0.99))
    mix = active.astype(float).copy()
    mix[0] = d + (1.0 - d) * active[0]
    mix[1:] = (1.0 - d) * active[1:]
    s = mix.sum()
    return mix / s if s > 0 else mix


def _mean(pmf: np.ndarray) -> float:
    return float(np.dot(np.arange(pmf.size, dtype=float), pmf))


@app.command()
def main(
    checkpoint_dir: str = typer.Option(..., "--checkpoint-dir"),
    out: str = typer.Option(..., "--out"),
):
    files = sorted(glob.glob(str(Path(checkpoint_dir) / "fold_*.parquet")))
    if not files:
        raise SystemExit(f"no fold_*.parquet in {checkpoint_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["p_dnp"] = df["p_dnp"].fillna(0.0).astype(float)
    print(f"loaded {len(files)} folds, {len(df):,} rows")

    n_fixed = 0
    for i in df.index[df["stat"].isin(OFFSET_STATS)]:
        d = float(df.at[i, "p_dnp"])
        # Stored pmf_json is the conditional (active) distribution at the minutes-adjusted mean.
        active = json_to_pmf(df.at[i, "pmf_json"]).astype(float)
        s = active.sum()
        if s > 0:
            active = active / s
        mixture = _blend(active, d)
        df.at[i, "active_pmf_json"] = pmf_to_json(active)
        df.at[i, "active_pmf_mean"] = _mean(active)
        df.at[i, "pmf_json"] = pmf_to_json(mixture)
        df.at[i, "availability_mixture_pmf_json"] = pmf_to_json(mixture)
        df.at[i, "pmf_mean"] = _mean(mixture)
        n_fixed += 1
    print(f"corrected ast/turnover construction on {n_fixed:,} rows (active + p_dnp == mixture)")

    # Verify the corrected invariant for every row (production-equivalent blend).
    bad = 0
    for _, r in df.iterrows():
        a = json_to_pmf(r["active_pmf_json"]).astype(float)
        m = json_to_pmf(r["pmf_json"]).astype(float)
        rb = _blend(a, float(r["p_dnp"]))
        n = max(rb.size, m.size); x = np.zeros(n); y = np.zeros(n); x[:rb.size]=rb; y[:m.size]=m
        if np.max(np.abs(x - y)) > 1e-6:
            bad += 1
    print(f"post-correction active⊕p_dnp==mixture violations (>1e-6): {bad}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")


if __name__ == "__main__":
    app()
