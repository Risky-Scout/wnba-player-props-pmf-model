#!/usr/bin/env python3
"""Fit chronological pick-engine reliability weights from historical OOF rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wnba_props_model.pick_engine.reliability import (
    default_reliability_weights,
    fit_reliability_weights,
    save_reliability_weights,
)


def _load_historical(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    # Normalize scored_candidates_g0v2 schema into pick-engine reliability schema.
    rename = {}
    if "model_prob_over_final" in df.columns and "pure_probability" not in df.columns:
        rename["model_prob_over_final"] = "pure_probability"
    if "market_prob_over_no_vig" in df.columns and "reference_market_probability" not in df.columns:
        rename["market_prob_over_no_vig"] = "reference_market_probability"
    if "prop" in df.columns and "stat" not in df.columns:
        rename["prop"] = "stat"
    if "outcome_over" in df.columns and "outcome_over" not in df.columns:
        pass
    df = df.rename(columns=rename)
    if "outcome_over" not in df.columns and "outcome" in df.columns:
        df["outcome_over"] = df["outcome"]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--historical",
        default="artifacts/market_feature_proof/G0_v2/scored_candidates_g0v2.parquet",
    )
    ap.add_argument(
        "--out",
        default="artifacts/pick_engine/reliability_weights.json",
    )
    args = ap.parse_args()
    hist_path = Path(args.historical)
    if not hist_path.exists():
        w = default_reliability_weights()
        save_reliability_weights(w, args.out)
        print(f"historical missing; wrote defaults -> {args.out} hash={w.weights_hash}")
        return 0
    df = _load_historical(hist_path)
    # Chronological: keep decision/is_primary rows when present.
    if "is_primary" in df.columns:
        df = df[df["is_primary"].astype(bool)]
    if "split" in df.columns:
        # Prefer out-of-fold / validation style splits when labeled.
        oof = df[df["split"].astype(str).str.contains("oof|val|test", case=False, na=False)]
        if len(oof) >= 100:
            df = oof
    need = {"pure_probability", "reference_market_probability", "outcome_over", "game_date", "stat"}
    if not need.issubset(df.columns):
        w = default_reliability_weights()
        save_reliability_weights(w, args.out)
        print(f"schema incomplete; wrote defaults -> {args.out}")
        return 0
    w = fit_reliability_weights(df)
    # Fill unsupported-stat gaps from conservative defaults (partial pool parent).
    defaults = default_reliability_weights().by_stat
    for stat, val in defaults.items():
        w.by_stat.setdefault(stat, val)
    from wnba_props_model.pick_engine.reliability import _hash_weights  # noqa: PLC0415

    w.weights_hash = _hash_weights(w)
    save_reliability_weights(w, args.out)
    print(
        f"fitted reliability weights n={w.n_training_rows} "
        f"global={w.global_weight:.3f} by_stat={w.by_stat} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
