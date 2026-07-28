#!/usr/bin/env python3
"""Pure (no-market) PBP-opportunity walk-forward OOF for FG3M and AST (owner directive step D).

This is a *pure model*: every input is a strictly-lagged PBP feature (built by
``build_pbp_features.py``); no market/line/odds signal ever enters. For each expanding-window fold we
fit only on games strictly before the validation window and emit a proper count PMF per validation
(player, game):

  * FG3M  = 3PA opportunity NB(mean = lagged 3PA/min x lagged minutes)  (x)  Beta(3P%) conversion,
            via the exact-quote-grade ``marginal_beta_binomial_pmf`` used by opportunity_v2.
  * AST   = assist opportunity NB(mean = lagged AST/min x lagged minutes).

The NB dispersion for each opportunity count is fit by method-of-moments on the training fold's
realized counts (falling back to Poisson when data are underdispersed). Output schema matches the
opportunity_v2 OOF (game_id, player_id, prop/stat, active_pmf_json, oof_fold, prediction_cutoff_utc,
actual) so it flows through the frozen ``build_canonical_scored_rows`` evaluator unchanged.

Usage::

  python3 scripts/build_pbp_opportunity_oof.py \
      --features data/processed/wnba_pbp_opportunity_features.parquet \
      --box data/processed/wnba_player_game_stats.parquet \
      --out data/oof/opportunity_v2_pbp/oof_pmfs.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.pmf_builders import (
    marginal_beta_binomial_pmf,
    pmf_mean,
    pmf_variance,
    poisson_or_nbinom_pmf,
)

# Frozen weekly folds identical to build_opportunity_oof.py (2026 season).
DEFAULT_FOLDS = [
    ("2026-05-08", "2026-05-14"), ("2026-05-15", "2026-05-21"), ("2026-05-22", "2026-05-28"),
    ("2026-05-29", "2026-06-04"), ("2026-06-05", "2026-06-11"), ("2026-06-12", "2026-06-18"),
    ("2026-06-19", "2026-06-25"), ("2026-06-26", "2026-07-02"), ("2026-07-03", "2026-07-09"),
    ("2026-07-10", "2026-07-16"), ("2026-07-17", "2026-07-23"), ("2026-07-24", "2026-07-30"),
]
MIN_HISTORY = 3
CONV_STRENGTH = 25.0
TAIL = 1e-8
CAP = 120


def _mom_dispersion(actual: np.ndarray, predicted_mean: np.ndarray) -> float | None:
    """Method-of-moments NB dispersion r from realized counts. None => Poisson (no overdispersion)."""
    actual = np.asarray(actual, float)
    m = float(np.mean(actual))
    v = float(np.var(actual))
    if m <= 1e-9 or v <= m:
        return None  # underdispersed / degenerate -> Poisson
    r = m * m / (v - m)
    return float(np.clip(r, 0.5, 500.0))


def _fg3m_pmf(e3pa: float, pct: float, r_att: float | None) -> np.ndarray:
    att = poisson_or_nbinom_pmf(max(e3pa, 1e-6), r_att, tail_tolerance=TAIL, maximum_cap=CAP)
    alpha = max(pct, 1e-3) * CONV_STRENGTH
    beta = max(1.0 - pct, 1e-3) * CONV_STRENGTH
    return marginal_beta_binomial_pmf(att, alpha, beta)


def _ast_pmf(east: float, r_ast: float | None) -> np.ndarray:
    return poisson_or_nbinom_pmf(max(east, 1e-6), r_ast, tail_tolerance=TAIL, maximum_cap=CAP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/wnba_pbp_opportunity_features.parquet")
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--out", default="data/oof/opportunity_v2_pbp/oof_pmfs.parquet")
    ap.add_argument("--props", nargs="+", default=["fg3m", "ast"])
    args = ap.parse_args()

    feats = pd.read_parquet(args.features)
    box = pd.read_parquet(args.box)
    for c in ("game_id", "player_id"):
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
        box[c] = pd.to_numeric(box[c], errors="coerce")
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce")
    feats["game_date"] = pd.to_datetime(feats["game_date"], errors="coerce")

    out_cols = ["game_id", "player_id", "game_date", "did_play", "minutes",
                "fg3m", "ast", "fg3a"]
    b = box[[c for c in out_cols if c in box.columns]].drop_duplicates(["game_id", "player_id"])
    df = feats.merge(b, on=["game_id", "player_id"], how="left", suffixes=("", "_box"))
    # prefer box outcome columns
    for c in ("did_play", "minutes"):
        bc = f"{c}_box"
        if bc in df.columns:
            df[c] = df[bc].where(df[bc].notna(), df.get(c))
    df = df[df["did_play"].astype("boolean").fillna(False)]
    df = df[df["player_games_played_prior"] >= MIN_HISTORY].copy()

    df["e3pa"] = (df["player_fg3a_per_min_ewma"] * df["player_minutes_ewma"]).clip(lower=0)
    df["east"] = (df["player_ast_per_min_ewma"] * df["player_minutes_ewma"]).clip(lower=0)
    df["pct3"] = df["player_fg3_pct_prior"].clip(0.02, 0.75)

    all_rows = []
    for fold_id, (vstart, vend) in enumerate(DEFAULT_FOLDS):
        vs = pd.Timestamp(vstart)
        ve = pd.Timestamp(vend) + pd.Timedelta(hours=23, minutes=59)
        train = df[df["game_date"] < vs]
        val = df[(df["game_date"] >= vs) & (df["game_date"] <= ve)]
        if len(train) < 100 or len(val) == 0:
            continue
        r_att = _mom_dispersion(train["fg3a"].fillna(0).to_numpy(),
                                train["e3pa"].to_numpy()) if "fg3a" in train else None
        r_ast = _mom_dispersion(train["ast"].fillna(0).to_numpy(), train["east"].to_numpy())

        for _, row in val.iterrows():
            for prop in args.props:
                if prop == "fg3m":
                    pmf = _fg3m_pmf(float(row["e3pa"]), float(row["pct3"]), r_att)
                    actual = row.get("fg3m")
                elif prop == "ast":
                    pmf = _ast_pmf(float(row["east"]), r_ast)
                    actual = row.get("ast")
                else:
                    continue
                all_rows.append({
                    "game_id": int(row["game_id"]), "player_id": int(row["player_id"]),
                    "game_date": row["game_date"], "stat": prop,
                    "candidate_id": "OPP_V2_PBP",
                    "active_pmf_json": json.dumps([round(float(x), 10) for x in pmf]),
                    "active_pmf_mean": pmf_mean(pmf), "active_pmf_variance": pmf_variance(pmf),
                    "oof_fold": fold_id,
                    "prediction_cutoff_utc": f"{vstart}T00:00:00+00:00",
                    "actual": float(actual) if pd.notna(actual) else np.nan,
                })
        print(f"[fold {fold_id}] train={len(train)} val={len(val)} r_att={r_att} r_ast={r_ast}")

    oof = pd.DataFrame(all_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.out, index=False)
    print(f"wrote {args.out} rows={len(oof)} props={sorted(oof['stat'].unique())}")


if __name__ == "__main__":
    main()
