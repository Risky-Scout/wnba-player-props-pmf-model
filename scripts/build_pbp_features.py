#!/usr/bin/env python3
"""Build strictly-lagged PBP opportunity features and run the leakage guard (step C).

Usage::

  python3 scripts/build_pbp_features.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wnba_props_model.data.pbp_features import (
    PBPFeatureConfig,
    assert_no_leakage,
    build_pbp_features,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="data/processed/wnba_pbp_player_game_counts.parquet")
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--out", default="data/processed/wnba_pbp_opportunity_features.parquet")
    ap.add_argument("--audit-out", default="artifacts/opportunity_v2/PBP_FEATURE_LEAKAGE_AUDIT.json")
    ap.add_argument("--halflife", type=float, default=6.0)
    args = ap.parse_args()

    parsed = pd.read_parquet(args.parsed)
    box = pd.read_parquet(args.box)
    cfg = PBPFeatureConfig(ewma_halflife_games=args.halflife)

    feats = build_pbp_features(parsed, box, cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)

    audit = assert_no_leakage(parsed, box, cfg, n_spot_checks=300)
    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_out).write_text(json.dumps(audit, indent=2, default=float))

    feat_cols = [c for c in feats.columns if c.startswith("player_") and c != "player_id"]
    print(json.dumps({
        "feature_rows": int(len(feats)),
        "players": int(feats["player_id"].nunique()),
        "games": int(feats["game_id"].nunique()),
        "n_feature_cols": len(feat_cols),
        "feature_cols": feat_cols,
        "leakage_guard": audit,
    }, indent=2))


if __name__ == "__main__":
    main()
