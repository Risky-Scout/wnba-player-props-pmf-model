#!/usr/bin/env python3
"""Temporal-purity audit for the Opportunity V2 feature frame (directive section 33).

Builds the point-in-time feature frame from canonical inputs and asserts no source timestamp exceeds
the prediction cutoff and no forbidden market column is present. Writes
artifacts/opportunity_v2/FEATURE_POINT_IN_TIME_AUDIT.json and exits non-zero on any violation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wnba_props_model.opportunity.audit import audit_temporal_purity
from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)

OUT = "artifacts/opportunity_v2/FEATURE_POINT_IN_TIME_AUDIT.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--games", default="data/processed/wnba_games.parquet")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    box = pd.read_parquet(args.box)
    games = pd.read_parquet(args.games)
    frame, manifest = build_opportunity_feature_frame(
        box, games, None, None, None, None, None, None, OpportunityFeatureConfig())

    res = audit_temporal_purity(
        frame, "prediction_cutoff_utc", manifest["source_timestamp_columns"],
        feature_columns=manifest["model_feature_columns"])
    payload = res.to_dict()
    payload["feature_manifest"] = manifest
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2, default=str)
    print(f"wrote {args.out} passed={res.passed} violations={res.violation_count} "
          f"forbidden_market={res.forbidden_market_columns}")
    if not res.passed:
        raise SystemExit("temporal purity audit FAILED")


if __name__ == "__main__":
    main()
