#!/usr/bin/env python3
"""Build the Opportunity V2 point-in-time feature frame and manifest (directive section 26).

Reads canonical box + games (and optional snapshot/quote inputs), builds the strictly-lagged feature
frame, and writes it plus data/processed/opportunity_v2_feature_manifest.json (with input hashes and
a temporal-audit result). Fails if any forbidden market field appears or temporal audit fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wnba_props_model.opportunity.audit import audit_temporal_purity
from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _opt(path: str | None):
    return pd.read_parquet(path) if path and Path(path).exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="data/processed/wnba_player_game_stats.parquet")
    ap.add_argument("--games", default="data/processed/wnba_games.parquet")
    ap.add_argument("--roster", default="data/processed/wnba_roster_intervals.parquet")
    ap.add_argument("--availability", default="data/snapshots/availability")
    ap.add_argument("--lineups", default="data/snapshots/lineups")
    ap.add_argument("--quotes", default=None)
    ap.add_argument("--out", default="data/processed/opportunity_v2_features.parquet")
    ap.add_argument("--manifest", default="data/processed/opportunity_v2_feature_manifest.json")
    args = ap.parse_args()

    box = pd.read_parquet(args.box)
    games = pd.read_parquet(args.games)
    roster = _opt(args.roster)
    avail = pd.read_parquet(args.availability) if Path(args.availability).exists() else None
    lineups = pd.read_parquet(args.lineups) if Path(args.lineups).exists() else None
    quotes = _opt(args.quotes)

    frame, manifest = build_opportunity_feature_frame(
        box, games, roster, avail, lineups, None, None, quotes, OpportunityFeatureConfig())

    audit = audit_temporal_purity(frame, "prediction_cutoff_utc",
                                  manifest["source_timestamp_columns"],
                                  feature_columns=manifest["model_feature_columns"])
    if manifest["forbidden_market_columns_found"]:
        raise SystemExit(f"forbidden market columns: {manifest['forbidden_market_columns_found']}")
    if not audit.passed:
        raise SystemExit(f"temporal audit failed: {audit.to_dict()}")

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    full = {
        "schema_version": "opportunity_v2_features_v1",
        "build_git_sha": git_sha,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": {"box": _sha(args.box), "games": _sha(args.games)},
        "temporal_audit_passed": audit.passed,
        "temporal_audit": audit.to_dict(),
        **manifest,
    }
    json.dump(full, open(args.manifest, "w"), indent=2, default=str)
    print(f"wrote {args.out} rows={len(frame)}; manifest {args.manifest}; "
          f"proof_eligible={manifest['proof_eligible_row_count']}; audit_passed={audit.passed}")


if __name__ == "__main__":
    main()
