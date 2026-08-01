#!/usr/bin/env python3
"""Run the WNBA pick engine for a slate and write delivery outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from wnba_props_model.pick_engine.constants import RETROSPECTIVE_LABEL
from wnba_props_model.pick_engine.engine import run_pick_engine, write_pick_engine_delivery
from wnba_props_model.pick_engine.reliability import load_or_fit_reliability_weights


def _load_quotes(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("quotes_*.parquet"))
        if not files:
            files = sorted(path.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_game_map(game_audit: Path | None) -> dict[str, dict[str, Any]]:
    if game_audit is None or not game_audit.exists():
        return {}
    df = pd.read_csv(game_audit)
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        eid = str(r.get("odds_api_event_id") or "")
        if not eid:
            continue
        out[eid] = {
            "game_id": r.get("bdl_game_id"),
            "scheduled_tip_utc": r.get("scheduled_tip_utc"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="Slate date YYYY-MM-DD")
    ap.add_argument("--quotes", required=True, help="Soft-book quotes parquet or directory")
    ap.add_argument("--pmfs", required=True, help="full_pmfs_wide.parquet or projections parquet")
    ap.add_argument("--fair-odds", default="", help="fair_odds_board.parquet")
    ap.add_argument("--identity", default="", help="player identity audit CSV")
    ap.add_argument("--game-audit", default="", help="GAME_AUDIT.csv")
    ap.add_argument("--injuries", default="", help="injuries JSON/CSV")
    ap.add_argument(
        "--weights",
        default="artifacts/pick_engine/reliability_weights.json",
    )
    ap.add_argument("--prediction-timestamp", default="")
    ap.add_argument("--asof-timestamp", default="")
    ap.add_argument("--lineage-hashes", default="", help="JSON file of lineage hashes")
    ap.add_argument("--out-root", default="deliveries/pick_engine")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--board-label", default="")
    ap.add_argument("--retrospective", action="store_true")
    ap.add_argument("--min-reference-books", type=int, default=2)
    args = ap.parse_args()

    quotes = _load_quotes(Path(args.quotes))
    pmfs = pd.read_parquet(args.pmfs)
    fair = pd.read_parquet(args.fair_odds) if args.fair_odds else None
    identity = pd.read_csv(args.identity) if args.identity else None
    game_map = _load_game_map(Path(args.game_audit) if args.game_audit else None)

    injuries = None
    if args.injuries:
        p = Path(args.injuries)
        if p.suffix == ".json":
            raw = json.loads(p.read_text())
            injuries = pd.DataFrame(raw if isinstance(raw, list) else raw.get("players", []))
        else:
            injuries = pd.read_csv(p)

    rejected = set()
    mismatch = set()
    if identity is not None and not identity.empty:
        if "audit_status" in identity.columns:
            rejected = set(
                identity.loc[
                    identity["audit_status"].astype(str).str.upper() == "REJECTED",
                    "canonical_player_id",
                ]
            )
        if "reject_reason" in identity.columns:
            mismatch = set(
                identity.loc[
                    identity["reject_reason"].astype(str).str.contains("mismatch", case=False, na=False),
                    "canonical_player_id",
                ]
            )

    hashes = {}
    if args.lineage_hashes:
        hashes = json.loads(Path(args.lineage_hashes).read_text())

    rel = load_or_fit_reliability_weights(args.weights)
    board_label = RETROSPECTIVE_LABEL if args.retrospective else args.board_label

    result = run_pick_engine(
        quotes=quotes,
        pmfs=pmfs,
        fair_odds=fair,
        identity=identity,
        injuries=injuries,
        game_map=game_map,
        reliability=rel,
        prediction_timestamp=args.prediction_timestamp or None,
        asof_timestamp=args.asof_timestamp or None,
        lineage_hashes=hashes,
        top_n=args.top_n,
        board_label=board_label,
        min_reference_books=args.min_reference_books,
        rejected_player_ids=rejected,
        team_mismatch_player_ids=mismatch,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / args.date / ts
    paths = write_pick_engine_delivery(result, out_dir)
    print(json.dumps({"out_dir": str(out_dir), **{k: v for k, v in paths.items()}, **result.manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
