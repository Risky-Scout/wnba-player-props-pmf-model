#!/usr/bin/env python3
"""Retrospective pick-engine board from frozen August 1 artifacts (read-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from wnba_props_model.pick_engine.constants import RETROSPECTIVE_LABEL
from wnba_props_model.pick_engine.engine import run_pick_engine, write_pick_engine_delivery
from wnba_props_model.pick_engine.reliability import load_or_fit_reliability_weights

SUPPORTED = {"pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage-dir",
        default="deliveries/pregame_verification/2026-08-01/20260731T223428Z",
    )
    ap.add_argument(
        "--quotes",
        default="data/snapshots/soft_book_quotes/snapshot_date_utc=2026-08-01",
    )
    ap.add_argument(
        "--identity",
        default="tests/fixtures/pick_engine/aug1_player_identity_audit.csv",
    )
    ap.add_argument(
        "--game-audit",
        default="tests/fixtures/pick_engine/aug1_game_audit.csv",
    )
    ap.add_argument(
        "--weights",
        default="artifacts/pick_engine/reliability_weights.json",
    )
    ap.add_argument("--artifact-dir", default="artifacts/pick_engine")
    ap.add_argument("--delivery-root", default="deliveries/pick_engine")
    args = ap.parse_args()

    stage = Path(args.stage_dir)
    pmf_path = stage / "full_pmfs_wide.parquet"
    fair_path = stage / "fair_odds_board.parquet"
    if not pmf_path.exists():
        raise SystemExit(f"frozen PMF path missing (read-only required): {pmf_path}")

    # Integrity fingerprints of frozen inputs (must not be rewritten by this script).
    frozen_hashes = {
        "full_pmfs_wide_sha256": _file_sha(pmf_path),
        "fair_odds_board_sha256": _file_sha(fair_path) if fair_path.exists() else "",
        "stage_dir": str(stage),
    }

    pmfs = pd.read_parquet(pmf_path)
    # Exclude combination markets at the PMF filter layer as well.
    pmfs = pmfs[pmfs["stat"].astype(str).isin(SUPPORTED)].copy()
    fair = pd.read_parquet(fair_path) if fair_path.exists() else None
    if fair is not None:
        fair = fair[fair["stat"].astype(str).isin(SUPPORTED)].copy()

    qpath = Path(args.quotes)
    if qpath.is_dir():
        files = sorted(qpath.rglob("*.parquet"))
        if not files:
            raise SystemExit(f"no quote snapshots under {qpath}")
        quotes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        quotes = pd.read_parquet(qpath)

    # Direct full-game markets only.
    if "market_key" in quotes.columns:
        quotes = quotes[
            ~quotes["market_key"].astype(str).str.contains("q1|first_quarter|combo|points_rebounds", case=False)
        ].copy()
    quotes = quotes[quotes["stat"].astype(str).isin(SUPPORTED | {"fg3m"})].copy()

    identity = pd.read_csv(args.identity) if Path(args.identity).exists() else None
    rejected = set()
    mismatch = set()
    if identity is not None:
        rejected = set(
            identity.loc[
                identity["audit_status"].astype(str).str.upper() == "REJECTED",
                "canonical_player_id",
            ]
        )
        mismatch = set(
            identity.loc[
                identity["reject_reason"].astype(str).str.contains("mismatch", case=False, na=False),
                "canonical_player_id",
            ]
        )

    game_map = {}
    ga = Path(args.game_audit)
    if ga.exists():
        gdf = pd.read_csv(ga)
        for _, r in gdf.iterrows():
            game_map[str(r["odds_api_event_id"])] = {
                "game_id": r.get("bdl_game_id"),
                "scheduled_tip_utc": r.get("scheduled_tip_utc"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
            }

    # Prediction timestamp from frozen stage (do not invent a new pre-tip claim).
    pred_ts = "2026-07-31T22:34:28Z"
    tip_manifest = stage / "slate_manifest.json"
    if tip_manifest.exists():
        try:
            man = json.loads(tip_manifest.read_text())
            pred_ts = man.get("prediction_timestamp") or man.get("generated_at") or pred_ts
        except Exception as exc:  # noqa: BLE001
            print(f"slate_manifest parse skipped: {exc}")

    rel = load_or_fit_reliability_weights(args.weights)
    result = run_pick_engine(
        quotes=quotes,
        pmfs=pmfs,
        fair_odds=fair,
        identity=identity,
        game_map=game_map,
        reliability=rel,
        prediction_timestamp=pred_ts,
        asof_timestamp="2026-08-01T03:30:58Z",
        lineage_hashes={
            "model_hash": "frozen_aug1_local_stage",
            "calibrator_hash": "frozen_aug1_local_stage",
        },
        top_n=10,
        board_label=RETROSPECTIVE_LABEL,
        min_reference_books=2,
        rejected_player_ids=rejected,
        team_mismatch_player_ids=mismatch,
        retrospective=True,
        quote_freshness_hours=18.0,
    )

    art = Path(args.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    replay_csv = art / "AUG1_PICK_ENGINE_REPLAY.csv"
    result.ranked.to_csv(replay_csv, index=False)

    # Verify frozen inputs unchanged.
    post_hashes = {
        "full_pmfs_wide_sha256": _file_sha(pmf_path),
        "fair_odds_board_sha256": _file_sha(fair_path) if fair_path.exists() else "",
    }
    frozen_unmodified = post_hashes == {
        k: frozen_hashes[k] for k in ("full_pmfs_wide_sha256", "fair_odds_board_sha256")
    }

    audit = {
        "label": RETROSPECTIVE_LABEL,
        "not_a_new_pre_tip_prediction": True,
        "frozen_inputs": frozen_hashes,
        "frozen_inputs_unmodified": frozen_unmodified,
        "n_quote_rows": int(len(quotes)),
        "n_valid_candidates": int(len(result.candidates)),
        "n_ranked_selections": int(len(result.ranked)),
        "n_provisional_picks": int(len(result.provisional)),
        "n_abstentions": int(len(result.abstentions)),
        "abstentions_by_reason": result.manifest.get("abstentions_by_reason", {}),
        "excluded": {
            "rejected_identities": sorted(int(x) for x in rejected if pd.notna(x)),
            "team_mismatch_player_ids": sorted(int(x) for x in mismatch if pd.notna(x)),
            "combination_markets": True,
            "rows_without_executable_prices": True,
        },
        "pure_alpha_source": "active_pmf_json -> settled_probabilities_from_pmf",
        "production_source": "fair_odds_board.p_over (separate column)",
        "zero_residual_does_not_suppress_alpha": True,
        "top_ranked": result.ranked.head(10).to_dict(orient="records") if not result.ranked.empty else [],
        "weights_hash": rel.weights_hash,
        "reliability_by_stat": rel.by_stat,
    }
    (art / "AUG1_PICK_ENGINE_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    write_pick_engine_delivery(result, Path(args.delivery_root) / "2026-08-01" / "RETROSPECTIVE")
    print(json.dumps({"replay_csv": str(replay_csv), **{k: audit[k] for k in (
        "n_ranked_selections", "n_provisional_picks", "n_valid_candidates", "frozen_inputs_unmodified"
    )}}, indent=2))
    return 0 if frozen_unmodified else 2


if __name__ == "__main__":
    raise SystemExit(main())
