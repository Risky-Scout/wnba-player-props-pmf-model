#!/usr/bin/env python3
"""Strict walk-forward OOF for Opportunity V2 (directive section 27).

Expanding-window chronological folds. Within each fold every submodel is fit ONLY on rows strictly
before the validation window; the bundle then predicts active PMFs for validation rows. Per-fold
immutable checkpoints are written with input hashes, and PMF/temporal invariants are enforced before
each checkpoint. No prior-only fallback, no P0 fallback, no silent exception handling.

Usage:
  python scripts/build_opportunity_oof.py --box data/processed/wnba_player_game_stats.parquet \
      --games data/processed/wnba_games.parquet \
      --quotes artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet \
      --out data/oof/opportunity_v2/oof_pmfs.parquet --props fg3m pts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wnba_props_model.opportunity.bundle import OpportunityModelBundleV2
from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)

# Frozen weekly fold boundaries aligned to the P0 baseline OOF (2026 season).
DEFAULT_FOLDS = [
    ("2026-05-08", "2026-05-14"), ("2026-05-15", "2026-05-21"), ("2026-05-22", "2026-05-28"),
    ("2026-05-29", "2026-06-04"), ("2026-06-05", "2026-06-11"), ("2026-06-12", "2026-06-18"),
    ("2026-06-19", "2026-06-25"), ("2026-06-26", "2026-07-02"), ("2026-07-03", "2026-07-09"),
    ("2026-07-10", "2026-07-16"), ("2026-07-17", "2026-07-23"), ("2026-07-24", "2026-07-30"),
]


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Deterministic prediction-cutoff policy (owner directive section 2).
# The games table has no scheduled tip timestamp, so a certified "scheduled tip minus lead"
# policy is impossible; game-date midnight is explicitly NOT a valid certified substitute.
# We therefore freeze ONE deterministic decision snapshot per game_id: the MAX quote
# decision_timestamp across ALL of that game's quote rows (independent of prop, book, and line).
# This removes the arbitrary drop_duplicates(game_id, player_id) timestamp selection.
CUTOFF_POLICY_ID = "deterministic_max_decision_snapshot_per_game_v1"

# Source files whose bytes determine a fold's computation (checkpoint hash-safety, section 3).
_CODE_FILES = [
    "scripts/build_opportunity_oof.py",
    "src/wnba_props_model/opportunity/bundle.py",
    "src/wnba_props_model/opportunity/feature_builder.py",
    "src/wnba_props_model/opportunity/team_environment.py",
    "src/wnba_props_model/opportunity/share_model.py",
    "src/wnba_props_model/opportunity/component_models.py",
    "src/wnba_props_model/opportunity/pmf_builders.py",
    "src/wnba_props_model/opportunity/pts_decomposition.py",
]


def _code_sha(repo: Path) -> str:
    h = hashlib.sha256()
    for rel in _CODE_FILES:
        p = repo / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"MISSING")
    return h.hexdigest()


def _build_deterministic_cutoffs(q: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One deterministic decision cutoff per game, broadcast to all players in that game.

    Returns (quote_cutoffs[game_id, player_id, quote_timestamp], policy_metadata).
    """
    ts_col = "decision_timestamp" if "decision_timestamp" in q.columns else "pair_timestamp"
    qc = q[["game_id", ts_col]].copy()
    qc[ts_col] = pd.to_datetime(qc[ts_col], utc=True, errors="coerce")
    per_game = (qc.dropna().groupby("game_id")[ts_col].max().reset_index()
                .rename(columns={ts_col: "game_cutoff_utc"}))
    players = box[["game_id", "player_id"]].dropna().drop_duplicates()
    players["game_id"] = pd.to_numeric(players["game_id"], errors="coerce")
    cut = players.merge(per_game, on="game_id", how="inner").rename(
        columns={"game_cutoff_utc": "quote_timestamp"})
    policy = {
        "cutoff_policy_id": CUTOFF_POLICY_ID,
        "cutoff_source": f"quote_{ts_col}_per_game_max",
        "scheduled_tip_available": False,
        "certified_mode": False,
        "reason": ("games table exposes no scheduled tip timestamp; per directive, game-date "
                   "midnight is NOT a certified substitute. Cutoff is one deterministic decision "
                   "snapshot per game, shared by all props/books/lines."),
        "games_with_cutoff": int(per_game.shape[0]),
    }
    return cut, policy


def _verify_checkpoint(manifest_path: Path, expected: dict) -> bool:
    """Hash-safe checkpoint reuse (section 3): only reuse a fold whose manifest matches ALL
    of code SHA, input hashes, config hash, fold dates, and candidate id. Missing/mismatch => recompute."""
    if not manifest_path.exists():
        return False
    try:
        man = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if man.get("code_sha") != expected["code_sha"]:
        return False
    if man.get("input_hashes") != expected["input_hashes"]:
        return False
    if man.get("validation") != expected["validation"]:
        return False
    if man.get("cutoff_policy_id") != expected["cutoff_policy_id"]:
        return False
    if man.get("candidate_id") != expected["candidate_id"]:
        return False
    return True


def _pmf_invariants(pred: pd.DataFrame) -> None:
    for js in pred["active_pmf_json"]:
        arr = np.asarray(json.loads(js), dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr < -1e-9):
            raise ValueError("OOF invariant: non-finite/negative PMF mass")
        if abs(arr.sum() - 1.0) > 1e-6:
            raise ValueError(f"OOF invariant: PMF not normalized (sum={arr.sum()})")
    if pred["active_pmf_mean"].isna().any():
        raise ValueError("OOF invariant: null active_pmf_mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--quotes", default=None, help="exact-quote scored rows (for proof cutoffs)")
    ap.add_argument("--config", default="config/model/opportunity_v2.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--props", nargs="+", default=["fg3m", "pts"])
    ap.add_argument("--candidate", default="OPP_V2_RAW",
                    help="OPP_V2_RAW | OPP_V2_TEAM_SHARE | OPP_V2_PTS_DECOMP")
    ap.add_argument("--pts-recon-labels", default="data/processed/pts_conversion_labels.parquet",
                    help="validated inferred PTS conversion labels (required for OPP_V2_PTS_DECOMP)")
    ap.add_argument("--strict-baseline", action="store_true", default=True)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load(open(args.config))
    box = pd.read_parquet(args.box)
    games = pd.read_parquet(args.games)
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce").dt.tz_localize(None)

    quote_cut = None
    cutoff_policy = {"cutoff_policy_id": "default_lead_fallback", "certified_mode": False}
    if args.quotes and Path(args.quotes).exists():
        q = pd.read_parquet(args.quotes)
        q["game_id"] = pd.to_numeric(q["game_id"], errors="coerce")
        q["player_id"] = pd.to_numeric(q["player_id"], errors="coerce")
        quote_cut, cutoff_policy = _build_deterministic_cutoffs(q, box)

    feat_cfg = OpportunityFeatureConfig(
        default_lead_minutes=cfg["default_lead_minutes"],
        ewma_halflife_games=cfg["history"]["ewma_halflife_games"],
        minimum_history_games=cfg["history"]["minimum_history_games"])

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.out).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {"box": _sha(args.box), "games": _sha(args.games), "config": _sha(args.config)}
    if args.quotes and Path(args.quotes).exists():
        input_hashes["quotes"] = _sha(args.quotes)
    code_sha = _code_sha(repo)

    # PTS full-decomposition candidate consumes validated inferred conversion labels (item 7/8).
    pts_recon = None
    props = list(args.props)
    if args.candidate == "OPP_V2_PTS_DECOMP":
        props = ["pts"]
        rp = Path(args.pts_recon_labels)
        if not rp.exists():
            raise SystemExit(f"OPP_V2_PTS_DECOMP requires --pts-recon-labels; missing {rp}")
        pts_recon = pd.read_parquet(rp)
        input_hashes["pts_recon"] = _sha(rp)

    # persist the frozen cutoff policy alongside the OOF output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cutoff_policy, open(Path(args.out).parent / "CUTOFF_POLICY.json", "w"), indent=2)

    all_pred = []
    for fold_id, (vstart, vend) in enumerate(DEFAULT_FOLDS):
        vstart_ts = pd.Timestamp(vstart, tz="UTC")
        vend_ts = pd.Timestamp(vend, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        ckpt_path = ckpt_dir / f"opp_v2_fold_{fold_id:02d}.parquet"
        manifest_path = ckpt_dir / f"opp_v2_fold_{fold_id:02d}_manifest.json"
        expected = {"code_sha": code_sha, "input_hashes": input_hashes,
                    "validation": [vstart, vend], "cutoff_policy_id": cutoff_policy["cutoff_policy_id"],
                    "candidate_id": args.candidate}
        # Hash-safe reuse: only load a checkpoint whose manifest matches code+data+config+fold+candidate.
        if ckpt_path.exists() and _verify_checkpoint(manifest_path, expected):
            all_pred.append(pd.read_parquet(ckpt_path))
            print(f"[fold {fold_id}] reused verified checkpoint")
            continue
        if ckpt_path.exists():
            print(f"[fold {fold_id}] checkpoint present but manifest MISMATCH -> recomputing")

        # Features built on the FULL history (all lags are strictly prior-game), then split by date.
        frame, _ = build_opportunity_feature_frame(
            box, games, None, None, None, None, None, quote_cut, feat_cfg)
        train = frame[frame["game_date"] < vstart_ts]
        train = train[train["player_games_played_prior"] >= feat_cfg.minimum_history_games]
        val = frame[(frame["game_date"] >= vstart_ts) & (frame["game_date"] <= vend_ts)]
        val = val[val["player_games_played_prior"] >= feat_cfg.minimum_history_games]
        if len(train) < 200 or len(val) == 0:
            continue

        bundle = OpportunityModelBundleV2(cfg).fit(train, train, pts_recon_labels=pts_recon)
        pred = bundle.predict_active_pmfs(val, val, props, candidate=args.candidate)
        pred["fold_id"] = fold_id
        pred["oof_fold"] = fold_id
        pred["fold_validation_start_date"] = vstart
        pred["fold_validation_end_date"] = vend
        pred["cutoff_policy_id"] = cutoff_policy["cutoff_policy_id"]
        # attach actual outcomes for information-gate scoring (single clean merge, no collisions)
        act_cols = ["game_id", "player_id", "did_play", "minutes"] + \
                   [p for p in ("fg3m", "pts") if p in val.columns]
        act = val[act_cols].drop_duplicates(["game_id", "player_id"])
        pred = pred.merge(act, on=["game_id", "player_id"], how="left")
        pred["actual_outcome"] = np.select(
            [pred["stat"] == p for p in ("fg3m", "pts")],
            [pred.get("fg3m"), pred.get("pts")], default=np.nan)
        pred["actual_minutes"] = pred["minutes"]

        _pmf_invariants(pred)
        pred.to_parquet(ckpt_path, index=False)
        manifest_path.write_text(json.dumps({
            "fold_id": fold_id, "validation": [vstart, vend],
            "train_rows": int(len(train)), "val_rows": int(len(val)),
            "pred_rows": int(len(pred)), "input_hashes": input_hashes,
            "code_sha": code_sha, "candidate_id": args.candidate,
            "cutoff_policy_id": cutoff_policy["cutoff_policy_id"],
            "model_bundle_hash": bundle.model_bundle_hash,
            "checkpoint_sha256": _sha(ckpt_path),
        }, indent=2))
        all_pred.append(pred)
        print(f"[fold {fold_id}] train={len(train)} val={len(val)} pred={len(pred)}")

    if not all_pred:
        raise SystemExit("build_opportunity_oof: no folds produced predictions")
    oof = pd.concat(all_pred, ignore_index=True)
    _pmf_invariants(oof)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.out, index=False)
    print(f"wrote {args.out} rows={len(oof)} props={sorted(oof['stat'].unique())}")


if __name__ == "__main__":
    main()
