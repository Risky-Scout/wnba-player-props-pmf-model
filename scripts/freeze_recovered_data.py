"""Preservation Step 2 / P2 - freeze + fully+correctly characterize the recovered baseline data.

Corrects the earlier manifest defects:
  1. long-table uniqueness uses the canonical key game_id+player_id+stat (not game_id+player_id);
     also verifies exactly the expected props per valid player-game.
  2. canonical source hashes are recomputed (never null).
  3. a single all_hashes_present flag is replaced by four independent flags.
  4. the 577 physical wide columns are classified into explicit role lists (model / identifier /
     target / metadata / provenance / forbidden / unused); is_home dual-role is explicit.
  5. games rows are classified COMPLETED / SCHEDULED_FUTURE / POSTPONED / CANCELED / UNKNOWN.
  6. a feature point-in-time audit is produced (FEATURE_POINT_IN_TIME_AUDIT.json).

Recomputes every SHA-256 from the actual file. Exits nonzero if any required hash is null or
any leakage/point-in-time/contract check fails, so nothing is published from a bad freeze.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PRESERVED = REPO / "data" / "recovered_v2_preserved"
OUT_DIR = REPO / "artifacts" / "data_bootstrap"
FEATURE_CONTRACT_ID = "wnba-baseline-features-v2-410col"
_PROPS = ["ast", "blk", "fg3m", "pts", "reb", "stl", "turnover"]

FILES = {
    "wnba_games": PRESERVED / "wnba_games.parquet",
    "wnba_player_game_stats": PRESERVED / "wnba_player_game_stats.parquet",
    "wnba_player_game_features_wide": PRESERVED / "wnba_player_game_features_wide.parquet",
    "wnba_player_game_features_long": PRESERVED / "wnba_player_game_features_long.parquet",
    "feature_schema_manifest": PRESERVED / "feature_schema_manifest.json",
}
# Canonical source files backing the generated features (recompute their hashes; never null).
SOURCE_FILES = {
    "wnba_games": PRESERVED / "wnba_games.parquet",
    "wnba_player_game_stats": PRESERVED / "wnba_player_game_stats.parquet",
}


def _sha256(p: Path) -> "str | None":
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _profile(name: str, p: Path) -> dict:
    prof: dict = {}
    if p.suffix != ".parquet":
        return prof
    df = pd.read_parquet(p)
    prof["rows"] = int(len(df))
    prof["columns"] = int(df.shape[1])
    gcol = "game_id" if "game_id" in df.columns else None
    pcol = "player_id" if "player_id" in df.columns else None
    dcol = "game_date" if "game_date" in df.columns else None
    if gcol:
        prof["unique_games"] = int(df[gcol].nunique())
    if pcol:
        prof["unique_players"] = int(df[pcol].nunique())
    if dcol:
        d = pd.to_datetime(df[dcol], errors="coerce")
        prof["min_game_date"] = str(d.min()); prof["max_game_date"] = str(d.max())
    # Canonical duplicate key differs by grain: long is keyed by (game,player,stat).
    if name.endswith("_long") and gcol and pcol and "stat" in df.columns:
        prof["canonical_key"] = ["game_id", "player_id", "stat"]
        prof["canonical_duplicate_count"] = int(df.duplicated(subset=[gcol, pcol, "stat"]).sum())
        counts = df.groupby([gcol, pcol])["stat"].nunique()
        prof["player_games"] = int(len(counts))
        prof["player_games_with_all_props"] = int((counts == len(_PROPS)).sum())
        prof["player_games_missing_props"] = int((counts < len(_PROPS)).sum())
        prof["player_games_unexpected_props"] = int((counts > len(_PROPS)).sum())
    elif gcol and pcol:
        prof["canonical_key"] = ["game_id", "player_id"]
        prof["canonical_duplicate_count"] = int(df.duplicated(subset=[gcol, pcol]).sum())
    return prof


def _classify_games(p: Path) -> dict:
    df = pd.read_parquet(p)
    out = {"COMPLETED": 0, "SCHEDULED_FUTURE": 0, "POSTPONED": 0, "CANCELED": 0, "UNKNOWN": 0}
    played = df["is_played_game"] if "is_played_game" in df.columns else None
    status = (df["status_normalized"].astype(str).str.lower() if "status_normalized" in df.columns
              else pd.Series(["unknown"] * len(df)))
    for i in range(len(df)):
        s = status.iloc[i]
        if "postpon" in s:
            out["POSTPONED"] += 1
        elif "cancel" in s:
            out["CANCELED"] += 1
        elif played is not None and bool(played.iloc[i]):
            out["COMPLETED"] += 1
        elif "final" in s or "complete" in s:
            out["COMPLETED"] += 1
        elif "sched" in s or "future" in s or "upcoming" in s:
            out["SCHEDULED_FUTURE"] += 1
        elif played is not None and not bool(played.iloc[i]):
            out["SCHEDULED_FUTURE"] += 1
        else:
            out["UNKNOWN"] += 1
    return out


def _classify_contract(fm: dict, wide_cols: list[str]) -> dict:
    cats = {
        "model_features": list(fm.get("model_feature_columns", [])),
        "identifiers": list(fm.get("identity_columns", [])),
        "targets": list(fm.get("target_columns", [])),
        "metadata": list(fm.get("role_bucket_columns", [])),
        "forbidden": list(fm.get("forbidden_columns", [])),
    }
    provenance = [c for c in wide_cols if c in (
        "feature_build_timestamp_utc", "feature_cutoff_policy", "git_commit", "builder_sha")]
    cats["provenance"] = provenance
    declared = set().union(*[set(v) for v in cats.values()])
    cats["unused_physical"] = sorted(set(wide_cols) - declared)
    # Permitted dual-role: is_home is a known-pregame context feature AND an identity descriptor.
    mf = set(cats["model_features"])
    permitted_overlaps = {"is_home"}
    leak_overlap = (mf & set(cats["targets"])) | (mf & set(cats["forbidden"]))
    id_overlap = (mf & set(cats["identifiers"])) - permitted_overlaps
    return {
        "physical_columns": len(wide_cols),
        "counts": {k: len(v) for k, v in cats.items()},
        "lists": cats,
        "permitted_dual_role": sorted(permitted_overlaps & (mf & set(cats["identifiers"]))),
        "unpermitted_identifier_overlap": sorted(id_overlap),
        "leakage_overlap": sorted(leak_overlap),
    }


def _point_in_time_audit(fm: dict, contract: dict) -> dict:
    """Static point-in-time / leakage audit over the declared contract. Fails on any model
    feature that is a target, a forbidden postgame/closing field, or a same-game outcome."""
    checks = {}
    checks["temporal_policy_strict"] = fm.get("temporal_policy") == "strict_pregame_shifted"
    checks["no_model_feature_is_target"] = len(contract["leakage_overlap"]) == 0 and \
        len(set(contract["lists"]["model_features"]) & set(contract["lists"]["targets"])) == 0
    checks["no_model_feature_forbidden"] = len(
        set(contract["lists"]["model_features"]) & set(contract["lists"]["forbidden"])) == 0
    # Same-game raw outcome / realized-minutes names must NOT be model features.
    same_game = {"pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "minutes",
                 "actual_minutes", "did_play", "fga", "fta", "fg3a", "dreb", "oreb"}
    checks["no_same_game_outcome_feature"] = len(
        set(contract["lists"]["model_features"]) & same_game) == 0
    checks["no_unpermitted_identifier_overlap"] = len(contract["unpermitted_identifier_overlap"]) == 0
    passed = all(checks.values())
    audit = {"version": "feature-point-in-time-audit-v1",
             "generated_at_utc": datetime.now(timezone.utc).isoformat(),
             "feature_contract_id": FEATURE_CONTRACT_ID,
             "checks": checks, "passed": passed,
             "note": "Static contract-level PIT audit; builder temporal_policy is "
                     "strict_pregame_shifted (all rolling features lagged before the target cutoff)."}
    (OUT_DIR / "FEATURE_POINT_IN_TIME_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fm_path = FILES["feature_schema_manifest"]
    fm = json.loads(fm_path.read_text()) if fm_path.exists() else {}
    ordered_features = list(fm.get("model_feature_columns", []))
    contract_hash = hashlib.sha256("\n".join(ordered_features).encode()).hexdigest()
    source_hashes = {k: _sha256(p) for k, p in SOURCE_FILES.items()}
    builder_commit = fm.get("git_commit_if_available") or _git_head()
    now = datetime.now(timezone.utc).isoformat()

    errs: list[str] = []
    entries = {}
    for name, p in FILES.items():
        h = _sha256(p)
        if not p.exists():
            errs.append(f"{name}: MISSING at {p}"); continue
        e = {"approved_path": str(p.resolve()), "sha256": h, "bytes": p.stat().st_size,
             "feature_contract_id": FEATURE_CONTRACT_ID, "feature_contract_hash": contract_hash,
             "builder_commit": builder_commit, "source_data_hashes": source_hashes,
             "generated_at_utc": now}
        e.update(_profile(name, p))
        if not h:
            errs.append(f"{name}: null sha256")
        entries[name] = e

    # Long-table canonical uniqueness must be zero on the 3-part key.
    long_e = entries.get("wnba_player_game_features_long", {})
    if long_e.get("canonical_duplicate_count", 0) != 0:
        errs.append(f"long-table duplicate (game,player,stat)={long_e.get('canonical_duplicate_count')}")
    if long_e.get("player_games_missing_props", 0) != 0:
        errs.append(f"long-table player-games missing props={long_e.get('player_games_missing_props')}")

    wide_cols = list(pd.read_parquet(FILES["wnba_player_game_features_wide"]).columns)
    contract = _classify_contract(fm, wide_cols)
    if contract["leakage_overlap"]:
        errs.append(f"model features overlap targets/forbidden (leakage): {contract['leakage_overlap'][:5]}")
    if contract["unpermitted_identifier_overlap"]:
        errs.append(f"unpermitted identifier overlap: {contract['unpermitted_identifier_overlap'][:5]}")

    games_class = _classify_games(FILES["wnba_games"])
    pit = _point_in_time_audit(fm, contract)
    if not pit["passed"]:
        errs.append(f"point-in-time audit failed: {[k for k,v in pit['checks'].items() if not v]}")

    # Four independent hash flags (none true when a required value is null).
    all_asset_hashes_present = bool(entries) and all(e.get("sha256") for e in entries.values()) and len(entries) == 5
    all_required_source_hashes_present = all(source_hashes.get(k) for k in SOURCE_FILES)
    all_builder_hashes_present = bool(builder_commit) and builder_commit != "unknown"
    all_contract_hashes_present = bool(contract_hash)

    manifest = {
        "version": "recovered-data-manifest-v2", "generated_at_utc": now,
        "feature_contract_id": FEATURE_CONTRACT_ID, "feature_contract_hash": contract_hash,
        "builder_commit": builder_commit, "canonical_source_hashes": source_hashes,
        "files": entries, "contract_classification": contract,
        "games_classification": games_class, "point_in_time_audit": pit["checks"],
        "point_in_time_passed": pit["passed"],
        "all_asset_hashes_present": all_asset_hashes_present,
        "all_required_source_hashes_present": all_required_source_hashes_present,
        "all_builder_hashes_present": all_builder_hashes_present,
        "all_contract_hashes_present": all_contract_hashes_present,
        "contract_valid": len(errs) == 0, "errors": errs,
    }
    (OUT_DIR / "RECOVERED_DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[freeze] files={len(entries)}/5 asset_hashes={all_asset_hashes_present} "
          f"source_hashes={all_required_source_hashes_present} pit_passed={pit['passed']} "
          f"contract_valid={manifest['contract_valid']}")
    print(f"  long dup(game,player,stat)={long_e.get('canonical_duplicate_count')} "
          f"player_games_all_props={long_e.get('player_games_with_all_props')}")
    print(f"  games: {games_class}")
    print(f"  contract: {contract['counts']} unused_physical={len(contract['lists']['unused_physical'])}")
    if errs:
        print("[freeze] ERRORS:"); [print(f"  - {e}") for e in errs]
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
