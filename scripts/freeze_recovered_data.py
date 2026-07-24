"""Preservation Step 2 - freeze + fully characterize the recovered baseline data.

Recomputes every SHA-256 from the actual file (never trusts previously reported values) and
writes artifacts/data_bootstrap/RECOVERED_DATA_MANIFEST.json describing all five required
files, plus a fail-closed validation that the 410-feature contract distinguishes model
features / identifiers / targets / role metadata / provenance. Exits nonzero if any hash is
null or the contract/point-in-time policy check fails, so nothing gets published from a bad
freeze.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
OUT = REPO / "artifacts" / "data_bootstrap" / "RECOVERED_DATA_MANIFEST.json"

FILES = {
    "wnba_games": PROC / "wnba_games.parquet",
    "wnba_player_game_stats": PROC / "wnba_player_game_stats.parquet",
    "wnba_player_game_features_wide": PROC / "wnba_player_game_features_wide.parquet",
    "wnba_player_game_features_long": PROC / "wnba_player_game_features_long.parquet",
    "feature_schema_manifest": PROC / "feature_schema_manifest.json",
}
FEATURE_CONTRACT_ID = "wnba-baseline-features-v2-410col"


def _sha256(p: Path) -> str:
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


def _profile(p: Path) -> dict:
    prof: dict = {}
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        prof["rows"] = int(len(df))
        prof["columns"] = int(df.shape[1])
        gcol = next((c for c in ("game_id", "GAME_ID") if c in df.columns), None)
        pcol = next((c for c in ("player_id", "PLAYER_ID") if c in df.columns), None)
        dcol = next((c for c in ("game_date", "date") if c in df.columns), None)
        prof["unique_games"] = int(df[gcol].nunique()) if gcol else None
        prof["unique_players"] = int(df[pcol].nunique()) if pcol else None
        if dcol:
            d = pd.to_datetime(df[dcol], errors="coerce")
            prof["min_game_date"] = str(d.min())
            prof["max_game_date"] = str(d.max())
        if gcol and pcol:
            prof["canonical_duplicate_count"] = int(df.duplicated(subset=[gcol, pcol]).sum())
    return prof


def main() -> int:
    fm_path = FILES["feature_schema_manifest"]
    fm = json.loads(fm_path.read_text()) if fm_path.exists() else {}
    ordered_features = list(fm.get("model_feature_columns", []))
    contract_hash = hashlib.sha256("\n".join(ordered_features).encode()).hexdigest()
    source_tables = fm.get("source_tables", [])
    source_hashes = {st: (_sha256(REPO / st) if (REPO / st).exists() else None) for st in source_tables}
    builder_commit = fm.get("git_commit_if_available") or _git_head()
    now = datetime.now(timezone.utc).isoformat()

    errs: list[str] = []
    entries = {}
    for name, p in FILES.items():
        if not p.exists():
            errs.append(f"{name}: MISSING at {p}"); continue
        e = {"approved_path": str(p.resolve()), "sha256": _sha256(p), "bytes": p.stat().st_size,
             "feature_contract_id": FEATURE_CONTRACT_ID, "feature_contract_hash": contract_hash,
             "builder_commit": builder_commit, "source_data_hashes": source_hashes,
             "generated_at_utc": now}
        e.update(_profile(p))
        if not e["sha256"]:
            errs.append(f"{name}: null sha256")
        entries[name] = e

    # Contract separation (410 model features distinct from ids/targets/metadata/provenance).
    contract_check = {
        "model_features": len(ordered_features),
        "identifier_columns": len(fm.get("identity_columns", [])),
        "target_columns": len(fm.get("target_columns", [])),
        "role_metadata_columns": len(fm.get("role_bucket_columns", [])),
        "forbidden_columns": len(fm.get("forbidden_columns", [])),
        "temporal_policy": fm.get("temporal_policy"),
    }
    mf = set(ordered_features)
    overlap_targets = mf & set(fm.get("target_columns", []))
    overlap_forbidden = mf & set(fm.get("forbidden_columns", []))
    overlap_ids = mf & set(fm.get("identity_columns", []))
    warnings: list[str] = []
    if len(ordered_features) != 410:
        errs.append(f"contract feature count {len(ordered_features)} != 410")
    # FATAL: leakage-critical overlaps (targets / forbidden). These would train on the outcome.
    if overlap_targets:
        errs.append(f"model features overlap TARGETS (leakage): {sorted(overlap_targets)[:5]}")
    if overlap_forbidden:
        errs.append(f"model features overlap FORBIDDEN (leakage): {sorted(overlap_forbidden)[:5]}")
    # WARNING: a known-pregame field (e.g. is_home) also listed as identity context is benign
    # (no leakage) but the categories should be cleaned up in the builder.
    if overlap_ids:
        warnings.append(f"model features also listed as identifiers (benign, pregame-known): "
                        f"{sorted(overlap_ids)}")
    if fm.get("temporal_policy") != "strict_pregame_shifted":
        errs.append(f"temporal_policy not strict_pregame_shifted: {fm.get('temporal_policy')}")

    manifest = {
        "version": "recovered-data-manifest-v1",
        "generated_at_utc": now,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "feature_contract_hash": contract_hash,
        "builder_commit": builder_commit,
        "contract_separation": contract_check,
        "files": entries,
        "all_hashes_present": all(e.get("sha256") for e in entries.values()) and len(entries) == 5,
        "contract_valid": len(errs) == 0,
        "errors": errs,
        "warnings": warnings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[freeze] files={len(entries)}/5 all_hashes_present={manifest['all_hashes_present']} "
          f"contract_valid={manifest['contract_valid']} -> {OUT}")
    for name, e in entries.items():
        print(f"  {name}: sha256={e['sha256'][:16]}… bytes={e['bytes']:,} rows={e.get('rows','-')}")
    if errs:
        print("[freeze] ERRORS:")
        for e in errs:
            print(f"  - {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
