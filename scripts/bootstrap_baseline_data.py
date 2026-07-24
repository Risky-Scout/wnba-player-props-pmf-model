"""Phase 3 - recover / rebuild / validate / publish / re-fetch the datasets W1 needs.

W1 (baseline OOF) requires canonical inputs (wnba_games, wnba_player_game_stats) and the
generated feature matrices (wnba_player_game_features_wide/long + feature_schema_manifest).
This tool inventories ONLY approved project locations, reports each candidate precisely, and
either (a) rebuilds features from canonical source data when present, or (b) emits an EXACT
owner blocker report when canonical inputs cannot be found or rebuilt.

It never publishes the old invalid 52-of-128 matrix as a complete matrix, and a feature
dataset may be published only when it exactly satisfies its declared, versioned contract.

Subcommands:
  inventory   scan approved locations, write BASELINE_DATA_INVENTORY.{json,md} + blocker report
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import typer

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "data_bootstrap"

# Approved locations ONLY (no unrelated personal directories).
APPROVED_DATA_DIRS = [REPO / "data" / "processed", REPO / "data" / "oof", REPO / "data" / "raw"]
REGISTRY = REPO / "config" / "data_registry.json"

# Datasets W1 depends on, with their role.
REQUIRED_CANONICAL = ["wnba_games", "wnba_player_game_stats"]
REQUIRED_GENERATED = ["wnba_player_game_features_wide", "wnba_player_game_features_long",
                      "feature_schema_manifest"]
# Props affected when the baseline feature matrix is unavailable.
AFFECTED_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]

app = typer.Typer(add_completion=False)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _profile_parquet(p: Path) -> dict:
    import pandas as pd
    prof = {"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)}
    try:
        df = pd.read_parquet(p)
        prof["rows"] = int(len(df))
        prof["columns"] = list(df.columns)
        prof["n_columns"] = int(df.shape[1])
        gcol = next((c for c in ("game_id", "GAME_ID", "gameId") if c in df.columns), None)
        pcol = next((c for c in ("player_id", "PLAYER_ID", "personId") if c in df.columns), None)
        dcol = next((c for c in ("game_date", "date") if c in df.columns), None)
        if gcol:
            prof["unique_games"] = int(df[gcol].nunique())
        if pcol:
            prof["unique_players"] = int(df[pcol].nunique())
        if dcol:
            prof["date_range"] = [str(df[dcol].min()), str(df[dcol].max())]
        if gcol and pcol:
            prof["duplicate_canonical_keys"] = int(df.duplicated(subset=[gcol, pcol]).sum())
    except Exception as exc:  # noqa: BLE001
        prof["error"] = f"unreadable: {exc}"
    return prof


def _registry_entries() -> dict:
    if not REGISTRY.exists():
        return {}
    reg = json.loads(REGISTRY.read_text())
    return reg.get("datasets", reg) if isinstance(reg, dict) else {}


@app.command()
def inventory() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = _registry_entries()
    found: dict[str, dict] = {}
    # Approved local files.
    for d in APPROVED_DATA_DIRS:
        if d.exists():
            for p in sorted(d.glob("*.parquet")):
                found[p.stem] = _profile_parquet(p)
    # Registry locators (published assets).
    registry_report = {}
    for name, meta in reg.items():
        registry_report[name] = {
            "path": meta.get("path"), "sha256": meta.get("sha256"),
            "published": meta.get("sha256") is not None,
            "release": meta.get("release_tag") or meta.get("release"),
        }

    def _status(name: str) -> str:
        stem = Path(reg.get(name, {}).get("path", name)).stem if name in reg else name
        if stem in found and "error" not in found[stem]:
            return "PRESENT_LOCAL"
        if reg.get(name, {}).get("sha256"):
            return "PUBLISHED_FETCHABLE"
        return "MISSING"

    required = {}
    for name in REQUIRED_CANONICAL + REQUIRED_GENERATED:
        required[name] = {"role": ("canonical" if name in REQUIRED_CANONICAL else "generated"),
                          "status": _status(name),
                          "registry": registry_report.get(name)}

    # Exact owner blocker report (3.4) for anything MISSING.
    blockers = []
    for name, meta in required.items():
        if meta["status"] == "MISSING":
            blockers.append({
                "dataset": name,
                "expected_path": f"data/processed/{name}.parquet"
                                 if name != "feature_schema_manifest"
                                 else "artifacts/models/stage4_baseline/feature_manifest.json",
                "role": meta["role"],
                "registry_entry_present": name in reg,
                "registry_hash_present": bool(reg.get(name, {}).get("sha256")),
                "affected_props": AFFECTED_PROPS if meta["role"] != "canonical" else "ALL (features cannot be rebuilt)",
                "api_redownload_possible": name in ("wnba_games", "wnba_player_game_stats"),
                "owner_action": (
                    "Provide/publish canonical source via scripts/publish_data.py, or run the "
                    "ingestion (BDL/nba_api) in an env with network+credentials, then re-fetch."),
            })

    inv = {
        "version": "baseline-data-inventory-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "approved_locations": [str(d) for d in APPROVED_DATA_DIRS],
        "local_parquet_found": found,
        "registry": registry_report,
        "required_for_w1": required,
        "blockers": blockers,
        "w1_ready": len(blockers) == 0,
        "note": ("A feature matrix may be published only when it exactly satisfies its declared "
                 "versioned contract; the old invalid 52-of-128 matrix must NOT be published as "
                 "a complete matrix. A corrected sub-128 baseline needs a NEW contract version."),
    }
    (OUT / "BASELINE_DATA_INVENTORY.json").write_text(json.dumps(inv, indent=2, default=str) + "\n")
    lines = ["# Baseline Data Inventory", "",
             f"- generated: {inv['generated_utc']}", f"- W1 ready: **{inv['w1_ready']}**", "",
             "## Required datasets for W1", "", "| dataset | role | status |", "|---|---|---|"]
    for name, meta in required.items():
        lines.append(f"| {name} | {meta['role']} | `{meta['status']}` |")
    if blockers:
        lines += ["", "## Owner blockers (exact)", ""]
        for b in blockers:
            lines.append(f"- **{b['dataset']}** ({b['role']}): expected `{b['expected_path']}`; "
                         f"registry_entry={b['registry_entry_present']}, "
                         f"api_redownload_possible={b['api_redownload_possible']}; "
                         f"affected_props={b['affected_props']}")
    (OUT / "BASELINE_DATA_INVENTORY.md").write_text("\n".join(lines) + "\n")

    print(f"[bootstrap] w1_ready={inv['w1_ready']} blockers={len(blockers)} -> {OUT}")
    for b in blockers:
        print(f"  BLOCKER: {b['dataset']} ({b['role']}) status=MISSING "
              f"api_redownload={b['api_redownload_possible']}")


if __name__ == "__main__":
    app()
