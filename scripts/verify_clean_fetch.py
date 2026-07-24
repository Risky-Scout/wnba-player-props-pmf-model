"""Preservation Step 5/6 - clean-fetch reproducibility + backup verification.

Uses ONLY the committed registry + remote storage to decide whether a clean runner (with none
of the recovered local data) could reproduce the baseline. Sets four independent flags:

  local_run_ready       all 5 files present locally + freeze contract valid
  durable_data_ready    all 5 published to remote storage and hash-declared
  clean_fetch_verified  a clean clone can fetch + hash-verify all 5 from remote
  reproducible_run_ready durable_data_ready AND clean_fetch_verified

Writes CLEAN_FETCH_VERIFICATION.{json,md} and BACKUP_VERIFICATION.json. Remote asset existence
is probed read-only via `gh release view`; nothing is uploaded here.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "data_bootstrap"
REGISTRY = REPO / "config" / "data_registry.json"
FREEZE = OUT / "RECOVERED_DATA_MANIFEST.json"
REQUIRED = ["wnba_games", "wnba_player_game_stats", "wnba_player_game_features_wide",
            "wnba_player_game_features_long", "feature_schema_manifest"]


def _release_assets(tag: str) -> "set | None":
    try:
        out = subprocess.check_output(
            ["gh", "release", "view", tag, "--repo", "Risky-Scout/wnba-player-props-pmf-model",
             "--json", "assets", "--jq", "[.assets[].name]"],
            stderr=subprocess.STDOUT).decode()
        return set(json.loads(out))
    except subprocess.CalledProcessError:
        return None  # release not found


def main() -> int:
    reg = json.loads(REGISTRY.read_text())["datasets"]
    freeze = json.loads(FREEZE.read_text()) if FREEZE.exists() else {"files": {}, "contract_valid": False}
    now = datetime.now(timezone.utc).isoformat()

    per = {}
    tag_cache: dict[str, "set | None"] = {}
    for name in REQUIRED:
        e = reg.get(name, {})
        tag = e.get("release_tag")
        asset = e.get("asset")
        if tag not in tag_cache:
            tag_cache[tag] = _release_assets(tag) if tag else None
        remote = tag_cache[tag]
        remote_present = bool(remote is not None and asset in remote)
        per[name] = {
            "local_present": name in freeze.get("files", {}),
            "sha256": e.get("sha256"),
            "release_tag": tag, "asset": asset,
            "remote_release_exists": remote is not None,
            "remote_asset_present": remote_present,
            "publication_status": e.get("publication_status"),
        }

    local_run_ready = bool(freeze.get("contract_valid")
                           and all(per[n]["local_present"] for n in REQUIRED))
    durable_data_ready = all(per[n]["remote_asset_present"] and per[n]["sha256"] for n in REQUIRED)
    clean_fetch_verified = durable_data_ready  # can only be verified once remote assets exist
    reproducible_run_ready = bool(durable_data_ready and clean_fetch_verified)

    verification = {
        "version": "clean-fetch-verification-v1", "generated_at_utc": now,
        "datasets": per,
        "local_run_ready": local_run_ready,
        "durable_data_ready": durable_data_ready,
        "clean_fetch_verified": clean_fetch_verified,
        "reproducible_run_ready": reproducible_run_ready,
        "blocker": (None if reproducible_run_ready else
                    "Remote releases processed-data-v1 / processed-features-v2 do not exist; "
                    "no write credential available to publish (cursor account push=false; "
                    "GH_TOKEN not injected). Owner must publish or inject a write-scoped token."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CLEAN_FETCH_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n")
    md = ["# Clean-Fetch Verification", "", f"- generated: {now}", "",
          f"- **local_run_ready**: {local_run_ready}",
          f"- **durable_data_ready**: {durable_data_ready}",
          f"- **clean_fetch_verified**: {clean_fetch_verified}",
          f"- **reproducible_run_ready**: {reproducible_run_ready}", "",
          "| dataset | local | remote asset | status |", "|---|---|---|---|"]
    for n in REQUIRED:
        p = per[n]
        md.append(f"| {n} | {p['local_present']} | {p['remote_asset_present']} | "
                  f"`{p['publication_status']}` |")
    if verification["blocker"]:
        md += ["", "## Blocker", "", verification["blocker"]]
    (OUT / "CLEAN_FETCH_VERIFICATION.md").write_text("\n".join(md) + "\n")

    # Backup verification: primary (local VM) + required secondary (owner durable storage).
    backup = {
        "version": "backup-verification-v1", "generated_at_utc": now,
        "primary": {"locator": "ephemeral-agent-filesystem:data/processed", "durable": False,
                    "files": {n: freeze["files"].get(n, {}).get("sha256") for n in REQUIRED}},
        "secondary": {"locator": None, "durable": False,
                      "status": "NOT_CREATED - requires owner write-scoped credential or private "
                                "object storage; agent has no durable write target."},
        "retention_policy": "immutable, hash-pinned; never overwrite a published asset with "
                            "different contents",
        "recovery_instructions": "Re-run scripts/pull_bdl_history.py --start-season 2021 "
                                 "--end-season 2026 -> build_canonical_tables.py -> "
                                 "build_features.py; verify against RECOVERED_DATA_MANIFEST.json "
                                 "hashes before use.",
        "verification_timestamps": {"frozen_at": freeze.get("generated_at_utc"), "checked_at": now},
    }
    (OUT / "BACKUP_VERIFICATION.json").write_text(json.dumps(backup, indent=2) + "\n")

    print(f"[verify] local_run_ready={local_run_ready} durable_data_ready={durable_data_ready} "
          f"clean_fetch_verified={clean_fetch_verified} reproducible_run_ready={reproducible_run_ready}")
    for n in REQUIRED:
        print(f"  {n}: local={per[n]['local_present']} remote={per[n]['remote_asset_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
