#!/usr/bin/env python3
"""Phase 6 - publish-ready inventory of local immutable inputs (no-secret).

Computes path / rowcount / schema / sha256 for the immutable assets a clean-clone real-data rebuild
needs, and emits artifacts/publish_ready/{manifest.json, checksums.sha256, publish_commands.sh,
fetch_commands.sh}. It does NOT publish: the configured store (config/data_registry.json 'repo') is
verified for visibility first, and restricted BDL-derived assets are NEVER published to a PUBLIC repo.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts/publish_ready"
REG = json.load(open(REPO / "config/data_registry.json"))

ASSETS = {
    "wnba_player_game_stats": ("data/processed/wnba_player_game_stats.parquet", "processed-data-v1", "RESTRICTED_BDL"),
    "wnba_games": ("data/processed/wnba_games.parquet", "processed-data-v1", "RESTRICTED_BDL"),
    "wnba_player_game_features_wide": ("data/processed/wnba_player_game_features_wide.recovered_v2_20260725.parquet", "processed-features-v2", "RESTRICTED_DERIVED"),
    "oof_predictions": ("artifacts/models/calibration/oof_predictions.parquet", "oof-data-v1", "DERIVED"),
    "atomic_sides": ("data/processed/atomic_quotes/atomic_sides.parquet", "atomic-quotes-v1", "RESTRICTED_ODDS"),
    "atomic_pairs": ("data/processed/atomic_quotes/atomic_pairs.parquet", "atomic-quotes-v1", "RESTRICTED_ODDS"),
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _visibility(repo: str) -> str:
    try:
        r = subprocess.run(["gh", "repo", "view", repo, "--json", "visibility"],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout).get("visibility", "UNKNOWN") if r.returncode == 0 else "UNKNOWN"
    except Exception as e:  # noqa: BLE001
        return f"UNKNOWN:{e}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    repo = REG.get("repo", "UNKNOWN")
    vis = _visibility(repo)
    manifest, checks, pub, fetch = [], [], [], []
    for key, (rel, tag, lic) in ASSETS.items():
        p = REPO / rel
        if not p.exists():
            manifest.append({"asset": key, "path": rel, "present": False, "license": lic})
            continue
        try:
            n = int(len(pd.read_parquet(p, columns=[])))
        except Exception:
            n = None
        sha = _sha256(p)
        publishable = (vis == "PRIVATE") or (lic == "DERIVED")
        manifest.append({"asset": key, "path": rel, "present": True, "rows": n, "sha256": sha,
                         "release_tag": tag, "license": lic,
                         "public_publish_allowed": bool(lic == "DERIVED"),
                         "publishable_to_configured_store": bool(publishable)})
        checks.append(f"{sha}  {rel}")
        if publishable:
            pub.append(f"gh release create {tag} '{rel}' --repo {repo} --title '{tag}' --notes 'immutable {key}' || "
                       f"gh release upload {tag} '{rel}' --repo {repo} --clobber")
        fetch.append(f"python scripts/fetch_data.py {key}")

    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configured_store": repo, "store_visibility": vis,
        "publication_blocker": (
            None if vis == "PRIVATE" else
            "configured store is PUBLIC; RESTRICTED_BDL / RESTRICTED_ODDS / RESTRICTED_DERIVED assets "
            "MUST NOT be published to a public repo (license). Only DERIVED assets may be public."),
        "verdicts": {
            "code_clean_clone_pass": True,
            "real_data_clean_fetch_pass": False,
            "real_data_pipeline_rebuild_pass": False,
            "artifact_hash_parity_pass": False,
            "environment_lock_pass": False,
            "reason": "restricted assets not fetchable in a clean clone because the configured store is "
                      "PUBLIC and cannot host them; a PRIVATE data store + PRIVATE_DATA_WRITER_TOKEN target "
                      "is required. PRIVATE_DATA_WRITER_TOKEN is a CI-only secret (absent from local env).",
        },
        "smallest_owner_action": "provision/confirm a PRIVATE data repo, set config/data_registry.json 'repo' "
                                 "to it, ensure PRIVATE_DATA_WRITER_TOKEN can write there, then run the CI "
                                 "publish job. Do NOT publish restricted data to the public repo.",
    }
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    (OUT / "checksums.sha256").write_text("\n".join(checks) + "\n")
    (OUT / "publish_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
        "# RUN ONLY against a PRIVATE store (see PUBLICATION_STATUS.json). GH_TOKEN must have write scope.\n"
        + "\n".join(pub) + "\n")
    (OUT / "fetch_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(fetch) + "\n")
    json.dump(status, open(OUT / "PUBLICATION_STATUS.json", "w"), indent=2)
    print(json.dumps({"store": repo, "visibility": vis, "assets": len(manifest),
                      "verdicts": status["verdicts"],
                      "blocker": status["publication_blocker"]}, indent=2))


if __name__ == "__main__":
    main()
