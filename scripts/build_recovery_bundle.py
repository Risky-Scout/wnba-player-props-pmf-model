"""Assemble the immutable recovery bundle for private, durable preservation.

The five frozen baseline feature assets (canonical games / player-game stats / wide
features / long features / schema manifest) are NOT committed to git (they are
gitignored under data/processed/) and were never published to a release. On a fresh
clone they are therefore absent and — with no BDL_API_KEY and no prior release — are
byte-unrecoverable here. Their known target hashes are recorded in
artifacts/data_bootstrap/RECOVERED_DATA_MANIFEST.json for a future authenticated rebuild.

What IS durable and preservation-worthy right now are the committed, real evaluation
artifacts derived from those assets: the 7-prop OOF PMFs, the atomic exact-quote store,
the closing-consensus table, the prequential ledger, and the G0 market-superiority
input. This script bundles those (plus the manifests/registry) into a single tarball
with a byte-exact manifest so publish_data.py can push it immutably to the private repo.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "recovery"
BUNDLE = OUT_DIR / "recovery_bundle_v1.tar.gz"
MANIFEST = REPO / "artifacts" / "data_bootstrap" / "RECOVERY_BUNDLE_MANIFEST.json"

# Real, committed evaluation artifacts that constitute the durable recovery payload.
PAYLOAD = [
    "artifacts/models/calibration/oof_predictions.parquet",
    "artifacts/p1/p1_quotes.parquet",
    "artifacts/p1/p1_closing_consensus.parquet",
    "artifacts/p1/market_superiority_input.parquet",
    "artifacts/p3/p3_prequential_ledger.parquet",
    "artifacts/tracking/tracking_game_crosswalk.parquet",
    "artifacts/data_bootstrap/RECOVERED_DATA_MANIFEST.json",
    "config/data_registry.json",
]

# The five frozen baseline assets (byte-unrecoverable on a fresh VM without BDL access).
FROZEN_ASSETS = [
    "wnba_games",
    "wnba_player_game_stats",
    "wnba_player_game_features_wide",
    "wnba_player_game_features_long",
    "feature_schema_manifest",
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    present, missing = [], []
    for rel in PAYLOAD:
        (present if (REPO / rel).exists() else missing).append(rel)

    with tarfile.open(BUNDLE, "w:gz") as tar:
        for rel in present:
            tar.add(REPO / rel, arcname=rel)

    members = []
    for rel in present:
        p = REPO / rel
        members.append({"path": rel, "sha256": _sha256(p), "bytes": p.stat().st_size})

    # Frozen-asset status: read known hashes from the recovered-data manifest (never fabricate bytes).
    rd_manifest_path = REPO / "artifacts" / "data_bootstrap" / "RECOVERED_DATA_MANIFEST.json"
    rd = json.loads(rd_manifest_path.read_text()) if rd_manifest_path.exists() else {}
    frozen_status = []
    proc = REPO / "data" / "processed"
    for name in FROZEN_ASSETS:
        ext = "json" if name == "feature_schema_manifest" else "parquet"
        local = proc / f"{name}.{ext}"
        info = rd.get("files", {}).get(name, {})
        frozen_status.append({
            "asset": name,
            "target_sha256": info.get("sha256"),
            "target_bytes": info.get("bytes"),
            "present_locally": local.exists(),
            "status": "PRESENT" if local.exists() else "UNRECOVERABLE_ON_THIS_VM",
        })

    manifest = {
        "version": "recovery-bundle-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(BUNDLE.relative_to(REPO)),
        "bundle_sha256": _sha256(BUNDLE),
        "bundle_bytes": BUNDLE.stat().st_size,
        "members": members,
        "missing_payload": missing,
        "frozen_baseline_assets": frozen_status,
        "frozen_assets_note": (
            "The five frozen baseline assets are gitignored (data/processed/) and were never "
            "released; with no BDL_API_KEY on this VM they are byte-unrecoverable here. Their "
            "target hashes are preserved for a future authenticated deterministic rebuild."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[recovery-bundle] {BUNDLE.relative_to(REPO)} "
          f"({manifest['bundle_bytes']:,} bytes, sha256={manifest['bundle_sha256'][:12]}…)")
    print(f"[recovery-bundle] payload members present={len(present)} missing={len(missing)}")
    for f in frozen_status:
        print(f"  frozen {f['asset']}: {f['status']} target_sha256={str(f['target_sha256'])[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
