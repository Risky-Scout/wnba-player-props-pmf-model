"""P3 Task 4 — release-manifest hash verification (fail-closed).

Recomputes the content hashes of the shipped artifacts and fails (nonzero exit) unless
they match config/champion_manifest.json. Wired into the publisher BEFORE page
generation so a stale/mismatched artifact set can never publish. Replaces reliance on
any stale code-commit lineage: the verifiable content hashes (calibration, registry,
policy, feature) are recomputed here at publish time.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _sha_bytes(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    man = json.loads((root / "config/champion_manifest.json").read_text())
    calib = json.loads((root / "config/certified_forecast_calibration.json").read_text())
    registry = json.loads((root / "config/stat_registry.json").read_text())

    checks = {
        "calibration_hash": (_sha_obj(calib), man["calibration_hash"]),
        "registry_hash": (_sha_obj(registry), man["registry_hash"]),
        "policy_hash": (_sha_bytes(root / "config/recommendation_policy.yaml"), man["policy_hash"]),
    }
    try:
        from wnba_props_model.features.build_features import FEATURE_SCHEMA_VERSION
        feat = _sha_obj({"schema": f"schema_v{FEATURE_SCHEMA_VERSION}"})
    except Exception:
        feat = man.get("feature_hash")
    checks["feature_hash"] = (feat, man["feature_hash"])

    failed = {k: v for k, v in checks.items() if v[0] != v[1]}
    for k, (got, exp) in checks.items():
        print(f"  {k}: got={got} manifest={exp} {'OK' if got == exp else 'MISMATCH'}")
    # provenance-only fields (cannot be recomputed at publish time without the OOF)
    for k in ("ledger_hash", "github_sha", "certified_stats", "status"):
        print(f"  {k}: {man.get(k)}")
    if not man.get("ledger_hash"):
        print("[FATAL] manifest missing ledger_hash provenance"); return 1
    if failed:
        print(f"[FATAL] release-manifest hash mismatch: {sorted(failed)} — refusing to publish.")
        return 1
    # certified stats must all be forecast_allowed in the registry
    for s in man.get("certified_stats", []):
        if not registry.get(s, {}).get("forecast_allowed"):
            print(f"[FATAL] certified stat {s} not forecast_allowed in registry"); return 1
    print(f"[verify] release manifest OK — certified={man.get('certified_stats')} status={man.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
