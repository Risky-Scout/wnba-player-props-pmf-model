"""Frozen production bundle I/O for wnba-pmf-production-v1.

Production loads one explicit immutable bundle. Ambient discovery is forbidden.
Integrity: MANIFEST.model_sha256 MUST equal the on-disk model_bundle.pkl digest.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wnba_props_model.sharp_v6.models import ModelBundle

BUNDLE_NAME = "wnba-pmf-production-v1"
DEFAULT_BUNDLE_DIR = Path("artifacts/releases") / BUNDLE_NAME
REQUIRED_BUNDLE_FILES = (
    "model_bundle.pkl",
    "MANIFEST.json",
    "FEATURE_CONTRACTS.json",
    "SELECTED_FAMILIES.json",
    "CALIBRATORS.json",
    "DEPENDENCE.json",
    "SHA256SUMS",
)


class BundleIntegrityError(RuntimeError):
    """Corrupt, incomplete, or hash-mismatched production bundle."""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def save_bundle(
    bundle: ModelBundle,
    out_dir: Path | str = DEFAULT_BUNDLE_DIR,
    *,
    meta: dict | None = None,
) -> dict[str, Any]:
    """Serialize the final bundle once, hash that exact file, and write sidecars."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    code_sha = _code_sha()
    incoming = {**(meta or {})}
    # Keep only semantic provenance inside the pickle. Volatile fields
    # (timestamps, prior hashes, integrity blobs) must not contaminate the digest.
    volatile = {
        "model_sha256", "saved_at_utc", "integrity", "manifest",
        "rewrap_from_claimed_sha",
    }
    base_meta = {
        k: v for k, v in (bundle.meta or {}).items()
        if k not in volatile and k != "saved_at_utc"
    }
    merged = {**base_meta, **{k: v for k, v in incoming.items() if k not in volatile}}
    semantic_meta = {
        "bundle_id": merged.get("bundle_id") or BUNDLE_NAME,
        "code_sha": code_sha,
        "training_cutoff": merged.get("training_cutoff"),
        "data_hashes": merged.get("data_hashes", {}),
        "identity_snapshot_hash": merged.get("identity_snapshot_hash"),
        "rollback_bundle": merged.get("rollback_bundle"),
        "rewrap_from": merged.get("rewrap_from"),
        "inference_function": "wnba_props_model.sharp_v6.inference.predict_slate",
        "retrain_in_daily": False,
    }
    # Drop Nones for stable pickle
    semantic_meta = {k: v for k, v in semantic_meta.items() if v is not None}
    bundle.meta = semantic_meta

    blob = pickle.dumps(bundle, protocol=4)
    model_sha = _sha256_bytes(blob)
    model_path = out / "model_bundle.pkl"
    model_path.write_bytes(blob)
    # Timestamps / digests live in sidecars and in-memory only.
    bundle.meta["model_sha256"] = model_sha
    bundle.meta["saved_at_utc"] = datetime.now(timezone.utc).isoformat()

    contracts = bundle.contracts or {}
    # Enrich contracts with explicit missingness classification
    enriched = {}
    for comp, cmeta in contracts.items():
        if not isinstance(cmeta, dict):
            cmeta = {"value": cmeta}
        enriched[comp] = {
            **cmeta,
            "missingness": cmeta.get(
                "missingness",
                "OPTIONAL_WITH_NATIVE_MISSING_SUPPORT",
            ),
            "feature_class_default": "OPTIONAL_WITH_NATIVE_MISSING_SUPPORT",
        }
    (out / "FEATURE_CONTRACTS.json").write_text(
        json.dumps(enriched, indent=2, default=str)
    )
    cal_meta = {
        s: {"method": c.method, "pit_ks_before": c.pit_ks_before, "pit_ks_after": c.pit_ks_after}
        for s, c in bundle.calibrators.items()
    }
    (out / "CALIBRATORS.json").write_text(json.dumps(cal_meta, indent=2))
    selected = bundle.selected_family
    (out / "SELECTED_FAMILIES.json").write_text(json.dumps(selected, indent=2))
    dep = None
    if bundle.dependence is not None:
        dep = {
            "method": bundle.dependence.method,
            "stats": bundle.dependence.stats,
            "corr": bundle.dependence.corr.tolist(),
            "status": bundle.dependence.status,
        }
    (out / "DEPENDENCE.json").write_text(json.dumps(dep, indent=2))
    (out / "GAME_ENVIRONMENT.json").write_text(json.dumps({
        "status": bundle.game_environment.status,
        "targets": list(bundle.game_environment.targets.keys()),
        "feature_hash": bundle.game_environment.feature_hash,
    }, indent=2))
    (out / "PARTICIPATION.json").write_text(json.dumps({
        "method": bundle.participation.method,
        "feature_hash": bundle.participation.feature_hash,
    }, indent=2))
    (out / "MINUTES.json").write_text(json.dumps({
        "family": bundle.minutes.family,
        "feature_hash": bundle.minutes.feature_hash,
        "team_regulation_minutes": 200,
        "team_q1_minutes": 50,
        "ot_shared": True,
        "reconciliation": "soft_scale_mu_to_team_target_before_pmf",
    }, indent=2))

    supported = list(selected.keys()) + [
        "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast",
        "q1_pts", "q1_reb", "q1_ast", "first_basket",
    ]
    unsupported = {
        "fantasy_points": "requires operator scoring configuration at runtime",
        "double_double": "derived from joint sims; enabled when dependence present",
        "triple_double": "derived from joint sims; enabled when dependence present",
    }
    feature_contract_hash = _sha256_bytes(
        json.dumps(enriched, sort_keys=True, default=str).encode()
    )
    manifest = {
        "artifact": "MANIFEST",
        "bundle_id": semantic_meta.get("bundle_id") or BUNDLE_NAME,
        "code_sha": code_sha,
        "training_cutoff": semantic_meta.get("training_cutoff"),
        "data_hashes": semantic_meta.get("data_hashes", {}),
        "identity_snapshot_hash": semantic_meta.get("identity_snapshot_hash"),
        "feature_hashes": {k: v.get("schema_hash") for k, v in enriched.items()},
        "feature_contract_hash": feature_contract_hash,
        "model_sha256": model_sha,
        "calibrator_hashes": {
            s: _sha256_bytes(json.dumps(cal_meta[s], sort_keys=True).encode()) for s in cal_meta
        },
        "dependence_hash": _sha256_bytes(json.dumps(dep, sort_keys=True).encode()) if dep else None,
        "random_seeds": {"SEED": 20260730},
        "supported_markets": supported,
        "unsupported_markets": unsupported,
        "rollback_bundle": semantic_meta.get("rollback_bundle"),
        "inference_function": "wnba_props_model.sharp_v6.inference.predict_slate",
        "retrain_in_daily": False,
        "selected_families": selected,
        "calibration_policy": {s: cal_meta[s]["method"] for s in cal_meta},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "MODEL_CARD.md").write_text(
        f"# {manifest['bundle_id']}\n\n"
        f"Authoritative WNBA player-prop PMF bundle (generated from MANIFEST).\n\n"
        f"- Inference: `{manifest['inference_function']}`\n"
        f"- Code SHA: `{code_sha}`\n"
        f"- Bundle hash (model_bundle.pkl): `{model_sha}`\n"
        f"- Feature-contract hash: `{feature_contract_hash}`\n"
        f"- Training cutoff: `{semantic_meta.get('training_cutoff')}`\n"
        f"- Selected families: `{json.dumps(selected)}`\n"
        f"- Calibration: `{json.dumps(manifest['calibration_policy'])}`\n"
        f"- Supported markets: `{json.dumps(supported)}`\n"
        f"- Withheld: `{json.dumps(unsupported)}`\n"
        f"- Daily inference loads this bundle and does not retrain.\n"
        f"- Command: `python scripts/run_wnba_pmf.py --bundle-dir {out.as_posix()}`\n"
    )
    try:
        lock = subprocess.check_output(["python3", "-m", "pip", "freeze"], text=True)
    except Exception:  # noqa: BLE001
        lock = ""
    (out / "dependency_lock.txt").write_text(lock)

    sums = []
    for p in sorted(out.iterdir()):
        if p.name == "SHA256SUMS" or p.is_dir():
            continue
        sums.append(f"{_sha256_file(p)}  {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return manifest


def verify_bundle_integrity(bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> dict[str, Any]:
    """Fail closed when required files, hashes, or schema fields are inconsistent."""
    out = Path(bundle_dir)
    missing = [n for n in REQUIRED_BUNDLE_FILES if not (out / n).exists()]
    if missing:
        raise BundleIntegrityError(f"missing required bundle files: {missing}")

    man = json.loads((out / "MANIFEST.json").read_text())
    pkl = out / "model_bundle.pkl"
    got = _sha256_file(pkl)
    expected = man.get("model_sha256")
    if not expected:
        raise BundleIntegrityError("MANIFEST missing model_sha256")
    if got != expected:
        raise BundleIntegrityError(
            f"model_bundle.pkl sha256 mismatch: file={got} manifest={expected}"
        )

    # Verify SHA256SUMS for every listed file
    sums = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest.strip()
    if "model_bundle.pkl" not in sums:
        raise BundleIntegrityError("SHA256SUMS missing model_bundle.pkl")
    if sums["model_bundle.pkl"] != got:
        raise BundleIntegrityError(
            f"SHA256SUMS model_bundle.pkl mismatch: sums={sums['model_bundle.pkl']} file={got}"
        )
    for name, digest in sums.items():
        p = out / name
        if not p.exists():
            raise BundleIntegrityError(f"SHA256SUMS lists missing file: {name}")
        if _sha256_file(p) != digest:
            raise BundleIntegrityError(f"SHA256SUMS mismatch for {name}")

    required_manifest = (
        "bundle_id", "inference_function", "retrain_in_daily",
        "selected_families", "supported_markets",
    )
    # selected_families may live only in SELECTED_FAMILIES.json for older bundles
    if "selected_families" not in man:
        sel = json.loads((out / "SELECTED_FAMILIES.json").read_text())
        man = {**man, "selected_families": sel}
    for k in ("bundle_id", "inference_function", "retrain_in_daily", "supported_markets"):
        if k not in man:
            raise BundleIntegrityError(f"MANIFEST missing required field: {k}")
    if man.get("retrain_in_daily") is not False:
        raise BundleIntegrityError("bundle must set retrain_in_daily=false")
    if "predict_slate" not in str(man.get("inference_function", "")):
        raise BundleIntegrityError("bundle inference_function must be predict_slate")

    selected = man.get("selected_families") or json.loads(
        (out / "SELECTED_FAMILIES.json").read_text()
    )
    expected_families = {
        "pts": "structural_shooting",
        "reb": "structural_oreb_dreb",
        "ast": "minutes_mixture_nb2",
        "fg3m": "minutes_mixture_nb2",
        "stl": "hurdle_nb2",
        "blk": "minutes_mixture_nb2",
        "turnover": "hurdle_nb2",
    }
    for stat, fam in expected_families.items():
        if selected.get(stat) != fam:
            raise BundleIntegrityError(
                f"selected family mismatch for {stat}: got={selected.get(stat)} expected={fam}"
            )

    return {
        "bundle_dir": str(out),
        "bundle_id": man.get("bundle_id"),
        "model_sha256": got,
        "code_sha": man.get("code_sha"),
        "feature_contract_hash": man.get("feature_contract_hash"),
        "data_hashes": man.get("data_hashes", {}),
        "selected_families": selected,
        "inference_function": man.get("inference_function"),
        "retrain_in_daily": man.get("retrain_in_daily"),
    }


def load_bundle(bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> ModelBundle:
    """Load and verify one explicit immutable production bundle."""
    out = Path(bundle_dir)
    info = verify_bundle_integrity(out)
    pkl = out / "model_bundle.pkl"
    try:
        bundle = pickle.loads(pkl.read_bytes())
    except Exception as e:  # noqa: BLE001
        raise BundleIntegrityError(f"corrupt model_bundle.pkl: {e}") from e
    man = json.loads((out / "MANIFEST.json").read_text())
    if "selected_families" not in man:
        man["selected_families"] = json.loads((out / "SELECTED_FAMILIES.json").read_text())
    bundle.meta = {
        **(bundle.meta or {}),
        "bundle_id": man.get("bundle_id", BUNDLE_NAME),
        "manifest": man,
        "model_sha256": info["model_sha256"],
        "integrity": info,
    }
    return bundle
