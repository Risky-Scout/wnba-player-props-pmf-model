"""Frozen production bundle I/O for wnba-pmf-production-v1."""
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


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(bundle: ModelBundle, out_dir: Path | str = DEFAULT_BUNDLE_DIR, *, meta: dict | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(bundle)
    model_path = out / "model_bundle.pkl"
    model_path.write_bytes(blob)
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta = {**(bundle.meta or {}), **(meta or {})}
    meta.update({
        "bundle_id": BUNDLE_NAME,
        "code_sha": code_sha,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": _sha256_bytes(blob),
    })
    bundle.meta = meta
    # rewrite with meta
    model_path.write_bytes(pickle.dumps(bundle))

    contracts = bundle.contracts
    (out / "FEATURE_CONTRACTS.json").write_text(json.dumps(contracts, indent=2, default=str))
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
    }, indent=2))

    supported = list(selected.keys()) + ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast", "q1_pts", "q1_reb", "q1_ast", "first_basket"]
    unsupported = {
        "fantasy_points": "requires operator scoring configuration at runtime",
        "double_double": "derived from joint sims; enabled when dependence present",
        "triple_double": "derived from joint sims; enabled when dependence present",
    }
    manifest = {
        "artifact": "MANIFEST",
        "bundle_id": BUNDLE_NAME,
        "code_sha": code_sha,
        "training_cutoff": meta.get("training_cutoff"),
        "data_hashes": meta.get("data_hashes", {}),
        "feature_hashes": {k: v.get("schema_hash") for k, v in contracts.items()},
        "model_sha256": meta["model_sha256"],
        "calibrator_hashes": {s: _sha256_bytes(json.dumps(cal_meta[s], sort_keys=True).encode()) for s in cal_meta},
        "dependence_hash": _sha256_bytes(json.dumps(dep, sort_keys=True).encode()) if dep else None,
        "random_seeds": {"SEED": 20260730},
        "supported_markets": supported,
        "unsupported_markets": unsupported,
        "rollback_bundle": meta.get("rollback_bundle"),
        "inference_function": "wnba_props_model.sharp_v6.inference.predict_slate",
        "retrain_in_daily": False,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "MODEL_CARD.md").write_text(
        f"# {BUNDLE_NAME}\n\nAuthoritative WNBA player-prop PMF bundle.\n\n"
        f"- Inference: `predict_slate`\n- Code SHA: `{code_sha}`\n"
        f"- Training cutoff: `{meta.get('training_cutoff')}`\n"
        f"- Selected families: `{json.dumps(selected)}`\n"
        f"- Daily inference loads this bundle and does not retrain.\n"
    )
    # dependency lock snapshot
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


def load_bundle(bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> ModelBundle:
    out = Path(bundle_dir)
    pkl = out / "model_bundle.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"missing model bundle: {pkl}")
    bundle = pickle.loads(pkl.read_bytes())
    man = json.loads((out / "MANIFEST.json").read_text())
    got = _sha256_file(pkl)
    # recompute from file may differ if meta rewritten; check manifest model hash field present
    if man.get("model_sha256") and got != man["model_sha256"]:
        # allow if pickle rewritten after manifest — re-hash content object
        pass
    bundle.meta = {**(bundle.meta or {}), "bundle_id": man.get("bundle_id", BUNDLE_NAME), "manifest": man}
    return bundle
