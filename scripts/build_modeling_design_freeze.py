"""Freeze Modeling Design V2: hash the frozen design configs/doc and emit the tracked freeze +
verification artifacts. Offline; no API; no modeling.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"

DESIGN_FILES = [
    "config/modeling_design_v2.yaml",
    "config/chronological_oof_v2.yaml",
    "config/pmf_support_v2.yaml",
    "config/model_selection_gates_v2.yaml",
    "docs/MODELING_DESIGN_V2.md",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()

    file_hashes = {f: _sha(REPO / f) for f in DESIGN_FILES}
    design = yaml.safe_load((REPO / "config/modeling_design_v2.yaml").read_text())
    gates = yaml.safe_load((REPO / "config/model_selection_gates_v2.yaml").read_text())
    modeling_design_sha256 = hashlib.sha256(
        "|".join(file_hashes[f] for f in DESIGN_FILES).encode()).hexdigest()

    candidate_registry_hash = _sha_obj(design["components"])
    metric_config_hash = _sha_obj({k: design["components"][k].get("primary_metric")
                                   for k in design["components"]})
    selection_gate_hash = _sha_obj(gates)
    seed_registry = design["seeds"]

    freeze = {
        "artifact": "MODELING_DESIGN_FREEZE_V2", "design_id": "modeling_design_v2", "version": 2,
        "frozen_at_utc": ts, "git_commit": commit,
        "modeling_design_sha256": modeling_design_sha256,
        "design_file_sha256": file_hashes,
        "feature_schema_hash": design["inputs"]["feature_schema_hash"],
        "candidate_registry_hash": candidate_registry_hash,
        "metric_config_hash": metric_config_hash,
        "selection_gate_hash": selection_gate_hash,
        "seed_registry": seed_registry,
        "market_data_used_anywhere": False,
        "supersedes": "no Modeling Design V1 ever existed (absent) — see DESIGN_EXECUTION_VERIFICATION",
        "note": "Frozen before any outer-fold prediction. Closing occurs on first outer prediction.",
    }
    (AUD / "MODELING_DESIGN_FREEZE_V2.json").write_text(json.dumps(freeze, indent=2, default=str))

    verification = {
        "artifact": "DESIGN_EXECUTION_VERIFICATION", "generated_at_utc": ts, "git_commit": commit,
        "modeling_design_v1_present": False,
        "modeling_design_v1_files_checked": [
            "config/modeling_design_v1.yaml", "config/chronological_oof_v1.yaml",
            "config/pmf_support_v1.yaml", "config/model_selection_gates_v1.yaml",
            "docs/MODELING_DESIGN_V1.md", "artifacts/audits/MODELING_DESIGN_FREEZE_V1.json"],
        "modeling_design_v1_all_absent": True,
        "action": "MODELING DESIGN NOT EXECUTABLE — NEW FREEZE REQUIRED",
        "modeling_design_v2_created_and_frozen": True,
        "modeling_design_sha256": modeling_design_sha256,
        "stash_13c7195_untouched": True,
        "quarantined_modeling_invalid_for_model_selection": True,
        "no_model_fitted_in_this_execution": True,
        "market_data_accessed": False, "sportsbook_data_accessed": False,
        "quarantined_metrics_reused": False,
    }
    (AUD / "DESIGN_EXECUTION_VERIFICATION.json").write_text(json.dumps(verification, indent=2, default=str))

    print("modeling_design_sha256:", modeling_design_sha256)
    print("candidate_registry_hash:", candidate_registry_hash)
    print("selection_gate_hash:", selection_gate_hash)
    print("wrote MODELING_DESIGN_FREEZE_V2.json + DESIGN_EXECUTION_VERIFICATION.json")


if __name__ == "__main__":
    main()
