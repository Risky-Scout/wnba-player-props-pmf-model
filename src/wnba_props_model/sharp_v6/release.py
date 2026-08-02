"""Unified release readiness matrix, gates, proof, and deployment receipt for V6."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wnba_props_model.sharp_v6.bundle import BUNDLE_NAME, DEFAULT_BUNDLE_DIR, _sha256_file, verify_bundle_integrity

REPO_ROOT = Path(__file__).resolve().parents[3]
POINTER_PATH = REPO_ROOT / "artifacts/releases/PRODUCTION_POINTER.json"
PROOF_PATH = REPO_ROOT / "artifacts/sharp_v6/ONE_PRODUCTION_MODEL_PROOF.json"
INFERENCE_FN = "wnba_props_model.sharp_v6.inference.predict_slate"
AUTHORITATIVE_CMD = (
    "python scripts/run_wnba_pmf.py "
    "--bundle-dir artifacts/releases/wnba-pmf-production-v1.1"
)

READINESS_LEVELS = (
    "STRUCTURALLY_VALID",
    "STATISTICALLY_EVALUABLE",
    "CALIBRATED",
    "PROSPECTIVELY_MONITORED",
    "MARKET_VALIDATED",
    "PRODUCTION_READY",
)


@dataclass
class GateResult:
    name: str
    status: str  # PASS | FAIL | NOT_EVALUABLE
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass
class ReleaseMatrix:
    levels: dict[str, str]
    market_status: dict[str, str]
    gates: list[GateResult]
    market_superiority: str = "NOT_PROVEN"
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": "V6_RELEASE_MATRIX",
            "generated_at_utc": self.generated_at_utc,
            "levels": self.levels,
            "market_family_status": self.market_status,
            "market_superiority": self.market_superiority,
            "gates": [
                {"name": g.name, "status": g.status, "detail": g.detail} for g in self.gates
            ],
            "all_structural_gates_passed": all(
                g.passed for g in self.gates if g.name.startswith("structural_")
            ),
            "production_ready": self.levels.get("PRODUCTION_READY") == "PASS",
            # Explicit: structural validity is not "all model gates passed"
            "summary_label": (
                "PRODUCTION_READY"
                if self.levels.get("PRODUCTION_READY") == "PASS"
                else (
                    "STRUCTURALLY_VALID_NOT_MARKET_VALIDATED"
                    if self.levels.get("STRUCTURALLY_VALID") == "PASS"
                    else "NOT_READY"
                )
            ),
        }


def _git_sha(ref: str = "HEAD") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _read_production_flags() -> dict[str, bool]:
    out = {}
    for ver in ("v3", "v4", "v5", "v6"):
        p = REPO_ROOT / f"src/wnba_props_model/sharp_{ver}/__init__.py"
        text = p.read_text() if p.exists() else ""
        tree = ast.parse(text) if text else None
        prod = False
        for node in ast.walk(tree) if tree else []:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "PRODUCTION":
                        if isinstance(node.value, ast.Constant):
                            prod = bool(node.value.value)
        out[ver] = prod
    return out


def gate_one_production_model() -> GateResult:
    flags = _read_production_flags()
    pointer = json.loads(POINTER_PATH.read_text()) if POINTER_PATH.exists() else {}
    bundle_path = str(pointer.get("production_bundle", ""))
    ok = (
        flags.get("v3") is False
        and flags.get("v4") is False
        and flags.get("v5") is False
        and flags.get("v6") is True
        and pointer.get("authoritative") is True
        and pointer.get("inference_function") == INFERENCE_FN
        and ("wnba-pmf-production-v1" in bundle_path)
    )
    return GateResult(
        "structural_one_production_model",
        "PASS" if ok else "FAIL",
        {"production_flags": flags, "pointer_authoritative": pointer.get("authoritative")},
    )


def gate_bundle_integrity(bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> GateResult:
    try:
        info = verify_bundle_integrity(bundle_dir)
        return GateResult("structural_bundle_integrity", "PASS", info)
    except Exception as e:  # noqa: BLE001
        return GateResult("structural_bundle_integrity", "FAIL", {"error": str(e)})


def gate_no_legacy_publish() -> GateResult:
    """Legacy daily_pipeline must not be able to overwrite V6 outputs under normal operation."""
    daily = REPO_ROOT / ".github/workflows/daily_pipeline.yml"
    v6 = REPO_ROOT / ".github/workflows/wnba_pmf_daily.yml"
    if not daily.exists() or not v6.exists():
        return GateResult("structural_no_legacy_publish", "FAIL", {"error": "workflow_missing"})
    daily_txt = daily.read_text()
    v6_txt = v6.read_text()
    # Hard requirements: V6 workflow is authoritative; legacy marked non-authoritative
    legacy_blocked = (
        "LEGACY_CONTROL" in daily_txt
        or "not authoritative" in daily_txt.lower()
        or "AUTHORITATIVE_PUBLISH: false" in daily_txt
        or "authoritative_publish: false" in daily_txt.lower()
    )
    v6_ok = "run_wnba_pmf.py" in v6_txt and "fit_wnba_pmf_bundle" not in v6_txt
    # Legacy must not write sharp_v6 deliveries in executable steps
    run_lines = [ln for ln in daily_txt.splitlines() if not ln.strip().startswith("#")]
    writes_v6 = any("deliveries/sharp_v6" in ln for ln in run_lines)
    publish_gated = "AUTHORITATIVE_PUBLISH" in daily_txt and (
        "AUTHORITATIVE_PUBLISH == 'true'" in daily_txt
        or 'AUTHORITATIVE_PUBLISH == "true"' in daily_txt
    )
    ok = legacy_blocked and v6_ok and not writes_v6 and publish_gated
    return GateResult(
        "structural_no_legacy_publish",
        "PASS" if ok else "FAIL",
        {
            "legacy_blocked": legacy_blocked,
            "v6_workflow_ok": v6_ok,
            "legacy_writes_sharp_v6": writes_v6,
            "publish_gated": publish_gated,
        },
    )


def gate_sample_size(
    n_obs: int,
    *,
    min_obs: int,
    name: str,
) -> GateResult:
    """Vacuous-pass prevention: zero/insufficient rows → NOT_EVALUABLE (never PASS)."""
    if n_obs <= 0:
        return GateResult(name, "NOT_EVALUABLE", {"n_obs": n_obs, "min_obs": min_obs})
    if n_obs < min_obs:
        return GateResult(name, "NOT_EVALUABLE", {"n_obs": n_obs, "min_obs": min_obs})
    return GateResult(name, "PASS", {"n_obs": n_obs, "min_obs": min_obs})


def gate_not_tautology(expression_true: bool, *, name: str, detail: dict | None = None) -> GateResult:
    """Mutation-testable gate wrapper — caller supplies the real boolean condition."""
    return GateResult(name, "PASS" if expression_true else "FAIL", detail or {})


def evaluate_release_matrix(
    *,
    bundle_dir: Path | str = DEFAULT_BUNDLE_DIR,
    n_stat_obs: int | None = None,
    n_games: int | None = None,
    n_players: int | None = None,
    calibration_explicit: bool = True,
    dependence_ok: bool = True,
    train_serve_parity: bool = False,
    reproducibility_ok: bool = False,
    ci_ok: bool = False,
    smoke_ok: bool = False,
    deployment_ok: bool = False,
    market_validated: bool = False,
) -> ReleaseMatrix:
    gates = [
        gate_one_production_model(),
        gate_bundle_integrity(bundle_dir),
        gate_no_legacy_publish(),
    ]
    if n_stat_obs is not None:
        gates.append(gate_sample_size(n_stat_obs, min_obs=500, name="stat_min_observations"))
    if n_games is not None:
        gates.append(gate_sample_size(n_games, min_obs=20, name="stat_min_games"))
    if n_players is not None:
        gates.append(gate_sample_size(n_players, min_obs=50, name="stat_min_players"))

    gates.append(gate_not_tautology(calibration_explicit, name="structural_calibration_explicit"))
    gates.append(gate_not_tautology(dependence_ok, name="structural_dependence_valid"))
    gates.append(gate_not_tautology(train_serve_parity, name="structural_train_serve_parity"))
    gates.append(gate_not_tautology(reproducibility_ok, name="structural_reproducibility"))
    gates.append(gate_not_tautology(ci_ok, name="ops_ci"))
    gates.append(gate_not_tautology(smoke_ok, name="ops_daily_smoke"))
    gates.append(gate_not_tautology(deployment_ok, name="ops_deployment_receipt"))

    # Vacuous guard: any NOT_EVALUABLE statistical gate blocks PRODUCTION_READY
    vacuous = [g for g in gates if g.status == "NOT_EVALUABLE"]
    structural = [g for g in gates if g.name.startswith("structural_")]
    structurally_valid = all(g.passed for g in structural)
    statistically_evaluable = not vacuous and all(
        g.passed for g in gates if g.name.startswith("stat_")
    ) if any(g.name.startswith("stat_") for g in gates) else False

    levels = {
        "STRUCTURALLY_VALID": "PASS" if structurally_valid else "FAIL",
        "STATISTICALLY_EVALUABLE": "PASS" if statistically_evaluable else "NOT_EVALUABLE" if vacuous else "FAIL",
        "CALIBRATED": "PASS" if calibration_explicit else "FAIL",
        "PROSPECTIVELY_MONITORED": "PASS" if smoke_ok else "FAIL",
        "MARKET_VALIDATED": "PASS" if market_validated else "NOT_PROVEN",
        "PRODUCTION_READY": (
            "PASS"
            if structurally_valid and statistically_evaluable and calibration_explicit
            and train_serve_parity and reproducibility_ok and ci_ok and smoke_ok and deployment_ok
            else "FAIL"
        ),
    }

    market_status = {
        "pts": "STRUCTURALLY_VALID",
        "reb": "STRUCTURALLY_VALID",
        "ast": "STRUCTURALLY_VALID",
        "fg3m": "STRUCTURALLY_VALID",
        "stl": "STRUCTURALLY_VALID",
        "blk": "STRUCTURALLY_VALID",
        "turnover": "STRUCTURALLY_VALID",
        "combinations": "STRUCTURALLY_VALID" if dependence_ok else "WITHHELD",
        "q1": "STRUCTURALLY_VALID",
        "first_basket": "STRUCTURALLY_VALID",
        "double_double": "WITHHELD",
        "triple_double": "WITHHELD",
        "fantasy_points": "WITHHELD",
    }

    return ReleaseMatrix(
        levels=levels,
        market_status=market_status,
        gates=gates,
        market_superiority="PASS" if market_validated else "NOT_PROVEN",
    )


def generate_one_production_model_proof(
    *,
    bundle_dir: Path | str = DEFAULT_BUNDLE_DIR,
    origin_main_sha: str | None = None,
) -> dict[str, Any]:
    """Generate ONE_PRODUCTION_MODEL_PROOF.json from repository facts (not manual assertion)."""
    flags = _read_production_flags()
    pointer = json.loads(POINTER_PATH.read_text()) if POINTER_PATH.exists() else {}
    origin = origin_main_sha or _git_sha("origin/main")
    head = _git_sha("HEAD")
    integrity: dict[str, Any]
    try:
        integrity = verify_bundle_integrity(bundle_dir)
        integrity_ok = True
    except Exception as e:  # noqa: BLE001
        integrity = {"error": str(e)}
        integrity_ok = False

    proof = {
        "artifact": "ONE_PRODUCTION_MODEL_PROOF",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from": "wnba_props_model.sharp_v6.release.generate_one_production_model_proof",
        "v3_production": flags.get("v3", True),
        "v3_deprecated": True,
        "v4_production": flags.get("v4", True),
        "v4_deprecated": True,
        "v5_production": flags.get("v5", True),
        "v5_deprecated": True,
        "v6_production": flags.get("v6", False),
        "pointer_authoritative": bool(pointer.get("authoritative")),
        "pointer_bundle": pointer.get("production_bundle"),
        "pointer_model_sha256": pointer.get("model_sha256"),
        "bundle_inference": pointer.get("inference_function") or INFERENCE_FN,
        "daily_retrain": False,
        "origin_main": origin,
        "head_sha": head,
        "bundle_integrity_ok": integrity_ok,
        "bundle_integrity": integrity,
        "authoritative_command": AUTHORITATIVE_CMD,
        "facts_consistent": (
            flags.get("v3") is False
            and flags.get("v4") is False
            and flags.get("v5") is False
            and flags.get("v6") is True
            and bool(pointer.get("authoritative"))
            and integrity_ok
        ),
    }
    return proof


def write_proof(path: Path | str = PROOF_PATH, **kwargs: Any) -> dict[str, Any]:
    proof = generate_one_production_model_proof(**kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def build_deployment_receipt(
    *,
    expected_origin_main_sha: str,
    expected_bundle_hash: str,
    bundle_dir: Path | str = DEFAULT_BUNDLE_DIR,
    deployment_environment: str = "github_actions",
    smoke_run_id: str | None = None,
    smoke_result: str = "UNKNOWN",
    sample_output_hash: str | None = None,
    staged_status: str = "local",
) -> dict[str, Any]:
    """Machine-readable deployment receipt — merge/CI alone is not deployment proof."""
    origin = _git_sha("origin/main")
    head = _git_sha("HEAD")
    pointer = json.loads(POINTER_PATH.read_text()) if POINTER_PATH.exists() else {}
    try:
        integrity = verify_bundle_integrity(bundle_dir)
        actual_hash = integrity.get("model_sha256", "")
        integrity_ok = True
    except Exception as e:  # noqa: BLE001
        integrity = {"error": str(e)}
        actual_hash = ""
        integrity_ok = False

    sha_match = origin == expected_origin_main_sha
    hash_match = actual_hash == expected_bundle_hash and bool(actual_hash)
    pointer_ok = (
        pointer.get("authoritative") is True
        and pointer.get("inference_function") == INFERENCE_FN
        and pointer.get("model_sha256") == expected_bundle_hash
    )
    verified = (
        sha_match and hash_match and pointer_ok and integrity_ok
        and smoke_result == "success"
    )
    return {
        "artifact": "DEPLOYMENT_RECEIPT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_environment": deployment_environment,
        "deployment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_origin_main_sha": expected_origin_main_sha,
        "actual_origin_main_sha": origin,
        "head_sha": head,
        "origin_main_sha_match": sha_match,
        "expected_bundle_hash": expected_bundle_hash,
        "actual_bundle_hash": actual_hash,
        "bundle_hash_match": hash_match,
        "authoritative_pointer_target": pointer.get("production_bundle"),
        "inference_entry_point": pointer.get("inference_function"),
        "pointer_ok": pointer_ok,
        "bundle_integrity": integrity,
        "staged_status": staged_status,
        "smoke_run_id": smoke_run_id,
        "smoke_result": smoke_result,
        "sample_output_hash": sample_output_hash,
        "deployment_verified": verified,
        "note": (
            "Deployment success requires SHA match, bundle hash match, "
            "authoritative pointer, and smoke success — not only local artifact generation."
        ),
    }


def content_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
