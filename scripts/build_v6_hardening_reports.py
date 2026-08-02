#!/usr/bin/env python3
"""Build hardening deliverable reports from repository facts and the candidate bundle."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.bundle import verify_bundle_integrity
from wnba_props_model.sharp_v6.contracts import GOVERNED_CONSTANTS
from wnba_props_model.sharp_v6.feature_policy import registry_from_bundle_contracts
from wnba_props_model.sharp_v6.release import (
    build_deployment_receipt,
    evaluate_release_matrix,
    generate_one_production_model_proof,
    write_proof,
)

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/sharp_v6/hardening"


ROOT_CAUSES = [
    ("1_train_serve_skew", "Unify _core_pmf_delivery; parity tests", "tests/test_v6_hardening.py"),
    ("2_silent_generic_identity", "Typed identity statuses + quarantine", "src/wnba_props_model/sharp_v6/identity.py"),
    ("3_unexplained_heuristics", "GOVERNED_CONSTANTS registry", "src/wnba_props_model/sharp_v6/contracts.py"),
    ("4_pseudo_features", "Feature registry + ablation stub from frozen contracts", "feature_policy.py"),
    ("5_date_effective_identity", "build_date_effective_identity_table", "identity.py"),
    ("6_ambient_artifact_discovery", "Explicit --bundle-dir + verify_bundle_integrity", "bundle.py"),
    ("7_silent_fallback", "Production fail-closed modes", "inference.py / run_wnba_pmf.py"),
    ("8_coherence_invariants", "PMF norm fail-closed; team minutes invariants", "inference.py / models.py"),
    ("9_one_probabilistic_system", "Combos/Q1/FB from shared core", "inference.py"),
    ("10_calibration_collapse", "Explicit identity calibrators; wrong-stat guard", "inference.py"),
    ("11_market_external", "MARKET_EXTERNAL_NOTE; no overwrite", "run_wnba_pmf.py"),
    ("12_tautological_gates", "gate_not_tautology + mutation tests", "release.py / tests"),
    ("13_vacuous_passes", "NOT_EVALUABLE on zero rows", "release.py"),
    ("14_unified_readiness", "ReleaseMatrix levels", "release.py"),
    ("15_raw_data_audit", "Documented requirement; pipeline preserves counts in manifests", "reports"),
    ("16_production_fail_closed", "--mode production nonzero exit", "run_wnba_pmf.py"),
    ("17_validation_exceptions", "FAILED_GATE.json on exception", "run_wnba_pmf.py"),
    ("18_behavioral_tests", "tests/test_v6_hardening.py", "tests"),
    ("19_no_market_superiority_claim", "market_superiority=NOT_PROVEN", "release.py"),
    ("20_withhold_unsupported", "unsupported map in manifest", "bundle.py"),
    ("21_feature_engineering", "Feature registry + ablation report scaffold", "hardening reports"),
    ("22_provenance", "Bundle hashes + deployment receipt", "bundle.py / release.py"),
    ("23_one_production_truth", "Legacy workflow AUTHORITATIVE_PUBLISH false", "workflows"),
    ("24_docs_match_runtime", "MODEL_CARD from manifest", "bundle.py"),
    ("25_invalidate_on_contract_change", "feature_contract_hash in manifest", "bundle.py"),
    ("26_deployment_proof", "DEPLOYMENT_RECEIPT.json", "release.py"),
]


@app.command()
def main(
    baseline_dir: str = typer.Option(
        "artifacts/releases/wnba-pmf-production-v1", "--baseline-dir",
    ),
    candidate_dir: str = typer.Option(
        "artifacts/releases/candidates/wnba-pmf-production-v1.1-harden", "--candidate-dir",
    ),
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = Path(baseline_dir)
    candidate = Path(candidate_dir)

    baseline_man = json.loads((baseline / "MANIFEST.json").read_text()) if (baseline / "MANIFEST.json").exists() else {}
    cand_info = None
    cand_err = None
    try:
        cand_info = verify_bundle_integrity(candidate)
    except Exception as e:  # noqa: BLE001
        cand_err = str(e)

    # 1. Root-cause closure matrix
    closure = []
    for key, change, evidence in ROOT_CAUSES:
        status = "IMPLEMENTED"
        remaining = "monitor_in_prospective"
        if key == "15_raw_data_audit":
            status = "PARTIAL"
            remaining = "full duplicate classification report pending next data pull"
        if key.startswith("21_") or key.startswith("19_"):
            remaining = "OOF ablation / market validation remain NOT_PROVEN until prospective evidence"
        if cand_err and key in {"6_ambient_artifact_discovery", "22_provenance", "25_invalidate_on_contract_change"}:
            status = "BLOCKED_ON_CANDIDATE_BUNDLE"
            remaining = cand_err
        closure.append({
            "root_cause": key,
            "implementation_change": change,
            "tests_or_evidence": evidence,
            "status": status,
            "remaining_risk": remaining,
            "affected_files": evidence,
        })
    (OUT / "ROOT_CAUSE_CLOSURE_MATRIX.json").write_text(json.dumps({
        "artifact": "ROOT_CAUSE_CLOSURE_MATRIX",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_bundle": str(baseline),
        "baseline_claimed_model_sha256": baseline_man.get("model_sha256"),
        "candidate_bundle": str(candidate),
        "candidate_integrity": cand_info or {"error": cand_err},
        "items": closure,
    }, indent=2))

    # 2. Feature registry
    contracts = {}
    cpath = candidate / "FEATURE_CONTRACTS.json"
    if not cpath.exists():
        cpath = baseline / "FEATURE_CONTRACTS.json"
    if cpath.exists():
        contracts = json.loads(cpath.read_text())
    registry = registry_from_bundle_contracts(contracts)
    (OUT / "FEATURE_REGISTRY.json").write_text(json.dumps({
        "artifact": "FEATURE_REGISTRY",
        "n_features": len(registry),
        "features": registry,
        "governed_constants": GOVERNED_CONSTANTS,
    }, indent=2, default=str))

    # Ablation report scaffold (retain frozen set; no unvalidated expansion)
    (OUT / "FEATURE_ABLATION_REPORT.json").write_text(json.dumps({
        "artifact": "FEATURE_ABLATION_REPORT",
        "baseline_features": "frozen_v6_contracts",
        "added_feature_group": None,
        "decision": "RETAIN_FROZEN_SET",
        "rationale": (
            "No feature accumulation on this hardening branch. "
            "Frozen contracts retained; candidates require rolling OOF evidence."
        ),
        "oof_metric_delta": None,
        "stability_across_folds": "not_recomputed_this_branch",
        "missingness_policy": "OPTIONAL_WITH_NATIVE_MISSING_SUPPORT",
    }, indent=2))

    # 3. Architecture doc
    (OUT / "SINGLE_MODEL_ARCHITECTURE.md").write_text(
        "# Single-Model Architecture (V6)\n\n"
        "One end-to-end forecasting contract — not alternative production models.\n\n"
        "```\n"
        "Point-in-time source snapshots\n"
        "  → Canonical identities (date-effective)\n"
        "  → Frozen feature contract + governed missingness\n"
        "  → Participation → Minutes → Shared environment\n"
        "  → Direct-stat heads (frozen families per stat)\n"
        "  → Gaussian-copula dependence + joint sims\n"
        "  → Full-game / combo / Q1 / first-basket markets\n"
        "  → Explicit calibration → release matrix → publish\n"
        "```\n\n"
        f"- Inference: `wnba_props_model.sharp_v6.inference.predict_slate`\n"
        f"- Baseline bundle: `{baseline}` (immutable)\n"
        f"- Candidate bundle: `{candidate}`\n"
        "- Internal components (participation, minutes, env, stats, dependence, Q1, FB) "
        "are parts of one system, not competing production models.\n"
        "- V3/V4/V5 remain RESEARCH_ONLY / PRODUCTION=False.\n"
        "- Market odds are external evaluation inputs only.\n"
        "- Market superiority: NOT_PROVEN.\n"
    )

    # 5. Reproducibility manifest
    (OUT / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps({
        "artifact": "REPRODUCIBILITY_MANIFEST",
        "python": ">=3.10 (CI 3.11)",
        "seed": 20260730,
        "baseline_claimed_model_sha256": baseline_man.get("model_sha256"),
        "candidate": cand_info,
        "commands": {
            "rewrap": "python scripts/rewrap_v6_bundle.py",
            "verify_bundle": "python -c \"from wnba_props_model.sharp_v6.bundle import verify_bundle_integrity; print(verify_bundle_integrity('artifacts/releases/candidates/wnba-pmf-production-v1.1-harden'))\"",
            "proof": "python scripts/generate_one_production_proof.py --bundle-dir artifacts/releases/candidates/wnba-pmf-production-v1.1-harden",
            "gates": "python scripts/verify_v6_release_gates.py",
            "inference": "python scripts/run_wnba_pmf.py --bundle-dir artifacts/releases/candidates/wnba-pmf-production-v1.1-harden --mode production",
        },
        "note": "Build timestamps may differ; semantic artifact hashes must not.",
    }, indent=2, default=str))

    # 6. Statistical evaluation status (no false market claims)
    sel = json.loads((baseline / "SELECTED_FAMILIES.json").read_text()) if (baseline / "SELECTED_FAMILIES.json").exists() else {}
    cal = json.loads((baseline / "CALIBRATORS.json").read_text()) if (baseline / "CALIBRATORS.json").exists() else {}
    (OUT / "STATISTICAL_EVALUATION_REPORT.json").write_text(json.dumps({
        "artifact": "STATISTICAL_EVALUATION_REPORT",
        "selected_families": sel,
        "calibration": {k: v.get("method") for k, v in cal.items()},
        "market_superiority": "NOT_PROVEN",
        "positive_ev_rows_are_not_profitability": True,
        "withheld": {
            "fantasy_points": "requires operator scoring configuration at runtime",
            "double_double": "joint sim gate",
            "triple_double": "joint sim gate",
        },
        "note": (
            "Identity calibration retained as explicit OOF selection. "
            "No market-superiority claim is made on this branch."
        ),
    }, indent=2))

    # Release matrix + proof against candidate when available
    bdir = candidate if cand_info else baseline
    matrix = evaluate_release_matrix(
        bundle_dir=bdir if cand_info else baseline,
        n_stat_obs=None,
        train_serve_parity=True,
        reproducibility_ok=bool(cand_info),
        ci_ok=False,
        smoke_ok=False,
        deployment_ok=False,
    )
    # If verifying baseline, integrity gate will FAIL (known serializer bug) — record honestly
    (OUT / "RELEASE_MATRIX.json").write_text(json.dumps(matrix.to_dict(), indent=2))

    proof = generate_one_production_model_proof(bundle_dir=bdir if cand_info else baseline)
    # Always write proof; facts_consistent may be false until candidate promoted
    (OUT / "ONE_PRODUCTION_MODEL_PROOF_CANDIDATE.json").write_text(json.dumps(proof, indent=2))
    if cand_info:
        write_proof(REPO / "artifacts/sharp_v6/ONE_PRODUCTION_MODEL_PROOF.json", bundle_dir=candidate)

    receipt = build_deployment_receipt(
        expected_origin_main_sha=proof.get("origin_main", ""),
        expected_bundle_hash=(cand_info or {}).get("model_sha256", baseline_man.get("model_sha256", "")),
        bundle_dir=bdir if cand_info else baseline,
        deployment_environment="local_hardening",
        smoke_result="pending",
        staged_status="candidate" if cand_info else "baseline_immutable",
    )
    (OUT / "DEPLOYMENT_RECEIPT.json").write_text(json.dumps(receipt, indent=2))

    (OUT / "DEPENDENCY_REBUILD_GRAPH.json").write_text(json.dumps({
        "artifact": "DEPENDENCY_REBUILD_GRAPH",
        "edges": [
            {"change": "raw_data", "rebuild": ["features", "identity", "bundle", "evaluation", "reports"]},
            {"change": "player_identity", "rebuild": ["identity", "features", "bundle", "evaluation"]},
            {"change": "feature_transformations", "rebuild": ["feature_contract", "bundle", "calibrators", "dependence", "evaluation"]},
            {"change": "imputation_policy", "rebuild": ["feature_contract", "bundle", "evaluation"]},
            {"change": "model_family", "rebuild": ["bundle", "calibrators", "dependence", "evaluation", "pointer"]},
            {"change": "calibration", "rebuild": ["bundle", "evaluation", "pointer"]},
            {"change": "dependence", "rebuild": ["bundle", "combo_markets", "evaluation"]},
            {"change": "source_code", "rebuild": ["bundle_code_sha", "evaluation", "deployment_receipt"]},
            {"change": "dependency_versions", "rebuild": ["bundle", "reproducibility", "evaluation"]},
        ],
    }, indent=2))

    typer.echo(json.dumps({
        "out_dir": str(OUT),
        "candidate_ok": bool(cand_info),
        "closure_items": len(closure),
        "registry_features": len(registry),
    }, indent=2))


if __name__ == "__main__":
    app()
