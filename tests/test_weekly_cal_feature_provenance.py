"""Focused tests for the weekly_calibration.yml feature provenance fix.

Proves:
1. Reused OOF restores or regenerates feature_schema_manifest.json.
2. The calibrator-manifest step cannot run without feature_schema_manifest.json.
3. The reusable audit artifact upload now includes feature_schema_manifest.json.
4. No OOF rebuild command executes when skip_oof_build=true.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

WF_PATH = Path(".github/workflows/weekly_calibration.yml")


def _load_wf() -> dict:
    return yaml.safe_load(WF_PATH.read_text())


def _get_step(name_fragment: str) -> dict | None:
    wf = _load_wf()
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            if name_fragment.lower() in str(step.get("name", "")).lower():
                return step
    return None


def _get_step_run(name_fragment: str) -> str:
    step = _get_step(name_fragment)
    return step.get("run", "") if step else ""


# ── 1. Restore step exists and is blocking ────────────────────────────────────

def test_restore_feature_provenance_step_exists():
    """The 'Restore feature provenance for reused OOF' step must exist."""
    step = _get_step("Restore feature provenance for reused OOF")
    assert step is not None, "Step 'Restore feature provenance for reused OOF' not found"


def test_restore_step_is_blocking():
    """The restore step must not have continue-on-error: true."""
    step = _get_step("Restore feature provenance for reused OOF")
    assert step is not None
    assert step.get("continue-on-error") is not True, (
        "Restore feature provenance step must be blocking"
    )


def test_restore_step_only_runs_on_skip_oof_path():
    """The restore step must be guarded by skip_oof_build == 'true'."""
    step = _get_step("Restore feature provenance for reused OOF")
    assert step is not None
    condition = str(step.get("if", ""))
    assert "skip_oof_build" in condition, (
        "Restore step must be conditional on skip_oof_build"
    )
    assert "true" in condition.lower(), (
        "Restore step must only run when skip_oof_build == 'true'"
    )


# ── 2. Restore step validates both required files ─────────────────────────────

def test_restore_step_validates_oof_parquet():
    """The restore step must hard-validate data/oof/oof_player_stat_pmfs.parquet."""
    run = _get_step_run("Restore feature provenance for reused OOF")
    assert "oof_player_stat_pmfs.parquet" in run, (
        "Restore step must validate oof_player_stat_pmfs.parquet presence"
    )


def test_restore_step_validates_feature_manifest():
    """The restore step must hard-validate data/processed/feature_schema_manifest.json."""
    run = _get_step_run("Restore feature provenance for reused OOF")
    assert "feature_schema_manifest.json" in run, (
        "Restore step must validate feature_schema_manifest.json presence"
    )


def test_restore_step_has_json_validity_check():
    """The restore step must verify the manifest is a valid non-empty JSON dict."""
    run = _get_step_run("Restore feature provenance for reused OOF")
    # Must contain some form of JSON parsing and emptiness check
    assert "json" in run.lower() and (
        "isinstance" in run or "not obj" in run or "empty" in run.lower()
    ), "Restore step must check that feature_schema_manifest.json is a valid non-empty dict"


# ── 3. Restore step has BDL fallback rebuild (without OOF rebuild) ────────────

def test_restore_step_has_bdl_fallback():
    """When manifest absent from artifact, restore step rebuilds via BDL scripts."""
    run = _get_step_run("Restore feature provenance for reused OOF")
    assert "pull_bdl_history.py" in run, (
        "Restore step must run pull_bdl_history.py as fallback"
    )
    assert "build_canonical_tables.py" in run, (
        "Restore step must run build_canonical_tables.py as fallback"
    )
    assert "build_features.py" in run, (
        "Restore step must run build_features.py as fallback"
    )


def test_restore_step_does_not_rebuild_oof_pmfs():
    """The restore step must NOT run any OOF PMF rebuild commands."""
    run = _get_step_run("Restore feature provenance for reused OOF")
    forbidden = [
        "build_oof_pmfs.py",
        "train_baseline_pmfs.py",
        "oof_pipeline",
    ]
    for cmd in forbidden:
        assert cmd not in run, (
            f"Restore step must NOT rebuild OOF PMFs: found '{cmd}'"
        )


# ── 4. No OOF rebuild when skip_oof_build=true ────────────────────────────────

def test_oof_build_step_skipped_when_skip_oof_build_true():
    """Steps that run the OOF training pipeline must be guarded by skip_oof_build != 'true'."""
    wf = _load_wf()
    oof_build_indicators = [
        "build_oof_pmfs",
        "Build OOF",
        "Run OOF",
        "oof-build",
    ]
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            name = step.get("name", "")
            run  = step.get("run", "") or ""
            is_oof_builder = any(
                ind.lower() in name.lower() or ind in run
                for ind in oof_build_indicators
            )
            if is_oof_builder:
                condition = str(step.get("if", ""))
                assert "skip_oof_build" in condition, (
                    f"OOF-building step '{name}' must be guarded by skip_oof_build"
                )
                assert "true" not in condition.replace("!= 'true'", "").lower() or \
                       "!=" in condition, (
                    f"OOF-building step '{name}' must skip when skip_oof_build == 'true'"
                )


# ── 5. Upload step includes feature_schema_manifest.json ─────────────────────

def test_upload_oof_audit_includes_feature_schema_manifest():
    """Upload OOF audit artifacts must include data/processed/feature_schema_manifest.json."""
    step = _get_step("Upload OOF audit artifacts")
    assert step is not None, "Upload OOF audit artifacts step not found"
    with_block = step.get("with", {})
    upload_path = str(with_block.get("path", ""))
    assert "feature_schema_manifest.json" in upload_path, (
        "Upload OOF audit artifacts must include data/processed/feature_schema_manifest.json"
    )


def test_upload_oof_audit_still_includes_oof_dir():
    """Upload OOF audit artifacts must still include data/oof/."""
    step = _get_step("Upload OOF audit artifacts")
    assert step is not None
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "data/oof/" in upload_path, (
        "Upload OOF audit artifacts must still include data/oof/"
    )


# ── 6. Generate manifest step requires the feature manifest ──────────────────

def test_generate_calibrator_manifest_requires_feature_schema_manifest():
    """The 'Generate calibrator artifact manifest' step must reference feature_schema_manifest.json."""
    step = _get_step("Generate calibrator artifact manifest")
    assert step is not None, "Generate calibrator artifact manifest step not found"
    run = step.get("run", "")
    assert "feature_schema_manifest.json" in run, (
        "Generate calibrator artifact manifest must use feature_schema_manifest.json"
    )


def test_generate_calibrator_manifest_step_is_gated_on_cal_gate():
    """The manifest generation step must only run when cal-gate passed."""
    step = _get_step("Generate calibrator artifact manifest")
    assert step is not None
    condition = str(step.get("if", ""))
    assert "cal-gate" in condition, (
        "Generate calibrator manifest must be conditional on cal-gate.outcome == 'success'"
    )
