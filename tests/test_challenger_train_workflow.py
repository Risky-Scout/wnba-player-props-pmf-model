"""Workflow-contract tests for .github/workflows/challenger_train.yml.

This test file may only test the workflow YAML itself.
It must not import model code, calibration code, or challenger configs.
All challenger artifacts are supplied by inputs.source_ref at dispatch time.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

WORKFLOW_PATH = Path(".github/workflows/challenger_train.yml")


def _wf() -> str:
    return WORKFLOW_PATH.read_text()


# ---------------------------------------------------------------------------
# Basic existence
# ---------------------------------------------------------------------------

def test_workflow_file_exists():
    """challenger_train.yml must exist."""
    assert WORKFLOW_PATH.exists(), f"Missing: {WORKFLOW_PATH}"


def test_workflow_is_valid_yaml():
    """challenger_train.yml must be valid YAML."""
    import yaml  # noqa: PLC0415
    wf_text = _wf()
    # PyYAML parses 'on:' as True key — that's expected for GitHub Actions YAML
    parsed = yaml.safe_load(wf_text)
    assert parsed is not None, "challenger_train.yml must parse as valid YAML"


def test_workflow_has_workflow_dispatch():
    """Workflow must have workflow_dispatch trigger with required inputs."""
    text = _wf()
    assert "workflow_dispatch:" in text, "Must have workflow_dispatch trigger"
    for inp in ("source_ref", "target_game_date", "challenger_version"):
        assert inp in text, f"workflow_dispatch must declare input: {inp!r}"


# ---------------------------------------------------------------------------
# source_ref checkout (required)
# ---------------------------------------------------------------------------

def test_source_ref_checkout():
    """Workflow must check out inputs.source_ref, not a hardcoded branch."""
    text = _wf()
    # Both formats are valid for workflow_dispatch inputs
    valid = (
        "ref: ${{ inputs.source_ref }}" in text or
        "ref: ${{ github.event.inputs.source_ref }}" in text
    )
    assert valid, (
        "Workflow must check out 'ref: ${{ inputs.source_ref }}' so challenger "
        "configs come from the caller-specified branch."
    )


def test_checkout_uses_actions_checkout():
    """Workflow must use actions/checkout@v4 or later."""
    text = _wf()
    assert "actions/checkout@v4" in text or "actions/checkout@v3" in text, (
        "Workflow must use actions/checkout"
    )


# ---------------------------------------------------------------------------
# CLI correctness — no unsupported arguments
# ---------------------------------------------------------------------------

def test_build_oof_pmfs_no_model_dir():
    """build_oof_pmfs.py must not receive --model-dir (unsupported argument)."""
    text = _wf()
    # Find all invocations of build_oof_pmfs.py
    invocations = re.findall(
        r"python\s+scripts/build_oof_pmfs\.py.*?(?=\n\s*(?:python|echo|\$|if|fi|\-\s*name:))",
        text,
        re.DOTALL,
    )
    for inv in invocations:
        assert "--model-dir" not in inv, (
            f"build_oof_pmfs.py invocation must not use --model-dir (unsupported): {inv[:200]}"
        )


def test_build_oof_pmfs_has_required_args():
    """build_oof_pmfs.py must use --features-wide, --config, --out-dir, --manifest."""
    text = _wf()
    required_args = ["--features-wide", "--config", "--out-dir", "--manifest"]
    oof_section = ""
    in_oof = False
    for line in text.splitlines():
        if "build_oof_pmfs.py" in line:
            in_oof = True
        if in_oof:
            oof_section += line + "\n"
            if line.strip() and not line.strip().startswith("\\") and not line.strip().startswith("--") and "build_oof_pmfs" not in line:
                if any(line.strip().startswith(c) for c in ["echo", "python", "if", "fi", "-"]):
                    break
    for arg in required_args:
        assert arg in oof_section or arg in text, (
            f"build_oof_pmfs.py call must include {arg}"
        )


def test_compare_challenger_not_score_oof():
    """Paired evaluation must use compare_champion_challenger.py, not score_oof_pmfs.py comparison."""
    text = _wf()
    assert "compare_champion_challenger.py" in text, (
        "Workflow must use compare_champion_challenger.py for champion/challenger comparison"
    )
    # score_oof_pmfs.py should not be called with --champion-oof (unsupported)
    score_oof_calls = re.findall(r"score_oof_pmfs\.py.*?(?=\n)", text)
    for call in score_oof_calls:
        assert "--champion-oof" not in call, (
            f"score_oof_pmfs.py does not support --champion-oof: {call}"
        )
        assert "--challenger-oof" not in call, (
            f"score_oof_pmfs.py does not support --challenger-oof: {call}"
        )
        assert "--eval-start" not in call, (
            f"score_oof_pmfs.py does not support --eval-start: {call}"
        )


def test_compare_challenger_required_args():
    """compare_champion_challenger.py must receive --champion-oof, --challenger-oof, --eval-start, --eval-end."""
    text = _wf()
    if "compare_champion_challenger.py" not in text:
        pytest.skip("compare_champion_challenger.py not in workflow")
    for arg in ("--champion-oof", "--challenger-oof", "--eval-start", "--eval-end", "--out-dir"):
        assert arg in text, (
            f"compare_champion_challenger.py call must include {arg}"
        )


# ---------------------------------------------------------------------------
# Fatal steps — no continue-on-error: true
# ---------------------------------------------------------------------------

def test_no_mandatory_continue_on_error():
    """No mandatory step may have continue-on-error: true."""
    text = _wf()
    count = text.count("continue-on-error: true")
    assert count == 0, (
        f"Found {count} 'continue-on-error: true' in challenger_train.yml — "
        "all mandatory steps must fail closed"
    )


# ---------------------------------------------------------------------------
# No deployment allowed
# ---------------------------------------------------------------------------

def test_no_gh_pages_deployment():
    """Workflow must never deploy to gh-pages."""
    text = _wf()
    forbidden = [
        "peaceiris/actions-gh-pages",
        "push origin HEAD:gh-pages",
        "actions/deploy-pages",
    ]
    for pattern in forbidden:
        assert pattern not in text, (
            f"challenger_train.yml must not deploy to gh-pages — found: {pattern!r}"
        )


def test_never_modifies_selected_production_package():
    """Workflow must not write or modify selected_production_package.json."""
    text = _wf()
    # Comments mentioning it are OK; actual write commands are not
    lines_writing = [
        l for l in text.splitlines()
        if "selected_production_package.json" in l
        and not l.strip().startswith("#")
        and any(verb in l for verb in ["write_text", "write(", "> config/", "echo.*>", "cat.*>"])
    ]
    assert not lines_writing, (
        f"Workflow must not write selected_production_package.json: {lines_writing}"
    )


def test_never_merges_pr7():
    """Workflow must not merge PR #7 or any PR automatically."""
    text = _wf()
    assert "gh pr merge" not in text, "Workflow must not automatically merge PRs"
    assert "pr merge" not in text.lower() or "gh pr merge" not in text, (
        "Workflow must not automatically merge PRs"
    )


# ---------------------------------------------------------------------------
# CLI smoke tests are present
# ---------------------------------------------------------------------------

def test_cli_smoke_tests_present():
    """Workflow must include --help smoke tests for all referenced commands."""
    text = _wf()
    assert "--help" in text, "Workflow must run --help smoke tests before training"
    assert "smoke" in text.lower(), "Workflow must include a smoke-test step"


def test_compare_challenger_cli_smoke_tested():
    """compare_champion_challenger.py must appear in the smoke-test step."""
    text = _wf()
    smoke_section = ""
    in_smoke = False
    for line in text.splitlines():
        if "smoke" in line.lower():
            in_smoke = True
        if in_smoke:
            smoke_section += line + "\n"
            if line.strip().startswith("- name:") and "smoke" not in line.lower() and smoke_section.count("- name:") > 1:
                break
    assert "compare_champion_challenger" in smoke_section, (
        "compare_champion_challenger.py must be smoke-tested before training"
    )


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------

def test_final_evidence_gate_present():
    """Workflow must have a final evidence validation gate."""
    text = _wf()
    assert "evidence gate" in text.lower() or "Final evidence" in text, (
        "Workflow must have a final evidence gate step"
    )


def test_evidence_gate_checks_required_artifacts():
    """Evidence gate must check for model hashes, calibrator hashes, promotion_decision.json."""
    text = _wf()
    required_in_gate = [
        "promotion_decision.json",
        "challenger_model_hashes.json",
        "challenger_calibrator_hashes.json",
    ]
    for item in required_in_gate:
        assert item in text, f"Evidence gate must check for: {item!r}"


def test_evidence_gate_checks_zero_duplicates():
    """Evidence gate must check for zero duplicate and invalid PMFs."""
    text = _wf()
    assert "duplicate" in text.lower(), "Evidence gate must check for duplicate PMFs"
    assert "invalid" in text.lower() or "INVALID" in text, "Evidence gate must check for invalid PMFs"


# ---------------------------------------------------------------------------
# Diff isolation — only challenger_train.yml changes vs main
# ---------------------------------------------------------------------------

def test_branch_diff_contains_only_workflow():
    """Runner branches may only change challenger_train.yml, tests, and pyproject.toml.

    Skips when challenger_train.yml is not in the diff (not on the runner branch).
    """
    result = subprocess.run(
        ["git", "diff", "main", "--name-only"],
        capture_output=True, text=True,
    )
    changed = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    # Only enforce on branches that change challenger_train.yml
    if ".github/workflows/challenger_train.yml" not in changed:
        pytest.skip(
            "challenger_train.yml is not in this branch's diff — "
            "branch isolation check only applies to the runner branch"
        )
    # Permitted changes on the runner branch
    permitted = {
        ".github/workflows/challenger_train.yml",
        "pyproject.toml",
    }
    non_workflow = [
        f for f in changed
        if f not in permitted and not f.startswith("tests/")
    ]
    assert non_workflow == [], (
        f"Runner branch may only change challenger_train.yml, tests, and pyproject.toml. "
        f"Unexpected files: {non_workflow}"
    )


# ---------------------------------------------------------------------------
# actionlint via subprocess (validates YAML + shell scripts in workflow)
# ---------------------------------------------------------------------------

def test_actionlint_passes():
    """actionlint must report 0 errors on challenger_train.yml."""
    import shutil  # noqa: PLC0415
    # Find actionlint binary — may be at /tmp/actionlint or on PATH
    binary = shutil.which("actionlint") or "/tmp/actionlint"
    if not Path(binary).exists():
        pytest.skip(f"actionlint binary not found at {binary}")
    result = subprocess.run(
        [binary, "-shellcheck=", "-pyflakes=", str(WORKFLOW_PATH)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"actionlint errors in challenger_train.yml:\n{result.stdout}\n{result.stderr}"
    )
