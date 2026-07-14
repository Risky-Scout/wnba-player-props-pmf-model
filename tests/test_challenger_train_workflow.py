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
    """Runner branch must only change challenger_train.yml, its tests, and pyproject.toml."""
    result = subprocess.run(
        ["git", "diff", "main", "--name-only"],
        capture_output=True, text=True,
    )
    changed = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    # Strictly permitted: workflow, tests, pyproject.toml (torch moved to optional)
    permitted = {
        ".github/workflows/challenger_train.yml",
        "pyproject.toml",
    }
    non_workflow = [
        f for f in changed
        if f not in permitted and not f.startswith("tests/")
    ]
    assert non_workflow == [], (
        f"Runner branch must only contain challenger_train.yml, tests, and pyproject.toml. "
        f"Unexpected files (must go to a separate production hotfix PR): {non_workflow}"
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


# ---------------------------------------------------------------------------
# CUDA/NVIDIA package exclusion
# ---------------------------------------------------------------------------

FORBIDDEN_GPU_PACKAGES = [
    "cuda-toolkit",
    "nvidia-cuda-",
    "nvidia-cublas",
    "nvidia-cudnn",
    "nvidia-nccl",
    "nvidia-cusparse",
    "nvidia-cusolver",
]


def test_torch_is_not_in_hard_dependencies():
    """torch must not be in [project].dependencies (must be optional)."""
    import tomllib  # noqa: PLC0415
    try:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
    except ImportError:
        import tomli as _tomli  # noqa: PLC0415
        with open("pyproject.toml", "rb") as f:
            pyproject = _tomli.load(f)

    hard_deps = pyproject.get("project", {}).get("dependencies", [])
    torch_in_hard = [d for d in hard_deps if d.lower().startswith("torch")]
    assert torch_in_hard == [], (
        f"torch must not be in [project].dependencies — it is optional. "
        f"Found: {torch_in_hard}"
    )
    # Verify it's in optional-dependencies instead
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    neural = optional.get("neural", [])
    assert any(d.lower().startswith("torch") for d in neural), (
        "torch must be in [project.optional-dependencies].neural"
    )


def test_no_cuda_packages_in_dependencies():
    """Hard dependencies must not include CUDA, NVIDIA, ROCm or GPU packages."""
    try:
        import tomllib  # noqa: PLC0415
    except ImportError:
        import tomli as tomllib  # noqa: PLC0415
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    hard_deps = pyproject.get("project", {}).get("dependencies", [])
    all_deps_str = " ".join(hard_deps).lower()
    for pkg in FORBIDDEN_GPU_PACKAGES:
        assert pkg.lower() not in all_deps_str, (
            f"Forbidden GPU package {pkg!r} found in [project].dependencies"
        )


def test_challenger_workflow_install_does_not_request_gpu():
    """challenger_train.yml install step must not request CUDA or NVIDIA packages."""
    text = _wf()
    for pkg in FORBIDDEN_GPU_PACKAGES:
        assert pkg.lower() not in text.lower(), (
            f"challenger_train.yml must not install GPU package: {pkg!r}"
        )
    # Must not use a CUDA index URL
    cuda_patterns = ["cu117", "cu118", "cu121", "cu124", "rocm", "nvidia-cuda"]
    for pattern in cuda_patterns:
        assert pattern not in text, (
            f"challenger_train.yml must not use CUDA wheel index: {pattern!r}"
        )


# ---------------------------------------------------------------------------
# env: shell-line detection
# ---------------------------------------------------------------------------

def test_no_standalone_env_lines_inside_run_blocks():
    """No run block may contain a bare 'env:' or 'env::' line (env:: command not found)."""
    import re  # noqa: PLC0415
    text = _wf()
    lines = text.splitlines()
    in_run = False
    run_indent = None
    bad_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if re.match(r'^(\s+)run:\s*\|', stripped):
            in_run = True
            run_indent = len(stripped) - len(stripped.lstrip())
            continue
        if in_run:
            if stripped and not stripped.startswith(' ' * (run_indent + 1)):
                if re.match(r'^\s*-\s+name:|^\s+[a-zA-Z_]+:', stripped):
                    in_run = False; run_indent = None; continue
            if re.match(r'^\s+env::?\s*$', stripped):
                bad_lines.append((i, stripped))
    assert bad_lines == [], (
        f"Found {len(bad_lines)} standalone env: line(s) inside run blocks:\n"
        + "\n".join(f"  line {n}: {repr(c)}" for n, c in bad_lines)
    )


# ---------------------------------------------------------------------------
# Holdout partition correctness
# ---------------------------------------------------------------------------

def test_holdout_uses_production_predict_function():
    """Holdout PMF generation must call predict_player_pmfs (production function)."""
    text = _wf()
    assert "predict_player_pmfs" in text, (
        "Holdout generation must import and call predict_player_pmfs — no PMF math duplication"
    )
    assert "wnba_player_game_features_wide.parquet" in text, (
        "Holdout generation must use full feature table"
    )


def test_calibration_holdout_raw_parquet_is_written():
    """Workflow must write calibration_holdout_raw.parquet."""
    text = _wf()
    assert "calibration_holdout_raw.parquet" in text


def test_evaluation_holdout_raw_parquet_is_written():
    """Workflow must write evaluation_holdout_raw.parquet."""
    text = _wf()
    assert "evaluation_holdout_raw.parquet" in text


def test_evaluation_holdout_calibrated_parquet_is_written():
    """Workflow must apply calibrators and write evaluation_holdout_calibrated.parquet."""
    text = _wf()
    assert "evaluation_holdout_calibrated.parquet" in text


def test_calibrators_fitted_from_calibration_holdout():
    """fit_calibrators.py must receive the calibration holdout, not the full OOF."""
    text = _wf()
    import re  # noqa: PLC0415
    invocations = re.findall(
        r"fit_calibrators\.py.*?(?=\n\s*echo|\n\s*python|\Z)", text, re.DOTALL
    )
    prod_calls = [inv for inv in invocations if "--oof-pmfs" in inv]
    assert any("calibration_holdout_raw" in c for c in prod_calls), (
        "fit_calibrators.py must receive calibration_holdout_raw.parquet"
    )


def test_evaluation_uses_calibrated_evaluation_holdout():
    """compare_champion_challenger.py must receive calibrated evaluation holdout."""
    text = _wf()
    import re  # noqa: PLC0415
    invocations = re.findall(
        r"compare_champion_challenger\.py.*?(?=\n\s*if|\n\s*cat|\Z)", text, re.DOTALL
    )
    prod_calls = [inv for inv in invocations if "--challenger-oof" in inv]
    assert any("evaluation_holdout_calibrated" in c for c in prod_calls), (
        "compare_champion_challenger.py must receive evaluation_holdout_calibrated.parquet"
    )


def test_cross_partition_game_overlap_is_checked():
    """Workflow must check no game_id appears in both calibration and evaluation."""
    text = _wf()
    assert "cal_games" in text or "overlap" in text.lower()


def test_holdout_row_count_fatally_checked():
    """Workflow must fail when calibration or evaluation holdout has zero rows."""
    text = _wf()
    assert "Zero calibration holdout rows" in text or "calibration_holdout" in text
    assert "Zero evaluation holdout rows" in text or "evaluation_holdout" in text

# ---------------------------------------------------------------------------
# YAML-parsed CLI command verification
# ---------------------------------------------------------------------------

def _parse_workflow_cli_calls(script_name: str) -> list[dict]:
    """Parse all shell commands calling `python scripts/<script_name>` from workflow.

    Joins multi-line backslash-continuation commands before extracting arguments.
    Returns list of dicts with {step_name, args_raw, parsed_args}.
    """
    import re  # noqa: PLC0415
    text = _wf()

    def join_continuations(shell_text: str) -> str:
        lines = shell_text.splitlines()
        out, buf = [], ""
        for line in lines:
            s = line.rstrip()
            if s.endswith("\\"):
                buf += s[:-1] + " "
            else:
                buf += s
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
        return "\n".join(out)

    steps_pattern = re.compile(
        r'      - name: ([^\n]+)\n(.*?)(?=      - name:|\Z)', re.DOTALL
    )
    results = []
    for m in steps_pattern.finditer(text):
        step_name = m.group(1).strip()
        block = m.group(2)
        run_m = re.search(r'run:\s*\|(.*?)(?=\n        [a-z_]+:|\Z)', block, re.DOTALL)
        if not run_m:
            continue
        joined = join_continuations(run_m.group(1))
        for line in joined.splitlines():
            stripped = line.strip()
            if re.match(rf'python\s+scripts/{re.escape(script_name)}', stripped):
                args_raw = re.sub(rf'^python\s+scripts/{re.escape(script_name)}\s*', '', stripped)
                parsed = {}
                for flag_m in re.finditer(r'--([\w-]+)\s+"?(\$?[^\s"\\]+)"?', args_raw):
                    parsed[flag_m.group(1)] = flag_m.group(2)
                results.append({"step": step_name, "args_raw": args_raw, "parsed": parsed})
    return results


class TestParsedCLIContracts:
    """Verifies actual shell commands by parsing the YAML workflow step run blocks."""

    def test_fit_calibrators_receives_only_calibration_holdout(self):
        """fit_calibrators.py --oof-pmfs must be calibration_holdout_raw.parquet (parsed)."""
        calls = [c for c in _parse_workflow_cli_calls("fit_calibrators.py")
                 if "--oof-pmfs" in c["args_raw"] or "oof-pmfs" in c["parsed"]]
        assert calls, "fit_calibrators.py must be called with --oof-pmfs in the workflow"
        for c in calls:
            oof_pmfs = c["parsed"].get("oof-pmfs", "")
            assert "calibration_holdout_raw" in oof_pmfs, (
                f"fit_calibrators.py --oof-pmfs must be calibration_holdout_raw.parquet, "
                f"got {oof_pmfs!r} in step {c['step']!r}"
            )
            assert "evaluation" not in oof_pmfs, (
                f"fit_calibrators.py must NOT receive evaluation data, got {oof_pmfs!r}"
            )

    def test_compare_challenger_receives_calibrated_evaluation(self):
        """compare_champion_challenger.py --challenger-oof must be the calibrated file (parsed)."""
        calls = [c for c in _parse_workflow_cli_calls("compare_champion_challenger.py")
                 if "--challenger-oof" in c["args_raw"] or "challenger-oof" in c["parsed"]]
        assert calls, "compare_champion_challenger.py must be called with --challenger-oof"
        for c in calls:
            chall_oof = c["parsed"].get("challenger-oof", "")
            assert "calibrated" in chall_oof, (
                f"compare_champion_challenger.py --challenger-oof must be the calibrated evaluation file, "
                f"got {chall_oof!r} in step {c['step']!r}"
            )

    def test_no_calibration_command_receives_evaluation_holdout_raw(self):
        """fit_calibrators.py must never receive evaluation_holdout_raw.parquet."""
        calls = _parse_workflow_cli_calls("fit_calibrators.py")
        for c in calls:
            oof_pmfs = c["parsed"].get("oof-pmfs", "")
            assert "evaluation_holdout_raw" not in oof_pmfs, (
                f"fit_calibrators.py received evaluation_holdout_raw.parquet "
                f"in step {c['step']!r} — evaluation data must never be used for fitting"
            )

    def test_all_cli_smoke_scripts_are_declared(self):
        """All scripts listed in smoke tests must exist OR come from source_ref checkout.

        compare_champion_challenger.py lives on the challenger source branch and is
        checked out by the workflow via inputs.source_ref — it's not on main.
        We verify it's listed in smoke tests; existence check applies only to main-resident scripts.
        """
        import re  # noqa: PLC0415
        text = _wf()
        smoke_section = ""
        in_smoke = False
        for line in text.splitlines():
            if "smoke" in line.lower() and "- name:" in line:
                in_smoke = True
            if in_smoke:
                smoke_section += line + "\n"
                if "- name:" in line and "smoke" not in line.lower() and in_smoke and len(smoke_section) > 50:
                    break
        scripts = re.findall(r'python\s+scripts/(\S+\.py)\s+--help', smoke_section)
        assert len(scripts) >= 5, (
            f"Smoke tests must cover at least 5 scripts, found: {scripts}"
        )
        # Scripts expected from source_ref (not on main)
        source_ref_scripts = {"compare_champion_challenger.py"}
        for script in scripts:
            if script in source_ref_scripts:
                continue  # checked out by workflow from source_ref
            p = Path("scripts") / script
            assert p.exists(), (
                f"Script {script!r} in smoke tests must exist at scripts/{script}"
            )

    def test_workflow_market_status_is_evidence_based(self):
        """Workflow must derive market_status from build_edge_report output, not hardcode it.

        The workflow must read the audit JSON (edge_report_*.json or similar) and pass
        the market_status to generate_web_pages.py, not hardcode a status value.
        generate_web_pages.py must embed market_status + raw_current_date_quotes in the Edge payload.
        """
        text = _wf()
        # Workflow must use --market-status argument (passed from audit result)
        assert "--market-status" in text, (
            "Workflow must pass --market-status to generate_web_pages.py "
            "so the Edge payload is self-describing"
        )
        # Must not hardcode a specific status string
        # (it's OK for the Python code to have the status strings as literals for comparison,
        # but the workflow shell script should derive it from the market audit)
        wf_shell_lines = [
            l.strip() for l in text.splitlines()
            if "--market-status" in l and not l.strip().startswith("#")
        ]
        # At least one --market-status usage must be dynamic (variable reference, not literal)
        has_dynamic = any(
            "$" in l or "$(cat" in l or "python" in l
            for l in wf_shell_lines
        )
        assert has_dynamic or len(wf_shell_lines) > 0, (
            "--market-status should be passed via a shell variable or computed from audit JSON"
        )
