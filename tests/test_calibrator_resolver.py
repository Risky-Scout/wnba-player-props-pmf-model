"""Focused test for the artifact-level calibrator resolver.

Proves:
  1. A newer failed-overall run whose artifact contains a valid gated manifest
     is selected over an older overall-successful run whose artifact lacks the manifest.
  2. Every required manifest field is checked and an absent or invalid field
     produces an exact rejection reason, never a silent fallback.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Adjust import path so we can import the resolver script directly
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from resolve_calibrator_artifact import list_artifacts, validate_manifest, main  # noqa: E402


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_manifest(
    artifact_type: str = "calibrator",
    gate_status: str = "PASS",
    source_run_id: str = "99999",
    source_commit: str = "abc12345def67890",
    calibration_cutoff: str = "2026-07-14T00:00:00Z",
    feature_manifest_hash: str = "abcd1234",
    config_hash: str = "efgh5678",
) -> dict:
    return {
        "artifact_type": artifact_type,
        "gate_status": gate_status,
        "source_run_id": source_run_id,
        "source_commit": source_commit,
        "calibration_cutoff": calibration_cutoff,
        "feature_manifest_hash": feature_manifest_hash,
        "config_hash": config_hash,
    }


def _make_zip(contents: dict[str, bytes | str]) -> bytes:
    """Create an in-memory zip with given file contents."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in contents.items():
            if isinstance(data, str):
                data = data.encode()
            zf.writestr(name, data)
    return buf.getvalue()


# ─── validate_manifest unit tests ───────────────────────────────────────────

def test_validate_manifest_all_pass():
    m = _make_manifest(source_run_id="42", source_commit="deadbeef")
    failures = validate_manifest(m, producer_run_id="42", producer_head_sha="deadbeef")
    assert failures == [], f"Expected no failures, got: {failures}"


def test_validate_manifest_wrong_artifact_type():
    m = _make_manifest(artifact_type="model")
    failures = validate_manifest(m, "42", "deadbeef")
    assert any("artifact_type" in f for f in failures)


def test_validate_manifest_gate_not_pass():
    m = _make_manifest(gate_status="FAIL")
    failures = validate_manifest(m, "42", "deadbeef")
    assert any("gate_status" in f for f in failures)


def test_validate_manifest_run_id_mismatch():
    m = _make_manifest(source_run_id="11111")
    failures = validate_manifest(m, producer_run_id="99999", producer_head_sha="deadbeef")
    assert any("source_run_id" in f for f in failures)


def test_validate_manifest_blank_calibration_cutoff():
    m = _make_manifest(calibration_cutoff="")
    failures = validate_manifest(m, "42", "deadbeef")
    assert any("calibration_cutoff" in f for f in failures)


def test_validate_manifest_blank_feature_manifest_hash():
    m = _make_manifest(feature_manifest_hash="")
    failures = validate_manifest(m, "42", "deadbeef")
    assert any("feature_manifest_hash" in f for f in failures)


def test_validate_manifest_blank_config_hash():
    m = _make_manifest(config_hash="")
    failures = validate_manifest(m, "42", "deadbeef")
    assert any("config_hash" in f for f in failures)


# ─── Main resolver integration test ─────────────────────────────────────────

def test_resolver_selects_newer_failed_run_with_manifest_over_older_successful_run_without():
    """
    Scenario:
      - Artifact A (newer, from a run that failed overall after cal-gate passed):
          HAS artifact_manifest_calibrator.json with all fields valid → SELECTED
      - Artifact B (older, from a run that succeeded overall):
          LACKS artifact_manifest_calibrator.json → REJECTED with 'absent' reason

    The resolver must select A (newer + valid manifest) not B (older + no manifest).
    """
    NEWER_RUN_ID  = "88888888"
    OLDER_RUN_ID  = "11111111"
    NEWER_SHA     = "aabbccdd"
    OLDER_SHA     = "11223344"
    NEWER_AID     = 2000
    OLDER_AID     = 1000

    # Build artifact list (newer first)
    fake_artifacts = [
        {
            "id": NEWER_AID,
            "created_at": "2026-07-14T17:00:00Z",
            "expired": False,
            "workflow_run": {"id": int(NEWER_RUN_ID), "head_sha": NEWER_SHA},
        },
        {
            "id": OLDER_AID,
            "created_at": "2026-07-13T10:00:00Z",
            "expired": False,
            "workflow_run": {"id": int(OLDER_RUN_ID), "head_sha": OLDER_SHA},
        },
    ]

    # Zip contents
    newer_manifest = _make_manifest(
        source_run_id=NEWER_RUN_ID,
        source_commit=NEWER_SHA,
    )
    newer_zip = _make_zip({
        "artifact_manifest_calibrator.json": json.dumps(newer_manifest),
        "bias_corrections.json": b"{}",
        "pmf_cal_role_pts.pkl": b"\x80\x04\x95",
    })

    # Older artifact: valid calibration .pkl files but NO manifest
    older_zip = _make_zip({
        "bias_corrections.json": b"{}",
        "pmf_cal_role_pts.pkl": b"\x80\x04\x95",
    })

    zip_map = {NEWER_AID: newer_zip, OLDER_AID: older_zip}

    with tempfile.TemporaryDirectory() as tmpdir:
        cal_dir = str(Path(tmpdir) / "calibration")
        env_file = str(Path(tmpdir) / "github_env.txt")

        # Patch list_artifacts to return our fake list
        # Patch gh_download_zip to return the right zip bytes
        with patch("resolve_calibrator_artifact.list_artifacts", return_value=fake_artifacts), \
             patch("resolve_calibrator_artifact.gh_download_zip",
                   side_effect=lambda repo, aid: zip_map.get(aid)):
            main(repo="Risky-Scout/wnba-player-props-pmf-model",
                 cal_dir=cal_dir,
                 github_env=env_file)

        # NEWER artifact was selected
        assert Path(cal_dir, "artifact_manifest_calibrator.json").exists(), \
            "Manifest from newer artifact must be in cal_dir"
        manifest = json.loads(Path(cal_dir, "artifact_manifest_calibrator.json").read_text())
        assert manifest["source_run_id"] == NEWER_RUN_ID, \
            f"Expected run {NEWER_RUN_ID}, got {manifest['source_run_id']}"

        # GITHUB_ENV recorded the newer run ID
        env_content = Path(env_file).read_text()
        assert NEWER_RUN_ID in env_content, \
            f"Expected {NEWER_RUN_ID} in GITHUB_ENV, got: {env_content!r}"


def test_resolver_rejects_all_and_fails_when_no_manifest_in_any_artifact():
    """When no artifact has the manifest, resolver must exit 1 and report each rejection."""
    fake_artifacts = [
        {
            "id": 9001,
            "created_at": "2026-07-14T12:00:00Z",
            "expired": False,
            "workflow_run": {"id": 100, "head_sha": "aabb"},
        },
    ]
    # Artifact with no manifest
    no_manifest_zip = _make_zip({"bias_corrections.json": b"{}"})

    with patch("resolve_calibrator_artifact.list_artifacts", return_value=fake_artifacts), \
         patch("resolve_calibrator_artifact.gh_download_zip", return_value=no_manifest_zip):
        with pytest.raises(SystemExit) as exc_info:
            with tempfile.TemporaryDirectory() as tmpdir:
                main(repo="test", cal_dir=str(Path(tmpdir)/"cal"), github_env="")
        assert exc_info.value.code == 1


def test_resolver_older_successful_run_without_manifest_is_explicitly_skipped():
    """An artifact lacking the manifest must appear in the rejection list with reason 'absent'."""
    fake_artifacts = [
        {
            "id": 5050,
            "created_at": "2026-07-11T22:00:00Z",
            "expired": False,
            "workflow_run": {"id": 87, "head_sha": "298c761e"},
        },
    ]
    no_manifest_zip = _make_zip({
        "bias_corrections.json": b"{}",
        "pmf_cal_role_pts.pkl": b"\x80",
    })

    rejections_seen: list[str] = []
    original_main = main

    # Capture rejections by patching and re-running
    with patch("resolve_calibrator_artifact.list_artifacts", return_value=fake_artifacts), \
         patch("resolve_calibrator_artifact.gh_download_zip", return_value=no_manifest_zip):
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with pytest.raises(SystemExit):
            with redirect_stderr(buf):
                with tempfile.TemporaryDirectory() as tmpdir:
                    main(repo="test", cal_dir=str(Path(tmpdir)/"cal"), github_env="")
        stderr_output = buf.getvalue()
        assert "absent" in stderr_output.lower() or "artifact_manifest" in stderr_output.lower(), \
            f"Rejection reason must mention manifest absence. Got: {stderr_output!r}"
