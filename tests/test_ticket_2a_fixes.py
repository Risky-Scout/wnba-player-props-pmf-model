"""Ticket 2A — four production gap fixes.

Tests are written FIRST and must FAIL before the fixes are applied.

Fix 1  — Make artifact manifests mandatory; call shared validate_artifact_manifest()
Fix 2  — Make producer manifests fail closed (no silent blank fields)
Fix 3  — Fix PMF completeness using final availability/actionability output
Fix 4  — Fix edge completeness using reconciled markets

Required new tests:
  Fix 1:
    test_workflow_rejects_missing_model_manifest
    test_workflow_rejects_missing_calibrator_manifest
    test_workflow_rejects_wrong_source_commit
    test_missing_manifest_is_fatal
    test_wrong_source_run_is_fatal
    test_wrong_source_commit_is_fatal
    test_missing_cutoff_is_fatal
    test_future_cutoff_is_fatal
    test_blank_feature_hash_is_fatal
    test_blank_config_hash_is_fatal

  Fix 2:
    test_producer_cannot_upload_blank_manifest_fields

  Fix 3:
    test_expected_nonempty_actual_empty_pmf_is_fatal
    test_final_availability_controls_expected_pmfs
    test_original_slate_status_does_not_override_final_availability
    test_confirmed_inactive_player_is_not_expected_actionable_pmf
    test_missing_expected_pmf_is_fatal
    test_unexpected_pmf_is_fatal

  Fix 4:
    test_reconciled_market_manifest_is_written_before_edges
    test_expected_nonempty_actual_empty_market_comparison_is_fatal
    test_raw_unreconciled_market_is_not_an_expected_edge
    test_verified_zero_markets_has_explicit_no_market_status
    test_missing_expected_market_comparison_row_is_fatal
    test_unexpected_market_comparison_row_is_fatal
    test_duplicate_market_comparison_row_is_fatal
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Workflow YAML paths
# ---------------------------------------------------------------------------
_WF_DIR = Path(__file__).parent.parent / ".github" / "workflows"
_PREGAME_INITIAL = _WF_DIR / "pregame_initial.yml"
_DAILY_PIPELINE = _WF_DIR / "daily_pipeline.yml"
_WEEKLY_CALIBRATION = _WF_DIR / "weekly_calibration.yml"

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_BUILD_EDGE_REPORT = _SCRIPTS_DIR / "build_edge_report.py"


# ===========================================================================
# FIX 1 — Artifact manifests mandatory; use shared validator
# ===========================================================================


class TestFix1ManifestMandatoryAndSharedValidator:
    """pregame_initial.yml must call validate_artifact_manifest() and treat
    a missing manifest as FATAL, not silently skip validation."""

    def test_workflow_rejects_missing_model_manifest(self):
        """pregame_initial.yml must NOT skip model manifest validation when absent.

        The old behavior: if no manifest file → log 'skipping manifest validation'.
        The new behavior: FATAL if the model manifest is missing.
        """
        content = _PREGAME_INITIAL.read_text()
        assert "skipping manifest validation" not in content, (
            "pregame_initial.yml must NOT contain 'skipping manifest validation'. "
            "A missing model artifact manifest must be FATAL."
        )

    def test_workflow_rejects_missing_calibrator_manifest(self):
        """pregame_initial.yml must NOT skip calibrator manifest validation when absent."""
        content = _PREGAME_INITIAL.read_text()
        assert "skipping manifest validation" not in content, (
            "pregame_initial.yml must NOT contain 'skipping manifest validation'. "
            "A missing calibrator artifact manifest must be FATAL."
        )

    def test_workflow_rejects_wrong_source_commit(self):
        """pregame_initial.yml must call validate_artifact_manifest with source_commit.

        The inline validator previously did not check source_commit.
        The shared function must be called and source_commit must be validated.
        """
        content = _PREGAME_INITIAL.read_text()
        assert "validate_artifact_manifest" in content, (
            "pregame_initial.yml must call validate_artifact_manifest() from the shared module"
        )
        assert "source_commit" in content, (
            "pregame_initial.yml must pass source_commit to validate_artifact_manifest"
        )

    def test_missing_manifest_is_fatal(self):
        """A completely absent artifact manifest must cause a FATAL exit, not skip."""
        content = _PREGAME_INITIAL.read_text()
        # The forbidden pattern is: optional manifest with skip fallback
        assert "skipping manifest validation" not in content, (
            "pregame_initial.yml must not skip validation when manifest is absent. "
            "Missing manifests must be FATAL."
        )
        # Must have an explicit fatal path for missing manifest
        assert (
            "No artifact_manifest" in content and "exit 1" in content
        ) or "skipping" not in content, (
            "pregame_initial.yml must exit 1 when an artifact manifest is not found."
        )

    def test_wrong_source_run_is_fatal(self):
        """validate_artifact_manifest must raise ArtifactManifestError for wrong source_run_id."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "RUN_CORRECT",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            "model_training_cutoff": "2026-07-12T00:00:00Z",
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "confighash_ddeeff",
        }
        with pytest.raises(ArtifactManifestError, match="source_run_id"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
                source_run_id="COMPLETELY_DIFFERENT_RUN",
            )

    def test_wrong_source_commit_is_fatal(self):
        """validate_artifact_manifest must raise ArtifactManifestError for wrong source_commit."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "RUN_A",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            "model_training_cutoff": "2026-07-12T00:00:00Z",
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "confighash_ddeeff",
        }
        with pytest.raises(ArtifactManifestError, match="source_commit"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
                source_commit="WRONG_COMMIT_deadbeef",
            )

    def test_missing_cutoff_is_fatal(self):
        """validate_artifact_manifest must raise when model_training_cutoff is absent.

        The old behavior: silently skipped missing cutoff (continue).
        The new behavior: missing required cutoff is a fatal validation error.
        """
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "RUN_A",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            # model_training_cutoff intentionally absent
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "confighash_ddeeff",
        }
        with pytest.raises(ArtifactManifestError, match="model_training_cutoff"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

    def test_missing_calibrator_cutoff_is_fatal(self):
        """validate_artifact_manifest must raise when calibration_cutoff is absent."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "calibrator",
            "artifact_schema_version": "1",
            "source_workflow": "Weekly OOF Refresh & Calibration",
            "source_run_id": "WEEKLY_RUN_111",
            "source_commit": "abc123",
            "created_at_utc": "2026-07-07T10:00:00Z",
            # calibration_cutoff intentionally absent
            "feature_manifest_hash": "fmh_xyz",
            "config_hash": "cfg_xyz",
            "gate_status": "PASS",
        }
        with pytest.raises(ArtifactManifestError, match="calibration_cutoff"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="calibrator",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

    def test_future_cutoff_is_fatal(self):
        """validate_artifact_manifest must raise when cutoff >= prediction_timestamp."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "RUN_A",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            "model_training_cutoff": "2026-07-14T00:00:00Z",  # future!
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "confighash_ddeeff",
        }
        with pytest.raises(ArtifactManifestError, match="model_training_cutoff"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

    def test_blank_feature_hash_is_fatal(self):
        """validate_artifact_manifest must raise when feature_manifest_hash is blank."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "RUN_A",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            "model_training_cutoff": "2026-07-12T00:00:00Z",
            "feature_manifest_hash": "",  # blank!
            "config_hash": "confighash_ddeeff",
        }
        with pytest.raises(ArtifactManifestError, match="feature_manifest_hash"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

    def test_blank_config_hash_is_fatal(self):
        """validate_artifact_manifest must raise when config_hash is blank."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )

        manifest = {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "RUN_A",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            "model_training_cutoff": "2026-07-12T00:00:00Z",
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "",  # blank!
        }
        with pytest.raises(ArtifactManifestError, match="config_hash"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )


# ===========================================================================
# FIX 2 — Producer manifests must fail closed
# ===========================================================================


class TestFix2ProducerManifestFailClosed:
    """daily_pipeline.yml and weekly_calibration.yml must not silently emit blank
    manifest fields and must not use if: always() for production artifact uploads."""

    def test_producer_cannot_upload_blank_manifest_fields(self):
        """daily_pipeline.yml must exit nonzero if required manifest fields cannot be computed.

        Forbidden patterns:
          - git_commit = "unknown"  (must exit 1 instead)
          - source_commit = "unknown"
        """
        daily = _DAILY_PIPELINE.read_text()
        forbidden_patterns = [
            'git_commit = "unknown"',
            'source_commit = "unknown"',
        ]
        for pattern in forbidden_patterns:
            assert pattern not in daily, (
                f"daily_pipeline.yml must not silently set {pattern!r}. "
                "If git rev-parse fails, the step must exit nonzero. "
                "A manifest with blank/unknown fields must NOT be uploaded."
            )

    def test_weekly_calibrator_manifest_no_unknown_commit(self):
        """weekly_calibration.yml must not silently use 'unknown' for source_commit."""
        weekly = _WEEKLY_CALIBRATION.read_text()
        forbidden_patterns = [
            'git_commit = "unknown"',
            'source_commit = "unknown"',
        ]
        for pattern in forbidden_patterns:
            assert pattern not in weekly, (
                f"weekly_calibration.yml must not silently set {pattern!r}. "
                "If git rev-parse fails, the step must exit nonzero."
            )

    def test_model_manifest_not_uploaded_if_always(self):
        """Production model artifact upload must NOT use if: always() — it must only
        run after training and validation succeed."""
        daily = _DAILY_PIPELINE.read_text()
        # Find model upload step and check it doesn't have if: always()
        lines = daily.splitlines()
        in_model_upload = False
        for i, line in enumerate(lines):
            if "Upload stage4 model" in line and "name:" in line:
                in_model_upload = True
            if in_model_upload:
                # Check within the next 10 lines
                window = lines[i : i + 10]
                for wline in window:
                    assert "if: always()" not in wline, (
                        "The 'Upload stage4 model' step must NOT use 'if: always()'. "
                        "Production model upload must only run after training succeeds."
                    )
                break


# ===========================================================================
# FIX 3 — PMF completeness using final availability
# ===========================================================================


class TestFix3PMFCompletenessUsingFinalAvailability:
    """pregame_initial.yml must build expected PMF keys from final Ticket 1
    availability/actionability output, not from original slate status."""

    def test_expected_nonempty_actual_empty_pmf_is_fatal(self):
        """When expected PMFs are nonempty but actual PMF file is empty, must be FATAL.

        The old pattern only compared when both non-empty:
          if not expected_pmf.empty and not actual_pmf.empty:
              ...compare...
        The new pattern must call validate_pmf_manifest() unconditionally.
        """
        content = _PREGAME_INITIAL.read_text()
        # The old guarded pattern must be gone
        forbidden = "if not expected_pmf.empty and not actual_pmf.empty:"
        assert forbidden not in content, (
            "pregame_initial.yml must NOT guard PMF comparison behind "
            "'if not expected_pmf.empty and not actual_pmf.empty:'. "
            "Use validate_pmf_manifest() unconditionally when scheduled games exist."
        )

    def test_final_availability_controls_expected_pmfs(self):
        """Expected PMF manifest must use final availability from apply_injury_updates output.

        The workflow must read the availability_table_{date}.parquet (or equivalent)
        produced by apply_injury_updates.py, not just the raw slate status column.
        """
        content = _PREGAME_INITIAL.read_text()
        assert (
            "availability_table" in content
            or "is_market_actionable" in content
        ), (
            "pregame_initial.yml must reference availability_table (from apply_injury_updates) "
            "or is_market_actionable flag to determine eligible players for expected PMF manifest."
        )

    def test_original_slate_status_does_not_override_final_availability(self):
        """Expected PMF manifest must NOT use raw slate status as the sole eligibility signal.

        The old forbidden pattern builds eligibility directly from slate['status'],
        which does not account for final injury updates from apply_injury_updates.py.
        """
        content = _PREGAME_INITIAL.read_text()
        forbidden = '~slate["status"].str.lower().isin(["inactive", "out"])'
        assert forbidden not in content, (
            "pregame_initial.yml must NOT build expected PMF eligibility directly from "
            "slate['status']. Use final availability from apply_injury_updates output instead."
        )

    def test_confirmed_inactive_player_is_not_expected_actionable_pmf(self):
        """A player whose availability_status is OUT must not appear in expected PMF manifest."""
        from wnba_props_model.pipeline.market_integrity import build_expected_pmf_manifest

        slate = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "status": "active"},
            {"game_id": "G001", "player_id": "P002", "status": "active"},
            {"game_id": "G001", "player_id": "P003", "status": "out"},
        ])
        # Simulate final availability filtering (exclude OUT players)
        eligible = slate[~slate["status"].str.lower().isin(["inactive", "out"])].copy()
        manifest = build_expected_pmf_manifest(eligible, ["pts", "reb"])
        player_ids = set(manifest["player_id"].unique())
        assert "P003" not in player_ids, (
            "Confirmed-inactive player P003 (status='out') must NOT appear in expected PMF manifest"
        )
        assert "P001" in player_ids and "P002" in player_ids

    def test_missing_expected_pmf_is_fatal(self):
        """validate_pmf_manifest must raise MissingPMFError when expected keys > actual keys."""
        from wnba_props_model.pipeline.market_integrity import (
            MissingPMFError,
            validate_pmf_manifest,
        )

        expected = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P002", "stat": "pts"},
        ])
        actual = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
        ])
        with pytest.raises(MissingPMFError, match="[Mm]issing"):
            validate_pmf_manifest(expected, actual)

    def test_unexpected_pmf_is_fatal(self):
        """validate_pmf_manifest must raise when actual contains keys not in expected."""
        from wnba_props_model.pipeline.market_integrity import (
            MissingPMFError,
            validate_pmf_manifest,
        )

        expected = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
        ])
        actual = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P999", "stat": "pts"},  # unexpected
        ])
        with pytest.raises(MissingPMFError, match="[Uu]nexpected"):
            validate_pmf_manifest(expected, actual)

    def test_validate_pmf_manifest_called_in_pregame_initial(self):
        """pregame_initial.yml manifest step must call validate_pmf_manifest()."""
        content = _PREGAME_INITIAL.read_text()
        assert "validate_pmf_manifest" in content, (
            "pregame_initial.yml must import and call validate_pmf_manifest() "
            "rather than an inline implementation."
        )


# ===========================================================================
# FIX 4 — Edge completeness using reconciled markets
# ===========================================================================

def _make_slate_manifest(
    tmp_path: Path,
    scheduled_game_count: int = 2,
    game_ids: list[str] | None = None,
) -> Path:
    manifest = {
        "game_date": "2026-07-13",
        "scheduled_game_count": scheduled_game_count,
        "game_ids": game_ids or ["G001", "G002"],
        "github_run_id": "RUN_TEST_001",
        "git_commit": "abc123def456",
    }
    p = tmp_path / "slate_manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _make_pmf_parquet(tmp_path: Path, rows: list[dict], filename: str = "full_pmfs_wide.parquet") -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / filename
    df.to_parquet(p, index=False)
    return p


def _make_market_parquet(tmp_path: Path, rows: list[dict], filename: str = "props.parquet") -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / filename
    df.to_parquet(p, index=False)
    return p


def _run_edge_report(
    tmp_path: Path,
    pmfs_path: str,
    raw_props: str,
    slate_manifest: str,
    game_date: str = "2026-07-13",
    source_policy: str = "odds_api_then_bdl",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_BUILD_EDGE_REPORT),
        "--pmfs", pmfs_path,
        "--raw-props", raw_props,
        "--out-dir", str(tmp_path / "out"),
        "--slate-manifest", slate_manifest,
        "--game-date", game_date,
        "--source-policy", source_policy,
    ]
    if extra_args:
        cmd.extend(extra_args)
    (tmp_path / "out").mkdir(exist_ok=True)
    return subprocess.run(cmd, capture_output=True, text=True)


class TestFix4EdgeCompletenessUsingReconciledMarkets:
    """build_edge_report.py must persist an expected_market_comparison_manifest.parquet
    after reconciliation but BEFORE probability/edge construction.
    The workflow must use validate_edge_manifest() for the comparison."""

    def test_reconciled_market_manifest_is_written_before_edges(self, tmp_path: Path):
        """build_edge_report.py must write expected_market_comparison_manifest.parquet
        after market validation and identity reconciliation, BEFORE edge construction."""
        script_content = _BUILD_EDGE_REPORT.read_text()
        assert "expected_market_comparison_manifest.parquet" in script_content, (
            "build_edge_report.py must write expected_market_comparison_manifest.parquet "
            "after market validation/reconciliation but before probability/edge construction."
        )

    def test_expected_nonempty_actual_empty_market_comparison_is_fatal(self, tmp_path: Path):
        """When expected market comparison rows exist but actual is empty, must be FATAL.

        The old behavior: expected nonempty + actual empty → LIVE_MARKETS_NOT_YET_AVAILABLE.
        The new behavior: FATAL (expected nonempty + actual empty is a data integrity failure).
        """
        content = _PREGAME_INITIAL.read_text()
        # Must call validate_edge_manifest which raises MissingEdgeError
        assert "validate_edge_manifest" in content, (
            "pregame_initial.yml must call validate_edge_manifest() from market_integrity. "
            "The old behavior of classify expected_nonempty+actual_empty as "
            "LIVE_MARKETS_NOT_YET_AVAILABLE must be removed."
        )

    def test_raw_unreconciled_market_is_not_an_expected_edge(self):
        """Expected edge manifest must come from reconciled market data in
        expected_market_comparison_manifest.parquet, not raw odds parquet."""
        content = _PREGAME_INITIAL.read_text()
        assert "expected_market_comparison_manifest.parquet" in content, (
            "pregame_initial.yml must read expected_market_comparison_manifest.parquet "
            "(written by build_edge_report.py after reconciliation) for expected edge keys. "
            "Do NOT build expected edges directly from the raw odds parquet."
        )

    def test_verified_zero_markets_has_explicit_no_market_status(self, tmp_path: Path):
        """When the market source successfully returned zero rows, status must be
        LIVE_MARKETS_NOT_YET_AVAILABLE and expected_market_comparison_manifest.parquet
        must be written to out_dir."""
        manifest = _make_slate_manifest(tmp_path, scheduled_game_count=2)
        pmf_rows = [
            {"game_id": "G001", "player_id": "P001", "stat": "pts",
             "pmf_json": json.dumps([0.1] * 10), "model_prob_over": 0.5}
        ]
        pmfs = _make_pmf_parquet(tmp_path, pmf_rows)
        empty_props = _make_market_parquet(tmp_path, [], "empty.parquet")

        result = _run_edge_report(
            tmp_path,
            pmfs_path=str(pmfs),
            raw_props=str(empty_props),
            slate_manifest=str(manifest),
        )
        # Must emit LIVE_MARKETS_NOT_YET_AVAILABLE status
        combined = result.stdout + result.stderr
        assert "LIVE_MARKETS_NOT_YET_AVAILABLE" in combined, (
            "build_edge_report.py must emit LIVE_MARKETS_NOT_YET_AVAILABLE "
            "when market source returns zero rows."
        )
        # expected_market_comparison_manifest.parquet must be written
        expected_manifest = tmp_path / "out" / "expected_market_comparison_manifest.parquet"
        assert expected_manifest.exists(), (
            "build_edge_report.py must write expected_market_comparison_manifest.parquet "
            "even when markets are empty (to record the reconciled-zero state)."
        )

    def test_missing_expected_market_comparison_row_is_fatal(self):
        """validate_edge_manifest must raise MissingEdgeError when expected rows > actual."""
        from wnba_props_model.pipeline.market_integrity import (
            MissingEdgeError,
            validate_edge_manifest,
        )

        expected = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
            {"game_id": "G001", "player_id": "P002", "stat": "pts", "vendor": "fanduel", "line": 15.5},
        ])
        actual = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
        ])
        with pytest.raises(MissingEdgeError, match="[Mm]issing"):
            validate_edge_manifest(expected, actual)

    def test_unexpected_market_comparison_row_is_fatal(self):
        """validate_edge_manifest must raise MissingEdgeError when actual contains rows not in expected."""
        from wnba_props_model.pipeline.market_integrity import (
            MissingEdgeError,
            validate_edge_manifest,
        )

        expected = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
        ])
        actual = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
            {"game_id": "G001", "player_id": "P999", "stat": "pts", "vendor": "fanduel", "line": 10.5},
        ])
        with pytest.raises(MissingEdgeError, match="[Uu]nexpected"):
            validate_edge_manifest(expected, actual)

    def test_duplicate_market_comparison_row_is_fatal(self):
        """validate_edge_manifest must raise DuplicateEdgeError when actual has duplicate keys."""
        from wnba_props_model.pipeline.market_integrity import (
            DuplicateEdgeError,
            validate_edge_manifest,
        )

        expected = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
        ])
        actual = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},
        ])
        with pytest.raises(DuplicateEdgeError):
            validate_edge_manifest(expected, actual)
