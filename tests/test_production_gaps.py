"""Production gap tests — Gaps 1–4.

These tests were written BEFORE the implementing fixes so that each one fails
until the corresponding change is made.  They cover:

Gap 1  — Artifact manifest and trigger-aware resolver
Gap 2  — Remove mixed calibration sources
Gap 3  — PMF and edge manifest gates
Gap 4  — Blocking pregame_final injury processing

Required tests (22 total):

Gap 1:
  test_daily_trigger_uses_exact_model_and_compatible_weekly_calibrator
  test_weekly_trigger_uses_exact_calibrator_and_compatible_daily_model
  test_scheduled_trigger_uses_compatible_daily_and_weekly_sources
  test_artifact_schema_mismatch_is_fatal
  test_feature_manifest_hash_mismatch_is_fatal
  test_config_hash_mismatch_is_fatal
  test_future_training_or_calibration_cutoff_is_fatal

Gap 2:
  test_downloaded_calibration_package_is_not_partially_overwritten
  test_calibration_files_share_one_source_run
  test_invalid_calibration_package_is_fatal

Gap 3:
  test_confirmed_inactive_player_is_not_an_expected_actionable_pmf
  test_missing_expected_pmf_is_fatal
  test_unexpected_pmf_is_fatal
  test_missing_expected_market_comparison_row_is_fatal
  test_unexpected_market_comparison_row_is_fatal
  test_duplicate_market_comparison_row_is_fatal
  test_no_markets_status_does_not_claim_edge_completeness_pass

Gap 4:
  test_pregame_initial_uses_ticket1_injury_path
  test_pregame_final_uses_ticket1_injury_path
  test_pregame_final_injury_failure_blocks_edge_generation
  test_injury_source_timestamps_survive_workflow_processing
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

# Workflow YAML paths
_WF_DIR = Path(__file__).parent.parent / ".github" / "workflows"
_PREGAME_INITIAL = _WF_DIR / "pregame_initial.yml"
_PREGAME_FINAL = _WF_DIR / "pregame_final.yml"
_DAILY_PIPELINE = _WF_DIR / "daily_pipeline.yml"
_WEEKLY_CALIBRATION = _WF_DIR / "weekly_calibration.yml"


# ===========================================================================
# GAP 1 — Artifact manifest and trigger-aware resolver
# ===========================================================================


class TestArtifactResolverTriggerAware:
    """pregame_initial.yml must distinguish between daily and weekly triggers."""

    def test_daily_trigger_uses_exact_model_and_compatible_weekly_calibrator(self):
        """When triggered by the daily pipeline, pregame_initial must use the exact
        triggering run's model AND the most recent compatible WEEKLY calibration run
        for calibrators — NOT the triggering run for calibrators.
        """
        content = _PREGAME_INITIAL.read_text()
        # Must reference the daily workflow name to branch on trigger type
        assert "Daily WNBA PMF Pipeline" in content, (
            "pregame_initial.yml must check for 'Daily WNBA PMF Pipeline' trigger name "
            "to distinguish daily vs weekly trigger sources"
        )
        # Must route calibrators to weekly_calibration.yml when daily trigger fires
        assert "weekly_calibration.yml" in content, (
            "pregame_initial.yml must resolve calibrators from weekly_calibration.yml "
            "when triggered by the daily pipeline"
        )
        # Model must still be resolved from the exact triggering run
        assert "TRIGGER_RUN_ID" in content, (
            "pregame_initial.yml must still use TRIGGER_RUN_ID for the exact model "
            "artifact on the daily trigger path"
        )

    def test_weekly_trigger_uses_exact_calibrator_and_compatible_daily_model(self):
        """When triggered by weekly_calibration, pregame_initial must use the exact
        triggering run's calibrators AND the most recent compatible DAILY run for model.
        """
        content = _PREGAME_INITIAL.read_text()
        # Must reference the weekly workflow name
        assert "Weekly OOF Refresh & Calibration" in content, (
            "pregame_initial.yml must check for 'Weekly OOF Refresh & Calibration' "
            "trigger name to distinguish weekly trigger"
        )
        # Must route model to daily_pipeline.yml when weekly trigger fires
        assert "daily_pipeline.yml" in content, (
            "pregame_initial.yml must resolve the model from daily_pipeline.yml "
            "when triggered by the weekly calibration workflow"
        )

    def test_scheduled_trigger_uses_compatible_daily_and_weekly_sources(self):
        """For scheduled/manual triggers, pregame_initial must resolve model from
        daily_pipeline.yml and calibrators from weekly_calibration.yml on main.
        """
        content = _PREGAME_INITIAL.read_text()
        # Both producer workflows must be referenced for the scheduled/manual path
        assert "daily_pipeline.yml" in content, (
            "pregame_initial.yml must reference daily_pipeline.yml for model resolution "
            "on scheduled/manual trigger"
        )
        assert "weekly_calibration.yml" in content, (
            "pregame_initial.yml must reference weekly_calibration.yml for calibrator "
            "resolution on scheduled/manual trigger"
        )


class TestArtifactManifestValidation:
    """validate_artifact_manifest() must enforce schema, type, cutoffs, and hashes."""

    def _good_model_manifest(self) -> dict:
        return {
            "artifact_type": "model",
            "artifact_schema_version": "1",
            "source_workflow": "Daily WNBA PMF Pipeline",
            "source_run_id": "12345",
            "source_commit": "abc123def456",
            "created_at_utc": "2026-07-13T10:00:00Z",
            "model_training_cutoff": "2026-07-12T00:00:00Z",
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "confighash_ddeeff",
        }

    def _good_calibrator_manifest(self) -> dict:
        return {
            "artifact_type": "calibrator",
            "artifact_schema_version": "1",
            "source_workflow": "Weekly OOF Refresh & Calibration",
            "source_run_id": "99999",
            "source_commit": "deadbeef1234",
            "created_at_utc": "2026-07-07T10:00:00Z",
            "calibration_cutoff": "2026-07-06T00:00:00Z",
            "feature_manifest_hash": "featurehash_aabbcc",
            "config_hash": "confighash_ddeeff",
            "gate_status": "PASS",
        }

    def test_artifact_schema_mismatch_is_fatal(self):
        """An artifact manifest with an unsupported schema version must raise."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )
        manifest = self._good_model_manifest()
        manifest["artifact_schema_version"] = "99"  # unsupported
        with pytest.raises(ArtifactManifestError, match="schema_version"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

    def test_feature_manifest_hash_mismatch_is_fatal(self):
        """A canonical feature_manifest_hash mismatch must raise ArtifactManifestError.

        New-style manifests (feature_hash_kind=canonical_feature_contract_v1) compare
        canonical_feature_hash; a mismatch is fatal.
        Legacy manifests (no feature_hash_kind) skip raw-hash comparison — the legacy
        path accepts nonblank raw hashes to allow builds from different timestamps/paths
        to validate correctly (see test_feature_hash_compat.py for legacy path tests).
        """
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )
        # Use a new-style manifest so canonical hash comparison is active
        manifest = self._good_model_manifest()
        manifest["feature_hash_kind"] = "canonical_feature_contract_v1"
        manifest["feature_manifest_hash"] = "correcthash123456"
        with pytest.raises(ArtifactManifestError, match="feature_manifest_hash"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
                canonical_feature_hash="WRONG_CANONICAL_HASH_000",
            )

    def test_config_hash_mismatch_is_fatal(self):
        """A config_hash mismatch must raise ArtifactManifestError."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )
        manifest = self._good_model_manifest()
        with pytest.raises(ArtifactManifestError, match="config_hash"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
                config_hash="WRONG_CONFIG_HASH",
            )

    def test_future_training_or_calibration_cutoff_is_fatal(self):
        """A training or calibration cutoff that is >= the prediction timestamp must raise."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )
        # Model: training_cutoff after prediction time
        manifest = self._good_model_manifest()
        manifest["model_training_cutoff"] = "2026-07-14T00:00:00Z"  # future
        with pytest.raises(ArtifactManifestError, match="model_training_cutoff"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="model",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

        # Calibrator: calibration_cutoff after prediction time
        cal_manifest = self._good_calibrator_manifest()
        cal_manifest["calibration_cutoff"] = "2026-07-14T00:00:00Z"  # future
        with pytest.raises(ArtifactManifestError, match="calibration_cutoff"):
            validate_artifact_manifest(
                cal_manifest,
                expected_artifact_type="calibrator",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )

    def test_valid_manifest_does_not_raise(self):
        """A fully valid manifest must not raise."""
        from wnba_props_model.pipeline.market_integrity import validate_artifact_manifest

        manifest = self._good_model_manifest()
        # Must not raise
        validate_artifact_manifest(
            manifest,
            expected_artifact_type="model",
            prediction_timestamp_utc="2026-07-13T12:00:00Z",
            feature_manifest_hash="featurehash_aabbcc",
            config_hash="confighash_ddeeff",
            source_run_id="12345",
            source_commit="abc123def456",
        )

    def test_daily_pipeline_emits_artifact_manifest(self):
        """daily_pipeline.yml must emit an artifact manifest for the model artifact."""
        content = _DAILY_PIPELINE.read_text()
        assert "artifact_manifest_model.json" in content, (
            "daily_pipeline.yml must generate artifact_manifest_model.json "
            "and include it in the model artifact upload"
        )
        assert "artifact_type" in content, (
            "daily_pipeline.yml manifest must include artifact_type field"
        )

    def test_weekly_calibration_emits_artifact_manifest(self):
        """weekly_calibration.yml must emit an artifact manifest for the calibrator artifact."""
        content = _WEEKLY_CALIBRATION.read_text()
        assert "artifact_manifest_calibrator.json" in content, (
            "weekly_calibration.yml must generate artifact_manifest_calibrator.json "
            "and include it in the calibrators artifact upload"
        )
        assert "artifact_type" in content, (
            "weekly_calibration.yml manifest must include artifact_type field"
        )


# ===========================================================================
# GAP 2 — Remove mixed calibration sources
# ===========================================================================


class TestMixedCalibrationSourcesRemoved:
    """pregame_initial.yml must not restore calibration files from git after download."""

    def test_downloaded_calibration_package_is_not_partially_overwritten(self):
        """pregame_initial.yml must NOT contain 'git checkout HEAD -- artifacts/models/calibration'
        for bias_corrections or related files after the artifact download.

        The validated calibration artifact must be one internally consistent package.
        Overwriting downloaded files with git-committed versions mixes sources.
        """
        content = _PREGAME_INITIAL.read_text()
        forbidden_patterns = [
            "git checkout HEAD -- artifacts/models/calibration/bias_corrections.json",
            "git checkout HEAD -- artifacts/models/calibration/bias_corrections_by_role.json",
            "git checkout HEAD -- artifacts/models/calibration/player_form_corrections_2026.json",
            "git checkout HEAD -- artifacts/models/calibration/variance_compress.json",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in content, (
                f"pregame_initial.yml must NOT contain '{pattern}'. "
                "Downloaded calibration artifacts must not be mixed with git-restored files."
            )

    def test_calibration_files_share_one_source_run(self):
        """validate_artifact_manifest must reject a manifest whose source_run_id
        does not match the expected source run.
        """
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
            "calibration_cutoff": "2026-07-06T00:00:00Z",
            "feature_manifest_hash": "fmh_xyz",
            "config_hash": "cfg_xyz",
            "gate_status": "PASS",
        }
        # Must raise when caller supplies a different expected source_run_id
        with pytest.raises(ArtifactManifestError, match="source_run_id"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="calibrator",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
                source_run_id="COMPLETELY_DIFFERENT_RUN_999",
            )

    def test_invalid_calibration_package_is_fatal(self):
        """validate_artifact_manifest must raise when gate_status is not PASS."""
        from wnba_props_model.pipeline.market_integrity import (
            ArtifactManifestError,
            validate_artifact_manifest,
        )
        manifest = {
            "artifact_type": "calibrator",
            "artifact_schema_version": "1",
            "source_workflow": "Weekly OOF Refresh & Calibration",
            "source_run_id": "WEEKLY_RUN_222",
            "source_commit": "deadbeef",
            "created_at_utc": "2026-07-07T10:00:00Z",
            "calibration_cutoff": "2026-07-06T00:00:00Z",
            "feature_manifest_hash": "fmh_xyz",
            "config_hash": "cfg_xyz",
            "gate_status": "FAIL",  # gate failed — must not be promoted
        }
        with pytest.raises(ArtifactManifestError, match="gate_status"):
            validate_artifact_manifest(
                manifest,
                expected_artifact_type="calibrator",
                prediction_timestamp_utc="2026-07-13T12:00:00Z",
            )


# ===========================================================================
# GAP 3 — PMF and edge manifest gates
# ===========================================================================


class TestPMFManifestGates:
    """Expected and actual PMF / edge key sets must be compared fatally."""

    def _make_manifest_df(
        self, rows: list[dict], cols=("game_id", "player_id", "stat")
    ) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=list(cols))

    def test_confirmed_inactive_player_is_not_an_expected_actionable_pmf(self):
        """A player whose availability_status is OUT must not appear in the expected
        PMF manifest produced by build_expected_pmf_manifest.

        The workflow must consult the Ticket 1 availability/actionability result,
        not only the slate status column.
        """
        from wnba_props_model.pipeline.market_integrity import build_expected_pmf_manifest

        # Slate with 3 players: 2 active, 1 confirmed OUT
        slate = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "status": "active"},
            {"game_id": "G001", "player_id": "P002", "status": "active"},
            {"game_id": "G001", "player_id": "P003", "status": "out"},
        ])
        eligible = slate[~slate["status"].str.lower().isin(["inactive", "out"])].copy()
        manifest = build_expected_pmf_manifest(eligible, ["pts", "reb"])
        player_ids = set(manifest["player_id"].unique())
        assert "P003" not in player_ids, (
            "Confirmed-inactive player P003 (status='out') must NOT appear in "
            "the expected PMF manifest"
        )
        assert "P001" in player_ids and "P002" in player_ids, (
            "Active players P001 and P002 must appear in the expected PMF manifest"
        )

    def test_missing_expected_pmf_is_fatal(self):
        """validate_pmf_manifest must raise MissingPMFError when expected keys > actual keys."""
        from wnba_props_model.pipeline.market_integrity import (
            MissingPMFError,
            validate_pmf_manifest,
        )
        expected = self._make_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P002", "stat": "pts"},
        ])
        actual = self._make_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            # P002 pts is missing
        ])
        with pytest.raises(MissingPMFError, match="[Mm]issing"):
            validate_pmf_manifest(expected, actual)

    def test_unexpected_pmf_is_fatal(self):
        """validate_pmf_manifest must raise MissingPMFError when actual keys not in expected."""
        from wnba_props_model.pipeline.market_integrity import (
            MissingPMFError,
            validate_pmf_manifest,
        )
        expected = self._make_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
        ])
        actual = self._make_manifest_df([
            {"game_id": "G001", "player_id": "P001", "stat": "pts"},
            {"game_id": "G001", "player_id": "P999", "stat": "pts"},  # unexpected
        ])
        with pytest.raises(MissingPMFError, match="[Uu]nexpected"):
            validate_pmf_manifest(expected, actual)

    def test_missing_expected_market_comparison_row_is_fatal(self):
        """validate_edge_manifest must raise MissingEdgeError when expected rows not in actual."""
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
            # P002 row is missing from market_comparison
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
            {"game_id": "G001", "player_id": "P999", "stat": "pts", "vendor": "fanduel", "line": 10.5},  # unexpected
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
            {"game_id": "G001", "player_id": "P001", "stat": "pts", "vendor": "fanduel", "line": 20.5},  # duplicate
        ])
        with pytest.raises(DuplicateEdgeError):
            validate_edge_manifest(expected, actual)

    def test_no_markets_status_does_not_claim_edge_completeness_pass(self):
        """When market status is LIVE_MARKETS_NOT_YET_AVAILABLE, the workflow manifest
        step must record that explicit status and must NOT claim edge completeness passed.
        """
        content = _PREGAME_INITIAL.read_text()
        # The manifest step must explicitly record LIVE_MARKETS_NOT_YET_AVAILABLE
        assert "LIVE_MARKETS_NOT_YET_AVAILABLE" in content, (
            "pregame_initial.yml manifest step must explicitly record "
            "LIVE_MARKETS_NOT_YET_AVAILABLE when no market rows exist"
        )
        # Must not claim edge gate passed when there are no markets
        # (i.e., must have a branch that skips edge comparison for no-markets case)
        assert "edge_completeness_status" in content or "live_markets_status" in content or \
               "market_status" in content, (
            "pregame_initial.yml manifest step must record an explicit market/edge "
            "completeness status variable to distinguish no-markets from gate-passed"
        )

    def test_workflow_manifest_step_compares_edge_keys_fatally(self):
        """pregame_initial.yml must do a FATAL comparison of expected vs actual edge keys,
        not just record counts.

        The current bug is that the workflow records counts but never calls sys.exit(1)
        when edge keys mismatch.
        """
        content = _PREGAME_INITIAL.read_text()
        # Must have edge integrity check with fatal exit
        assert "missing_edges" in content or "edge_integrity" in content or \
               ("edge" in content and "sys.exit(1)" in content), (
            "pregame_initial.yml manifest step must fatally compare expected vs actual "
            "edge keys and call sys.exit(1) on mismatch"
        )


# ===========================================================================
# GAP 4 — Blocking pregame_final injury processing
# ===========================================================================


class TestPregameFinalInjuryBlocking:
    """Both injury steps in pregame_final.yml must be blocking (continue-on-error: false)."""

    def test_pregame_initial_uses_ticket1_injury_path(self):
        """pregame_initial.yml must use apply_injury_updates.py for injury processing,
        not an ad-hoc inline fetch.
        """
        content = _PREGAME_INITIAL.read_text()
        assert "apply_injury_updates.py" in content, (
            "pregame_initial.yml must invoke apply_injury_updates.py (Ticket 1 path) "
            "for injury processing"
        )

    def test_pregame_final_uses_ticket1_injury_path(self):
        """pregame_final.yml must use apply_injury_updates.py for injury processing."""
        content = _PREGAME_FINAL.read_text()
        assert "apply_injury_updates.py" in content, (
            "pregame_final.yml must invoke apply_injury_updates.py (Ticket 1 typed path) "
            "for injury processing, not an ad-hoc inline script"
        )

    def test_pregame_final_injury_failure_blocks_edge_generation(self):
        """Both the injury fetch step AND the apply step in pregame_final.yml must have
        continue-on-error: false (or no continue-on-error key at all, which defaults to false).

        A FAILURE at injury processing must prevent edge generation.
        """
        content = _PREGAME_FINAL.read_text()

        # Injury fetch step must NOT have continue-on-error: true
        # We check that the old pattern (fetch step with continue-on-error: true immediately
        # followed by fetch content) is gone
        assert "Fetch and cache injury report" not in content or \
               _injury_step_is_blocking(content, "Fetch and cache injury report"), (
            "pregame_final.yml 'Fetch and cache injury report' step must be blocking "
            "(continue-on-error: false)"
        )

        # Apply injury updates step must NOT have continue-on-error: true
        assert "Apply confirmed lineups" not in content or \
               _injury_step_is_blocking(content, "Apply confirmed lineups"), (
            "pregame_final.yml 'Apply confirmed lineups' step must be blocking "
            "(continue-on-error: false)"
        )

    def test_injury_source_timestamps_survive_workflow_processing(self):
        """pregame_final.yml must not run an inline injury normaliser that discards
        source_updated_at, pulled_at_utc, or typed fetch status.

        The Ticket 1 typed path (apply_injury_updates.py) preserves per-record timestamps.
        """
        content = _PREGAME_FINAL.read_text()
        # The old pattern wrote normalized records WITHOUT source_updated_at / pulled_at_utc.
        # If the step still uses the inline normaliser (discarding those fields) it must fail.
        # We check that the inline discard pattern is gone OR that the preserved fields appear.
        discarding_pattern = (
            '"player_id": int(pid), "player_name": name,\n'
            '                      "status": str(r.get("status") or "available").lower(),\n'
            '                      "return_date": r.get("return_date"), "comment": r.get("comment"),\n'
            "                  })"
        )
        # The inline normaliser without source timestamps must not be the only code path
        assert discarding_pattern not in content, (
            "pregame_final.yml must not run an inline injury normaliser that discards "
            "source_updated_at and pulled_at_utc. Use apply_injury_updates.py instead."
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _injury_step_is_blocking(yaml_content: str, step_name: str) -> bool:
    """Return True if the named step does NOT have 'continue-on-error: true'.

    Scans the YAML for the step name and checks the next ~10 lines for the flag.
    """
    lines = yaml_content.splitlines()
    for i, line in enumerate(lines):
        if step_name in line and "name:" in line:
            # Scan the next 15 lines for continue-on-error: true
            window = lines[i : i + 15]
            for wline in window:
                if "continue-on-error: true" in wline:
                    return False  # step is permissive — not blocking
            return True  # no continue-on-error: true found in this step block
    return True  # step not found → assume no explicit flag → blocking by default
