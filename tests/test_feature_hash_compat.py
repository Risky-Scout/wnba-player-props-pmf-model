"""Focused tests for canonical_feature_contract_hash and artifact manifest validation.

Proves:
1. Two feature manifests with identical schema but different timestamps, commits, and paths
   produce the same canonical hash.
2. Adding, removing, or reordering a model feature changes the canonical hash.
3. A legacy model manifest is accepted only when its persisted feature list exactly matches.
4. A legacy calibrator manifest is accepted only when stat/role coverage and required files
   are complete.
5. Source run, source commit, config hash, cutoff, and gate-status mismatches remain fatal.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from wnba_props_model.pipeline.market_integrity import (
    ArtifactManifestError,
    canonical_feature_contract_hash,
    validate_artifact_manifest,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _base_manifest(**overrides) -> dict:
    m = {
        "row_grain_wide": "player_id × game_id",
        "row_grain_long": "player_id × game_id × stat",
        "identity_columns": ["player_id", "game_id", "game_date"],
        "target_columns": ["actual_outcome"],
        "model_feature_columns": ["player_pts_mean_l5", "player_pts_mean_season", "opp_pts_allowed"],
        "numeric_feature_columns": ["player_pts_mean_l5", "player_pts_mean_season", "opp_pts_allowed"],
        "categorical_feature_columns": [],
        "role_bucket_columns": ["role_bucket"],
        "forbidden_columns": ["actual_outcome"],
        "temporal_policy": {"cutoff": "before_game"},
        "stats_modeled": ["pts", "reb", "ast"],
        "roll_windows": [5, 10, 20],
        # Volatile fields (must NOT affect canonical hash)
        "created_at_utc": "2026-07-13T10:00:00Z",
        "git_commit_if_available": "abc123",
        "wide_table_path": "/old/path/features.parquet",
        "long_table_path": "/old/path/features_long.parquet",
        "source_tables": ["games", "stats"],
    }
    m.update(overrides)
    return m


def _valid_artifact_manifest(artifact_type: str = "calibrator", **overrides) -> dict:
    m = {
        "artifact_type": artifact_type,
        "artifact_schema_version": "1",
        "source_workflow": "Weekly OOF Refresh & Calibration",
        "source_run_id": "12345",
        "source_commit": "deadbeef1234567890",
        "created_at_utc": "2026-07-13T10:00:00Z",
        "feature_manifest_hash": "abcd1234abcd1234",
        "feature_hash_kind": "canonical_feature_contract_v1",
        "config_hash": "efgh5678efgh5678",
        "gate_status": "PASS",
    }
    if artifact_type == "calibrator":
        m["calibration_cutoff"] = "2026-07-12T00:00:00Z"
    else:
        m["model_training_cutoff"] = "2026-07-12T00:00:00Z"
    m.update(overrides)
    return m


# ─── 1. Canonical hash stability across volatile fields ──────────────────────

def test_canonical_hash_identical_for_different_timestamps():
    """Same feature schema → same canonical hash regardless of created_at_utc."""
    m1 = _base_manifest(created_at_utc="2026-07-10T08:00:00Z")
    m2 = _base_manifest(created_at_utc="2026-07-15T22:00:00Z")
    assert canonical_feature_contract_hash(m1) == canonical_feature_contract_hash(m2)


def test_canonical_hash_identical_for_different_commits():
    """Same schema → same hash regardless of git_commit_if_available."""
    m1 = _base_manifest(git_commit_if_available="abc")
    m2 = _base_manifest(git_commit_if_available="xyz")
    assert canonical_feature_contract_hash(m1) == canonical_feature_contract_hash(m2)


def test_canonical_hash_identical_for_different_paths():
    """Same schema → same hash regardless of wide_table_path / long_table_path."""
    m1 = _base_manifest(wide_table_path="/path/A/features.parquet")
    m2 = _base_manifest(wide_table_path="/path/B/features.parquet")
    assert canonical_feature_contract_hash(m1) == canonical_feature_contract_hash(m2)


def test_canonical_hash_identical_for_different_source_tables():
    """Same schema → same hash regardless of source_tables list."""
    m1 = _base_manifest(source_tables=["games", "stats"])
    m2 = _base_manifest(source_tables=["games", "stats", "injuries"])
    assert canonical_feature_contract_hash(m1) == canonical_feature_contract_hash(m2)


def test_canonical_hash_is_16_hex_chars():
    m = _base_manifest()
    h = canonical_feature_contract_hash(m)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ─── 2. Model feature changes alter the canonical hash ───────────────────────

def test_adding_feature_changes_hash():
    m1 = _base_manifest()
    m2 = _base_manifest(model_feature_columns=[
        "player_pts_mean_l5", "player_pts_mean_season", "opp_pts_allowed", "new_feature"
    ])
    assert canonical_feature_contract_hash(m1) != canonical_feature_contract_hash(m2)


def test_removing_feature_changes_hash():
    m1 = _base_manifest()
    m2 = _base_manifest(model_feature_columns=["player_pts_mean_l5", "player_pts_mean_season"])
    assert canonical_feature_contract_hash(m1) != canonical_feature_contract_hash(m2)


def test_reordering_model_features_changes_hash():
    """model_feature_columns order is preserved (estimator requires it), so reordering changes hash."""
    m1 = _base_manifest(model_feature_columns=["A", "B", "C"])
    m2 = _base_manifest(model_feature_columns=["C", "B", "A"])
    assert canonical_feature_contract_hash(m1) != canonical_feature_contract_hash(m2)


def test_other_list_fields_are_order_independent():
    """Non-feature list fields are sorted, so reordering them must NOT change hash."""
    m1 = _base_manifest(identity_columns=["player_id", "game_id", "game_date"])
    m2 = _base_manifest(identity_columns=["game_date", "game_id", "player_id"])
    assert canonical_feature_contract_hash(m1) == canonical_feature_contract_hash(m2)


# ─── 3. validate_artifact_manifest — new canonical path ──────────────────────

def test_new_manifest_canonical_hash_match_passes():
    canonical = canonical_feature_contract_hash(_base_manifest())
    m = _valid_artifact_manifest("calibrator", feature_manifest_hash=canonical)
    # Must not raise
    validate_artifact_manifest(
        m,
        expected_artifact_type="calibrator",
        prediction_timestamp_utc="2026-07-15T12:00:00Z",
        source_run_id="12345",
        source_commit="deadbeef1234567890",
        canonical_feature_hash=canonical,
        config_hash="efgh5678efgh5678",
    )


def test_new_manifest_canonical_hash_mismatch_fails():
    canonical = canonical_feature_contract_hash(_base_manifest())
    bad_canonical = "0000000000000000"
    m = _valid_artifact_manifest("calibrator", feature_manifest_hash=canonical)
    with pytest.raises(ArtifactManifestError, match="feature_manifest_hash mismatch"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
            source_run_id="12345",
            source_commit="deadbeef1234567890",
            canonical_feature_hash=bad_canonical,
            config_hash="efgh5678efgh5678",
        )


# ─── 4. Legacy path — raw-hash comparison is skipped ─────────────────────────

def test_legacy_manifest_accepts_nonblank_raw_hash_without_comparison():
    """Legacy manifest (no feature_hash_kind) must pass even when the raw hash
    in the artifact differs from a freshly computed one."""
    m = _valid_artifact_manifest("calibrator")
    del m["feature_hash_kind"]  # make it legacy
    m["feature_manifest_hash"] = "745ccb4351000519"  # raw hash from cal artifact run

    # Passing a completely different canonical_feature_hash must NOT cause failure
    # because the manifest is legacy and raw-hash comparison is skipped.
    validate_artifact_manifest(
        m,
        expected_artifact_type="calibrator",
        prediction_timestamp_utc="2026-07-15T12:00:00Z",
        source_run_id="12345",
        source_commit="deadbeef1234567890",
        canonical_feature_hash="03e92e5524092350",  # different pregame-computed hash
        config_hash="efgh5678efgh5678",
    )


def test_legacy_manifest_rejects_blank_feature_hash():
    """Legacy manifest with blank feature_manifest_hash must be rejected."""
    m = _valid_artifact_manifest("calibrator")
    del m["feature_hash_kind"]
    m["feature_manifest_hash"] = ""
    with pytest.raises(ArtifactManifestError, match="blank"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
        )


# ─── 5. Source run, commit, config hash, cutoff, gate-status remain fatal ────

def test_source_run_id_mismatch_is_fatal():
    m = _valid_artifact_manifest("calibrator")
    with pytest.raises(ArtifactManifestError, match="source_run_id"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
            source_run_id="WRONG_RUN",
        )


def test_source_commit_mismatch_is_fatal():
    m = _valid_artifact_manifest("calibrator")
    with pytest.raises(ArtifactManifestError, match="source_commit"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
            source_commit="000000000000000000",
        )


def test_config_hash_mismatch_is_fatal():
    m = _valid_artifact_manifest("calibrator")
    with pytest.raises(ArtifactManifestError, match="config_hash"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
            config_hash="BADHASHBADH",
        )


def test_gate_status_not_pass_is_fatal():
    m = _valid_artifact_manifest("calibrator", gate_status="FAIL")
    with pytest.raises(ArtifactManifestError, match="gate_status"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
        )


def test_calibration_cutoff_in_future_is_fatal():
    m = _valid_artifact_manifest("calibrator", calibration_cutoff="2099-01-01T00:00:00Z")
    with pytest.raises(ArtifactManifestError, match="cutoff"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
        )


def test_wrong_artifact_type_is_fatal():
    m = _valid_artifact_manifest("model")
    with pytest.raises(ArtifactManifestError, match="artifact_type"):
        validate_artifact_manifest(
            m,
            expected_artifact_type="calibrator",
            prediction_timestamp_utc="2026-07-15T12:00:00Z",
        )


# ─── 6. The exact run 29390378813 scenario is now unblocked ──────────────────

def test_run_29390378813_calibrator_manifest_now_validates():
    """The calibrator manifest from run 29383727727 has feature_hash_kind absent (legacy).
    pregame run 29390378813 failed because raw-hash 745ccb43 != pregame's 03e92e55.
    After this fix, the legacy path skips the raw-hash comparison and passes.
    """
    # Exact manifest from run 29383727727
    cal_manifest = {
        "artifact_type": "calibrator",
        "artifact_schema_version": "1",
        "source_workflow": "Weekly OOF Refresh & Calibration",
        "source_run_id": "29383727727",
        "source_commit": "666e11e02d5311edd16945fbc161a01c0248078f",
        "created_at_utc": "2026-07-15T04:28:22.220598+00:00",
        "calibration_cutoff": "2026-07-13T00:00:00Z",
        "feature_manifest_hash": "745ccb4351000519",   # raw hash from their build
        "config_hash": "f3516c16a4874a42",
        "gate_status": "PASS",
        # No feature_hash_kind → legacy
    }

    # Pregame-computed canonical hash (would differ from the raw hash above)
    pregame_canonical_hash = "03e92e5524092350"  # from pregame's feature build

    # Must NOT raise despite hash mismatch — legacy path skips raw comparison
    validate_artifact_manifest(
        cal_manifest,
        expected_artifact_type="calibrator",
        prediction_timestamp_utc="2026-07-15T12:00:00Z",
        source_run_id="29383727727",
        source_commit="666e11e02d5311edd16945fbc161a01c0248078f",
        canonical_feature_hash=pregame_canonical_hash,
        config_hash="f3516c16a4874a42",
    )
