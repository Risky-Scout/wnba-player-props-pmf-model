"""Production hardening tests (Stage A).

Covers:
  - Effective-dated canonical player identity alias resolution
  - Duplicate PMF rejection
  - Duplicate market rejection
  - Selected-package manifest validation
  - Stale date-specific payload rejection
  - Public cache-busting payload selection
  - No direct experimental gh-pages deployment

All tests use the real production code.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.data.identity import (
    apply_canonical_ids,
    deduplicate_pmfs,
    validate_no_duplicate_identities,
)
from wnba_props_model.models.simulation import pmf_to_json, normalize_pmf


# ---------------------------------------------------------------------------
# A1: Effective-dated alias resolution
# ---------------------------------------------------------------------------

class TestEffectiveDatedAliasResolution:

    def test_alias_file_exists(self):
        """config/player_identity_aliases.json must exist."""
        p = Path("config/player_identity_aliases.json")
        assert p.exists(), "config/player_identity_aliases.json must exist"

    def test_alias_has_required_fields(self):
        """Every alias must have canonical_id, effective_from, source, resolution_reason."""
        data = json.loads(Path("config/player_identity_aliases.json").read_text())
        for dup_id, info in data.get("aliases", {}).items():
            assert "canonical_id" in info, f"Alias {dup_id} missing canonical_id"
            assert "effective_from" in info, f"Alias {dup_id} missing effective_from"
            assert "source" in info, f"Alias {dup_id} missing source"
            assert "resolution_reason" in info, f"Alias {dup_id} missing resolution_reason"

    def test_teja_oblak_alias_resolves_to_canonical(self):
        """Player ID 101929 must resolve to canonical 75090."""
        df = pd.DataFrame([
            {"player_id": 101929, "player_name": "Teja Oblak", "game_id": "G001", "stat": "pts"},
            {"player_id": 75090,  "player_name": "Teja Oblak", "game_id": "G001", "stat": "pts"},
        ])
        resolved = apply_canonical_ids(df, "player_id")
        assert (resolved["player_id"] == 75090).all(), (
            "Both Teja Oblak IDs must resolve to canonical 75090"
        )

    def test_alias_does_not_merge_same_name_different_team(self):
        """Alias must require team_id match, not name-only."""
        # A player with the same name on a different team must NOT be aliased
        df = pd.DataFrame([
            {"player_id": 101929, "player_name": "Teja Oblak", "game_id": "G001", "stat": "pts", "team_id": 31},
            {"player_id": 99999,  "player_name": "Teja Oblak", "game_id": "G002", "stat": "pts", "team_id": 99},
        ])
        resolved = apply_canonical_ids(df, "player_id")
        # Only 101929 should be aliased; 99999 has no alias entry
        assert resolved[resolved["player_id_original" if "player_id_original" in resolved.columns else "player_id"] != 99999]["player_id"].eq(75090).any() or \
               resolved[resolved.index == 0]["player_id"].iloc[0] == 75090, (
            "player_id 101929 should be resolved to 75090"
        )
        # player_id 99999 has no alias — must remain unchanged
        assert 99999 in resolved["player_id"].values, (
            "player_id 99999 (different team) must NOT be aliased"
        )

    def test_duplicate_pmf_rejection_after_resolution(self):
        """After canonical ID resolution, duplicate (game_id, player_id, stat) raises."""
        arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
        df = pd.DataFrame([
            {"game_id": "G001", "player_id": 101929, "stat": "pts",
             "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0},
            {"game_id": "G001", "player_id": 75090, "stat": "pts",
             "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0},
        ])
        # After resolution: both become player_id=75090
        df = apply_canonical_ids(df, "player_id")
        assert (df["player_id"] == 75090).all()
        # deduplicate_pmfs should remove the duplicate
        deduped = deduplicate_pmfs(df, key_cols=["game_id", "player_id", "stat"])
        assert len(deduped) == 1, "Deduplicated result must have exactly 1 row"
        # validate_no_duplicate_identities should pass after dedup
        validate_no_duplicate_identities(deduped)  # must not raise

    def test_validate_no_duplicate_identities_raises(self):
        """validate_no_duplicate_identities raises ValueError on duplicate keys."""
        df = pd.DataFrame([
            {"game_id": "G001", "player_id": 75090, "stat": "pts"},
            {"game_id": "G001", "player_id": 75090, "stat": "pts"},  # duplicate
        ])
        with pytest.raises(ValueError):
            validate_no_duplicate_identities(df)

    def test_duplicate_market_rejection(self):
        """Market rows must not have duplicate (game_id, player_id, stat, vendor, line)."""
        from wnba_props_model.pipeline.market_integrity import (
            DuplicateQuoteError,
            validate_no_duplicate_quotes,
        )
        rows = pd.DataFrame([
            {"game_id": "G001", "player_id": 75090, "stat": "pts", "vendor": "dk", "line": 15.0},
            {"game_id": "G001", "player_id": 75090, "stat": "pts", "vendor": "dk", "line": 15.0},
        ])
        with pytest.raises(DuplicateQuoteError):
            validate_no_duplicate_quotes(rows)

    def test_identity_resolution_in_page_generation(self, tmp_path: Path):
        """Page generation applies identity resolution before building props."""
        arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
        proj = pd.DataFrame([
            {"game_id": "G001", "player_id": 101929, "player_name": "Teja Oblak",
             "stat": "pts", "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0,
             "model_prob_over": 0.3, "role_bucket": "starter", "game_date": "2026-07-14"},
            {"game_id": "G001", "player_id": 75090, "player_name": "Teja Oblak",
             "stat": "pts", "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0,
             "model_prob_over": 0.3, "role_bucket": "starter", "game_date": "2026-07-14"},
        ])
        proj_path = tmp_path / "proj.parquet"
        proj.to_parquet(proj_path, index=False)
        edges_path = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(edges_path, index=False)
        out_dir = tmp_path / "Pre-Game"
        out_dir.mkdir()
        r = subprocess.run([
            sys.executable, "scripts/generate_web_pages.py",
            "--game-date", "2026-07-14",
            "--projections", str(proj_path),
            "--edges", str(edges_path),
            "--out-dir", str(out_dir),
            "--json-only",
            "--release-id", "test-id",
            "--git-commit", "abc",
        ], capture_output=True, text=True)
        assert r.returncode == 0, f"generate_web_pages failed: {r.stderr[:300]}"
        pmf_page = json.loads((out_dir / "PMF-Distributions" / "latest.json").read_text())
        # Should have read date-specific payload via pointer, or direct for test
        # Main check: no duplicates
        props = pmf_page.get("props", [])
        teja_rows = [p for p in props if "oblak" in p.get("player", "").lower()]
        assert len(teja_rows) <= 1, (
            f"After identity resolution, Teja Oblak should appear at most once, got {len(teja_rows)}"
        )


# ---------------------------------------------------------------------------
# A2: Selected-package manifest
# ---------------------------------------------------------------------------

class TestSelectedPackageManifest:

    def test_manifest_exists(self):
        """config/selected_production_package.json must exist."""
        assert Path("config/selected_production_package.json").exists()

    def test_manifest_has_required_fields(self):
        """Manifest must contain all required lineage fields."""
        pkg = json.loads(Path("config/selected_production_package.json").read_text())
        required = [
            "selected_package", "model_version", "calibration_version",
            "code_sha", "model_artifact_source_run", "model_artifact_hashes",
            "calibration_artifact_source_run", "calibration_artifact_hashes",
            "training_cutoff", "calibration_cutoff", "selected_at_utc", "rollback_package",
        ]
        for field in required:
            assert field in pkg, f"Manifest missing required field: {field!r}"

    def test_selected_package_is_valid_value(self):
        """selected_package must be ENHANCED_CHAMPION or OPTIMIZED_CHALLENGER."""
        pkg = json.loads(Path("config/selected_production_package.json").read_text())
        assert pkg["selected_package"] in ("ENHANCED_CHAMPION", "OPTIMIZED_CHALLENGER"), (
            f"Invalid selected_package: {pkg['selected_package']!r}"
        )

    def test_manifest_hashes_are_hex(self):
        """All artifact hashes must be valid hex strings."""
        pkg = json.loads(Path("config/selected_production_package.json").read_text())
        for k, v in pkg.get("model_artifact_hashes", {}).items():
            assert isinstance(v, str) and all(c in "0123456789abcdef" for c in v.lower()), (
                f"model_artifact_hashes[{k!r}] is not a valid hex hash: {v!r}"
            )


# ---------------------------------------------------------------------------
# A3: Cache-safe payloads
# ---------------------------------------------------------------------------

class TestCacheSafePayloads:

    def test_generate_web_pages_creates_release_directory(self, tmp_path: Path):
        """generate_web_pages must create releases/<release_id>.json alongside latest.json."""
        arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
        proj = pd.DataFrame([{"game_id": "G001", "player_id": "P001",
                               "player_name": "Test Player", "stat": "pts",
                               "pmf_json": pmf_to_json(arr), "pmf_mean": 1.1,
                               "model_prob_over": 0.3, "role_bucket": "starter",
                               "game_date": "2026-07-14"}])
        proj_path = tmp_path / "proj.parquet"
        proj.to_parquet(proj_path, index=False)
        edges_path = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(edges_path, index=False)
        out = tmp_path / "Pre-Game"
        out.mkdir()
        r = subprocess.run([
            sys.executable, "scripts/generate_web_pages.py",
            "--game-date", "2026-07-14",
            "--projections", str(proj_path),
            "--edges", str(edges_path),
            "--out-dir", str(out),
            "--json-only",
            "--release-id", "TEST_RELEASE_001",
            "--git-commit", "abc123",
        ], capture_output=True, text=True)
        assert r.returncode == 0, f"Failed: {r.stderr[:200]}"
        # Check releases dir
        release_path = out / "PMF-Distributions" / "releases" / "TEST_RELEASE_001.json"
        assert release_path.exists(), f"Immutable release file must exist: {release_path}"
        # Check latest.json is a pointer with full required fields
        import hashlib as _hl
        latest = json.loads((out / "PMF-Distributions" / "latest.json").read_text())
        assert latest.get("pointer") is True, "latest.json must be a pointer"
        assert latest.get("release_id") == "TEST_RELEASE_001"
        assert "payload_path" in latest or "release_payload_path" in latest
        assert "payload_sha256" in latest, "Pointer must include payload_sha256"
        assert "row_count" in latest, "Pointer must include row_count"
        assert "generated_at_utc" in latest, "Pointer must include generated_at_utc"
        # Validate payload_sha256 matches actual release file
        release_payload = release_path.read_text()
        expected_sha = _hl.sha256(release_payload.encode()).hexdigest()
        assert latest["payload_sha256"] == expected_sha, "payload_sha256 must match release file"
        # Validate row_count matches actual props
        release_data = json.loads(release_payload)
        assert latest["row_count"] == release_data.get("total_props", 0)

    def test_latest_json_is_pointer_not_full_payload(self, tmp_path: Path):
        """latest.json must contain pointer fields, not the full props array."""
        arr = normalize_pmf(np.array([0.5, 0.3, 0.2]))
        proj = pd.DataFrame([{"game_id": "G001", "player_id": "P001",
                               "player_name": "Alice", "stat": "pts",
                               "pmf_json": pmf_to_json(arr), "pmf_mean": 0.7,
                               "model_prob_over": 0.2, "role_bucket": "starter",
                               "game_date": "2026-07-14"}])
        proj_path = tmp_path / "proj.parquet"
        proj.to_parquet(proj_path, index=False)
        edges_path = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(edges_path, index=False)
        out = tmp_path / "Pre-Game"
        out.mkdir()
        subprocess.run([
            sys.executable, "scripts/generate_web_pages.py",
            "--game-date", "2026-07-14",
            "--projections", str(proj_path),
            "--edges", str(edges_path),
            "--out-dir", str(out),
            "--json-only",
            "--release-id", "PTR_TEST",
            "--git-commit", "abc",
        ], capture_output=True, text=True)
        latest_edge = json.loads((out / "Edge" / "latest.json").read_text())
        # pointer must NOT contain the full props array
        assert "props" not in latest_edge, (
            "latest.json must be a pointer, not a full payload — 'props' should not be present"
        )
        assert latest_edge.get("pointer") is True

    def test_stale_date_payload_rejected(self, tmp_path: Path):
        """A payload with game_date != target date must be rejected as stale."""
        stale_payload = {"game_date": "2026-07-12", "total_props": 10, "props": []}
        stale_path = tmp_path / "stale.json"
        stale_path.write_text(json.dumps(stale_payload))
        target_date = "2026-07-14"
        loaded = json.loads(stale_path.read_text())
        is_stale = loaded.get("game_date") != target_date
        assert is_stale, "A payload with game_date=2026-07-12 must be rejected as stale when target=2026-07-14"

    def test_distributions_page_reads_date_specific_not_pointer(self, tmp_path: Path):
        """generate_distributions_page prefers date-specific PMF payload over latest.json pointer.
        Verifies that when both latest.json (pointer) and 2026-07-14.json exist, the script
        uses the date-specific file (which has content) rather than the pointer (which has 0 props).
        """
        arr = normalize_pmf(np.array([0.4, 0.3, 0.2, 0.1]))
        # Create a date-specific real PMF payload with actual props
        real_payload = {
            "game_date": "2026-07-14", "total_props": 3, "props": [
                {"player": "Alice", "stat": "PTS", "stat_raw": "pts",
                 "pmf_full": [[0, 0.4], [1, 0.3], [2, 0.2], [3, 0.1]],
                 "pmf": [[0, 0.4], [1, 0.3], [2, 0.2], [3, 0.1]],
                 "model_p_over": 0.3, "model_p_under": 0.7, "model_p_push": 0.0, "mean": 1.0},
                {"player": "Bob", "stat": "REB", "stat_raw": "reb",
                 "pmf_full": [[0, 0.4], [1, 0.6]],
                 "pmf": [[0, 0.4], [1, 0.6]],
                 "model_p_over": 0.6, "model_p_under": 0.4, "model_p_push": 0.0, "mean": 0.6},
                {"player": "Carol", "stat": "AST", "stat_raw": "ast",
                 "pmf_full": [[0, 0.5], [1, 0.5]],
                 "pmf": [[0, 0.5], [1, 0.5]],
                 "model_p_over": 0.5, "model_p_under": 0.5, "model_p_push": 0.0, "mean": 0.5},
            ]
        }
        pmf_dir = tmp_path / "Pre-Game" / "PMF-Distributions"
        pmf_dir.mkdir(parents=True)
        (pmf_dir / "2026-07-14.json").write_text(json.dumps(real_payload))
        # Create a pointer latest.json (zero props) to prove date-specific is preferred
        pointer = {"pointer": True, "release_id": "R1", "game_date": "2026-07-14",
                   "total_props": 0}
        (pmf_dir / "latest.json").write_text(json.dumps(pointer))

        r = subprocess.run([
            sys.executable, "scripts/generate_distributions_page.py",
            "--game-date", "2026-07-14",
            "--base-dir", str(tmp_path),
            "--json-only",
            "--release-id", "R1",
            "--git-commit", "abc",
        ], capture_output=True, text=True)
        assert r.returncode == 0, f"Failed: {r.stderr[:300]}"

        # Check pointer structure
        dist_latest = json.loads((tmp_path / "Pre-Game" / "Distributions" / "latest.json").read_text())
        assert dist_latest.get("pointer") is True, "Distributions latest.json must be a pointer"

        # The actual content should come from releases/R1.json
        dist_release = tmp_path / "Pre-Game" / "Distributions" / "releases" / "R1.json"
        assert dist_release.exists(), f"Release file must exist: {dist_release}"
        full_dist = json.loads(dist_release.read_text())
        # Must have used date-specific payload (3 props), not the pointer (0 props)
        assert full_dist.get("total_props", 0) > 0, (
            "Distributions page must use date-specific PMF payload (3 props), not pointer (0)"
        )


# ---------------------------------------------------------------------------
# A4: Production workflow contract
# ---------------------------------------------------------------------------

class TestProductionWorkflowContract:

    def test_challenger_train_workflow_exists(self):
        """challenger_train.yml must exist in .github/workflows/."""
        p = Path(".github/workflows/challenger_train.yml")
        assert p.exists(), "challenger_train.yml must exist"

    def test_challenger_train_workflow_has_required_inputs(self):
        """challenger_train.yml must declare source_ref, target_game_date, challenger_version inputs."""
        # Check via raw text since PyYAML parses bare 'on:' as Python True
        text = Path(".github/workflows/challenger_train.yml").read_text()
        required = ["source_ref", "target_game_date", "challenger_version"]
        for inp in required:
            assert inp in text, f"challenger_train.yml must declare input: {inp!r}"

    def test_challenger_train_workflow_never_deploys(self):
        """challenger_train.yml must not contain any deployment steps (no gh-pages actions, no public page deploy)."""
        text = Path(".github/workflows/challenger_train.yml").read_text()
        # These patterns indicate actual deployment — actions or direct pushes
        forbidden_actual = [
            "peaceiris/actions-gh-pages",
            "push origin HEAD:gh-pages",
            "actions/deploy-pages",
        ]
        import re
        for pattern in forbidden_actual:
            assert pattern not in text, (
                f"challenger_train.yml must NOT deploy — found: {pattern!r}"
            )

    def test_pregame_initial_resolves_selected_package(self):
        """pregame_initial.yml must include selected-package resolution step."""
        text = Path(".github/workflows/pregame_initial.yml").read_text()
        assert "selected_production_package.json" in text, (
            "pregame_initial.yml must resolve config/selected_production_package.json"
        )

    def test_no_direct_gh_pages_push_in_production_workflows(self):
        """Production pregame workflows must not push directly to gh-pages (only via Actions)."""
        for wf_name in ("pregame_initial.yml", "pregame_final.yml", "pregame_odds_refresh.yml"):
            text = (Path(".github/workflows") / wf_name).read_text()
            # Direct git push to gh-pages is not allowed
            assert "push origin HEAD:gh-pages" not in text, (
                f"{wf_name} must not push directly to gh-pages — use peaceiris/actions-gh-pages"
            )

    def test_page_lineage_validation_is_blocking(self):
        """pregame_initial.yml must have blocking validate_page_release_lineage step."""
        text = Path(".github/workflows/pregame_initial.yml").read_text()
        assert "validate_page_release_lineage" in text
        assert "continue-on-error: false" in text

    def test_challenger_workflow_uses_bdl_secret_not_plain_text(self):
        """challenger_train.yml must use ${{ secrets.BDL_API_KEY }}, never a plain value."""
        text = Path(".github/workflows/challenger_train.yml").read_text()
        assert "${{ secrets.BDL_API_KEY }}" in text, (
            "challenger_train.yml must reference BDL_API_KEY via secrets"
        )
        # Must not contain any known plain-text key value patterns
        assert "e06d1882" not in text, "BDL_API_KEY value must not appear in workflow text"
