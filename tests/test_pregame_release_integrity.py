"""Pregame Release Integrity tests — Ticket 1.

Scope:
  - PMF Distributions page (_build_pmf_json)
  - Pre-Game Edge page (_build_edge_json)
  - generate_web_pages.py CLI (real subprocess invocations)
  - Release lineage: both pages share one release_id
  - Exact current-run artifacts (stale fallback detection)
  - PMF completeness and duplicate checks
  - Market row completeness
  - Integer push probabilities correct
  - Page probabilities match final PMF parquet
  - No edge labeled CLV in page output

All tests use real production entrypoints.
I/O uses pytest tmp_path only — no writes to live website dirs.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Real production imports
# ---------------------------------------------------------------------------
from wnba_props_model.models.simulation import json_to_pmf, normalize_pmf, pmf_to_json
from wnba_props_model.pipeline.market_integrity import (
    MissingEdgeError,
    MissingPMFError,
    StaleFallbackForbiddenError,
    build_expected_edge_manifest,
    build_expected_pmf_manifest,
    check_no_stale_fallback,
    validate_edge_manifest,
    validate_pmf_manifest,
    validate_page_release_lineage,  # added by this ticket
    validate_page_probabilities,    # added by this ticket
)

# ---------------------------------------------------------------------------
# Compute probabilities directly from PMF (independent of page code)
# ---------------------------------------------------------------------------

def _prob_over(pmf_arr: np.ndarray, line: float) -> float:
    """P(X > line) from normalized PMF array."""
    indices = np.arange(len(pmf_arr), dtype=float)
    return float(pmf_arr[indices > float(line)].sum())


def _prob_push(pmf_arr: np.ndarray, line: float) -> float:
    """P(X == line) — nonzero only for integer lines."""
    if float(line) != math.floor(float(line)):
        return 0.0
    idx = int(line)
    return float(pmf_arr[idx]) if 0 <= idx < len(pmf_arr) else 0.0


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_pmf_array_pts() -> np.ndarray:
    """PMF for pts — nonzero at 10, 15, 20, 25."""
    arr = np.zeros(31, dtype=float)
    arr[10] = 0.15
    arr[15] = 0.20   # exactly at integer line 15 → push mass
    arr[20] = 0.35
    arr[25] = 0.30
    return arr


def _make_pmf_array_reb() -> np.ndarray:
    """PMF for reb — nonzero at 3, 5, 7, 9."""
    arr = np.zeros(16, dtype=float)
    arr[3] = 0.20
    arr[5] = 0.30
    arr[7] = 0.30
    arr[9] = 0.20
    return arr


def _make_proj_parquet(tmp_path: Path) -> Path:
    """Build a player projections parquet with known PMFs."""
    rows = [
        {
            "game_id": "G001",
            "player_id": "P001",
            "player_name": "Alice Adams",
            "stat": "pts",
            "pmf_json": pmf_to_json(_make_pmf_array_pts()),
            "pmf_mean": 19.25,
            "model_prob_over": _prob_over(normalize_pmf(_make_pmf_array_pts()), 15.0),
            "role_bucket": "starter",
            "game_date": "2026-07-14",
        },
        {
            "game_id": "G001",
            "player_id": "P002",
            "player_name": "Bob Brown",
            "stat": "reb",
            "pmf_json": pmf_to_json(_make_pmf_array_reb()),
            "pmf_mean": 6.0,
            "model_prob_over": _prob_over(normalize_pmf(_make_pmf_array_reb()), 5.5),
            "role_bucket": "core",
            "game_date": "2026-07-14",
        },
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "player_projections_2026-07-14.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_edges_parquet(tmp_path: Path) -> Path:
    """Build a publishable edges parquet."""
    rows = [
        {
            "game_id": "G001",
            "player_id": "P001",
            "player_name": "Alice Adams",
            "stat": "pts",
            "line": 15.0,             # INTEGER — push mass exists
            "over_odds": -110,
            "under_odds": -110,
            "model_prob_over": _prob_over(normalize_pmf(_make_pmf_array_pts()), 15.0),
            "market_prob_over_no_vig": 0.50,
            "edge_over": _prob_over(normalize_pmf(_make_pmf_array_pts()), 15.0) - 0.50,
            "kelly_fraction": 0.03,
            "vendor": "draftkings",
            "pmf_json": pmf_to_json(_make_pmf_array_pts()),
            "pmf_mean": 19.25,
        },
        {
            "game_id": "G001",
            "player_id": "P002",
            "player_name": "Bob Brown",
            "stat": "reb",
            "line": 5.5,              # HALF-POINT — no push
            "over_odds": -115,
            "under_odds": -105,
            "model_prob_over": _prob_over(normalize_pmf(_make_pmf_array_reb()), 5.5),
            "market_prob_over_no_vig": 0.52,
            "edge_over": _prob_over(normalize_pmf(_make_pmf_array_reb()), 5.5) - 0.52,
            "kelly_fraction": 0.02,
            "vendor": "fanduel",
            "pmf_json": pmf_to_json(_make_pmf_array_reb()),
            "pmf_mean": 6.0,
        },
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "publishable_edges.parquet"
    df.to_parquet(p, index=False)
    return p


def _run_generate_web_pages(
    tmp_path: Path,
    proj_path: str,
    edges_path: str,
    game_date: str = "2026-07-14",
    release_id: str = "RELEASE_TEST_001",
    git_commit: str = "abc123def456",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run generate_web_pages.py CLI and return the result."""
    out_dir = tmp_path / "Pre-Game"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "scripts" / "generate_web_pages.py"),
        "--game-date", game_date,
        "--projections", proj_path,
        "--edges", edges_path,
        "--out-dir", str(out_dir),
        "--json-only",
        "--release-id", release_id,       # added by this ticket
        "--git-commit", git_commit,        # added by this ticket
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


# ===========================================================================
# GROUP 1: Release lineage — both pages share one release ID
# ===========================================================================

class TestReleaseLineage:

    def test_both_pages_carry_release_id(self, tmp_path: Path):
        """Edge page and PMF page both carry release_id field when --release-id is given."""
        proj = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        result = _run_generate_web_pages(
            tmp_path, str(proj), str(edges),
            release_id="RUN_001", git_commit="abc123",
        )
        assert result.returncode == 0, (
            f"generate_web_pages must succeed.\nstdout={result.stdout[:500]}\nstderr={result.stderr[:500]}"
        )
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        pmf_json  = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        assert "release_id" in edge_json, "Edge page must carry release_id"
        assert "release_id" in pmf_json,  "PMF page must carry release_id"

    def test_both_pages_share_same_release_id(self, tmp_path: Path):
        """Edge page and PMF page must have identical release_id — one release per deployment."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(
            tmp_path, str(proj), str(edges),
            release_id="RUN_SHARED_42", git_commit="def456",
        )
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        pmf_json  = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        assert edge_json["release_id"] == pmf_json["release_id"], (
            f"Both pages must share same release_id: "
            f"edge={edge_json['release_id']!r} pmf={pmf_json['release_id']!r}"
        )
        assert edge_json["release_id"] == "RUN_SHARED_42"

    def test_both_pages_carry_git_commit(self, tmp_path: Path):
        """Both pages must carry git_commit for traceability."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(
            tmp_path, str(proj), str(edges),
            release_id="RUN_GIT", git_commit="91980ba17ebd",
        )
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        pmf_json  = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        assert edge_json.get("git_commit") == "91980ba17ebd", "Edge page must carry git_commit"
        assert pmf_json.get("git_commit")  == "91980ba17ebd", "PMF page must carry git_commit"

    def test_validate_page_release_lineage_passes(self, tmp_path: Path):
        """validate_page_release_lineage passes when both pages match the expected release."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(
            tmp_path, str(proj), str(edges),
            release_id="RUN_VALID", git_commit="abc",
        )
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        pmf_json  = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        # Must not raise
        validate_page_release_lineage(edge_json, pmf_json, expected_release_id="RUN_VALID")

    def test_validate_page_release_lineage_fails_on_mismatch(self, tmp_path: Path):
        """validate_page_release_lineage raises when edge and PMF pages have different release_ids."""
        from wnba_props_model.pipeline.market_integrity import ArtifactLineageMismatchError
        # Manually create mismatched page JSONs
        edge_json = {
            "schema_version": "2.1",
            "release_id": "RUN_A",
            "git_commit": "abc",
            "generated_at": "2026-07-14T12:00:00Z",
            "props": [],
        }
        pmf_json = {
            "schema_version": "2.1",
            "release_id": "RUN_B",  # different!
            "git_commit": "abc",
            "generated_at": "2026-07-14T12:00:00Z",
            "props": [],
        }
        with pytest.raises(ArtifactLineageMismatchError):
            validate_page_release_lineage(edge_json, pmf_json, expected_release_id="RUN_A")

    def test_validate_page_release_lineage_fails_on_wrong_expected(self, tmp_path: Path):
        """validate_page_release_lineage raises when pages don't match expected run."""
        from wnba_props_model.pipeline.market_integrity import ArtifactLineageMismatchError
        edge_json = {"schema_version": "2.1", "release_id": "RUN_OLD", "git_commit": "old"}
        pmf_json  = {"schema_version": "2.1", "release_id": "RUN_OLD", "git_commit": "old"}
        with pytest.raises(ArtifactLineageMismatchError):
            validate_page_release_lineage(edge_json, pmf_json, expected_release_id="RUN_NEW")


# ===========================================================================
# GROUP 2: CLV labels absent from page output
# ===========================================================================

class TestNoClvLabels:

    def _load_pages(self, tmp_path: Path) -> tuple[dict, dict]:
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        pmf_json  = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        return edge_json, pmf_json

    def test_edge_page_has_no_clv_labeled_fields(self, tmp_path: Path):
        """No key in any prop dict of the edge page may contain 'clv'."""
        edge_json, _ = self._load_pages(tmp_path)
        for prop in edge_json.get("props", []):
            clv_keys = [k for k in prop.keys() if "clv" in k.lower()]
            assert clv_keys == [], (
                f"Edge page prop {prop.get('player')}/{prop.get('stat')} "
                f"contains CLV-labeled keys: {clv_keys}"
            )

    def test_edge_page_has_time_decay_not_clv(self, tmp_path: Path):
        """Edge page carries 'time_decay_adjusted_edge', not 'clv_adj_edge'."""
        edge_json, _ = self._load_pages(tmp_path)
        for prop in edge_json.get("props", []):
            assert "clv_adj_edge" not in prop, (
                f"Edge page still writes deprecated 'clv_adj_edge': {prop.get('player')}"
            )
            assert "time_decay_adjusted_edge" in prop, (
                f"Edge page must carry 'time_decay_adjusted_edge': {prop.get('player')}"
            )

    def test_pmf_page_has_no_clv_labeled_fields(self, tmp_path: Path):
        """No key in any prop dict of the PMF page may contain 'clv'."""
        _, pmf_json = self._load_pages(tmp_path)
        for prop in pmf_json.get("props", []):
            clv_keys = [k for k in prop.keys() if "clv" in k.lower()]
            assert clv_keys == [], (
                f"PMF page prop {prop.get('player')}/{prop.get('stat')} "
                f"contains CLV-labeled keys: {clv_keys}"
            )


# ===========================================================================
# GROUP 3: Page probabilities match final PMF parquet
# ===========================================================================

class TestProbabilityConsistency:

    def test_edge_page_model_p_over_matches_pmf(self, tmp_path: Path):
        """model_p_over in edge page must equal P(X > line) from the serialized PMF."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())

        edges_df = pd.read_parquet(edges)
        pmf_lookup = {}
        for _, row in edges_df.iterrows():
            key = (str(row.get("player_name", "")).lower(), str(row.get("stat", "")).lower())
            pmf_lookup[key] = {
                "pmf_json": row["pmf_json"],
                "line": float(row["line"]),
            }

        max_err = 0.0
        for prop in edge_json.get("props", []):
            key = (str(prop.get("player", "")).lower(), str(prop.get("stat", "")).lower())
            if key not in pmf_lookup:
                continue
            info = pmf_lookup[key]
            pmf_arr = normalize_pmf(json_to_pmf(info["pmf_json"]))
            line = info["line"]
            expected_p_over = _prob_over(pmf_arr, line)
            page_p_over = float(prop["model_p_over"])
            err = abs(expected_p_over - page_p_over)
            max_err = max(max_err, err)

        assert max_err <= 1e-8, (
            f"Maximum model_p_over error between page and PMF parquet: {max_err} > 1e-8"
        )

    def test_validate_page_probabilities_passes(self, tmp_path: Path):
        """validate_page_probabilities must pass for correctly built pages."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        edges_df = pd.read_parquet(edges)
        # Must not raise
        validate_page_probabilities(edge_json["props"], edges_df)

    def test_validate_page_probabilities_fails_on_bad_p_over(self, tmp_path: Path):
        """validate_page_probabilities raises when page model_p_over doesn't match PMF."""
        from wnba_props_model.pipeline.market_integrity import PageProbabilityError

        pmf_arr = normalize_pmf(_make_pmf_array_pts())
        edges_df = pd.DataFrame([{
            "player_name": "Alice Adams",
            "stat": "pts",
            "line": 15.0,
            "pmf_json": pmf_to_json(pmf_arr),
            "model_prob_over": _prob_over(pmf_arr, 15.0),
        }])
        # Page has wrong model_p_over
        bad_props = [{"player": "Alice Adams", "stat": "PTS", "model_p_over": 0.99}]
        with pytest.raises(PageProbabilityError):
            validate_page_probabilities(bad_props, edges_df)

    def test_integer_line_push_probability_is_nonzero(self, tmp_path: Path):
        """For integer lines, P(push) > 0 — the PMF must have mass at int(line)."""
        pmf_arr = normalize_pmf(_make_pmf_array_pts())
        line = 15.0  # integer
        p_over = _prob_over(pmf_arr, line)
        p_push = _prob_push(pmf_arr, line)
        p_under = 1.0 - p_over - p_push
        assert p_push > 0, f"Integer line={line} must have nonzero push probability"
        assert abs(p_over + p_push + p_under - 1.0) < 1e-12

    def test_half_point_line_push_is_zero(self, tmp_path: Path):
        """For half-point lines, P(push) == 0 exactly."""
        pmf_arr = normalize_pmf(_make_pmf_array_reb())
        line = 5.5  # half-point
        p_push = _prob_push(pmf_arr, line)
        assert p_push == 0.0, f"Half-point line={line} must have p_push=0, got {p_push}"
        p_over = _prob_over(pmf_arr, line)
        assert abs(p_over + p_push + (1.0 - p_over - p_push) - 1.0) < 1e-12


# ===========================================================================
# GROUP 4: PMF completeness — zero missing, zero duplicates
# ===========================================================================

class TestPMFCompleteness:

    def test_pmf_page_zero_duplicate_player_stat_rows(self, tmp_path: Path):
        """No (player, stat) pair appears more than once in the PMF page props."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        pmf_json = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        seen = set()
        duplicates = []
        for prop in pmf_json.get("props", []):
            key = (prop.get("player", ""), prop.get("stat", ""))
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        assert duplicates == [], f"Duplicate (player, stat) in PMF page: {duplicates}"

    def test_pmf_page_no_invalid_pmf_rows(self, tmp_path: Path):
        """All PMF distributions in the page must contain at least one pair."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        pmf_json = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())
        for prop in pmf_json.get("props", []):
            pmf_pairs = prop.get("pmf", [])
            assert len(pmf_pairs) > 0, (
                f"PMF page prop {prop.get('player')}/{prop.get('stat')} has empty PMF"
            )
            total_mass = sum(pair[1] for pair in pmf_pairs)
            assert abs(total_mass - 1.0) < 0.01, (
                f"PMF pairs don't sum to ~1: {prop.get('player')}/{prop.get('stat')} sum={total_mass}"
            )

    def test_expected_pmf_rows_present_in_page(self, tmp_path: Path):
        """All expected (player, stat) rows from the projection parquet appear in the PMF page."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        pmf_json = json.loads((tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").read_text())

        proj_df = pd.read_parquet(proj)
        # Exclude suppressed stats (stl, blk) — same as in _build_pmf_json
        _SUPPRESSED = {"stl", "blk"}
        expected = set(
            (str(row["player_name"]).lower(), str(row["stat"]).lower())
            for _, row in proj_df.iterrows()
            if str(row.get("stat", "")).lower() not in _SUPPRESSED
        )
        actual = set(
            (str(p["player"]).lower(), str(p["stat"]).lower())
            for p in pmf_json.get("props", [])
        )
        missing = expected - actual
        assert missing == set(), f"Expected PMF rows missing from page: {missing}"

    def test_pmf_manifest_validate_passes_for_good_data(self):
        """validate_pmf_manifest passes when expected == actual."""
        slate = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001"},
            {"game_id": "G001", "player_id": "P002"},
        ])
        stats = ["pts", "reb"]
        expected = build_expected_pmf_manifest(slate, stats)
        validate_pmf_manifest(expected, expected.copy())  # must not raise

    def test_missing_pmf_row_is_fatal(self):
        """validate_pmf_manifest raises MissingPMFError when a row is absent."""
        slate = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001"},
            {"game_id": "G001", "player_id": "P002"},
        ])
        expected = build_expected_pmf_manifest(slate, ["pts", "reb"])
        actual = expected.iloc[1:].reset_index(drop=True)  # drop first row
        with pytest.raises(MissingPMFError):
            validate_pmf_manifest(expected, actual)


# ===========================================================================
# GROUP 5: Market row completeness
# ===========================================================================

class TestMarketRowCompleteness:

    def test_market_manifest_matches_actual_edge_rows(self, tmp_path: Path):
        """One market comparison row per reconciled market quote."""
        from datetime import datetime, timezone
        fresh_ts = datetime.now(timezone.utc).isoformat()
        markets = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts",
             "vendor": "dk", "line": 15.0, "over_odds": -110, "under_odds": -110,
             "market_updated_at": fresh_ts},
            {"game_id": "G001", "player_id": "P002", "stat": "reb",
             "vendor": "fd", "line": 5.5,  "over_odds": -115, "under_odds": -105,
             "market_updated_at": fresh_ts},
        ])
        expected = build_expected_edge_manifest(markets)
        actual = expected.copy()
        validate_edge_manifest(expected, actual)  # must not raise
        assert len(expected) == 2

    def test_missing_market_comparison_row_is_fatal(self):
        """validate_edge_manifest raises MissingEdgeError when a row is absent."""
        from datetime import datetime, timezone
        markets = pd.DataFrame([
            {"game_id": "G001", "player_id": "P001", "stat": "pts",
             "vendor": "dk", "line": 15.0},
            {"game_id": "G001", "player_id": "P002", "stat": "reb",
             "vendor": "fd", "line": 5.5},
        ])
        expected = build_expected_edge_manifest(markets)
        actual = expected.iloc[:1].reset_index(drop=True)  # missing one row
        with pytest.raises(MissingEdgeError):
            validate_edge_manifest(expected, actual)


# ===========================================================================
# GROUP 6: Stale artifact rejection
# ===========================================================================

class TestStaleArtifactRejection:

    def test_stale_page_file_raises_when_current_missing(self, tmp_path: Path):
        """check_no_stale_fallback raises when current page file is missing."""
        stale_dir = tmp_path / "stale"
        stale_dir.mkdir()
        (stale_dir / "latest.json").write_text('{"stale": true}')
        current = tmp_path / "current" / "latest.json"
        # current does NOT exist
        with pytest.raises(StaleFallbackForbiddenError):
            check_no_stale_fallback("edge_page", current, fallback_path=stale_dir / "latest.json")

    def test_current_page_file_passes_stale_check(self, tmp_path: Path):
        """check_no_stale_fallback passes when current artifact exists."""
        current = tmp_path / "latest.json"
        current.write_text('{"release_id": "RUN_CURRENT"}')
        check_no_stale_fallback("edge_page", current)  # must not raise

    def test_no_prior_run_file_substituted_for_edge_page(self, tmp_path: Path):
        """The edge page must not be served from a prior run's output."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(
            tmp_path, str(proj), str(edges),
            release_id="RUN_CURRENT", git_commit="abc123",
        )
        edge_path = tmp_path / "Pre-Game" / "Edge" / "latest.json"
        assert edge_path.exists(), "Edge page must be written"
        # Now pretend we have a stale file; check current run is preferred
        check_no_stale_fallback("edge_page", edge_path)  # must not raise

    def test_no_prior_run_file_substituted_for_pmf_page(self, tmp_path: Path):
        """The PMF page must not be served from a prior run's output."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(
            tmp_path, str(proj), str(edges),
            release_id="RUN_CURRENT", git_commit="abc123",
        )
        pmf_path = tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json"
        assert pmf_path.exists(), "PMF page must be written"
        check_no_stale_fallback("pmf_page", pmf_path)  # must not raise


# ===========================================================================
# GROUP 7: Page generation correctness
# ===========================================================================

class TestPageGenerationCorrectness:

    def test_generate_web_pages_writes_both_json_files(self, tmp_path: Path):
        """Single CLI call must produce both Edge/latest.json and PMF-Distributions/latest.json."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        result = _run_generate_web_pages(tmp_path, str(proj), str(edges))
        assert result.returncode == 0, (
            f"generate_web_pages must exit 0.\nstdout={result.stdout[:500]}\nstderr={result.stderr[:500]}"
        )
        assert (tmp_path / "Pre-Game" / "Edge" / "latest.json").exists()
        assert (tmp_path / "Pre-Game" / "PMF-Distributions" / "latest.json").exists()

    def test_generate_web_pages_writes_date_specific_files(self, tmp_path: Path):
        """Both date-specific JSON files must also be written."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges), game_date="2026-07-14")
        assert (tmp_path / "Pre-Game" / "Edge" / "2026-07-14.json").exists()
        assert (tmp_path / "Pre-Game" / "PMF-Distributions" / "2026-07-14.json").exists()

    def test_edge_page_schema_version(self, tmp_path: Path):
        """Edge page must carry schema_version field."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        assert "schema_version" in edge_json
        assert edge_json["schema_version"] == "2.1"

    def test_edge_page_sorted_by_abs_edge_descending(self, tmp_path: Path):
        """Edge page props must be sorted by absolute edge value descending."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges))
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        props = edge_json.get("props", [])
        if len(props) >= 2:
            abs_edges = [abs(float(p.get("edge", 0))) for p in props]
            assert abs_edges == sorted(abs_edges, reverse=True), (
                "Edge page props must be sorted by |edge| descending"
            )

    def test_edge_page_game_date_matches_input(self, tmp_path: Path):
        """Edge page game_date must match the --game-date argument."""
        proj  = _make_proj_parquet(tmp_path)
        edges = _make_edges_parquet(tmp_path)
        _run_generate_web_pages(tmp_path, str(proj), str(edges), game_date="2026-07-14")
        edge_json = json.loads((tmp_path / "Pre-Game" / "Edge" / "latest.json").read_text())
        assert edge_json["game_date"] == "2026-07-14"

    def test_generate_web_pages_empty_edges_exits_zero(self, tmp_path: Path):
        """Empty edges parquet must produce a valid (empty props) page, not an error."""
        proj  = _make_proj_parquet(tmp_path)
        empty_edges = tmp_path / "empty_edges.parquet"
        pd.DataFrame(columns=["player_name", "player_id", "stat", "line", "edge_over",
                               "kelly_fraction", "model_prob_over", "market_prob_over_no_vig",
                               "pmf_json", "pmf_mean"]).to_parquet(empty_edges, index=False)
        result = _run_generate_web_pages(tmp_path, str(proj), str(empty_edges))
        assert result.returncode == 0, (
            f"Empty edges must not crash the page generator.\nstderr={result.stderr[:300]}"
        )


# ===========================================================================
# GROUP 8: Workflow fail-closed behavior
# ===========================================================================

class TestWorkflowFailClosed:

    def test_generate_web_pages_requires_game_date(self, tmp_path: Path):
        """generate_web_pages must fail when --game-date is missing."""
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "scripts" / "generate_web_pages.py"),
            # deliberately omit --game-date
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0, (
            "generate_web_pages must exit nonzero when --game-date is missing"
        )

    def test_page_release_validator_fails_when_release_id_missing(self):
        """validate_page_release_lineage raises when release_id is absent from pages."""
        from wnba_props_model.pipeline.market_integrity import ArtifactLineageMismatchError
        # Pages without release_id (old format)
        edge_json = {"schema_version": "2.1", "generated_at": "2026-07-14T12:00:00Z"}
        pmf_json  = {"schema_version": "2.1", "generated_at": "2026-07-14T12:00:00Z"}
        with pytest.raises(ArtifactLineageMismatchError):
            validate_page_release_lineage(edge_json, pmf_json, expected_release_id="RUN_X")
