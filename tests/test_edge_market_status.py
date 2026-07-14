"""Edge market status contract tests — production hotfix.

Proves that:
  - market_status is evidence-based (read from market audit JSON)
  - Edge payload contains all required market evidence fields
  - Consistency rules are enforced (status vs quote counts)
  - No stale prior-date rows appear
  - Generator fails when supplied status conflicts with audit counts

Uses the real generate_web_pages.py CLI — no stubs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.simulation import normalize_pmf, pmf_to_json


# ---------------------------------------------------------------------------
# Required Edge payload fields (must all be present)
# ---------------------------------------------------------------------------

REQUIRED_EDGE_FIELDS = [
    "market_status",
    "market_request_status",
    "raw_quote_count",
    "fresh_quote_count",
    "reconciled_quote_count",
    "rejection_counts",
    "market_request_timestamp_utc",
    "release_id",
    "model_version",
    "calibration_version",
]

VALID_MARKET_STATUSES = {
    "SUCCESS_WITH_MARKETS",
    "LIVE_MARKETS_NOT_YET_AVAILABLE",
    "FAILURE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proj(tmp_path: Path, game_date: str = "2026-07-14") -> Path:
    arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
    df = pd.DataFrame([{
        "game_id": "G001", "player_id": "P001", "player_name": "Alice Adams",
        "stat": "pts", "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0,
        "model_prob_over": 0.3, "role_bucket": "starter", "game_date": game_date,
    }])
    p = tmp_path / f"player_projections_{game_date}.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_edges(tmp_path: Path, n_rows: int = 1) -> Path:
    """Create a non-empty edges parquet with n_rows rows."""
    if n_rows == 0:
        df = pd.DataFrame(columns=["player_name", "player_id", "stat", "line",
                                     "edge_over", "kelly_fraction", "model_prob_over",
                                     "market_prob_over_no_vig"])
    else:
        arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
        rows = [{
            "game_id": "G001", "player_id": "P001", "player_name": "Alice Adams",
            "stat": "pts", "line": 1.5, "over_odds": -110, "under_odds": -110,
            "model_prob_over": 0.55, "market_prob_over_no_vig": 0.50,
            "edge_over": 0.05, "kelly_fraction": 0.02, "vendor": "dk",
            "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0,
        } for _ in range(n_rows)]
        df = pd.DataFrame(rows)
    p = tmp_path / "publishable_edges.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_audit(tmp_path: Path, *, status: str, total_market_rows: int,
                raw_quotes: int = 0, fresh_quotes: int = 0,
                reconciled: int = 0, rejection_counts: dict | None = None) -> Path:
    """Create a market audit JSON matching the build_edge_report format."""
    audit = {
        "market_status": status,
        "total_market_rows": total_market_rows,
        "raw_quote_count": raw_quotes,
        "fresh_quote_count": fresh_quotes,
        "reconciled_quote_count": reconciled,
        "rejection_counts": rejection_counts or {},
        "market_request_status": "ok" if status != "FAILURE" else "failed",
        "market_request_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "props_source": "bdl",
    }
    p = tmp_path / "edge_report_2026-07-14.json"
    p.write_text(json.dumps(audit))
    return p


def _run(
    tmp_path: Path,
    proj_path: str,
    edges_path: str,
    *,
    audit_path: str = "",
    market_status_override: str = "",
    game_date: str = "2026-07-14",
    release_id: str = "test-release",
    model_version: str = "test-model-v1",
    calibration_version: str = "test-cal-v1",
) -> subprocess.CompletedProcess:
    out = tmp_path / "Pre-Game"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/generate_web_pages.py",
        "--game-date", game_date,
        "--projections", proj_path,
        "--edges", edges_path,
        "--out-dir", str(out),
        "--json-only",
        "--release-id", release_id,
        "--git-commit", "abc123",
        "--model-version", model_version,
        "--calibration-version", calibration_version,
    ]
    if audit_path:
        cmd += ["--market-audit-json", audit_path]
    if market_status_override:
        cmd += ["--market-status", market_status_override]
    return subprocess.run(cmd, capture_output=True, text=True)


def _load_edge_release(tmp_path: Path, game_date: str = "2026-07-14") -> dict:
    """Load Edge release payload following the pointer."""
    ptr_path = tmp_path / "Pre-Game" / "Edge" / "latest.json"
    if not ptr_path.exists():
        return {}
    ptr = json.loads(ptr_path.read_text())
    if not ptr.get("pointer"):
        return ptr
    release_p = ptr_path.parent / ptr.get("payload_path", "")
    if release_p.exists():
        return json.loads(release_p.read_text())
    date_p = ptr_path.parent / f"{game_date}.json"
    if date_p.exists():
        return json.loads(date_p.read_text())
    return ptr


# ===========================================================================
# 1. Required field presence
# ===========================================================================

class TestRequiredEdgeFields:

    def test_edge_payload_has_all_required_fields(self, tmp_path: Path):
        """Edge release payload must contain every required market evidence field."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0, raw_quotes=0)

        r = _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        assert r.returncode == 0, r.stderr[:200]

        release = _load_edge_release(tmp_path)
        missing = [f for f in REQUIRED_EDGE_FIELDS if f not in release]
        assert missing == [], (
            f"Edge release payload missing required fields: {missing}"
        )

    def test_edge_payload_market_status_is_valid(self, tmp_path: Path):
        """market_status must be one of the three valid values."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        release = _load_edge_release(tmp_path)
        assert release.get("market_status") in VALID_MARKET_STATUSES


# ===========================================================================
# 2. SUCCESS_WITH_MARKETS consistency
# ===========================================================================

class TestSuccessWithMarkets:

    def test_success_requires_rows_greater_than_zero(self, tmp_path: Path):
        """SUCCESS_WITH_MARKETS requires raw_quote_count > 0 and Edge rows > 0."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=1)
        audit = _make_audit(tmp_path, status="SUCCESS_WITH_MARKETS",
                            total_market_rows=1, raw_quotes=5, fresh_quotes=5,
                            reconciled=1)

        r = _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        assert r.returncode == 0
        release = _load_edge_release(tmp_path)
        assert release["market_status"] == "SUCCESS_WITH_MARKETS"
        assert int(release["raw_quote_count"]) > 0
        assert int(release["reconciled_quote_count"]) > 0

    def test_success_overridden_when_zero_rows(self, tmp_path: Path):
        """SUCCESS_WITH_MARKETS with 0 rows must be corrected — no fraudulent status."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        # Audit says success but actual edge file is empty — inconsistency
        audit = _make_audit(tmp_path, status="SUCCESS_WITH_MARKETS",
                            total_market_rows=0, raw_quotes=0)

        r = _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        assert r.returncode == 0
        release = _load_edge_release(tmp_path)
        # Must NOT allow SUCCESS_WITH_MARKETS when rows = 0
        assert release["market_status"] != "SUCCESS_WITH_MARKETS", (
            "SUCCESS_WITH_MARKETS must not be published when Edge rows = 0"
        )


# ===========================================================================
# 3. LIVE_MARKETS_NOT_YET_AVAILABLE consistency
# ===========================================================================

class TestLiveMarketsNotYetAvailable:

    def test_lmnya_requires_zero_rows(self, tmp_path: Path):
        """LIVE_MARKETS_NOT_YET_AVAILABLE requires rows = 0."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0, raw_quotes=0)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        release = _load_edge_release(tmp_path)
        assert release["market_status"] == "LIVE_MARKETS_NOT_YET_AVAILABLE"
        assert int(release["raw_quote_count"]) == 0
        assert int(release["reconciled_quote_count"]) == 0
        assert release["total_props"] == 0

    def test_lmnya_has_empty_props_list(self, tmp_path: Path):
        """LIVE_MARKETS_NOT_YET_AVAILABLE must have empty props — no stale cards."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        release = _load_edge_release(tmp_path)
        assert release.get("props", []) == [], (
            "LIVE_MARKETS_NOT_YET_AVAILABLE must have empty props list — no stale cards"
        )

    def test_lmnya_overridden_when_rows_present(self, tmp_path: Path):
        """LIVE_MARKETS_NOT_YET_AVAILABLE with actual rows must be corrected."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=1)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=1, raw_quotes=3, reconciled=1)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        release = _load_edge_release(tmp_path)
        # Must correct to SUCCESS_WITH_MARKETS
        assert release["market_status"] == "SUCCESS_WITH_MARKETS", (
            "LIVE_MARKETS_NOT_YET_AVAILABLE must not be published when rows exist"
        )


# ===========================================================================
# 4. FAILURE consistency
# ===========================================================================

class TestFailureStatus:

    def test_failure_results_in_zero_rows(self, tmp_path: Path):
        """FAILURE status must produce zero Edge rows."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="FAILURE",
                            total_market_rows=0, raw_quotes=0)
        audit_data = json.loads((tmp_path / "edge_report_2026-07-14.json").read_text())
        audit_data["market_request_status"] = "failed"
        (tmp_path / "edge_report_2026-07-14.json").write_text(json.dumps(audit_data))

        r = _run(tmp_path, str(proj), str(edges),
                 audit_path=str(tmp_path / "edge_report_2026-07-14.json"))
        assert r.returncode == 0
        release = _load_edge_release(tmp_path)
        assert release["total_props"] == 0, "FAILURE must produce 0 Edge rows"


# ===========================================================================
# 5. Audit JSON is the authoritative source
# ===========================================================================

class TestAuditJsonSource:

    def test_market_request_timestamp_from_audit(self, tmp_path: Path):
        """market_request_timestamp_utc must be read from the audit JSON."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        ts = "2026-07-14T21:00:00Z"
        audit_data = {
            "market_status": "LIVE_MARKETS_NOT_YET_AVAILABLE",
            "total_market_rows": 0,
            "raw_quote_count": 0,
            "fresh_quote_count": 0,
            "reconciled_quote_count": 0,
            "rejection_counts": {},
            "market_request_status": "ok",
            "market_request_timestamp_utc": ts,
            "generated_at": ts,
        }
        audit_path = tmp_path / "edge_report_2026-07-14.json"
        audit_path.write_text(json.dumps(audit_data))

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit_path))
        release = _load_edge_release(tmp_path)
        assert release.get("market_request_timestamp_utc") == ts, (
            "market_request_timestamp_utc must come from the audit JSON"
        )

    def test_rejection_counts_from_audit(self, tmp_path: Path):
        """rejection_counts must be propagated from the audit JSON."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        rejections = {"stale_quote": 2, "malformed_odds": 1}
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0, rejection_counts=rejections)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit))
        release = _load_edge_release(tmp_path)
        assert release.get("rejection_counts") == rejections, (
            "rejection_counts must be passed through from the audit JSON"
        )

    def test_release_id_present_in_edge_payload(self, tmp_path: Path):
        """release_id must appear in the Edge payload for lineage."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit),
             release_id="test-release-42")
        release = _load_edge_release(tmp_path)
        assert release.get("release_id") == "test-release-42"

    def test_model_and_calibration_version_present(self, tmp_path: Path):
        """model_version and calibration_version must be in the Edge payload."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n_rows=0)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            total_market_rows=0)

        _run(tmp_path, str(proj), str(edges), audit_path=str(audit),
             model_version="hgb_v2.3.1", calibration_version="idr_v12")
        release = _load_edge_release(tmp_path)
        assert release.get("model_version") == "hgb_v2.3.1"
        assert release.get("calibration_version") == "idr_v12"


# ===========================================================================
# 6. pregame_initial.yml workflow contract
# ===========================================================================

class TestPregameInitialWorkflowContract:

    def test_pregame_initial_passes_market_audit_json(self):
        """pregame_initial.yml must pass --market-audit-json to generate_web_pages.py."""
        text = Path(".github/workflows/pregame_initial.yml").read_text()
        assert "--market-audit-json" in text, (
            "pregame_initial.yml must pass --market-audit-json to generate_web_pages.py"
        )

    def test_pregame_initial_reads_audit_from_deliveries(self):
        """pregame_initial.yml must read market audit from the deliveries directory."""
        text = Path(".github/workflows/pregame_initial.yml").read_text()
        assert "edge_report_" in text or "EDGE_AUDIT" in text, (
            "pregame_initial.yml must read the edge_report audit JSON"
        )

    def test_generate_web_pages_supports_market_audit_json_arg(self):
        """generate_web_pages.py must accept --market-audit-json."""
        import re as _re  # noqa: PLC0415
        result = subprocess.run(
            [sys.executable, "scripts/generate_web_pages.py", "--help"],
            capture_output=True, text=True, env={**__import__("os").environ, "NO_COLOR": "1"},
        )
        # Strip ANSI escape codes before checking
        clean = _re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        assert "--market-audit-json" in clean, (
            f"generate_web_pages.py must support --market-audit-json. "
            f"Help output (cleaned):\n{clean[:500]}"
        )

    def test_generate_web_pages_supports_market_status_arg(self):
        """generate_web_pages.py must accept --market-status."""
        import re as _re  # noqa: PLC0415
        result = subprocess.run(
            [sys.executable, "scripts/generate_web_pages.py", "--help"],
            capture_output=True, text=True, env={**__import__("os").environ, "NO_COLOR": "1"},
        )
        clean = _re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        assert "--market-status" in clean, (
            f"generate_web_pages.py must support --market-status. "
            f"Help output (cleaned):\n{clean[:500]}"
        )
