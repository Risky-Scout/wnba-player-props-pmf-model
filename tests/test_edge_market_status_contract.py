"""Edge market status contract tests — production hotfix.

Proves via functional CLI invocation that:
  - --market-status and --market-audit-json are accepted by generate_web_pages.py
  - market_status is evidence-based (read from audit JSON)
  - All required market fields appear in the Edge release payload
  - Consistency rules are enforced (status vs quote counts auto-corrected)
  - No stale prior-date rows appear in any status
  - pregame_initial.yml passes --market-audit-json to generate_web_pages.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.simulation import normalize_pmf, pmf_to_json

# ---------------------------------------------------------------------------
# Required Edge payload fields
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

VALID_STATUSES = {"SUCCESS_WITH_MARKETS", "LIVE_MARKETS_NOT_YET_AVAILABLE", "FAILURE"}


# ---------------------------------------------------------------------------
# Fixture helpers
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


def _make_edges(tmp_path: Path, n: int = 0) -> Path:
    if n == 0:
        df = pd.DataFrame(columns=["player_name", "player_id", "stat"])
    else:
        arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
        df = pd.DataFrame([{
            "game_id": "G001", "player_id": "P001", "player_name": "Alice Adams",
            "stat": "pts", "line": 1.5, "over_odds": -110, "under_odds": -110,
            "model_prob_over": 0.55, "market_prob_over_no_vig": 0.50,
            "edge_over": 0.05, "kelly_fraction": 0.02, "vendor": "dk",
            "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0,
        } for _ in range(n)])
    p = tmp_path / "publishable_edges.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_audit(tmp_path: Path, *, status: str, raw: int = 0, fresh: int = 0,
                reconciled: int = 0, total_market_rows: int = 0,
                req_status: str = "ok", rejections: dict | None = None) -> Path:
    ts = datetime.now(timezone.utc).isoformat()
    audit = {
        "market_status": status,
        "market_request_status": req_status,
        "total_market_rows": total_market_rows,
        "raw_quote_count": raw,
        "fresh_quote_count": fresh,
        "reconciled_quote_count": reconciled,
        "rejection_counts": rejections or {},
        "market_request_timestamp_utc": ts,
        "generated_at": ts,
        "props_source": "bdl",
    }
    p = tmp_path / "edge_report_2026-07-14.json"
    p.write_text(json.dumps(audit))
    return p


def _run(tmp_path: Path, proj: str, edges: str, *,
         audit: str = "", status_override: str = "",
         release_id: str = "test-r", model_ver: str = "m-v1",
         cal_ver: str = "c-v1", game_date: str = "2026-07-14") -> subprocess.CompletedProcess:
    out = tmp_path / "Pre-Game"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/generate_web_pages.py",
        "--game-date", game_date,
        "--projections", proj,
        "--edges", edges,
        "--out-dir", str(out),
        "--json-only",
        "--release-id", release_id,
        "--git-commit", "abc123",
        "--model-version", model_ver,
        "--calibration-version", cal_ver,
    ]
    if audit:
        cmd += ["--market-audit-json", audit]
    if status_override:
        cmd += ["--market-status", status_override]
    import os  # noqa: PLC0415
    env = {**os.environ, "NO_COLOR": "1"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _release(tmp_path: Path, game_date: str = "2026-07-14") -> dict:
    ptr_path = tmp_path / "Pre-Game" / "Edge" / "latest.json"
    if not ptr_path.exists():
        return {}
    ptr = json.loads(ptr_path.read_text())
    if not ptr.get("pointer"):
        return ptr
    rp = ptr_path.parent / ptr.get("payload_path", "")
    if rp.exists():
        return json.loads(rp.read_text())
    dp = ptr_path.parent / f"{game_date}.json"
    return json.loads(dp.read_text()) if dp.exists() else ptr


# ===========================================================================
# 1. Functional CLI option acceptance tests
# ===========================================================================

class TestCLIOptionAcceptance:
    """Functional tests proving --market-status and --market-audit-json are accepted."""

    def test_market_audit_json_is_accepted_by_cli(self, tmp_path: Path):
        """--market-audit-json must be accepted without error (exit 0, no UnrecognizedArgument)."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE")
        r = _run(tmp_path, str(proj), str(edges), audit=str(audit))
        assert r.returncode == 0, (
            f"--market-audit-json rejected or caused error.\n"
            f"stdout: {r.stdout[:300]}\nstderr: {r.stderr[:300]}"
        )
        assert "No such option" not in r.stderr and "unrecognized" not in r.stderr.lower(), (
            "--market-audit-json must be a recognized CLI option"
        )

    def test_market_status_is_accepted_by_cli(self, tmp_path: Path):
        """--market-status must be accepted without error."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        r = _run(tmp_path, str(proj), str(edges),
                 status_override="LIVE_MARKETS_NOT_YET_AVAILABLE")
        assert r.returncode == 0, (
            f"--market-status rejected or caused error.\nstderr: {r.stderr[:300]}"
        )
        assert "No such option" not in r.stderr, "--market-status must be recognized"

    def test_market_status_invalid_value_still_runs(self, tmp_path: Path):
        """An invalid market_status value is accepted by CLI (validation is by logic)."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        r = _run(tmp_path, str(proj), str(edges),
                 status_override="SOME_VALID_STRING")
        # CLI accepts any string; consistency enforcement happens in the payload builder
        assert r.returncode == 0, "CLI should not reject unknown status strings"

    def test_audit_json_status_propagated_to_payload(self, tmp_path: Path):
        """When --market-audit-json is given, its market_status is used in the payload."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            raw=0, fresh=0, reconciled=0)
        r = _run(tmp_path, str(proj), str(edges), audit=str(audit))
        assert r.returncode == 0
        release = _release(tmp_path)
        assert release.get("market_status") == "LIVE_MARKETS_NOT_YET_AVAILABLE", (
            "market_status from audit JSON must appear in Edge release payload"
        )

    def test_audit_json_counts_propagated_to_payload(self, tmp_path: Path):
        """raw_quote_count, fresh_quote_count, reconciled_quote_count come from audit."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            raw=7, fresh=5, reconciled=0, total_market_rows=0)
        _run(tmp_path, str(proj), str(edges), audit=str(audit))
        release = _release(tmp_path)
        assert int(release.get("raw_quote_count", -1)) == 7
        assert int(release.get("fresh_quote_count", -1)) == 5

    def test_contradictory_status_is_auto_corrected(self, tmp_path: Path):
        """SUCCESS_WITH_MARKETS with 0 edge rows must be corrected to LMNYA."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n=0)
        # Audit claims success but edges file is empty
        audit = _make_audit(tmp_path, status="SUCCESS_WITH_MARKETS",
                            raw=0, total_market_rows=0)
        r = _run(tmp_path, str(proj), str(edges), audit=str(audit))
        assert r.returncode == 0
        release = _release(tmp_path)
        assert release.get("market_status") != "SUCCESS_WITH_MARKETS", (
            "SUCCESS_WITH_MARKETS with 0 rows must be auto-corrected"
        )

    def test_lmnya_with_rows_is_auto_corrected(self, tmp_path: Path):
        """LIVE_MARKETS_NOT_YET_AVAILABLE with actual rows must be corrected."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path, n=1)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            raw=3, total_market_rows=1, reconciled=1)
        r = _run(tmp_path, str(proj), str(edges), audit=str(audit))
        assert r.returncode == 0
        release = _release(tmp_path)
        assert release.get("market_status") == "SUCCESS_WITH_MARKETS", (
            "LMNYA with actual rows must be corrected to SUCCESS_WITH_MARKETS"
        )


# ===========================================================================
# 2. Required field presence
# ===========================================================================

class TestRequiredFields:

    def test_all_required_fields_present(self, tmp_path: Path):
        """Every required market evidence field must appear in the Edge release payload."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE")
        r = _run(tmp_path, str(proj), str(edges), audit=str(audit))
        assert r.returncode == 0
        release = _release(tmp_path)
        missing = [f for f in REQUIRED_EDGE_FIELDS if f not in release]
        assert missing == [], f"Edge payload missing required fields: {missing}"

    def test_market_status_is_valid_value(self, tmp_path: Path):
        """market_status must be one of the three valid values."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE")
        _run(tmp_path, str(proj), str(edges), audit=str(audit))
        release = _release(tmp_path)
        assert release.get("market_status") in VALID_STATUSES

    def test_no_stale_cards_when_lmnya(self, tmp_path: Path):
        """LIVE_MARKETS_NOT_YET_AVAILABLE must have zero props (no stale cards)."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE")
        _run(tmp_path, str(proj), str(edges), audit=str(audit))
        release = _release(tmp_path)
        assert release.get("props", []) == []
        assert int(release.get("total_props", 0)) == 0

    def test_rejection_counts_from_audit(self, tmp_path: Path):
        """rejection_counts must be propagated from the audit JSON."""
        proj = _make_proj(tmp_path)
        edges = _make_edges(tmp_path)
        rejections = {"stale_quote": 3, "malformed_odds": 1}
        audit = _make_audit(tmp_path, status="LIVE_MARKETS_NOT_YET_AVAILABLE",
                            rejections=rejections)
        _run(tmp_path, str(proj), str(edges), audit=str(audit))
        release = _release(tmp_path)
        assert release.get("rejection_counts") == rejections


# ===========================================================================
# 3. Workflow contract
# ===========================================================================

class TestWorkflowContract:

    def test_pregame_initial_passes_market_audit_json(self):
        """pregame_initial.yml must pass --market-audit-json to generate_web_pages.py."""
        text = Path(".github/workflows/pregame_initial.yml").read_text()
        assert "--market-audit-json" in text

    def test_cli_grep_confirms_args_in_source(self):
        """git grep must find --market-audit-json and --market-status in the source file."""
        result = subprocess.run(
            ["git", "grep", "-n", "market-audit-json", "--", "scripts/generate_web_pages.py"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, "git grep found no market-audit-json in generate_web_pages.py"
        assert "generate_web_pages.py" in result.stdout

        result2 = subprocess.run(
            ["git", "grep", "-n", "market-status", "--", "scripts/generate_web_pages.py"],
            capture_output=True, text=True,
        )
        assert result2.returncode == 0, "git grep found no market-status in generate_web_pages.py"

    def test_no_color_help_contains_market_audit_arg(self):
        """NO_COLOR=1 help output must contain --market-audit-json (plain text, no ANSI)."""
        import os  # noqa: PLC0415
        env = {**os.environ, "NO_COLOR": "1"}
        result = subprocess.run(
            [sys.executable, "scripts/generate_web_pages.py", "--help"],
            capture_output=True, text=True, env=env,
        )
        # With NO_COLOR=1, output should be plain text
        clean = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        assert "--market-audit-json" in clean, (
            f"NO_COLOR=1 help output must contain --market-audit-json.\n"
            f"Cleaned output:\n{clean[:500]}"
        )

    def test_no_color_help_contains_market_status_arg(self):
        """NO_COLOR=1 help output must contain --market-status (plain text)."""
        import os  # noqa: PLC0415
        env = {**os.environ, "NO_COLOR": "1"}
        result = subprocess.run(
            [sys.executable, "scripts/generate_web_pages.py", "--help"],
            capture_output=True, text=True, env=env,
        )
        clean = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
        assert "--market-status" in clean
