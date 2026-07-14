"""Edge market status tests — production hotfix.

Verifies:
  - Edge payload embeds market_status, raw_quote_count, reconciled_quote_count
  - market_status is consistent with actual row count
  - SUCCESS_WITH_MARKETS iff rows > 0
  - LIVE_MARKETS_NOT_YET_AVAILABLE iff request succeeded and rows == 0
  - FAILURE iff request failed
  - generate_web_pages.py --market-audit-json reads status from audit file
  - pregame_initial.yml passes --market-audit-json to generate_web_pages.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models.simulation import normalize_pmf, pmf_to_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_proj(tmp_path: Path, game_date: str = "2026-07-14") -> Path:
    arr = normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))
    proj = pd.DataFrame([{
        "game_id": "G001", "player_id": "P001", "player_name": "Alice",
        "stat": "pts", "pmf_json": pmf_to_json(arr), "pmf_mean": 1.0,
        "model_prob_over": 0.3, "role_bucket": "starter", "game_date": game_date,
    }])
    p = tmp_path / f"player_projections_{game_date}.parquet"
    proj.to_parquet(p, index=False)
    return p


def _run_generate(
    tmp_path: Path, proj_path: str, edges_path: str, *,
    market_status: str = "", market_audit_json: str = "",
    game_date: str = "2026-07-14",
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
        "--release-id", "test-release",
        "--git-commit", "abc123",
    ]
    if market_status:
        cmd += ["--market-status", market_status]
    if market_audit_json:
        cmd += ["--market-audit-json", market_audit_json]
    return subprocess.run(cmd, capture_output=True, text=True)


def _load_edge_release(tmp_path: Path, game_date: str = "2026-07-14") -> dict:
    ptr_path = tmp_path / "Pre-Game" / "Edge" / "latest.json"
    if not ptr_path.exists():
        return {}
    ptr = json.loads(ptr_path.read_text())
    if not ptr.get("pointer"):
        return ptr
    release_p = ptr_path.parent / ptr.get("payload_path", "")
    if release_p.exists():
        return json.loads(release_p.read_text())
    # Try date-specific
    date_p = ptr_path.parent / f"{game_date}.json"
    if date_p.exists():
        return json.loads(date_p.read_text())
    return ptr


# ---------------------------------------------------------------------------
# Core market-status consistency tests
# ---------------------------------------------------------------------------

class TestMarketStatusConsistency:

    def test_live_markets_not_available_when_empty_edges(self, tmp_path: Path):
        """Empty edges → market_status = LIVE_MARKETS_NOT_YET_AVAILABLE."""
        proj = _make_minimal_proj(tmp_path)
        empty_edges = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(empty_edges, index=False)

        r = _run_generate(tmp_path, str(proj), str(empty_edges))
        assert r.returncode == 0, r.stderr[:200]

        release = _load_edge_release(tmp_path)
        assert release.get("market_status") == "LIVE_MARKETS_NOT_YET_AVAILABLE", (
            f"Expected LIVE_MARKETS_NOT_YET_AVAILABLE, got {release.get('market_status')!r}"
        )
        assert int(release.get("raw_quote_count", -1)) == 0
        assert int(release.get("reconciled_quote_count", -1)) == 0
        assert release.get("total_props") == 0

    def test_success_with_markets_when_edges_present(self, tmp_path: Path):
        """Non-empty edges → market_status = SUCCESS_WITH_MARKETS."""
        proj = _make_minimal_proj(tmp_path)
        edges = pd.DataFrame([{
            "game_id": "G001", "player_id": "P001", "player_name": "Alice",
            "stat": "pts", "line": 1.5, "over_odds": -110, "under_odds": -110,
            "model_prob_over": 0.55, "market_prob_over_no_vig": 0.50,
            "edge_over": 0.05, "kelly_fraction": 0.02, "vendor": "dk",
            "pmf_json": pmf_to_json(normalize_pmf(np.array([0.3, 0.4, 0.2, 0.1]))),
            "pmf_mean": 1.0,
        }])
        edges_path = tmp_path / "edges.parquet"
        edges.to_parquet(edges_path, index=False)

        r = _run_generate(tmp_path, str(proj), str(edges_path))
        assert r.returncode == 0, r.stderr[:200]

        release = _load_edge_release(tmp_path)
        assert release.get("market_status") == "SUCCESS_WITH_MARKETS", (
            f"Expected SUCCESS_WITH_MARKETS, got {release.get('market_status')!r}"
        )

    def test_market_status_consistency_override(self, tmp_path: Path):
        """When --market-status=SUCCESS_WITH_MARKETS but rows=0, auto-correct to LMNYA."""
        proj = _make_minimal_proj(tmp_path)
        empty_edges = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(empty_edges, index=False)

        r = _run_generate(tmp_path, str(proj), str(empty_edges),
                         market_status="SUCCESS_WITH_MARKETS")
        assert r.returncode == 0
        release = _load_edge_release(tmp_path)
        # Should be corrected to LIVE_MARKETS_NOT_YET_AVAILABLE
        assert release.get("market_status") == "LIVE_MARKETS_NOT_YET_AVAILABLE", (
            "market_status=SUCCESS_WITH_MARKETS with 0 rows must be corrected"
        )

    def test_market_audit_json_overrides_status(self, tmp_path: Path):
        """--market-audit-json reads market_status from the audit file."""
        proj = _make_minimal_proj(tmp_path)
        empty_edges = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(empty_edges, index=False)

        audit = {
            "market_status": "LIVE_MARKETS_NOT_YET_AVAILABLE",
            "total_market_rows": 0,
            "generated_at": "2026-07-14T21:00:00Z",
            "props_source": "bdl",
        }
        audit_path = tmp_path / "edge_report_2026-07-14.json"
        audit_path.write_text(json.dumps(audit))

        r = _run_generate(tmp_path, str(proj), str(empty_edges),
                         market_audit_json=str(audit_path))
        assert r.returncode == 0, r.stderr[:200]
        release = _load_edge_release(tmp_path)
        assert release.get("market_status") == "LIVE_MARKETS_NOT_YET_AVAILABLE"
        assert "raw_quote_count" in release

    def test_edge_payload_has_required_market_fields(self, tmp_path: Path):
        """Edge release payload must contain all required market evidence fields."""
        proj = _make_minimal_proj(tmp_path)
        empty_edges = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(empty_edges, index=False)

        r = _run_generate(tmp_path, str(proj), str(empty_edges))
        assert r.returncode == 0
        release = _load_edge_release(tmp_path)

        required = ["market_status", "raw_quote_count", "reconciled_quote_count"]
        for field in required:
            assert field in release, (
                f"Edge release payload must contain '{field}' field"
            )

    def test_no_stale_prior_date_cards_when_no_markets(self, tmp_path: Path):
        """When market_status=LIVE_MARKETS_NOT_YET_AVAILABLE, props must be empty list."""
        proj = _make_minimal_proj(tmp_path)
        empty_edges = tmp_path / "edges.parquet"
        pd.DataFrame().to_parquet(empty_edges, index=False)

        r = _run_generate(tmp_path, str(proj), str(empty_edges))
        assert r.returncode == 0
        release = _load_edge_release(tmp_path)
        assert release.get("market_status") == "LIVE_MARKETS_NOT_YET_AVAILABLE"
        assert release.get("total_props") == 0
        assert release.get("props", []) == []

    def test_generate_web_pages_has_market_status_arg(self):
        """generate_web_pages.py CLI must support --market-status argument."""
        result = subprocess.run(
            [sys.executable, "scripts/generate_web_pages.py", "--help"],
            capture_output=True, text=True,
        )
        assert "--market-status" in result.stdout, (
            "generate_web_pages.py must support --market-status argument"
        )

    def test_generate_web_pages_has_market_audit_json_arg(self):
        """generate_web_pages.py CLI must support --market-audit-json argument."""
        result = subprocess.run(
            [sys.executable, "scripts/generate_web_pages.py", "--help"],
            capture_output=True, text=True,
        )
        assert "--market-audit-json" in result.stdout, (
            "generate_web_pages.py must support --market-audit-json argument"
        )

    def test_pregame_initial_passes_market_audit_to_generate_pages(self):
        """pregame_initial.yml must pass --market-audit-json to generate_web_pages.py."""
        text = Path(".github/workflows/pregame_initial.yml").read_text()
        assert "--market-audit-json" in text, (
            "pregame_initial.yml must pass --market-audit-json to generate_web_pages.py"
        )
