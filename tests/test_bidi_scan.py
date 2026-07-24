"""A9: hidden/bidirectional Unicode control scanner + CI wiring."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN = REPO / "scripts" / "scan_bidi_controls.py"


def _mod():
    spec = importlib.util.spec_from_file_location("bidi", SCAN)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_detects_forbidden_controls(tmp_path):
    m = _mod()
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\u202ehidden\u202c\n")            # RLO ... PDF
    hits = m.scan_file(bad)
    cps = {cp for _, cp, _ in hits}
    assert 0x202E in cps and 0x202C in cps


def test_clean_file_has_no_hits(tmp_path):
    m = _mod()
    ok = tmp_path / "ok.py"
    ok.write_text("def f():\n    return 42\n")
    assert m.scan_file(ok) == []


def test_repo_is_clean_via_cli():
    r = subprocess.run([sys.executable, str(SCAN)], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_cli_fails_on_bad_file(tmp_path):
    bad = tmp_path / "bad.txt"; bad.write_text("a\u200fb\n")   # RLM
    r = subprocess.run([sys.executable, str(SCAN), str(bad)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "BIDI" in (r.stdout + r.stderr)


def test_ci_wires_the_scan():
    wf = (REPO / ".github" / "workflows" / "foundation_lock.yml").read_text()
    assert "scan_bidi_controls.py" in wf