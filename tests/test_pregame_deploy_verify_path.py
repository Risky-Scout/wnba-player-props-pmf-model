"""Focused regression tests for the post-deployment verification path fix.

The custom domain (sportsodds.wizardofodds.com) is served by nginx from the FTP
root /WNBA/ (ftp_deploy.py REMOTE_BASE). Two coupled defects made the BLOCKING
"Post-deployment verification — custom domain" step fail even after a successful
deploy:

1. The verification fetched the gh-pages layout path
   (/tools/odds-scanner/predictions/WNBA/...) instead of the live /WNBA/ path,
   so it read stale content.
2. ftp_deploy.py uploaded only top-level files (non-recursive), so the immutable
   release payloads at releases/<release_id>.json — referenced by latest.json's
   payload_path and by the page JS — 404'd on the live domain.

These tests lock both fixes.
"""
from __future__ import annotations

import ftplib
import io
from pathlib import Path

import pytest

import scripts.ftp_deploy as ftp_deploy

WF_PATH = Path(".github/workflows/pregame_initial.yml")


# ── Fix 1: verification hits the live /WNBA/ path, not the gh-pages layout ─────

def test_verification_uses_live_wnba_path():
    content = WF_PATH.read_text()
    assert 'BASE = f"{CDN}/WNBA/Pre-Game"' in content, (
        "Post-deploy verification must check the live custom-domain /WNBA/ path "
        "that FTP deploys to"
    )


def test_verification_does_not_use_ghpages_layout_for_custom_domain():
    """The custom-domain payload fetches must resolve against BASE (/WNBA/), not
    the gh-pages /tools/odds-scanner/predictions/WNBA/ layout."""
    content = WF_PATH.read_text()
    # Extract the verification step body (from its name to the next step marker).
    start = content.index("Post-deployment verification — custom domain")
    tail = content[start:]
    end = tail.index("\n      - name:", 1) if "\n      - name:" in tail[1:] else len(tail)
    step = tail[:end]
    # Match the actual fetch interpolation, not prose comments.
    assert "{CDN}/tools/odds-scanner" not in step, (
        "Post-deploy verification must NOT fetch the gh-pages layout path from the "
        "custom domain; use the live /WNBA/ BASE for pointer and payload fetches"
    )
    assert '{BASE}/Edge/' in step and '{BASE}/Distributions/' in step, (
        "payload fetches should be relative to the corrected BASE"
    )


# ── Fix 2: ftp_deploy uploads recursively (releases/<id>.json subdir) ──────────

class _FakeFTP:
    """Minimal fake FTP that records STOR targets and supports the context manager."""

    def __init__(self, *_a, **_k):
        self.stored: list[str] = []
        self.dirs: set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, *_a, **_k):
        return "230 OK"

    def getwelcome(self):
        return "220 fake"

    def cwd(self, path):
        # Simulate: dirs we've "made" (or root) exist; unknown dirs raise perm error.
        if path == "/" or path in self.dirs:
            return "250 OK"
        raise ftplib.error_perm(f"550 no such dir {path}")

    def mkd(self, path):
        self.dirs.add(path)
        return path

    def sendcmd(self, *_a, **_k):
        return "200 OK"

    def delete(self, *_a, **_k):
        return "250 OK"

    def nlst(self, *_a, **_k):
        return []

    def storbinary(self, cmd, _fp):
        # cmd == "STOR /WNBA/Pre-Game/Edge/releases/RID.json"
        self.stored.append(cmd.split(" ", 1)[1])
        return "226 OK"


def test_ftp_deploy_uploads_nested_release_payload(tmp_path, monkeypatch):
    # Build a fake local page dir with a top-level pointer + a nested release file.
    local_base = tmp_path / "WNBA"
    edge = local_base / "Pre-Game" / "Edge"
    (edge / "releases").mkdir(parents=True)
    (edge / "latest.json").write_text('{"payload_path": "releases/run123.json"}')
    (edge / "2026-07-16.json").write_text("{}")
    (edge / "releases" / "run123.json").write_text('{"total_props": 130}')

    monkeypatch.setattr(ftp_deploy, "LOCAL_BASE", local_base)
    monkeypatch.setenv("FTP_HOST", "h")
    monkeypatch.setenv("FTP_USER", "u")
    monkeypatch.setenv("FTP_PASS", "p")

    fake = _FakeFTP()
    monkeypatch.setattr(ftp_deploy.ftplib, "FTP", lambda *a, **k: fake)

    ftp_deploy.deploy(dirs=["Pre-Game/Edge"], wipe=False)

    assert "/WNBA/Pre-Game/Edge/latest.json" in fake.stored
    assert "/WNBA/Pre-Game/Edge/releases/run123.json" in fake.stored, (
        "Recursive upload must deploy the immutable release payload subdir so the "
        "pointer's payload_path resolves on the live domain"
    )
