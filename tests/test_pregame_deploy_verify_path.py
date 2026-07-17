"""Regression tests for deploying to the real product page path.

The live product pages are served by nginx at
  https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/...
A prior regression (commit 3efe49d) pointed the FTP deploy at /WNBA/ instead,
so the real Edge/Distributions pages went stale. Additionally the pipeline's
latest.json was a pointer, but the deployed page shells read a SELF-CONTAINED
latest.json (data.props inline).

These tests lock the corrected behavior:
1. ftp_deploy targets /tools/odds-scanner/predictions/WNBA and only touches
   data files (.json), never the existing index.html shells.
2. The post-deploy verification checks the real /tools/odds-scanner/ path and
   reads the self-contained latest.json directly.
3. The generators write a self-contained latest.json (props inline).
"""
from __future__ import annotations

import ftplib
from pathlib import Path

import scripts.ftp_deploy as ftp_deploy

WF_PATH = Path(".github/workflows/pregame_initial.yml")
REAL_BASE = "/tools/odds-scanner/predictions/WNBA"


# ── Deploy target + shell preservation ────────────────────────────────────────

def test_ftp_deploy_targets_real_product_path():
    assert ftp_deploy.REMOTE_BASE == REAL_BASE, (
        "ftp_deploy must publish to the real product path the page shells read"
    )


def test_ftp_deploy_preserves_html_shells():
    # .html must never be uploaded or wiped — the branded shells already exist
    # and must not be regenerated/overwritten.
    assert ".html" not in ftp_deploy.UPLOAD_EXTENSIONS, "must not upload/overwrite index.html shells"
    assert ".html" not in ftp_deploy.WIPE_EXTENSIONS, "must not wipe index.html shells"
    assert ".json" in ftp_deploy.UPLOAD_EXTENSIONS and ".json" in ftp_deploy.WIPE_EXTENSIONS


# ── post-deploy verification checks the real path, self-contained ─────────────

def test_verification_uses_real_product_path():
    content = WF_PATH.read_text()
    assert 'BASE = f"{CDN}/tools/odds-scanner/predictions/WNBA/Pre-Game"' in content, (
        "Post-deploy verification must check the real /tools/odds-scanner/ product path"
    )


def test_verification_reads_self_contained_latest_json():
    content = WF_PATH.read_text()
    start = content.index("Post-deployment verification — custom domain")
    tail = content[start:]
    end = tail.index("\n      - name:", 1) if "\n      - name:" in tail[1:] else len(tail)
    step = tail[:end]
    # latest.json is the payload itself — no pointer follow via payload_path.
    assert "edge_payload = edge_ptr" in step and "dist_payload = dist_ptr" in step, (
        "verification must treat latest.json as the self-contained payload"
    )
    assert "payload_path" not in step, "verification must not follow a payload_path pointer"


# ── generators write a self-contained latest.json ────────────────────────────

def test_generators_write_self_contained_latest_json():
    web = Path("scripts/generate_web_pages.py").read_text()
    dist = Path("scripts/generate_distributions_page.py").read_text()
    assert '(edge_dir / "latest.json").write_text(_edge_payload_str)' in web
    assert '(pmf_dir  / "latest.json").write_text(_pmf_payload_str)' in web
    assert '(out_dir / "latest.json").write_text(_payload_str)' in dist


# ── ftp_deploy uploads recursively to the real path (releases/<id>.json) ──────

class _FakeFTP:
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
        self.stored.append(cmd.split(" ", 1)[1])
        return "226 OK"


def test_ftp_deploy_uploads_nested_release_to_real_path(tmp_path, monkeypatch):
    local_base = tmp_path / "WNBA"
    edge = local_base / "Pre-Game" / "Edge"
    (edge / "releases").mkdir(parents=True)
    (edge / "latest.json").write_text('{"props": [], "total_props": 0}')
    (edge / "index.html").write_text("<html>shell</html>")  # must NOT be uploaded
    (edge / "releases" / "run123.json").write_text('{"total_props": 130}')

    monkeypatch.setattr(ftp_deploy, "LOCAL_BASE", local_base)
    monkeypatch.setenv("FTP_HOST", "h")
    monkeypatch.setenv("FTP_USER", "u")
    monkeypatch.setenv("FTP_PASS", "p")

    fake = _FakeFTP()
    monkeypatch.setattr(ftp_deploy.ftplib, "FTP", lambda *a, **k: fake)

    ftp_deploy.deploy(dirs=["Pre-Game/Edge"], wipe=False)

    assert f"{REAL_BASE}/Pre-Game/Edge/latest.json" in fake.stored
    assert f"{REAL_BASE}/Pre-Game/Edge/releases/run123.json" in fake.stored, (
        "recursive upload must deploy the release payload to the real product path"
    )
    assert not any(s.endswith("index.html") for s in fake.stored), (
        "index.html shells must never be uploaded/overwritten"
    )
