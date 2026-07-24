"""Shared helpers for the data-durability registry (fetch/publish/verify).

The registry (config/data_registry.json) is the single source of truth for where
every large/gitignored dataset lives (a GitHub Release asset) and its sha256, so
any clone — laptop, cloud VM, or server — can pull and verify byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "config" / "data_registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def save_registry(reg: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    """Full sha256 hex digest of a file, streamed (safe for large parquets)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_gh() -> str:
    """Return the gh executable path or raise a clear error."""
    gh = shutil.which("gh")
    if not gh:
        raise RuntimeError(
            "GitHub CLI `gh` not found. Install it (brew install gh) and run "
            "`gh auth login` so datasets can be pushed to / pulled from Releases."
        )
    return gh


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)
