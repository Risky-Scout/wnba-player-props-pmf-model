"""Deploy WNBA prediction pages to sportsodds.wizardofodds.com via FTP.

Uploads every file in the following directories (preserving structure):
  Pre-Game/Edge/          → index.html + latest.json + date JSONs
  Pre-Game/Distributions/ → index.html + latest.json
  Pre-Game/PMF-Distributions/ → index.html + latest.json
  Pre-Game/Pricer/        → index.html
  In-Play/Edges/          → index.html + latest.json
  In-Play/Pricer/         → index.html

All files are uploaded to /tools/odds-scanner/predictions/WNBA/<same path>
on the server.

Environment variables:
    FTP_HOST   server hostname or IP
    FTP_USER   FTP username
    FTP_PASS   FTP password
"""
from __future__ import annotations

import ftplib
import io
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Remote base where all WNBA files live on the server
REMOTE_BASE = "/tools/odds-scanner/predictions/WNBA"

# Local base directory containing all the pages
LOCAL_BASE = REPO_ROOT / "tools" / "odds-scanner" / "predictions" / "WNBA"

# Which subdirectories to deploy (relative to LOCAL_BASE / REMOTE_BASE)
DEPLOY_DIRS = [
    "Pre-Game/Edge",
    "Pre-Game/Distributions",
    "Pre-Game/PMF-Distributions",
    "Pre-Game/Pricer",
    "In-Play/Edges",
    "In-Play/Pricer",
]

# File extensions to upload (skip .parquet, .pkl, etc.)
UPLOAD_EXTENSIONS = {".html", ".json", ".js", ".css", ".txt", ".ico", ".png", ".svg"}


def _ensure_remote_dir(ftp: ftplib.FTP, path: str) -> None:
    """Create remote directory path recursively, chmod 755 each level."""
    parts = [p for p in path.split("/") if p]
    current = ""
    for part in parts:
        current += f"/{part}"
        try:
            ftp.cwd(current)
        except ftplib.error_perm:
            try:
                ftp.mkd(current)
                print(f"  Created remote dir: {current}")
            except ftplib.error_perm as e:
                if "exists" not in str(e).lower() and "exist" not in str(e).lower():
                    raise
        try:
            ftp.sendcmd(f"SITE CHMOD 755 {current}")
        except Exception:
            pass


def _upload_file(ftp: ftplib.FTP, local_path: Path, remote_path: str) -> None:
    """Upload a single file, deleting any existing version first."""
    content = local_path.read_bytes()
    try:
        ftp.delete(remote_path)
    except Exception:
        pass
    ftp.storbinary(f"STOR {remote_path}", io.BytesIO(content))
    try:
        ftp.sendcmd(f"SITE CHMOD 644 {remote_path}")
    except Exception:
        pass
    print(f"  ✓ {remote_path} ({len(content) / 1024:.1f} KB)")


def deploy() -> None:
    host = os.environ.get("FTP_HOST", "").strip()
    user = os.environ.get("FTP_USER", "").strip()
    password = os.environ.get("FTP_PASS", "").strip()

    if not host or not user or not password:
        print("ERROR: FTP_HOST, FTP_USER, and FTP_PASS must be set in .env or environment")
        sys.exit(1)

    print(f"Connecting to {host}…")
    with ftplib.FTP(host, timeout=60) as ftp:
        ftp.login(user, password)
        print(f"  Connected: {ftp.getwelcome()[:80]}")

        uploaded = 0
        skipped = 0

        for subdir in DEPLOY_DIRS:
            local_dir = LOCAL_BASE / Path(subdir)
            remote_dir = f"{REMOTE_BASE}/{subdir}"

            if not local_dir.exists():
                print(f"  SKIP (not found locally): {local_dir}")
                skipped += 1
                continue

            _ensure_remote_dir(ftp, remote_dir)

            for local_file in sorted(local_dir.iterdir()):
                if not local_file.is_file():
                    continue
                if local_file.suffix.lower() not in UPLOAD_EXTENSIONS:
                    continue
                remote_file = f"{remote_dir}/{local_file.name}"
                _upload_file(ftp, local_file, remote_file)
                uploaded += 1

        print(f"\nDeploy complete: {uploaded} files uploaded, {skipped} dirs skipped.")


if __name__ == "__main__":
    deploy()
