"""Deploy WNBA prediction pages to sportsodds.wizardofodds.com via FTP.

Uploads every file in the following directories (preserving structure):
  Pre-Game/Edge/          → index.html + latest.json + date JSONs
  Pre-Game/Distributions/ → index.html + latest.json
  Pre-Game/PMF-Distributions/ → index.html + latest.json
  Pre-Game/Pricer/        → index.html
  In-Play/Edges/          → index.html + latest.json
  In-Play/Pricer/         → index.html

Data files are uploaded to /tools/odds-scanner/predictions/WNBA/<same path> on
the server, which nginx serves at
https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/<same path>.
Only data files (.json) and assets are uploaded — existing index.html page
shells are preserved (never overwritten or wiped).

When --wipe is set (default True for pre-game dirs), existing HTML/JSON files
in the three pre-game player-facing directories are deleted before uploading
fresh ones. This prevents stale player pages (e.g. Kahleah Copper on a slate
she's not playing on) from persisting on the site.

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

# Remote base where all WNBA files live on the server.
# nginx serves the live product pages at
#   https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/WNBA/...
# so files must be deployed to that directory (the FTP login root maps to the
# nginx web root). Deploying to /WNBA/ (a previous regression) published to a
# path the product pages do NOT read, leaving the live Edge/Distributions pages
# stale. Keep this in sync with the URLs the page shells are served from.
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

# Pre-game dirs whose stale player files are wiped before uploading fresh ones.
# Wipe prevents players not on today's slate from persisting on the live site.
WIPE_DIRS = {
    "Pre-Game/Edge",
    "Pre-Game/Distributions",
    "Pre-Game/PMF-Distributions",
}

# File extensions to upload (skip .parquet, .pkl, etc.)
# NOTE: .html is intentionally EXCLUDED. The page shells (index.html) already
# exist on the server and must not be regenerated/overwritten — the pipeline's
# job is to refresh the DATA the shells consume (latest.json / <date>.json /
# releases/*.json), not the shells themselves.
UPLOAD_EXTENSIONS = {".json", ".js", ".css", ".txt", ".ico", ".png", ".svg"}

# Extensions targeted by the remote wipe. Only stale data files are wiped;
# .html shells are preserved so the existing page shells are never deleted.
WIPE_EXTENSIONS = {".json"}


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


def _wipe_remote_dir(ftp: ftplib.FTP, remote_dir: str) -> int:
    """Delete all HTML/JSON files in a remote directory (non-recursive).

    Only targets files with WIPE_EXTENSIONS so subdirectories and assets
    (images, CSS, JS) are preserved. Returns the count of deleted files.
    """
    deleted = 0
    try:
        ftp.cwd(remote_dir)
    except ftplib.error_perm:
        print(f"  WIPE SKIP (dir not found remotely): {remote_dir}")
        return 0

    # Collect filenames via NLST (name list only, no stat info)
    try:
        entries = ftp.nlst()
    except ftplib.error_perm:
        entries = []

    for entry in entries:
        # Strip any leading path that NLST might include
        name = entry.split("/")[-1]
        if not name or name.startswith("."):
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in WIPE_EXTENSIONS:
            continue
        remote_file = f"{remote_dir}/{name}"
        try:
            ftp.delete(remote_file)
            print(f"  ✗ wiped {remote_file}")
            deleted += 1
        except Exception as exc:
            print(f"  WIPE WARN: could not delete {remote_file}: {exc}")

    return deleted


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


def deploy(dirs: list[str] | None = None, wipe: bool = True) -> None:
    """Deploy local prediction pages to the FTP server.

    Parameters
    ----------
    dirs:
        Subdirectories to deploy (relative to LOCAL_BASE). Defaults to all DEPLOY_DIRS.
    wipe:
        When True (default), deletes all existing HTML/JSON files in WIPE_DIRS on the
        remote server before uploading fresh ones. This prevents stale player pages for
        players not on today's slate from persisting on the live site.
    """
    host = os.environ.get("FTP_HOST", "").strip()
    user = os.environ.get("FTP_USER", "").strip()
    password = os.environ.get("FTP_PASS", "").strip()

    if not host or not user or not password:
        print("ERROR: FTP_HOST, FTP_USER, and FTP_PASS must be set in .env or environment")
        sys.exit(1)

    deploy_dirs = dirs if dirs is not None else DEPLOY_DIRS

    print(f"Connecting to {host}…")
    print(f"Deploying dirs: {deploy_dirs}")
    print(f"Wipe mode: {'ON' if wipe else 'OFF'}")
    with ftplib.FTP(host, timeout=60) as ftp:
        ftp.login(user, password)
        print(f"  Connected: {ftp.getwelcome()[:80]}")

        wiped = 0
        uploaded = 0
        skipped = 0

        # Step 1: Wipe stale player files from pre-game dirs before uploading
        if wipe:
            print("\n--- Wiping stale remote files ---")
            for subdir in deploy_dirs:
                if subdir not in WIPE_DIRS:
                    continue
                remote_dir = f"{REMOTE_BASE}/{subdir}"
                n = _wipe_remote_dir(ftp, remote_dir)
                wiped += n
            print(f"Wipe complete: {wiped} stale file(s) deleted.\n")

        # Step 2: Upload all local files
        print("--- Uploading fresh files ---")
        for subdir in deploy_dirs:
            local_dir = LOCAL_BASE / Path(subdir)
            remote_dir = f"{REMOTE_BASE}/{subdir}"

            if not local_dir.exists():
                print(f"  SKIP (not found locally): {local_dir}")
                skipped += 1
                continue

            _ensure_remote_dir(ftp, remote_dir)

            # Recurse so nested assets — notably the immutable release payloads at
            # releases/<release_id>.json referenced by latest.json's payload_path and
            # by the page JS — are deployed too. nginx serves them at the same relative
            # path the pointer references. Without this, payload_path 404s on the live
            # custom domain and the pages/post-deploy verification cannot load the release.
            ensured_remote_dirs: set[str] = {remote_dir}
            for local_file in sorted(local_dir.rglob("*")):
                if not local_file.is_file():
                    continue
                if local_file.suffix.lower() not in UPLOAD_EXTENSIONS:
                    continue
                rel = local_file.relative_to(local_dir)
                remote_file = f"{remote_dir}/{rel.as_posix()}"
                if rel.parent != Path("."):
                    parent_remote = f"{remote_dir}/{rel.parent.as_posix()}"
                    if parent_remote not in ensured_remote_dirs:
                        _ensure_remote_dir(ftp, parent_remote)
                        ensured_remote_dirs.add(parent_remote)
                _upload_file(ftp, local_file, remote_file)
                uploaded += 1

        print(f"\nDeploy complete: {wiped} wiped, {uploaded} files uploaded, {skipped} dirs skipped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTP deploy WNBA prediction pages")
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help="Subdirs to deploy (default: all). E.g. --dirs In-Play/Edges In-Play/Pricer",
    )
    parser.add_argument(
        "--wipe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wipe existing HTML/JSON files in pre-game dirs before uploading (default: on). "
            "Use --no-wipe to skip the wipe step (e.g. for in-play-only deploys)."
        ),
    )
    args = parser.parse_args()
    deploy(dirs=args.dirs, wipe=args.wipe)
