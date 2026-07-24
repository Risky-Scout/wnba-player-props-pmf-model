"""A9 - scan text files for hidden/bidirectional Unicode control characters.

GitHub (and terminals) can render code differently than it executes when bidi/invisible
controls are present. This scanner FAILS CLOSED when any of the following appear in a tracked
text file:

    U+202A..U+202E  (LRE, RLE, PDF, LRO, RLO)
    U+2066..U+2069  (LRI, RLI, FSI, PDI)
    U+200E, U+200F  (LRM, RLM)
    U+061C          (Arabic Letter Mark)

Usage:
    python3 scripts/scan_bidi_controls.py                 # scan git-tracked text files
    python3 scripts/scan_bidi_controls.py --base origin/main   # scan only changed files
    python3 scripts/scan_bidi_controls.py path1 path2 ...      # scan explicit paths
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FORBIDDEN = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F, 0x061C}
_NAMES = {
    0x202A: "LRE", 0x202B: "RLE", 0x202C: "PDF", 0x202D: "LRO", 0x202E: "RLO",
    0x2066: "LRI", 0x2067: "RLI", 0x2068: "FSI", 0x2069: "PDI",
    0x200E: "LRM", 0x200F: "RLM", 0x061C: "ALM",
}
_SKIP_SUFFIXES = {".parquet", ".pkl", ".joblib", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
                  ".zip", ".gz", ".ico", ".woff", ".woff2", ".ttf", ".webp", ".pyc"}


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=str(REPO)).decode()
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _changed_files(base: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...HEAD"], cwd=str(REPO)).decode()
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def scan_file(path: Path) -> list[tuple[int, int, str]]:
    if path.suffix.lower() in _SKIP_SUFFIXES or not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []                                            # binary/unreadable -> skip
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for ch in line:
            cp = ord(ch)
            if cp in FORBIDDEN:
                hits.append((lineno, cp, _NAMES.get(cp, f"U+{cp:04X}")))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan for hidden/bidirectional Unicode controls.")
    ap.add_argument("paths", nargs="*", help="Explicit files to scan (default: git-tracked).")
    ap.add_argument("--base", default=None, help="Scan only files changed vs this base ref.")
    args = ap.parse_args()

    if args.paths:
        files = args.paths
    elif args.base:
        files = _changed_files(args.base)
    else:
        files = _tracked_files()

    problems = 0
    for rel in files:
        p = (REPO / rel) if not Path(rel).is_absolute() else Path(rel)
        for lineno, cp, name in scan_file(p):
            problems += 1
            print(f"[BIDI] {rel}:{lineno}: forbidden control U+{cp:04X} ({name})", file=sys.stderr)
    if problems:
        print(f"[BIDI SCAN FAIL] {problems} hidden/bidirectional control(s) found.", file=sys.stderr)
        return 1
    print(f"[BIDI SCAN PASS] {len(files)} file(s) scanned; no hidden/bidirectional controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
