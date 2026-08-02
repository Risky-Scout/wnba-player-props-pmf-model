"""DEPRECATED alias — redirects to the authoritative V6 daily runner.

Use ``scripts/run_wnba_pmf.py`` → ``sharp_v6.predict_slate``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [sys.executable, str(REPO / "scripts" / "run_wnba_pmf.py"), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO)))


if __name__ == "__main__":
    main()
