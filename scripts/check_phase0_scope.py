#!/usr/bin/env python3
"""Phase-0 scope protection for the Foundation Lock PR.

Foundation Lock must not modify Phase-0 / live-delivery correctness surfaces. This guard
fails when the branch changes any protected path unless an explicit approved exception is
declared in config/phase0_scope_exception.json:

    {"approved_paths": ["<path>", ...], "reason": "...", "approved_by": "..."}

Absence of that file means NO exceptions are approved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXCEPTION_FILE = REPO / "config" / "phase0_scope_exception.json"

# Exact protected files (Phase-0 market/delivery/recommendation surfaces).
PROTECTED_FILES = {
    "src/wnba_props_model/models/market.py",
    "src/wnba_props_model/pipeline/deliver.py",
    "src/wnba_props_model/pipeline/calibrate.py",
    "scripts/build_scored_candidates.py",
    "config/recommendation_policy.yaml",
    "config/stat_registry.json",
}

# Live delivery workflows (by basename prefix / exact name under .github/workflows/).
_LIVE_WORKFLOW_PREFIXES = ("live_", "pregame")
_LIVE_WORKFLOW_EXACT = {
    "daily_pipeline.yml", "deploy_forecast_shell.yml", "post_game_scoring.yml",
}


def is_protected(path: str) -> bool:
    p = path.replace("\\", "/")
    if p in PROTECTED_FILES:
        return True
    if p.startswith(".github/workflows/"):
        base = p.split("/")[-1]
        if base in _LIVE_WORKFLOW_EXACT:
            return True
        if any(base.startswith(pre) for pre in _LIVE_WORKFLOW_PREFIXES):
            return True
    return False


def violations(changed_paths: list[str], approved_paths: list[str]) -> list[str]:
    approved = set(approved_paths)
    return sorted(p for p in changed_paths if is_protected(p) and p not in approved)


def _changed_paths(base: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...HEAD"], cwd=str(REPO)).decode()
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _load_approved() -> list[str]:
    if not EXCEPTION_FILE.exists():
        return []
    data = json.loads(EXCEPTION_FILE.read_text())
    return list(data.get("approved_paths", []))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-0 scope protection for Foundation Lock.")
    ap.add_argument("--base", default="origin/main", help="Base ref to diff against.")
    args = ap.parse_args()

    try:
        changed = _changed_paths(args.base)
    except subprocess.CalledProcessError as exc:
        print(f"[SCOPE FAIL] could not compute diff against {args.base}: {exc}", file=sys.stderr)
        return 1

    approved = _load_approved()
    bad = violations(changed, approved)
    if bad:
        print("[SCOPE FAIL] Foundation Lock modifies protected Phase-0 / live-delivery paths "
              "without an approved exception:", file=sys.stderr)
        for p in bad:
            print(f"  - {p}", file=sys.stderr)
        print("Add an approved config/phase0_scope_exception.json to override.", file=sys.stderr)
        return 1
    print(f"[SCOPE PASS] no protected Phase-0 paths modified vs {args.base} "
          f"({len(changed)} changed file(s); {len(approved)} approved exception(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
