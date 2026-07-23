#!/usr/bin/env python3
"""Phase-0 scope protection with a SINGLE-USE exception (Foundation Lock + PR 1A).

Foundation Lock must not modify Phase-0 / live-delivery correctness surfaces. Any PR that
changes a protected path must declare a single-use exception in
config/phase0_scope_exception.json that is bound to:

    approved_branch   - the exact branch authorized
    approved_base_sha - the exact base commit authorized
    approved_paths    - an explicit allowlist (NO wildcards)
    expiration        - an explicit YYYY-MM-DD date

The checker FAILS closed when a protected path changed and: there is no exception; the
exception's branch or base SHA differs; a changed protected path is not listed; a wildcard
is present; or the exception is expired. A PR that changes no protected path always passes
(the persisted exception is irrelevant to it), so an old exception cannot block unrelated
PRs - but it also cannot be reused by a later PR that DOES touch protected paths.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXCEPTION_FILE = REPO / "config" / "phase0_scope_exception.json"

PROTECTED_FILES = {
    "src/wnba_props_model/models/market.py",
    "src/wnba_props_model/pipeline/deliver.py",
    "src/wnba_props_model/pipeline/calibrate.py",
    "scripts/build_scored_candidates.py",
    "config/recommendation_policy.yaml",
    "config/stat_registry.json",
}
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


def exception_errors(exc: dict | None, *, current_branch: str, base_sha: str,
                     today: str) -> list[str]:
    """Return fail-closed problems with the exception (empty list == usable)."""
    if exc is None:
        return ["no phase0_scope_exception.json present"]
    errs: list[str] = []
    if exc.get("schema_version") != 1:
        errs.append("exception schema_version must be 1")
    approved_branch = exc.get("approved_branch")
    if approved_branch != current_branch:
        errs.append(f"exception approved_branch={approved_branch!r} != current branch "
                    f"{current_branch!r} (single-use: cannot be reused by another PR)")
    approved_base = exc.get("approved_base_sha")
    if not approved_base or (base_sha and not base_sha.startswith(str(approved_base)[:12])
                             and str(approved_base) != base_sha):
        errs.append(f"exception approved_base_sha={approved_base!r} != base {base_sha!r}")
    paths = exc.get("approved_paths")
    if not isinstance(paths, list) or not paths:
        errs.append("exception approved_paths must be a non-empty list")
    else:
        for p in paths:
            if "*" in str(p) or "?" in str(p) or str(p).endswith("/"):
                errs.append(f"exception approved_paths must not contain wildcards: {p!r}")
    exp = exc.get("expiration")
    if not exp:
        errs.append("exception has no expiration")
    else:
        try:
            if str(today) > str(exp):
                errs.append(f"exception expired on {exp} (today {today})")
        except Exception:
            errs.append(f"exception expiration not comparable: {exp!r}")
    return errs


def evaluate(changed_paths: list[str], exc: dict | None, *, current_branch: str,
             base_sha: str, today: str) -> tuple[bool, list[str]]:
    """Return (ok, messages)."""
    protected_changed = sorted(p for p in changed_paths if is_protected(p))
    if not protected_changed:
        return True, [f"no protected Phase-0 paths modified ({len(changed_paths)} changed)"]
    errs = exception_errors(exc, current_branch=current_branch, base_sha=base_sha, today=today)
    if errs:
        return False, [f"protected paths changed {protected_changed} but exception is unusable:"] + errs
    approved = set(exc.get("approved_paths", []))
    unlisted = sorted(p for p in protected_changed if p not in approved)
    if unlisted:
        return False, [f"protected paths not in the exception allowlist: {unlisted}"]
    return True, [f"protected paths {protected_changed} authorized by single-use exception "
                  f"(branch {current_branch}, base {base_sha[:12]})"]


def _changed_paths(base: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", f"{base}...HEAD"], cwd=str(REPO)).decode()
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _current_branch() -> str:
    env = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    if env:
        return env
    try:
        b = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                    cwd=str(REPO)).decode().strip()
        return b
    except Exception:
        return "unknown"


def _base_sha(base: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", base], cwd=str(REPO)).decode().strip()
    except Exception:
        return ""


def _load_exception() -> dict | None:
    if not EXCEPTION_FILE.exists():
        return None
    return json.loads(EXCEPTION_FILE.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-0 scope protection (single-use exception).")
    ap.add_argument("--base", default="origin/main", help="Base ref to diff against.")
    ap.add_argument("--branch", default=None, help="Override current branch (testing).")
    ap.add_argument("--base-sha", default=None, help="Override resolved base SHA (testing).")
    ap.add_argument("--today", default=None, help="Override today YYYY-MM-DD (testing).")
    args = ap.parse_args()

    try:
        changed = _changed_paths(args.base)
    except subprocess.CalledProcessError as exc:
        print(f"[SCOPE FAIL] could not compute diff against {args.base}: {exc}", file=sys.stderr)
        return 1

    branch = args.branch or _current_branch()
    base_sha = args.base_sha or _base_sha(args.base)
    today = args.today or _dt.date.today().isoformat()

    ok, msgs = evaluate(changed, _load_exception(), current_branch=branch,
                        base_sha=base_sha, today=today)
    if not ok:
        print("[SCOPE FAIL] " + msgs[0], file=sys.stderr)
        for m in msgs[1:]:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print("[SCOPE PASS] " + msgs[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
