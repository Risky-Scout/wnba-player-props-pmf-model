#!/usr/bin/env python3
"""Verify (or update) the Foundation Lock manifest.

The Foundation Lock manifest (config/foundation_lock_v1.json) pins every completed
foundational component by SHA-256 and records its status, invariants, required tests,
required CI job, evidence, limitations, and promotion-eligibility.

Verification (default) FAILS on:
  * a missing in-repo path;
  * a hash mismatch (a changed artifact without an explicit manifest update);
  * a missing required test file;
  * a manifest/schema mismatch (missing keys or bad status);
  * a component labeled promotion-eligible while its limitations prohibit that status.

Modes:
  (default)        verify the manifest against the working tree; exit 1 on any failure.
  --update         recompute in-repo path hashes and rewrite the manifest (maintainer
                   action, like refreshing a lockfile). Never run in CI.
  --write-report   regenerate artifacts/foundation_lock/FOUNDATION_LOCK_REPORT.md.

Data-artifact paths (availability="data_artifact_untracked", e.g. the git-ignored feature
parquet) are verified when present and reported as DEFERRED (explicitly, never silently)
when absent from the checkout. All in-repo locked artifacts and required tests are always
checked and can never be silently skipped.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "config" / "foundation_lock_v1.json"
REPORT_PATH = REPO / "artifacts" / "foundation_lock" / "FOUNDATION_LOCK_REPORT.md"

ALLOWED_STATUS = {"locked", "exploratory_locked", "not_landed"}
REQUIRED_ITEM_KEYS = {
    "id", "title", "status", "promotion_eligible", "prohibits_promotion",
    "paths", "invariants", "required_tests", "required_ci_job",
    "evidence_artifacts", "limitations",
}
REQUIRED_TOP_KEYS = {"schema_version", "foundation_version", "generated_from_commit",
                     "created_utc", "items"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generated_from_commit() -> str:
    """Base commit the evidence was generated from (merge-base with origin/main).

    Never the commit that will contain this manifest (no self-referential provenance)."""
    for args in (["git", "merge-base", "origin/main", "HEAD"],
                 ["git", "merge-base", "origin/HEAD", "HEAD"]):
        try:
            return subprocess.check_output(
                args, cwd=str(REPO), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return "unknown"


def verify(manifest: dict) -> tuple[list[str], list[str]]:
    """Return (failures, deferrals)."""
    failures: list[str] = []
    deferrals: list[str] = []

    if manifest.get("schema_version") != 1:
        failures.append("top: schema_version must be 1")
    missing_top = REQUIRED_TOP_KEYS - set(manifest)
    if missing_top:
        failures.append(f"top: missing keys {sorted(missing_top)}")

    for item in manifest.get("items", []):
        iid = item.get("id", "<no-id>")
        missing = REQUIRED_ITEM_KEYS - set(item)
        if missing:
            failures.append(f"{iid}: missing keys {sorted(missing)}")
            continue
        if item["status"] not in ALLOWED_STATUS:
            failures.append(f"{iid}: bad status {item['status']!r}")
        # Promotion-eligibility rule.
        if item["prohibits_promotion"] and item["promotion_eligible"]:
            failures.append(f"{iid}: labeled promotion_eligible while limitations prohibit it")
        # Paths + hashes.
        for entry in item["paths"]:
            rel = entry["path"]
            p = REPO / rel
            avail = entry.get("availability", "in_repo")
            if not p.exists():
                if avail == "data_artifact_untracked":
                    deferrals.append(f"{iid}: {rel} (data artifact) not present -> DEFERRED")
                    continue
                failures.append(f"{iid}: missing in-repo path {rel}")
                continue
            actual = _sha256(p)
            recorded = entry.get("sha256")
            if recorded is None:
                failures.append(f"{iid}: no recorded sha256 for {rel}")
            elif actual != recorded:
                failures.append(f"{iid}: hash mismatch {rel} "
                                f"(recorded {recorded[:12]}, actual {actual[:12]})")
        # Required tests must exist.
        for t in item["required_tests"]:
            if not (REPO / t).exists():
                failures.append(f"{iid}: missing required test {t}")
    return failures, deferrals


def authorization_errors(auth: dict | None, *, current_branch: str, base_sha: str,
                         today: str) -> list[str]:
    """Fail-closed validation of a Foundation Lock change authorization."""
    if auth is None:
        return ["no change-authorization file provided"]
    errs: list[str] = []
    if auth.get("schema_version") != 1:
        errs.append("authorization schema_version must be 1")
    if auth.get("authorized_branch") != current_branch:
        errs.append(f"authorized_branch={auth.get('authorized_branch')!r} != current branch "
                    f"{current_branch!r}")
    ab = auth.get("authorized_base_sha")
    if not ab or (base_sha and str(ab) != base_sha and not base_sha.startswith(str(ab)[:12])):
        errs.append(f"authorized_base_sha={ab!r} != base {base_sha!r}")
    if not auth.get("single_use"):
        errs.append("authorization must be single_use")
    paths = auth.get("allowed_locked_paths")
    if not isinstance(paths, list) or not paths:
        errs.append("allowed_locked_paths must be a non-empty list")
    else:
        for p in paths:
            if "*" in str(p) or "?" in str(p) or str(p).endswith("/"):
                errs.append(f"allowed_locked_paths must not contain wildcards: {p!r}")
    exp = auth.get("expiration")
    if exp and str(today) > str(exp):
        errs.append(f"authorization expired on {exp} (today {today})")
    return errs


def authorized_update(manifest: dict, auth: dict, *, manifest_path: Path,
                      auth_rel: str, current_branch: str, base_sha: str, today: str):
    """Controlled, single-use re-pin: only re-pin hashes for explicitly authorized locked
    paths; refuse any unauthorized (broad) change; increment the lock revision and record the
    previous manifest hash. Never a blanket re-pin."""
    errs = authorization_errors(auth, current_branch=current_branch, base_sha=base_sha, today=today)
    if errs:
        raise SystemExit("[update] REFUSED - invalid authorization:\n  - " + "\n  - ".join(errs))
    allowed = set(auth["allowed_locked_paths"])
    prev_manifest_sha = _sha256(manifest_path)
    changed: list[tuple[str, str | None, str]] = []
    for item in manifest.get("items", []):
        for entry in item.get("paths", []):
            rel = entry["path"]
            p = REPO / rel
            avail = entry.get("availability", "in_repo")
            if not p.exists():
                if avail == "data_artifact_untracked":
                    continue
                raise SystemExit(f"[update] REFUSED - missing in-repo path: {rel}")
            new = _sha256(p)
            old = entry.get("sha256")
            if new != old:
                if rel not in allowed:
                    raise SystemExit(
                        f"[update] REFUSED - unauthorized change to locked path {rel} "
                        "(not in allowed_locked_paths); broad/automatic re-pinning is forbidden")
                entry["sha256"] = new
                changed.append((rel, old, new))
    manifest["previous_manifest_sha256"] = prev_manifest_sha
    manifest["previous_lock_commit"] = base_sha
    manifest["lock_revision"] = int(manifest.get("lock_revision", 1)) + 1
    manifest["change_authorization"] = auth_rel
    manifest["generated_from_commit"] = _generated_from_commit()
    manifest["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return manifest, changed, prev_manifest_sha


def overall_status(failures: list[str], deferrals: list[str], manifest: dict) -> str:
    if failures:
        return "FAIL"
    has_not_landed = any(it.get("status") == "not_landed" for it in manifest.get("items", []))
    if deferrals or has_not_landed:
        return "PASS_WITH_DECLARED_DEFERRALS"
    return "PASS"


def write_report(manifest: dict, failures: list[str], deferrals: list[str]) -> None:
    status = overall_status(failures, deferrals, manifest)
    lines = [
        "# Foundation Lock Report",
        "",
        f"- Foundation version: **{manifest.get('foundation_version')}**",
        f"- Generated from commit: `{manifest.get('generated_from_commit')}` "
        "(base commit; NOT the commit containing this manifest)",
        f"- Generated (manifest): {manifest.get('created_utc')}",
        f"- Overall: **{status}**"
        f"  (failures: {len(failures)}, deferrals: {len(deferrals)})",
        "",
        "This report classifies each locked component by what it does and does not prove.",
        "No component below is promotion-eligible; the market edge is NOT proven by any of them.",
        "",
        "| # | Component | Status | Promotion-eligible | Paths | Required tests |",
        "|---|---|---|---|---:|---:|",
    ]
    for i, item in enumerate(manifest.get("items", []), 1):
        lines.append(
            f"| {i} | {item['title']} | `{item['status']}` | "
            f"{'yes' if item['promotion_eligible'] else 'NO'} | "
            f"{len(item['paths'])} | {len(item['required_tests'])} |")
    lines += ["", "## Per-component limitations", ""]
    for item in manifest.get("items", []):
        lines.append(f"### {item['title']} (`{item['id']}`)")
        lines.append(f"- Status: `{item['status']}`; promotion-eligible: "
                     f"{'yes' if item['promotion_eligible'] else 'NO'}; required CI job: "
                     f"`{item['required_ci_job']}`")
        for lim in item["limitations"]:
            lines.append(f"- Limitation: {lim}")
        lines.append("")
    if deferrals:
        lines += ["## Deferred (declared data artifacts absent from this checkout)", ""]
        lines += [f"- {d}" for d in deferrals] + [""]
    if failures:
        lines += ["## FAILURES", ""] + [f"- {f}" for f in failures] + [""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _current_branch() -> str:
    import os
    env = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       cwd=str(REPO)).decode().strip()
    except Exception:
        return "unknown"


def _rev(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", ref], cwd=str(REPO)).decode().strip()
    except Exception:
        return ""


CHANGELOG_PATH = REPO / "artifacts" / "foundation_lock" / "FOUNDATION_LOCK_CHANGELOG.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify/update the Foundation Lock manifest.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--update", action="store_true",
                    help="Controlled re-pin. REQUIRES --authorization; only re-pins authorized "
                         "locked paths (broad re-pinning is refused).")
    ap.add_argument("--authorization", default=None, help="Path to a change-authorization JSON.")
    ap.add_argument("--branch", default=None, help="Override current branch (testing).")
    ap.add_argument("--base", default="origin/main", help="Base ref for merge-base/base SHA.")
    ap.add_argument("--base-sha", default=None, help="Override resolved base SHA (testing).")
    ap.add_argument("--today", default=None, help="Override today YYYY-MM-DD (testing).")
    ap.add_argument("--write-report", action="store_true", help="Regenerate the lock report.")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    manifest = json.loads(mpath.read_text())

    if args.update:
        if not args.authorization:
            print("[update] REFUSED - broad/unguarded re-pin is forbidden; pass --authorization "
                  "<config/foundation_lock_change_*.json>", file=sys.stderr)
            return 1
        auth_path = Path(args.authorization)
        auth = json.loads(auth_path.read_text())
        branch = args.branch or _current_branch()
        base_sha = args.base_sha or _rev(args.base)
        import datetime as _d
        today = args.today or _d.date.today().isoformat()
        auth_rel = str(auth_path.relative_to(REPO)) if auth_path.is_absolute() else str(auth_path)
        manifest, changed, prev_sha = authorized_update(
            manifest, auth, manifest_path=mpath, auth_rel=auth_rel,
            current_branch=branch, base_sha=base_sha, today=today)
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
        # Append an immutable changelog entry.
        CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "lock_revision": manifest["lock_revision"],
            "authorization_id": auth.get("authorization_id"),
            "authorized_pr": auth.get("authorized_pr"),
            "generated_from_commit": manifest["generated_from_commit"],
            "previous_lock_commit": manifest["previous_lock_commit"],
            "previous_manifest_sha256": prev_sha,
            "repinned_paths": [{"path": r, "old_sha256": o, "new_sha256": n} for r, o, n in changed],
            "created_utc": manifest["created_utc"],
        }
        with open(CHANGELOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[foundation-lock] authorized re-pin: revision {manifest['lock_revision']}, "
              f"{len(changed)} path(s) re-pinned; changelog -> {CHANGELOG_PATH}")
        for r, o, n in changed:
            print(f"  - {r}: {str(o)[:12]} -> {n[:12]}")
        return 0

    failures, deferrals = verify(manifest)

    if args.write_report:
        write_report(manifest, failures, deferrals)
        print(f"[foundation-lock] wrote report -> {REPORT_PATH}")

    for d in deferrals:
        print(f"[DEFERRED] {d}")
    if failures:
        print(f"\n[FOUNDATION LOCK FAIL] {len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    status = overall_status(failures, deferrals, manifest)
    print(f"\n[FOUNDATION LOCK {status}] {len(manifest.get('items', []))} components verified; "
          f"{len(deferrals)} deferred data artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
