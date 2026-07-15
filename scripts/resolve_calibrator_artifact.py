"""Artifact-level calibrator resolver for pregame_initial.yml.

Queries ALL unexpired ``calibrators-latest`` artifacts newest-first and selects
the first one whose ``artifact_manifest_calibrator.json`` satisfies every
required field.  Selection is by manifest validity, NOT by overall workflow
conclusion — a weekly_calibration run that succeeds through the cal-gate but
later fails in an unrelated step (e.g. auto-trigger) still produces a valid
artifact that this resolver will correctly select.

Exit codes
----------
0  Compatible artifact found and downloaded to --cal-dir.
1  No compatible artifact found (all rejections printed to stderr).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def gh_download_zip(repo: str, artifact_id: int) -> bytes | None:
    """Download artifact zip via `gh api`.  Returns raw bytes or None on failure."""
    r = subprocess.run(
        ["gh", "api",
         f"repos/{repo}/actions/artifacts/{artifact_id}/zip",
         "--header", "Accept: application/vnd.github+json"],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        return None
    return r.stdout


def list_artifacts(repo: str, name: str, per_page: int = 30) -> list[dict]:
    """Return unexpired artifacts with the given name, newest-first."""
    r = subprocess.run(
        ["gh", "api",
         f"repos/{repo}/actions/artifacts?name={name}&per_page={per_page}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    try:
        data = json.loads(r.stdout)
    except Exception:
        return []
    arts = [a for a in data.get("artifacts", []) if not a.get("expired", True)]
    arts.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return arts


def validate_manifest(manifest: dict, producer_run_id: str, producer_head_sha: str) -> list[str]:
    """Return a list of failing check descriptions (empty = all pass)."""
    failures: list[str] = []

    if manifest.get("artifact_type") != "calibrator":
        failures.append(f"artifact_type={manifest.get('artifact_type')!r} (expected 'calibrator')")

    if manifest.get("gate_status") != "PASS":
        failures.append(f"gate_status={manifest.get('gate_status')!r} (expected 'PASS')")

    manifest_run = str(manifest.get("source_run_id", ""))
    if manifest_run != producer_run_id:
        failures.append(
            f"source_run_id={manifest_run!r} != producer_run_id={producer_run_id!r}"
        )

    manifest_sha = str(manifest.get("source_commit", ""))
    # Accept prefix match (short SHA) in either direction
    short = min(len(manifest_sha), len(producer_head_sha), 8)
    if short == 0 or manifest_sha[:short] != producer_head_sha[:short]:
        failures.append(
            f"source_commit={manifest_sha[:12]!r} does not match producer sha={producer_head_sha[:12]!r}"
        )

    for field in ("calibration_cutoff", "feature_manifest_hash", "config_hash"):
        if not str(manifest.get(field, "")).strip():
            failures.append(f"{field} is blank")

    return failures


def main(
    repo: str,
    cal_dir: str,
    github_env: str,
    artifact_name: str = "calibrators-latest",
) -> None:
    artifacts = list_artifacts(repo, artifact_name)
    if not artifacts:
        print(f"[FATAL] No unexpired {artifact_name!r} artifacts found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(artifacts)} unexpired {artifact_name!r} artifact(s) — probing newest-first")

    rejections: list[str] = []
    selected_aid: int | None = None
    selected_run_id: str = ""

    for art in artifacts:
        aid         = art["id"]
        run_id      = str((art.get("workflow_run") or {}).get("id", ""))
        head_sha    = str((art.get("workflow_run") or {}).get("head_sha", ""))
        created     = art.get("created_at", "")
        print(
            f"\nProbing artifact {aid}"
            f" (created={created} run={run_id} sha={head_sha[:8]})"
        )

        raw_zip = gh_download_zip(repo, aid)
        if not raw_zip:
            reason = "download failed"
            print(f"  REJECTED: {reason}")
            rejections.append(f"artifact {aid}: {reason}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "cal.zip"
            zip_path.write_bytes(raw_zip)
            extract_dir = Path(tmpdir) / "cal"
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_dir)
            except Exception as exc:
                reason = f"zip extract failed: {exc}"
                print(f"  REJECTED: {reason}")
                rejections.append(f"artifact {aid}: {reason}")
                continue

            mpath = extract_dir / "artifact_manifest_calibrator.json"
            if not mpath.exists():
                reason = "artifact_manifest_calibrator.json absent"
                print(f"  REJECTED: {reason}")
                rejections.append(f"artifact {aid}: {reason}")
                continue

            try:
                manifest = json.loads(mpath.read_text())
            except Exception as exc:
                reason = f"manifest unreadable: {exc}"
                print(f"  REJECTED: {reason}")
                rejections.append(f"artifact {aid}: {reason}")
                continue

            failures = validate_manifest(manifest, run_id, head_sha)
            if failures:
                reason = "; ".join(failures)
                print(f"  REJECTED: {reason}")
                rejections.append(f"artifact {aid}: {reason}")
                continue

            # All checks passed — copy to calibration dir
            dest = Path(cal_dir)
            dest.mkdir(parents=True, exist_ok=True)
            for f in extract_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
            print(f"  SELECTED artifact {aid} from run {run_id}")
            selected_aid = aid
            selected_run_id = run_id
            break

    if selected_aid is None:
        print(
            f"\n[FATAL] No compatible {artifact_name!r} artifact found.",
            file=sys.stderr,
        )
        print(
            f"[FATAL] {len(rejections)} artifact(s) inspected; all rejected:",
            file=sys.stderr,
        )
        for r in rejections:
            print(f"  - {r}", file=sys.stderr)
        sys.exit(1)

    # Export SOURCE_CALIBRATORS_RUN_ID to GITHUB_ENV
    if github_env:
        with open(github_env, "a") as fh:
            fh.write(f"SOURCE_CALIBRATORS_RUN_ID={selected_run_id}\n")

    print(
        f"\nCalibrators downloaded OK"
        f" (artifact_id={selected_aid}, source_run={selected_run_id})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",        required=True)
    parser.add_argument("--cal-dir",     default="artifacts/models/calibration")
    parser.add_argument("--github-env",  default=os.environ.get("GITHUB_ENV", ""))
    parser.add_argument("--artifact-name", default="calibrators-latest")
    args = parser.parse_args()

    main(
        repo=args.repo,
        cal_dir=args.cal_dir,
        github_env=args.github_env,
        artifact_name=args.artifact_name,
    )
