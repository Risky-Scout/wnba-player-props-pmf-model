"""Fetch registered datasets from GitHub Releases and verify their sha256.

Never-lose-a-file, receiving end. Reads config/data_registry.json, downloads each
dataset's release asset (skipping ones already present AND hash-correct), and fails
loudly on any checksum mismatch (corruption / wrong version).

Usage:
    python3 scripts/fetch_data.py --all
    python3 scripts/fetch_data.py wnba_tracking wnba_hustle
    python3 scripts/fetch_data.py --all --check   # verify-only, download nothing
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_registry_lib import (  # noqa: E402
    ROOT,
    load_registry,
    require_gh,
    run,
    sha256_file,
)

app = typer.Typer(add_completion=False)


class PrivateAuthError(RuntimeError):
    """Private asset requested but no authentication token is available (fail closed)."""


def _resolve_repo(value: str) -> str:
    """Resolve a repository field, expanding a ``${ENV_VAR}`` reference from the environment.

    The private data repository name is a protected secret and is NEVER persisted in the
    committed registry; entries store the literal ``${DATA_ASSET_REPOSITORY}`` and it is
    expanded here at runtime."""
    v = str(value).strip()
    if v.startswith("${") and v.endswith("}"):
        resolved = os.environ.get(v[2:-1])
        if not resolved:
            raise PrivateAuthError(f"registry repository refers to {v} but it is not set")
        return resolved
    return v


def _auth_env(private: bool) -> "dict | None":
    """Return a child-process env carrying a token for PRIVATE assets, or None for public.

    Priority: PRIVATE_DATA_WRITER_TOKEN, PRIVATE_DATA_GH_TOKEN, then GH_TOKEN; fail closed for
    private assets when none is set. The token is placed ONLY in the child env (never in argv,
    URLs, logs, or messages)."""
    if not private:
        return None
    token = (os.environ.get("PRIVATE_DATA_WRITER_TOKEN")
             or os.environ.get("PRIVATE_DATA_GH_TOKEN")
             or os.environ.get("GH_TOKEN"))
    if not token:
        raise PrivateAuthError(
            "private data asset requires PRIVATE_DATA_GH_TOKEN or GH_TOKEN in the environment")
    env = dict(os.environ)
    env["GH_TOKEN"] = token  # gh reads GH_TOKEN from env; never passed as an argument
    return env


def _download(repo: str, tag: str, asset: str, dest: Path, *, private: bool = False) -> None:
    gh = require_gh()
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = _auth_env(private)
    res = run([gh, "release", "download", tag, "--pattern", asset,
               "--dir", str(dest.parent), "--clobber", "--repo", repo], env=env)
    if res.returncode != 0:
        # Never surface token material; gh errors reference repo/tag/asset only.
        raise RuntimeError(f"gh release download failed for {repo} {tag}/{asset} "
                           f"(rc={res.returncode})")


@app.command()
def main(
    names: list[str] = typer.Argument(None, help="Dataset name(s) to fetch. Omit with --all."),
    all_: bool = typer.Option(False, "--all", help="Fetch every dataset in the registry."),
    check: bool = typer.Option(False, "--check", help="Verify only; do not download."),
    verify: bool = typer.Option(False, "--verify",
                                help="Alias for --check --all (verify every dataset, no download)."),
) -> None:
    reg = load_registry()
    default_repo = reg["repo"]
    datasets = reg["datasets"]
    if verify:
        check = True
        all_ = True
    wanted = list(datasets) if all_ else (names or [])
    if not wanted:
        typer.echo("Nothing to do. Pass --all or a dataset name.", err=True)
        raise typer.Exit(2)

    failures, fetched, ok, skipped = [], [], [], []
    for n in wanted:
        if n not in datasets:
            failures.append(f"{n}: not in registry"); continue
        d = datasets[n]
        path = ROOT / d["path"]
        want = d.get("sha256")
        private = str(d.get("visibility", "public")).lower() == "private"
        try:
            repo = _resolve_repo(d.get("repository", default_repo))
        except PrivateAuthError as exc:
            failures.append(f"{n}: {exc}"); continue
        # A private asset is only fetchable when its publication is confirmed durable.
        unpublished = (not want) or str(d.get("publication_status", "")) == "LOCAL_ONLY_UNPUBLISHED"

        if path.exists() and want and sha256_file(path) == want:
            ok.append(n); typer.echo(f"[ok] {n}: present and hash-verified"); continue
        if path.exists() and want and sha256_file(path) != want:
            failures.append(f"{n}: sha256 MISMATCH vs registry (corruption/drift)"); continue

        if unpublished:
            # Unpublished: can't fetch or verify. Informational unless it's the sole target.
            skipped.append(n)
            typer.echo(f"[skip] {n}: not yet durably published "
                       f"(status={d.get('publication_status')}) — run publish_data.py", err=True)
            continue
        if check:
            skipped.append(n); typer.echo(f"[missing] {n}: not present (fetch to retrieve)"); continue

        typer.echo(f"[fetch] {n} <- {'PRIVATE ' if private else ''}release "
                   f"{repo} {d['release_tag']}/{d['asset']}")
        try:
            _download(repo, d["release_tag"], d["asset"], path, private=private)
        except (RuntimeError, PrivateAuthError) as exc:
            failures.append(f"{n}: {exc}"); continue
        got = sha256_file(path) if path.exists() else None
        if got != want:
            failures.append(f"{n}: sha256 mismatch after download (got={got} want={want})"); continue
        fetched.append(n); typer.echo(f"[ok] {n}: downloaded and hash-verified")

    typer.echo(f"\nfetched={len(fetched)} verified_present={len(ok)} "
               f"skipped={len(skipped)} failed={len(failures)}")
    for f in failures:
        typer.echo(f"  [FAIL] {f}", err=True)
    raise typer.Exit(1 if failures else 0)


if __name__ == "__main__":
    app()
