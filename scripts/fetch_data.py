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


def _download(repo: str, tag: str, asset: str, dest: Path) -> None:
    gh = require_gh()
    dest.parent.mkdir(parents=True, exist_ok=True)
    res = run([gh, "release", "download", tag, "--pattern", asset,
               "--dir", str(dest.parent), "--clobber", "--repo", repo])
    if res.returncode != 0:
        raise RuntimeError(f"gh release download failed for {tag}/{asset}:\n{res.stderr.strip()}")


@app.command()
def main(
    names: list[str] = typer.Argument(None, help="Dataset name(s) to fetch. Omit with --all."),
    all_: bool = typer.Option(False, "--all", help="Fetch every dataset in the registry."),
    check: bool = typer.Option(False, "--check", help="Verify only; do not download."),
) -> None:
    reg = load_registry()
    repo = reg["repo"]
    datasets = reg["datasets"]
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

        if path.exists() and want and sha256_file(path) == want:
            ok.append(n); typer.echo(f"[ok] {n}: present and hash-verified"); continue
        if path.exists() and want and sha256_file(path) != want:
            failures.append(f"{n}: sha256 MISMATCH vs registry (corruption/drift)"); continue

        if not want:
            # Unpublished: can't fetch or verify. Informational unless it's the sole target.
            skipped.append(n)
            typer.echo(f"[skip] {n}: not yet published (no sha256) — run publish_data.py", err=True)
            continue
        if check:
            skipped.append(n); typer.echo(f"[missing] {n}: not present (fetch to retrieve)"); continue

        typer.echo(f"[fetch] {n} <- release {d['release_tag']}/{d['asset']}")
        try:
            _download(repo, d["release_tag"], d["asset"], path)
        except RuntimeError as exc:
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
