"""Publish a dataset to a GitHub Release and record its sha256 in the registry.

Never-lose-a-file, sending end. Uploads the local file to its release (creating the
release if needed), then writes the file's sha256/bytes back into
config/data_registry.json. Commit the registry change so every clone can fetch it.

Usage:
    python3 scripts/publish_data.py --name wnba_games
    python3 scripts/publish_data.py --all            # publish every dataset present locally
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_registry_lib import (  # noqa: E402
    ROOT,
    load_registry,
    require_gh,
    run,
    save_registry,
    sha256_file,
)

app = typer.Typer(add_completion=False)


def _ensure_release(gh: str, repo: str, tag: str) -> None:
    if run([gh, "release", "view", tag, "--repo", repo]).returncode != 0:
        res = run([gh, "release", "create", tag, "--repo", repo,
                   "--title", tag, "--notes", f"Data assets ({tag})"])
        if res.returncode != 0:
            raise RuntimeError(f"gh release create {tag} failed:\n{res.stderr.strip()}")


def _upload(gh: str, repo: str, tag: str, path: Path) -> None:
    res = run([gh, "release", "upload", tag, str(path), "--clobber", "--repo", repo])
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload {tag} failed:\n{res.stderr.strip()}")


@app.command()
def main(
    name: list[str] = typer.Option(None, "--name", help="Dataset name(s) to publish."),
    all_: bool = typer.Option(False, "--all", help="Publish every dataset present locally."),
) -> None:
    gh = require_gh()
    reg = load_registry()
    repo = reg["repo"]
    datasets = reg["datasets"]
    names = list(datasets) if all_ else (name or [])
    if not names:
        typer.echo("Nothing to do. Pass --all or --name <dataset>.", err=True)
        raise typer.Exit(2)

    published, failures = [], []
    for n in names:
        if n not in datasets:
            failures.append(f"{n}: not in registry"); continue
        d = datasets[n]
        path = ROOT / d["path"]
        if not path.exists():
            if all_:
                typer.echo(f"[skip] {n}: not present locally"); continue
            failures.append(f"{n}: local file missing ({path})"); continue

        typer.echo(f"[publish] {n} -> release {d['release_tag']}/{d['asset']}")
        try:
            _ensure_release(gh, repo, d["release_tag"])
            _upload(gh, repo, d["release_tag"], path)
        except RuntimeError as exc:
            failures.append(f"{n}: {exc}"); continue

        d["sha256"] = sha256_file(path)
        d["bytes"] = path.stat().st_size
        d["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        published.append(n)
        typer.echo(f"[ok] {n}: uploaded, sha256={d['sha256'][:12]}… ({d['bytes']:,} bytes)")

    if published:
        save_registry(reg)
        typer.echo(f"\nUpdated {load_registry.__module__ and 'config/data_registry.json'} — "
                   f"commit it so other clones can fetch. published={published}")
    for f in failures:
        typer.echo(f"  [FAIL] {f}", err=True)
    raise typer.Exit(1 if failures else 0)


if __name__ == "__main__":
    app()
