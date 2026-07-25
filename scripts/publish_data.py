"""Publish a dataset to a GitHub Release and record its sha256 in the registry.

Never-lose-a-file, sending end. Uploads the local file to its release (creating the
release if needed), then writes the file's sha256/bytes back into
config/data_registry.json. Commit the registry change so every clone can fetch it.

Private assets: a dataset may declare its own ``repository`` (default: the registry
``repo``) and ``visibility`` ("public"|"private"). PRIVATE assets are pushed with a
token taken from the environment (priority PRIVATE_DATA_WRITER_TOKEN, then
PRIVATE_DATA_GH_TOKEN, then GH_TOKEN); the token is placed ONLY in the child gh
env — never in argv, URLs, logs, or messages. On success a private asset's
``publication_status`` is set to PUBLISHED so fetch_data.py will retrieve it.

Usage:
    python3 scripts/publish_data.py --name wnba_games
    python3 scripts/publish_data.py --all            # publish every dataset present locally
"""
from __future__ import annotations

import os
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


class PrivateAuthError(RuntimeError):
    """Private asset publish requested but no write token is available (fail closed)."""


def _resolve_repo(value: str) -> str:
    """Resolve a repository field, expanding a ``${ENV_VAR}`` reference from the environment.

    The private data repository name is a protected secret and is NEVER persisted in the
    committed registry; entries store the literal ``${DATA_ASSET_REPOSITORY}`` and it is
    expanded here at runtime."""
    v = value.strip()
    if v.startswith("${") and v.endswith("}"):
        env_name = v[2:-1]
        resolved = os.environ.get(env_name)
        if not resolved:
            raise PrivateAuthError(f"registry repository refers to ${{{env_name}}} but it is "
                                   "not set in the environment")
        return resolved
    return v


def _auth_env(private: bool) -> "dict | None":
    """Child-process env carrying a WRITE token for PRIVATE assets, or None for public.

    Priority: PRIVATE_DATA_WRITER_TOKEN, PRIVATE_DATA_GH_TOKEN, GH_TOKEN. Fail closed for
    private assets when none is set. The token is placed ONLY in the child env (never in
    argv, URLs, logs, or messages)."""
    if not private:
        return None
    token = (os.environ.get("PRIVATE_DATA_WRITER_TOKEN")
             or os.environ.get("PRIVATE_DATA_GH_TOKEN")
             or os.environ.get("GH_TOKEN"))
    if not token:
        raise PrivateAuthError(
            "private data asset requires PRIVATE_DATA_WRITER_TOKEN / PRIVATE_DATA_GH_TOKEN "
            "/ GH_TOKEN in the environment")
    env = dict(os.environ)
    env["GH_TOKEN"] = token  # gh reads GH_TOKEN from env; never passed as an argument
    return env


def _repo_exists(gh: str, repo: str, env: "dict | None") -> bool:
    return run([gh, "repo", "view", repo, "--json", "name"], env=env).returncode == 0


def _ensure_release(gh: str, repo: str, tag: str, env: "dict | None") -> None:
    if run([gh, "release", "view", tag, "--repo", repo], env=env).returncode != 0:
        res = run([gh, "release", "create", tag, "--repo", repo,
                   "--title", tag, "--notes", f"Data assets ({tag})"], env=env)
        if res.returncode != 0:
            raise RuntimeError(f"gh release create {tag} failed (rc={res.returncode})")


def _upload(gh: str, repo: str, tag: str, path: Path, env: "dict | None") -> None:
    res = run([gh, "release", "upload", tag, str(path), "--clobber", "--repo", repo], env=env)
    if res.returncode != 0:
        raise RuntimeError(f"gh release upload {tag} failed (rc={res.returncode})")


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

        private = str(d.get("visibility", "public")).lower() == "private"
        try:
            target_repo = _resolve_repo(d.get("repository", repo))
            env = _auth_env(private)
        except PrivateAuthError as exc:
            failures.append(f"{n}: {exc}"); continue

        if private and not _repo_exists(gh, target_repo, env):
            # Fine-grained tokens scoped to a repo cannot create it; the owner must create
            # the empty private repo first. Fail closed with an actionable, token-safe message.
            failures.append(
                f"{n}: PRIVATE_DATA_REPO_MISSING ({target_repo} not reachable with the "
                f"provided token — create the empty private repo and grant Contents:read/write)")
            continue

        typer.echo(f"[publish] {n} -> {'PRIVATE ' if private else ''}{target_repo} "
                   f"release {d['release_tag']}/{d['asset']}")
        try:
            _ensure_release(gh, target_repo, d["release_tag"], env)
            _upload(gh, target_repo, d["release_tag"], path, env)
        except RuntimeError as exc:
            failures.append(f"{n}: {exc}"); continue

        d["sha256"] = sha256_file(path)
        d["bytes"] = path.stat().st_size
        d["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if private:
            d["publication_status"] = "PUBLISHED"
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
