"""Guard: pick-engine PR must not commit secrets or private live quote dumps."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_SUBSTRINGS = (
    "ODDS_API_KEY=",
    "BDL_API_KEY=",
    "API_KEY=sk-",
    "BEGIN RSA PRIVATE KEY",
)


def test_pick_engine_paths_contain_no_secrets():
    roots = [
        Path("src/wnba_props_model/pick_engine"),
        Path("scripts/run_pick_engine.py"),
        Path("scripts/replay_aug1_pick_engine.py"),
        Path("scripts/fit_pick_engine_reliability.py"),
        Path("artifacts/pick_engine"),
        Path("tests/pick_engine"),
        Path("tests/fixtures/pick_engine"),
        Path("docs/PICK_ENGINE.md"),
    ]
    files = []
    for r in roots:
        if r.is_file():
            files.append(r)
        elif r.is_dir():
            files.extend([p for p in r.rglob("*") if p.is_file()])
    assert files
    for path in files:
        if path.suffix in {".parquet", ".png", ".jpg", ".pyc"}:
            continue
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("test_no_secrets"):
            continue
        text = path.read_text(errors="ignore")
        for bad in FORBIDDEN_SUBSTRINGS:
            assert bad not in text, f"{path} contains forbidden secret marker {bad}"


def test_snapshots_dir_gitignored():
    gi = Path(".gitignore").read_text()
    assert "data/snapshots/" in gi
