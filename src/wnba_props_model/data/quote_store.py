"""A8 - crash-safe, partitioned, append-only storage for raw side snapshots and quote pairs.

Physical safety (not merely logical append-only):

    write new immutable partition -> fsync/close -> validate schema/hash/row counts
    -> write SHA-256 manifest (to two retained locations) -> atomically update a pointer.

A historical partition is NEVER rewritten. Repeated quote ids are deduplicated both WITHIN
the incoming batch and AGAINST existing partitions.

Layout:
    <base>/raw/date=YYYY-MM-DD/run=<run_id>/part.parquet
    <base>/pairs/date=YYYY-MM-DD/run=<run_id>/pairs.parquet
    <base>/<kind>/LATEST.json                     (atomic pointer)
    <base>/_manifests/<kind>-<date>-<run>.json    (retained manifest copy #2)
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_PART_FILE = {"raw": "part.parquet", "pairs": "pairs.parquet"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_path(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    _fsync_path(tmp)
    os.replace(tmp, path)


def _collect_existing_ids(base: Path, kind: str, id_col: str) -> set:
    ids: set = set()
    root = base / kind
    if not root.exists():
        return ids
    for part in root.rglob(_PART_FILE[kind]):
        try:
            col = pd.read_parquet(part, columns=[id_col])[id_col].astype(str)
            ids.update(col.tolist())
        except Exception:  # noqa: BLE001
            continue
    return ids


def write_partition(base_dir, kind: str, date: str, run_id: str, df: pd.DataFrame, *,
                    id_col: str, expected_columns: "list[str] | None" = None) -> dict:
    """Crash-safe append of one immutable partition. Returns a summary dict.

    Refuses to rewrite an existing partition. Deduplicates ids in-batch and across partitions.
    """
    if kind not in _PART_FILE:
        raise ValueError(f"kind must be one of {list(_PART_FILE)}")
    if id_col not in df.columns:
        raise ValueError(f"df missing id column {id_col!r}")
    base = Path(base_dir)
    part_dir = base / kind / f"date={date}" / f"run={run_id}"
    final = part_dir / _PART_FILE[kind]
    if final.exists():
        raise FileExistsError(f"refusing to rewrite historical partition: {final}")

    # In-batch dedup, then cross-partition dedup against everything already stored.
    n_in = len(df)
    df = df.drop_duplicates(subset=[id_col]).copy()
    n_after_batch = len(df)
    existing_ids = _collect_existing_ids(base, kind, id_col)
    df = df[~df[id_col].astype(str).isin(existing_ids)].copy()
    n_after_cross = len(df)

    if expected_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"partition missing expected columns: {missing}")

    part_dir.mkdir(parents=True, exist_ok=True)
    tmp = part_dir / (_PART_FILE[kind] + ".tmp")
    df.to_parquet(tmp, index=False)
    _fsync_path(tmp)

    # Validate BEFORE publishing: reread + row-count + schema check.
    check = pd.read_parquet(tmp)
    if len(check) != n_after_cross:
        raise IOError(f"partition validation failed: wrote {n_after_cross} read {len(check)}")
    if expected_columns and [c for c in expected_columns if c not in check.columns]:
        raise IOError("partition validation failed: schema mismatch after write")

    os.replace(tmp, final)
    _fsync_path(final)
    sha = _sha256(final)

    manifest = {
        "kind": kind, "date": date, "run_id": run_id,
        "path": str(final).replace("\\", "/"), "rows": int(len(check)),
        "sha256": sha, "columns": list(check.columns),
        "n_incoming": int(n_in), "n_after_batch_dedup": int(n_after_batch),
        "n_after_cross_partition_dedup": int(n_after_cross),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Two retained manifest locations (partition-local + central).
    _atomic_write_json(part_dir / "manifest.json", manifest)
    central = base / "_manifests"
    central.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(central / f"{kind}-{date}-{run_id}.json", manifest)

    # Atomic pointer update (last, after data + manifests are durable).
    _atomic_write_json(base / kind / "LATEST.json",
                       {"date": date, "run_id": run_id, "path": manifest["path"], "sha256": sha})
    return manifest


def read_kind(base_dir, kind: str) -> pd.DataFrame:
    """Read all partitions for a kind (append-only union)."""
    base = Path(base_dir)
    root = base / kind
    if not root.exists():
        return pd.DataFrame()
    parts = [pd.read_parquet(p) for p in sorted(root.rglob(_PART_FILE[kind]))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verify_partition_manifest(base_dir, kind: str, date: str, run_id: str) -> bool:
    """Recompute the partition hash and compare against its manifest (fail-closed integrity)."""
    base = Path(base_dir)
    part_dir = base / kind / f"date={date}" / f"run={run_id}"
    manifest = json.loads((part_dir / "manifest.json").read_text())
    final = part_dir / _PART_FILE[kind]
    return final.exists() and _sha256(final) == manifest["sha256"]
