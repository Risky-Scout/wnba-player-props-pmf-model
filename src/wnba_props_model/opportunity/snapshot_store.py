"""Append-only, contract-enforcing snapshot storage for Opportunity V2.

Snapshots are immutable point-in-time records. Writes are append-only and atomic (temp file ->
fsync -> rename). A record identity may never be overwritten with a differing payload; only exact
duplicates are de-duplicated. This is the substrate that makes strict as-of joins trustworthy.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


class SnapshotContractError(ValueError):
    """Raised on any snapshot storage contract violation."""


def canonicalize_utc(series: pd.Series) -> pd.Series:
    """Parse a series to timezone-aware UTC timestamps, rejecting unparseable values.

    Empty / all-null series are returned as UTC-typed. A value that cannot be parsed (becomes NaT
    while its input was non-null) raises ``SnapshotContractError``.
    """
    s = pd.Series(series)
    parsed = pd.to_datetime(s, utc=True, errors="coerce")
    # Non-null inputs that failed to parse -> contract error.
    was_present = s.notna() & (s.astype("string").str.strip() != "")
    failed = was_present & parsed.isna()
    if bool(failed.any()):
        bad = s[failed].unique()[:5].tolist()
        raise SnapshotContractError(f"canonicalize_utc: unparseable timestamp value(s): {bad!r}")
    return parsed


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over a payload mapping (sorted keys, compact, str-coerced)."""
    import hashlib

    blob = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    tmp = Path(tmp_name)
    try:
        os.close(fd)
        frame.to_parquet(tmp, index=False)
        # Best-effort fsync of the file contents before the atomic rename.
        f = open(tmp, "rb")
        try:
            os.fsync(f.fileno())
        finally:
            f.close()
        os.replace(tmp, path)  # atomic within a filesystem
        # fsync the directory so the rename is durable.
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def append_snapshot_partition(
    frame: pd.DataFrame,
    root: Path,
    required_columns: Sequence[str],
    unique_columns: Sequence[str],
) -> list[Path]:
    """Append snapshot rows into Hive-style partitions under ``root``, immutably.

    Partitioning is by ``snapshot_date_utc`` and ``source`` (both required). Behavior:

    1. Validate required columns are present.
    2. Canonicalize every ``*_utc`` column to UTC (rejects unparseable timestamps).
    3. Reject duplicated snapshot identities WITHIN the input.
    4. For each partition, read existing records and:
       - de-duplicate exact duplicate identities (same ``unique_columns`` AND ``payload_sha256``);
       - RAISE if an identity already exists with a DIFFERENT ``payload_sha256`` (no overwrite).
    5. Write through a temp file, fsync, atomically rename.
    6. Return the list of written partition paths.
    """
    root = Path(root)
    if frame is None or len(frame) == 0:
        return []
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise SnapshotContractError(f"append_snapshot_partition: missing column(s) {missing}")
    for part_key in ("snapshot_date_utc", "source"):
        if part_key not in frame.columns:
            raise SnapshotContractError(f"append_snapshot_partition: missing partition key {part_key!r}")
    if not unique_columns:
        raise SnapshotContractError("append_snapshot_partition: unique_columns must be non-empty")

    df = frame.copy()
    # Canonicalize UTC timestamp columns.
    for col in df.columns:
        if col.endswith("_utc") and col != "snapshot_date_utc":
            df[col] = canonicalize_utc(df[col])
    # snapshot_date_utc normalized to a date (string form for stable partition names).
    df["snapshot_date_utc"] = pd.to_datetime(df["snapshot_date_utc"], utc=True, errors="coerce").dt.date
    if bool(df["snapshot_date_utc"].isna().any()):
        raise SnapshotContractError("append_snapshot_partition: unparseable snapshot_date_utc value(s)")

    # payload_sha256 must exist for identity comparison; compute from the row if absent.
    if "payload_sha256" not in df.columns:
        df["payload_sha256"] = [payload_sha256({k: r[k] for k in df.columns}) for _, r in df.iterrows()]

    uniq = list(unique_columns)
    dup_mask = df.duplicated(subset=uniq, keep=False)
    if bool(dup_mask.any()):
        # Only exact-payload duplicates are tolerable inside a single input; conflicting ones raise.
        conflict = df[dup_mask].groupby(uniq)["payload_sha256"].nunique()
        bad = conflict[conflict > 1]
        if len(bad):
            raise SnapshotContractError(
                f"append_snapshot_partition: input has {len(bad)} identity(ies) with conflicting "
                f"payloads: {bad.index.tolist()[:5]}")
        df = df.drop_duplicates(subset=uniq + ["payload_sha256"], keep="first")

    written: list[Path] = []
    for (snap_date, source), part in df.groupby(["snapshot_date_utc", "source"], sort=True):
        part_dir = root / f"snapshot_date_utc={snap_date}" / f"source={source}"
        data_path = part_dir / "snapshots.parquet"
        existing = pd.read_parquet(data_path) if data_path.exists() else None

        combined = part
        if existing is not None and len(existing):
            # Enforce no-overwrite: identity present with a different payload -> raise.
            merged = existing.merge(
                part[uniq + ["payload_sha256"]].rename(columns={"payload_sha256": "_new_sha"}),
                on=uniq, how="inner",
            )
            conflict = merged[merged["payload_sha256"] != merged["_new_sha"]]
            if len(conflict):
                raise SnapshotContractError(
                    f"append_snapshot_partition: {len(conflict)} identity(ies) already stored with a "
                    f"DIFFERENT payload in {data_path} (immutable; refusing to overwrite): "
                    f"{conflict[uniq].head(5).to_dict('records')}")
            # Drop exact duplicates already present, keep only genuinely-new rows.
            existing_keys = set(map(tuple, existing[uniq].astype("string").fillna("<NA>").values.tolist()))
            new_keys = list(map(tuple, part[uniq].astype("string").fillna("<NA>").values.tolist()))
            keep_mask = [k not in existing_keys for k in new_keys]
            part_new = part[pd.Series(keep_mask, index=part.index)]
            if len(part_new) == 0:
                continue  # nothing new for this partition
            combined = pd.concat([existing, part_new], ignore_index=True)

        _atomic_write_parquet(combined.reset_index(drop=True), data_path)
        written.append(data_path)
    return written
