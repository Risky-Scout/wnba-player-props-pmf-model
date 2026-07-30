"""Verify + hash the private recovered inputs and emit PRIVATE_INPUT_MANIFEST.json.

Never commits licensed payloads; only records logical name, physical location, shape, date span,
SHA-256, and retrieval command. Training loads this manifest and fails closed on hash mismatch.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "sharp_v3"
OUT.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "pregame_features_t12": "data/recovered_v2/modeling/wnba_pregame_features_t12.parquet",
    "player_targets": "data/recovered_v2/modeling/wnba_player_targets.parquet",
    "atomic_quotes": "data/atomic_quotes/atomic_quotes.parquet",
    "decision_pairs": "data/atomic_quotes/decision_pairs.parquet",
    "closing_pairs": "data/atomic_quotes/closing_pairs.parquet",
    "settled_quote_pairs": "data/atomic_quotes/settled_quote_pairs.parquet",
    "bdl_player_game_stats": "data/recovered_v2/wnba_player_game_stats.parquet",
    "bdl_games": "data/recovered_v2/wnba_games.parquet",
    "bdl_players": "data/recovered_v2/wnba_players.parquet",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _date_span(df: pd.DataFrame) -> tuple[str | None, str | None]:
    for c in ("game_date", "decision_timestamp_utc", "pair_timestamp", "scheduled_tip_utc"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True).dropna()
            if len(s):
                return str(s.min().date()), str(s.max().date())
    return None, None


def main() -> None:
    records = {}
    for name, rel in ARTIFACTS.items():
        p = REPO / rel
        if not p.exists():
            records[name] = {"status": "MISSING", "physical_location": rel}
            continue
        try:
            df = pd.read_parquet(p)
            nrows, ncols = int(df.shape[0]), int(df.shape[1])
            d0, d1 = _date_span(df)
        except Exception:  # noqa: BLE001
            nrows = ncols = None; d0 = d1 = None
            df = None
        records[name] = {
            "status": "PRESENT", "physical_location": rel, "rows": nrows, "cols": ncols,
            "first_date": d0, "last_date": d1, "sha256": _sha256(p),
            "size_bytes": p.stat().st_size, "license_status": "PRIVATE_LICENSED_DO_NOT_PUBLISH",
            "retrieval_command": f"read_parquet('{rel}')  # from private volume / PR#97 workspace",
        }
    manifest = {
        "artifact": "PRIVATE_INPUT_MANIFEST", "release": "wnba-sharp-pmf-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Private, gitignored, licensed inputs. Training fails closed if hashes drift.",
        "inputs": records,
    }
    (OUT / "PRIVATE_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    present = sum(1 for r in records.values() if r.get("status") == "PRESENT")
    print(f"PRIVATE_INPUT_MANIFEST: {present}/{len(ARTIFACTS)} artifacts present")
    for n, r in records.items():
        print(f"  {r.get('status'):8} {n:26} rows={r.get('rows')} sha={str(r.get('sha256'))[:12]}")


if __name__ == "__main__":
    main()
