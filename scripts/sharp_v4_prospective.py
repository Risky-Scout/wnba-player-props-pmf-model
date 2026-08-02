"""Append-only prospective prediction registry (prequential). Registers the live pre-tip forecasts
immutably (hashed); refuses to overwrite an existing prediction_id.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "sharp_v4"
REG_DIR = REPO / "deliveries" / "sharp_v4" / "prospective"
REG = REG_DIR / "registry.parquet"


def _pmf_hash(g):
    a = g.sort_values("atom_value")["atom_probability"].to_numpy()
    return hashlib.sha256(np.round(a, 8).tobytes()).hexdigest()[:16]


def register(slate_date: str = "2026-07-31") -> None:
    REG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    live = REPO / "deliveries" / "sharp_v4" / slate_date / "T-live" / "active_atom_pmfs.parquet"
    if not live.exists():
        _status({"error": "no live atom file"}, ts); return
    atoms = pd.read_parquet(live)
    rows = []
    for (gid, pid, tgt), g in atoms.groupby(["game_id", "canonical_player_id", "target"]):
        h = _pmf_hash(g)
        pid_str = f"{gid}:{pid}:{tgt}:{slate_date}"
        rows.append({"prediction_id": hashlib.sha256(pid_str.encode()).hexdigest()[:24],
                     "forecast_timestamp": g["prediction_timestamp"].iloc[0], "scheduled_tip": slate_date,
                     "game_id": int(gid), "canonical_player_id": int(pid), "target": tgt,
                     "atom_pmf_hash": h, "p_active": float(g["p_active"].iloc[0]),
                     "model_version": "wnba-sharp-pmf-v4", "design_hash": g["design_hash"].iloc[0],
                     "code_sha": g["code_sha"].iloc[0], "source_track": "PURE_PMF_V4",
                     "pricing_status": "PROSPECTIVE_EVIDENCE_ACCUMULATING", "settled": False})
    new = pd.DataFrame(rows)
    if REG.exists():
        old = pd.read_parquet(REG)
        existing = set(old["prediction_id"])
        add = new[~new["prediction_id"].isin(existing)]        # append-only; never overwrite
        combined = pd.concat([old, add], ignore_index=True)
        appended = len(add)
    else:
        combined = new; appended = len(new)
    combined.to_parquet(REG, index=False)
    _status({"registry_rows": len(combined), "appended": int(appended),
             "distinct_game_dates": int(combined["scheduled_tip"].nunique()),
             "distinct_predictions": int(combined["prediction_id"].nunique()),
             "append_only": True, "prospective_start": ts,
             "certification_threshold": {"min_game_dates": 30, "min_settled_rows": 300},
             "status": "PROSPECTIVE_EVIDENCE_ACCUMULATING"}, ts)
    print(f"prospective registry: {len(combined)} rows (+{appended}) at {REG.relative_to(REPO)}")


def _status(extra, ts):
    (OUT / "PROSPECTIVE_REGISTRY_STATUS.json").write_text(json.dumps(
        {"artifact": "PROSPECTIVE_REGISTRY_STATUS", "generated_at_utc": ts,
         "mode": "append_only_prequential", "registry_path": "deliveries/sharp_v4/prospective/registry.parquet (gitignored)",
         **extra}, indent=2, default=str))


if __name__ == "__main__":
    register()
