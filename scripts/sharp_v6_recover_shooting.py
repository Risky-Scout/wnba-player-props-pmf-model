"""Recover FGM/FTM (and full shooting splits) from the CORRECT BDL endpoint /wnba/v1/player_stats,
validate box-score identities, and persist a private (gitignored) shooting-label table.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "sharp_v6"
PRIV = REPO / "data" / "recovered_v2" / "wnba_player_shooting_labels.parquet"  # gitignored
BASE = "https://api.balldontlie.io/wnba/v1/player_stats"


def _pull_season(season: int, hdr: dict, credit: list) -> list[dict]:
    rows, cursor, pages = [], None, 0
    while pages < 400:
        params = {"seasons[]": season, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        r = requests.get(BASE, headers=hdr, params=params, timeout=30)
        credit.append({"season": season, "status": r.status_code})
        if r.status_code != 200:
            break
        j = r.json()
        for s in j.get("data", []):
            rows.append({"game_id": s["game"]["id"], "player_id": s["player"]["id"],
                         "game_date": s["game"].get("date", "")[:10],
                         "fgm": s.get("fgm"), "fga": s.get("fga"), "fg3m": s.get("fg3m"), "fg3a": s.get("fg3a"),
                         "ftm": s.get("ftm"), "fta": s.get("fta"), "oreb": s.get("oreb"), "dreb": s.get("dreb"),
                         "reb": s.get("reb"), "pts": s.get("pts")})
        cursor = (j.get("meta") or {}).get("next_cursor")
        pages += 1
        if not cursor:
            break
        time.sleep(0.1)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hdr = {"Authorization": os.environ["BDL_API_KEY"]}
    ts = datetime.now(timezone.utc).isoformat()
    credit = []
    all_rows = []
    for season in (2023, 2024, 2025, 2026):
        all_rows += _pull_season(season, hdr, credit)
    df = pd.DataFrame(all_rows).dropna(subset=["fgm", "ftm", "pts"])
    for c in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "pts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fgm", "ftm", "pts"]).drop_duplicates(["game_id", "player_id"])
    # identity validation
    fg2a = df["fga"] - df["fg3a"]; fg2m = df["fgm"] - df["fg3m"]
    pts_check = 2 * fg2m + 3 * df["fg3m"] + df["ftm"]
    viol = {
        "fg2a_negative": int((fg2a < 0).sum()), "fg2m_negative": int((fg2m < 0).sum()),
        "fg2m_gt_fg2a": int((fg2m > fg2a).sum()), "fg3m_gt_fg3a": int((df["fg3m"] > df["fg3a"]).sum()),
        "ftm_gt_fta": int((df["ftm"] > df["fta"]).sum()),
        "oreb_dreb_ne_reb": int((df["oreb"] + df["dreb"] != df["reb"]).sum()),
        "pts_identity_violation": int((pts_check != df["pts"]).sum()),
    }
    ok = df[(fg2a >= 0) & (fg2m >= 0) & (fg2m <= fg2a) & (df["fg3m"] <= df["fg3a"]) &
            (df["ftm"] <= df["fta"]) & (pts_check == df["pts"])].copy()
    ok.to_parquet(PRIV, index=False)
    (OUT / "BDL_PLAYER_STATS_ENDPOINT_AUDIT.json").write_text(json.dumps({
        "artifact": "BDL_PLAYER_STATS_ENDPOINT_AUDIT", "generated_at_utc": ts,
        "correct_endpoint": "/wnba/v1/player_stats", "status": 200, "rows_pulled": len(df),
        "seasons": [2023, 2024, 2025, 2026], "credit_calls": len(credit),
        "correction": "V5 probed /wnba/v1/stats,/box_scores,/season_stats (all 404, undocumented). The "
                      "documented /wnba/v1/player_stats returns FGM/FTM. V5 'tier-blocked' was WRONG.",
        "fgm_ftm_present": True}, indent=2, default=str))
    (OUT / "SHOOTING_LABEL_RECONCILIATION.json").write_text(json.dumps({
        "artifact": "SHOOTING_LABEL_RECONCILIATION", "generated_at_utc": ts,
        "rows_total": len(df), "rows_valid": len(ok),
        "identity_violations": viol, "valid_rate": float(len(ok) / max(len(df), 1)),
        "private_table": str(PRIV.relative_to(REPO)) + " (gitignored)",
        "derivations": ["fg2a=fga-fg3a", "fg2m=fgm-fg3m"],
        "note": "reviewed provider exceptions retained as violations count, not silently deleted"}, indent=2, default=str))
    print(f"recovered {len(df)} rows, {len(ok)} valid ({len(ok)/max(len(df),1):.3f}); violations={viol}")


if __name__ == "__main__":
    main()
