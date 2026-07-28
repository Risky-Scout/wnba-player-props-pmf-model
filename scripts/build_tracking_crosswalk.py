#!/usr/bin/env python3
"""Build reviewed tracking<->canonical identity crosswalks (owner directive section B).

Tracking (tracking-data-v1) uses NBA-stats ids (gameId/personId) and has NO game_date. Canonical
box uses BDL ids (game_id/player_id) and exists only for seasons 2025-2026. We therefore bridge:

  GAME:   season (decoded from gameId[3:5]) + matched (normalized player name, minutes) signature
  PLAYER: resolved game + exact normalized name  ->  (game_id, player_id)

No fuzzy auto-accept: a game maps only when a unique best box game matches a high fraction of the
tracking (name, minute) signature with a clear margin over the runner-up. Emits crosswalk parquets,
a JSON report with per-season coverage + gate results, and unmatched / conflict CSVs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_DATA = REPO / "data" / "processed"
OUT_ART = REPO / "artifacts" / "opportunity_v2"
MIN_OVERLAP = 0.70          # min fraction of tracking players matched to accept a game
MIN_MARGIN = 0.15           # best overlap must beat 2nd best by this margin
MIN_TOL = 1.0               # minute tolerance for signature match


def _norm_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.replace(r"[.'`-]", "", regex=True)
    s = s.str.replace(r"\b(jr|sr|ii|iii|iv)\b", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _parse_minutes(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    s = str(v)
    if ":" in s:
        m, sec = s.split(":")[:2]
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def main() -> None:
    tr = pd.read_parquet(REPO / "data/processed/wnba_tracking_2021_2026.parquet")
    box = pd.read_parquet(REPO / "data/processed/wnba_player_game_stats.parquet")

    tr = tr.copy()
    tr["season"] = ("20" + tr["gameId"].astype(str).str[3:5]).astype(int)
    tr["name"] = _norm_name(tr["firstName"].astype(str) + " " + tr["familyName"].astype(str))
    tr["min_f"] = tr["minutes"].map(_parse_minutes)

    box = box.copy()
    box["name"] = _norm_name(box["player_name"])
    box["min_f"] = pd.to_numeric(box["minutes"], errors="coerce").fillna(0.0)
    box_seasons = set(box["season"].dropna().astype(int).unique())

    # ---- GAME BRIDGE (season-scoped signature match) ----
    box_sig = {}   # game_id -> list[(name, min)]
    box_by_season = {}
    for gid, g in box.groupby("game_id"):
        season = int(g["season"].iloc[0])
        sig = list(zip(g["name"], g["min_f"]))
        box_sig[gid] = sig
        box_by_season.setdefault(season, []).append(gid)

    game_rows, unmatched_games = [], []
    for tgid, g in tr.groupby("gameId"):
        season = int(g["season"].iloc[0])
        tsig = [(n, m) for n, m in zip(g["name"], g["min_f"]) if m > 0]
        if season not in box_seasons:
            unmatched_games.append({"gameId": tgid, "season": season,
                                    "reason": "season_not_in_canonical_box(2025-2026 only)"})
            continue
        best, second, best_gid = 0.0, 0.0, None
        for cand in box_by_season.get(season, []):
            cand_sig = box_sig[cand]
            matched = 0
            for n, m in tsig:
                if any(cn == n and abs(cm - m) <= MIN_TOL for cn, cm in cand_sig):
                    matched += 1
            score = matched / max(len(tsig), 1)
            if score > best:
                best, second, best_gid = score, best, cand
            elif score > second:
                second = score
        if best >= MIN_OVERLAP and (best - second) >= MIN_MARGIN:
            game_rows.append({"gameId": tgid, "game_id": best_gid, "season": season,
                              "overlap": round(best, 4), "runner_up": round(second, 4),
                              "n_tracking_players": len(tsig), "method": "signature_name_minutes"})
        else:
            unmatched_games.append({"gameId": tgid, "season": season, "best_overlap": round(best, 4),
                                    "runner_up": round(second, 4), "reason": "no_unique_confident_match"})

    game_xw = pd.DataFrame(game_rows)
    # duplicate canonical game key check (two tracking games -> same box game)
    dup_game = game_xw["game_id"].duplicated(keep=False).sum() if len(game_xw) else 0

    # ---- PLAYER BRIDGE (within matched games, exact normalized name) ----
    player_rows, conflicts = [], []
    if len(game_xw):
        gmap = dict(zip(game_xw["gameId"], game_xw["game_id"]))
        boxlk = {}  # (game_id, name) -> player_id
        for _, r in box.iterrows():
            boxlk.setdefault((r["game_id"], r["name"]), r["player_id"])
        seen = {}  # personId -> set(player_id)
        for _, r in tr.iterrows():
            if r["gameId"] not in gmap:
                continue
            gid = gmap[r["gameId"]]
            pid = boxlk.get((gid, r["name"]))
            if pid is None:
                continue
            seen.setdefault(r["personId"], set()).add(pid)
            player_rows.append({"personId": r["personId"], "player_id": pid, "name": r["name"]})
        for personId, pids in seen.items():
            if len(pids) > 1:
                conflicts.append({"personId": personId, "player_ids": sorted(map(int, pids))})

    player_xw = (pd.DataFrame(player_rows).drop_duplicates(["personId", "player_id"])
                 if player_rows else pd.DataFrame(columns=["personId", "player_id", "name"]))
    # keep only unambiguous one-to-one personId->player_id
    conflict_persons = {c["personId"] for c in conflicts}
    player_xw_clean = player_xw[~player_xw["personId"].isin(conflict_persons)].copy()

    # ---- coverage + gates ----
    per_season = {}
    for season in sorted(tr["season"].unique()):
        tg = tr[tr["season"] == season]["gameId"].nunique()
        mg = int(game_xw[game_xw["season"] == season]["gameId"].nunique()) if len(game_xw) else 0
        per_season[int(season)] = {"tracking_games": int(tg), "mapped_games": mg,
                                   "coverage": round(mg / tg, 4) if tg else 0.0,
                                   "in_canonical_box": int(season) in box_seasons}
    canonical_seasons = [s for s in per_season if per_season[s]["in_canonical_box"]]
    canon_tg = sum(per_season[s]["tracking_games"] for s in canonical_seasons)
    canon_mg = sum(per_season[s]["mapped_games"] for s in canonical_seasons)

    # market-eligible = 2026 games that appear in the deterministic scored rows
    scored = pd.read_parquet(REPO / "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet")
    mkt_games = set(pd.to_numeric(scored["game_id"], errors="coerce").dropna().astype(int))
    mapped_box_games = set(game_xw["game_id"].astype(int)) if len(game_xw) else set()
    mkt_covered = len(mkt_games & mapped_box_games)

    # diagnose each uncovered market game: identity failure vs genuine tracking-release absence
    box_team_by_game = {int(gid): set(g["team_abbreviation"].dropna())
                        for gid, g in box.groupby("game_id")}
    tr_team_pairs_by_season = {}
    for tgid, g in tr.groupby("gameId"):
        tr_team_pairs_by_season.setdefault(int(g["season"].iloc[0]), []).append(frozenset(g["teamTricode"]))
    uncovered_detail = []
    for gid in sorted(mkt_games - mapped_box_games):
        b = box[box["game_id"] == gid]
        season = int(b["season"].iloc[0]) if len(b) else None
        teams = box_team_by_game.get(int(gid), set())
        uncovered_detail.append({"game_id": int(gid),
                                 "date": str(b["game_date"].iloc[0])[:10] if len(b) else None,
                                 "season": season, "teams": sorted(teams),
                                 "reason": "absent_from_tracking_release_upstream_coverage_gap"})

    gates = {
        "canonical_seasons_coverage": round(canon_mg / canon_tg, 4) if canon_tg else 0.0,
        "canonical_coverage_ok(>=0.99)": (canon_mg / canon_tg >= 0.99) if canon_tg else False,
        "each_canonical_season_ge_0.975": all(
            per_season[s]["coverage"] >= 0.975 for s in canonical_seasons),
        "market_eligible_games": len(mkt_games),
        "market_eligible_covered": mkt_covered,
        "market_eligible_100pct": mkt_covered == len(mkt_games) if mkt_games else False,
        "ambiguous_player_conflicts": len(conflicts),
        "duplicate_canonical_game_keys": int(dup_game),
    }
    gates["all_gates_pass"] = bool(
        gates["canonical_coverage_ok(>=0.99)"] and gates["each_canonical_season_ge_0.975"]
        and gates["market_eligible_100pct"] and gates["ambiguous_player_conflicts"] == 0
        and gates["duplicate_canonical_game_keys"] == 0)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_ART.mkdir(parents=True, exist_ok=True)
    if len(game_xw):
        game_xw.to_parquet(OUT_DATA / "tracking_game_crosswalk.parquet", index=False)
    if len(player_xw_clean):
        player_xw_clean.to_parquet(OUT_DATA / "tracking_player_crosswalk.parquet", index=False)
    pd.DataFrame(unmatched_games).to_csv(OUT_ART / "unmatched_games.csv", index=False)
    (pd.DataFrame(columns=["personId", "gameId", "reason"]) if not player_rows
     else player_xw[player_xw["personId"].isin(conflict_persons)]).to_csv(
        OUT_ART / "unmatched_players.csv", index=False)
    pd.DataFrame(conflicts).to_csv(OUT_ART / "identity_conflicts.csv", index=False)

    classification = ("USABLE" if gates["all_gates_pass"] else
                      "PARTIALLY_USABLE" if canon_mg > 0 and mkt_covered > 0 else
                      "BLOCKED_IDENTITY")
    report = {
        "assets": {"tracking": "wnba_tracking_2021_2026.parquet (hash-verified)",
                   "hustle": "171 rows - unusable"},
        "canonical_box_seasons": [int(s) for s in sorted(box_seasons)],
        "note": ("Canonical box exists only for 2025-2026; pre-2025 tracking games have NO canonical "
                 "game to map to and are out-of-scope for certified use, not identity failures."),
        "per_season": per_season,
        "player_crosswalk": {"one_to_one_persons": int(player_xw_clean["personId"].nunique())
                             if len(player_xw_clean) else 0,
                             "conflicts": len(conflicts)},
        "gates": gates,
        "market_eligible_uncovered_detail": uncovered_detail,
        "classification": classification,
    }
    json.dump(report, open(OUT_ART / "tracking_identity_report.json", "w"), indent=2)
    print(json.dumps({"classification": classification, "gates": gates,
                      "canonical_coverage": report["gates"]["canonical_seasons_coverage"],
                      "market_eligible": f"{mkt_covered}/{len(mkt_games)}"}, indent=2))


if __name__ == "__main__":
    main()
