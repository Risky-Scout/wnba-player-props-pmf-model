"""LANE 2 (W3) - fail-closed identity crosswalk: GAME first, then PLAYER.

Tracking rows use provider ids (stats.nba.com `gameId`/`personId`). Before any tracking
feature can touch the model, each provider id must resolve to a CANONICAL `game_id` /
`player_id`. Resolution is EXACT only:

  * Game first: exact provider `gameId` -> canonical `game_id`, or (game_date, team-set) exact.
  * Player second, WITHIN a resolved game: exact (game_id, normalized_name[/team]) match.

There is NO fuzzy auto-accept. Ambiguous or unmatched ids are recorded and, if coverage falls
below a frozen threshold, resolution FAILS CLOSED (so a stale/mismatched table can never
silently produce a partial, misleading crosswalk).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


class CrosswalkCoverageError(Exception):
    """Raised when exact-match coverage is below the fail-closed threshold."""


def normalize_name(name) -> str:
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("'", "").replace(".", " ").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_game_crosswalk(provider_games: pd.DataFrame, canonical_games: pd.DataFrame, *,
                         provider_id_col: str = "gameId", canonical_id_col: str = "game_id",
                         reviewed_id_map: "dict | None" = None, manual_map: "dict | None" = None,
                         provider_season_col: "str | None" = None, canonical_season_col: "str | None" = None,
                         provider_date_col: "str | None" = None, canonical_date_col: "str | None" = None,
                         provider_home_col: "str | None" = None, provider_away_col: "str | None" = None,
                         canonical_home_col: "str | None" = None, canonical_away_col: "str | None" = None,
                         min_coverage: float = 0.98) -> pd.DataFrame:
    """Resolve provider game ids to canonical game ids with a TIERED, fail-closed matcher:

      1. stable reviewed provider-game-id map (reviewed_id_map)
      2. EXACT (season, game_date, normalized home, normalized away) match
      3. reviewed manual map (manual_map)

    There is NO fuzzy auto-accept and NO silent identical-id default: id equivalence is only
    honored when explicitly declared via reviewed_id_map. A (date, team-set) match that is not
    UNIQUE is CONFLICT (never guessed). Coverage below min_coverage fails closed."""
    reviewed_id_map = {str(k): str(v) for k, v in (reviewed_id_map or {}).items()}
    manual_map = {str(k): str(v) for k, v in (manual_map or {}).items()}
    canon_ids = set(canonical_games[canonical_id_col].dropna().astype(str))

    # Tier-2 lookup: (season, date, {home,away}) -> set(canonical_game_id).
    dt_lut: dict[tuple, set] = {}
    have_dt = all([provider_date_col, canonical_date_col, provider_home_col, provider_away_col,
                   canonical_home_col, canonical_away_col])
    if have_dt:
        for _, c in canonical_games.iterrows():
            season = str(c[canonical_season_col]) if canonical_season_col else ""
            date = str(c[canonical_date_col])[:10]
            teams = frozenset({normalize_name(c[canonical_home_col]), normalize_name(c[canonical_away_col])})
            dt_lut.setdefault((season, date, teams), set()).add(str(c[canonical_id_col]))

    # One representative provider row per game id (season/date/teams are game-level).
    prov = provider_games.drop_duplicates(subset=[provider_id_col])
    rows = []
    for _, p in prov.iterrows():
        gid = str(p[provider_id_col])
        if gid in reviewed_id_map and reviewed_id_map[gid] in canon_ids:
            rows.append({"provider_game_id": gid, "canonical_game_id": reviewed_id_map[gid],
                         "match_method": "reviewed_id_map", "status": "RESOLVED"}); continue
        matched = None
        if have_dt:
            season = str(p[provider_season_col]) if provider_season_col else ""
            date = str(p[provider_date_col])[:10]
            teams = frozenset({normalize_name(p[provider_home_col]), normalize_name(p[provider_away_col])})
            cands = dt_lut.get((season, date, teams), set())
            if len(cands) == 1:
                matched = next(iter(cands))
            elif len(cands) > 1:
                rows.append({"provider_game_id": gid, "canonical_game_id": None,
                             "match_method": "exact_date_team", "status": "CONFLICT_GAME"}); continue
        if matched is not None:
            rows.append({"provider_game_id": gid, "canonical_game_id": matched,
                         "match_method": "exact_date_team", "status": "RESOLVED"}); continue
        if gid in manual_map and manual_map[gid] in canon_ids:
            rows.append({"provider_game_id": gid, "canonical_game_id": manual_map[gid],
                         "match_method": "manual_map", "status": "RESOLVED"}); continue
        rows.append({"provider_game_id": gid, "canonical_game_id": None,
                     "match_method": None, "status": "UNMATCHED_GAME"})
    out = pd.DataFrame(rows)
    cov = float((out["status"] == "RESOLVED").mean()) if len(out) else 0.0
    if len(out) and cov < min_coverage:
        raise CrosswalkCoverageError(
            f"game crosswalk coverage {cov:.3f} < {min_coverage} "
            f"({int((out['status'] != 'RESOLVED').sum())}/{len(out)} unresolved) - failing closed")
    return out


def build_player_crosswalk(provider_players: pd.DataFrame, canonical_players: pd.DataFrame, *,
                           game_crosswalk: pd.DataFrame,
                           provider_game_col: str = "gameId", provider_pid_col: str = "personId",
                           provider_name_col: str = "player_name",
                           canonical_game_col: str = "game_id", canonical_pid_col: str = "player_id",
                           canonical_name_col: str = "player_name",
                           reviewed_pid_map: "dict | None" = None, manual_pid_map: "dict | None" = None,
                           min_coverage: float = 0.98) -> pd.DataFrame:
    """Resolve provider player ids to canonical player ids WITHIN a resolved game, TIERED:

      1. stable reviewed person-id map (reviewed_pid_map: provider_person_id -> canonical id)
      2. EXACT (canonical_game_id, normalized_name) match
      3. reviewed manual map (manual_pid_map)

    No fuzzy auto-accept; a normalized name mapping to >1 canonical player in the same game is
    AMBIGUOUS (never guessed). Fails closed when coverage < threshold."""
    reviewed_pid_map = {str(k): str(v) for k, v in (reviewed_pid_map or {}).items()}
    manual_pid_map = {str(k): str(v) for k, v in (manual_pid_map or {}).items()}
    gmap = {r["provider_game_id"]: r["canonical_game_id"]
            for _, r in game_crosswalk.iterrows() if r["status"] == "RESOLVED"}
    canon_pids = set(canonical_players[canonical_pid_col].dropna().astype(str))
    # canonical (game_id, normalized_name) -> set(player_id) to detect ambiguity.
    canon = canonical_players.copy()
    canon["_n"] = canon[canonical_name_col].map(normalize_name)
    lut: dict[tuple, set] = {}
    for _, c in canon.iterrows():
        lut.setdefault((str(c[canonical_game_col]), c["_n"]), set()).add(str(c[canonical_pid_col]))

    prov = provider_players.drop_duplicates(subset=[provider_game_col, provider_pid_col]).copy()
    rows = []
    for _, p in prov.iterrows():
        cg = gmap.get(str(p[provider_game_col]))
        if cg is None:
            rows.append({"provider_game_id": str(p[provider_game_col]),
                         "provider_player_id": str(p[provider_pid_col]),
                         "canonical_game_id": None, "canonical_player_id": None,
                         "status": "GAME_UNRESOLVED"}); continue
        pid = str(p[provider_pid_col])
        # Tier 1: stable reviewed person-id map.
        if pid in reviewed_pid_map and reviewed_pid_map[pid] in canon_pids:
            rows.append({"provider_game_id": str(p[provider_game_col]), "provider_player_id": pid,
                         "canonical_game_id": str(cg), "canonical_player_id": reviewed_pid_map[pid],
                         "status": "RESOLVED", "match_method": "reviewed_pid_map"}); continue
        nm = normalize_name(p.get(provider_name_col))
        cands = lut.get((str(cg), nm), set())
        if len(cands) == 1:
            rows.append({"provider_game_id": str(p[provider_game_col]),
                         "provider_player_id": pid,
                         "canonical_game_id": str(cg), "canonical_player_id": next(iter(cands)),
                         "status": "RESOLVED", "match_method": "exact_game_name"})
        elif pid in manual_pid_map and manual_pid_map[pid] in canon_pids:
            rows.append({"provider_game_id": str(p[provider_game_col]), "provider_player_id": pid,
                         "canonical_game_id": str(cg), "canonical_player_id": manual_pid_map[pid],
                         "status": "RESOLVED", "match_method": "manual_map"})
        elif len(cands) > 1:
            rows.append({"provider_game_id": str(p[provider_game_col]),
                         "provider_player_id": str(p[provider_pid_col]),
                         "canonical_game_id": str(cg), "canonical_player_id": None,
                         "status": "AMBIGUOUS_PLAYER"})    # never auto-accept
        else:
            rows.append({"provider_game_id": str(p[provider_game_col]),
                         "provider_player_id": str(p[provider_pid_col]),
                         "canonical_game_id": str(cg), "canonical_player_id": None,
                         "status": "UNMATCHED_PLAYER"})
    out = pd.DataFrame(rows)
    resolvable = out[out["status"] != "GAME_UNRESOLVED"]
    cov = float((resolvable["status"] == "RESOLVED").mean()) if len(resolvable) else 0.0
    if len(resolvable) and cov < min_coverage:
        raise CrosswalkCoverageError(
            f"player crosswalk coverage {cov:.3f} < {min_coverage} "
            f"({int((resolvable['status'] != 'RESOLVED').sum())}/{len(resolvable)} unresolved) - failing closed")
    return out
