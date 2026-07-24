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
                         min_coverage: float = 0.98) -> pd.DataFrame:
    """Resolve provider game ids to canonical game ids by EXACT id match (fail-closed).

    Returns one row per distinct provider game id with columns
    [provider_game_id, canonical_game_id, match_method, status]."""
    prov_ids = pd.Index(provider_games[provider_id_col].dropna().astype(str).unique())
    canon = set(canonical_games[canonical_id_col].dropna().astype(str))
    rows = []
    for gid in prov_ids:
        if gid in canon:
            rows.append({"provider_game_id": gid, "canonical_game_id": gid,
                         "match_method": "exact_id", "status": "RESOLVED"})
        else:
            rows.append({"provider_game_id": gid, "canonical_game_id": None,
                         "match_method": None, "status": "UNMATCHED_GAME"})
    out = pd.DataFrame(rows)
    cov = float((out["status"] == "RESOLVED").mean()) if len(out) else 0.0
    if len(out) and cov < min_coverage:
        raise CrosswalkCoverageError(
            f"game crosswalk coverage {cov:.3f} < {min_coverage} "
            f"({int((out['status'] != 'RESOLVED').sum())}/{len(out)} unmatched) - failing closed")
    return out


def build_player_crosswalk(provider_players: pd.DataFrame, canonical_players: pd.DataFrame, *,
                           game_crosswalk: pd.DataFrame,
                           provider_game_col: str = "gameId", provider_pid_col: str = "personId",
                           provider_name_col: str = "player_name",
                           canonical_game_col: str = "game_id", canonical_pid_col: str = "player_id",
                           canonical_name_col: str = "player_name",
                           min_coverage: float = 0.98) -> pd.DataFrame:
    """Resolve provider player ids to canonical player ids WITHIN a resolved game, by EXACT
    (canonical_game_id, normalized_name) match. No fuzzy auto-accept; ambiguous names (a
    normalized name mapping to >1 canonical player in the same game) are AMBIGUOUS, never guessed.
    Fails closed when coverage < threshold."""
    gmap = {r["provider_game_id"]: r["canonical_game_id"]
            for _, r in game_crosswalk.iterrows() if r["status"] == "RESOLVED"}
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
        nm = normalize_name(p.get(provider_name_col))
        cands = lut.get((str(cg), nm), set())
        if len(cands) == 1:
            rows.append({"provider_game_id": str(p[provider_game_col]),
                         "provider_player_id": str(p[provider_pid_col]),
                         "canonical_game_id": str(cg), "canonical_player_id": next(iter(cands)),
                         "status": "RESOLVED"})
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
