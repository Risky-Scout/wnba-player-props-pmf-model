"""Deterministic player-identity resolution (Stage 7). No fuzzy matching, no API.

Resolution order within the resolved game's BDL roster only:
  1. exact normalized full name
  2. exact approved alias
  3. deterministic punctuation / suffix / diacritic normalization
  4. otherwise unresolved

Every automatic acceptance is deterministic and tested. A resolution that would collide two
different roster players is refused (returns unresolved with a COLLISION reason).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def fold_ascii(s: object) -> str:
    """NFKD diacritic fold to ASCII (Núñez -> nunez, Dāmiris -> damiris)."""
    t = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in t if not unicodedata.combining(c))


def normalize_strict(s: object) -> str:
    """Exact-normalized name: lowercase, ASCII-folded, non-alphanumeric removed."""
    return re.sub(r"[^a-z0-9]", "", fold_ascii(s).lower())


def normalize_relaxed(s: object) -> str:
    """Deterministic relaxed key: ASCII-folded, suffixes dropped, tokens sorted-joined.
    Handles 'A.J. Player Jr.' vs 'AJ Player' and diacritic differences."""
    folded = fold_ascii(s).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", folded) if t and t not in _SUFFIXES]
    return "".join(tokens)


def _roster_maps(roster_sub: pd.DataFrame) -> tuple[dict, dict]:
    strict, relaxed = {}, {}
    for pn, pid in zip(roster_sub["player_name"], roster_sub["player_id"]):
        strict.setdefault(normalize_strict(pn), set()).add(str(pid))
        relaxed.setdefault(normalize_relaxed(pn), set()).add(str(pid))
    return strict, relaxed


def resolve_player(name: str, game_id, roster: pd.DataFrame,
                   aliases: dict | None = None) -> tuple[str | None, str]:
    """Return (player_id, method). method in
    {exact_roster_name, approved_alias, normalized_relaxed, unmatched, collision}."""
    if roster is None or roster.empty or game_id is None:
        return None, "unmatched"
    sub = roster[roster["game_id"].astype(str) == str(game_id)]
    if sub.empty:
        return None, "unmatched"
    strict, relaxed = _roster_maps(sub)

    key = normalize_strict(name)
    if key in strict and len(strict[key]) == 1:
        return next(iter(strict[key])), "exact_roster_name"

    if aliases:
        alias_target = aliases.get(str(name)) or aliases.get(key)
        if alias_target:
            akey = normalize_strict(alias_target)
            if akey in strict and len(strict[akey]) == 1:
                return next(iter(strict[akey])), "approved_alias"

    rkey = normalize_relaxed(name)
    if rkey in relaxed:
        if len(relaxed[rkey]) == 1:
            return next(iter(relaxed[rkey])), "normalized_relaxed"
        return None, "collision"       # never force a collision

    return None, "unmatched"
