"""Stage 7: deterministic identity resolution (no fuzzy matching, no forced collisions)."""
from __future__ import annotations

import pandas as pd

from wnba_props_model.data.identity_resolution import (
    fold_ascii,
    normalize_relaxed,
    normalize_strict,
    resolve_player,
)


def _roster(*names):
    return pd.DataFrame([{"game_id": "g1", "player_id": 100 + i, "player_name": n}
                         for i, n in enumerate(names)])


def test_diacritic_and_suffix_folding():
    assert fold_ascii("Núñez") == "Nunez"
    assert normalize_strict("A'ja Wilson") == "ajawilson"
    assert normalize_relaxed("A.J. Player Jr.") == normalize_relaxed("AJ Player")


def test_exact_normalized_match():
    pid, m = resolve_player("A'ja Wilson", "g1", _roster("A'ja Wilson", "Kelsey Plum"))
    assert m == "exact_roster_name" and pid == "100"


def test_diacritic_resolves_at_strict_level():
    # strict normalization already ASCII-folds, so a diacritic difference matches exactly
    pid, m = resolve_player("Damiris Dantas", "g1", _roster("Dâmiris Dantas", "Other Player"))
    assert m == "exact_roster_name" and pid == "100"


def test_suffix_resolves_via_relaxed():
    pid, m = resolve_player("A.J. Player", "g1", _roster("AJ Player Jr", "Zzz Other"))
    assert m == "normalized_relaxed" and pid == "100"


def test_approved_alias():
    pid, m = resolve_player("Chelsea Gray", "g1", _roster("C. Gray"),
                            aliases={"Chelsea Gray": "C. Gray"})
    assert m == "approved_alias" and pid == "100"


def test_collision_is_refused_not_forced():
    # two roster players collapse to the same relaxed key -> refuse
    pid, m = resolve_player("A Player", "g1", _roster("A. Player", "A Player."))
    assert pid is None and m == "collision"


def test_unmatched_returns_none():
    pid, m = resolve_player("Nobody Here", "g1", _roster("A'ja Wilson"))
    assert pid is None and m == "unmatched"
