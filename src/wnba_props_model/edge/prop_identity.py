"""Exact-match player identity resolution for Odds API prop quotes (Path B, req 1).

The Odds API returns player *names* only (``outcome.description``). Path B's
acceptance gate forbids name-only edges: every displayed row must carry a canonical
``player_id``. This module resolves a provider name to a canonical id via an EXACT,
fail-closed name index (built from a canonical player roster). There is NO fuzzy
auto-accept — an unmatched or ambiguous name resolves to ``None`` with an explicit
status so the caller can reject the row with a reason.

Design mirrors ``identity_crosswalk`` (normalize -> exact lookup -> fail closed):
  * ``build_name_index`` builds ``{normalized_name: {player_id, ...}}``.
  * ``resolve_player_id`` returns ``(player_id | None, status)`` where status is one of
    RESOLVED / UNMATCHED_PLAYER / AMBIGUOUS_PLAYER. A name mapping to >1 canonical id is
    AMBIGUOUS and never guessed.
"""
from __future__ import annotations

from wnba_props_model.data.identity_crosswalk import normalize_name

STATUS_RESOLVED = "RESOLVED"
STATUS_UNMATCHED = "UNMATCHED_PLAYER"
STATUS_AMBIGUOUS = "AMBIGUOUS_PLAYER"


def build_name_index(
    players,
    name_key: str = "player_name",
    id_key: str = "player_id",
) -> dict[str, set]:
    """Build a ``{normalized_name: set(canonical_id)}`` index from a roster.

    ``players`` is an iterable of mappings (dicts / rows). Rows missing a name or id
    are skipped. A normalized name that appears for multiple distinct ids yields a set
    with >1 element and will resolve as AMBIGUOUS (never auto-accepted).
    """
    index: dict[str, set] = {}
    for p in players or []:
        try:
            name = p.get(name_key) if hasattr(p, "get") else p[name_key]
            pid = p.get(id_key) if hasattr(p, "get") else p[id_key]
        except (KeyError, TypeError):
            continue
        if pid is None:
            continue
        nm = normalize_name(name)
        if not nm:
            continue
        index.setdefault(nm, set()).add(pid)
    return index


def resolve_player_id(name, index: dict[str, set]) -> tuple[object | None, str]:
    """Resolve one provider name to a canonical id via the exact name index.

    Returns ``(player_id, "RESOLVED")`` on a unique exact match, ``(None,
    "AMBIGUOUS_PLAYER")`` when the normalized name maps to more than one id, and
    ``(None, "UNMATCHED_PLAYER")`` when there is no match. Never fuzzy-matches.
    """
    nm = normalize_name(name)
    if not nm or not index:
        return None, STATUS_UNMATCHED
    cands = index.get(nm)
    if not cands:
        return None, STATUS_UNMATCHED
    if len(cands) > 1:
        return None, STATUS_AMBIGUOUS
    return next(iter(cands)), STATUS_RESOLVED
