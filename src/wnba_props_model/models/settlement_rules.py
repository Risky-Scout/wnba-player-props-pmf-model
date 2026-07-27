"""Sportsbook DNP (did-not-play) settlement rules for player props (owner item 6).

The correct binary-settlement basis depends on how the *book* settles a wager when the
player does not play:

  * ``VOID_DNP``            — a DNP voids / refunds the wager. The settled Over/Under
                             probability must be computed from the ACTIVE
                             (conditional-on-appearance) PMF.
  * ``SETTLES_DNP_AS_UNDER`` — a DNP settles as an Under (for line > 0). The
                             availability-mixture PMF (DNP mass folded onto 0) is the
                             correct basis.
  * ``UNKNOWN``            — the rule is not established for this book. Fail closed: no
                             certified probability, no proof row, no Edge Board row.

For US-regulated sportsbooks the standard practice for player-prop over/under markets is to
VOID a wager when the player does not appear. The default map below encodes that; unrecognized
books resolve to ``UNKNOWN`` and fail closed rather than silently assuming a rule.
"""
from __future__ import annotations

VOID_DNP = "VOID_DNP"
SETTLES_DNP_AS_UNDER = "SETTLES_DNP_AS_UNDER"
UNKNOWN = "UNKNOWN"

# Settlement-basis strings persisted on each delivered / scored row.
BASIS_ACTIVE = "active_pmf_push_safe_void_on_dnp"
BASIS_MIXTURE = "availability_mixture_push_safe_dnp_as_under"
BASIS_FAIL_CLOSED = "unknown_book_dnp_rule_fail_closed"


def _norm_book(book: object) -> str:
    """Normalize a book/vendor label to lowercase alphanumerics for stable matching."""
    return "".join(ch for ch in str(book or "").lower() if ch.isalnum())


# Known US player-prop books: a DNP voids the wager (settle from the ACTIVE PMF).
# Keys are stored normalized (see ``_norm_book``).
_RAW_VOID_DNP_BOOKS = (
    "draftkings", "fanduel", "betmgm", "caesars", "williamhill_us",
    "pointsbet", "pointsbetus", "betrivers", "espnbet", "fanatics",
    "hardrockbet", "hard_rock_bet", "bovada", "betonlineag", "mybookieag",
    "wynnbet", "unibet_us", "superbook", "twinspires", "barstool", "foxbet",
    "betus", "lowvig", "ballybet", "fliff",
)
_DEFAULT_DNP_RULES: dict[str, str] = {_norm_book(b): VOID_DNP for b in _RAW_VOID_DNP_BOOKS}


def resolve_dnp_settlement_rule(
    book: object,
    *,
    overrides: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(sportsbook_rule_id, dnp_settlement_rule)`` for a book/vendor label.

    ``sportsbook_rule_id`` is the normalized book key (or ``"unknown"`` when empty).
    Unrecognized books resolve to :data:`UNKNOWN` (the caller must fail closed).
    ``overrides`` (book label -> rule) lets a frozen config extend/override the map.
    """
    key = _norm_book(book)
    table = dict(_DEFAULT_DNP_RULES)
    if overrides:
        table.update({_norm_book(k): v for k, v in overrides.items()})
    rule = table.get(key, UNKNOWN)
    return (key or "unknown", rule)


def settlement_basis_for_rule(rule: str) -> str:
    """Map a DNP settlement rule to the persisted settlement-basis string."""
    if rule == VOID_DNP:
        return BASIS_ACTIVE
    if rule == SETTLES_DNP_AS_UNDER:
        return BASIS_MIXTURE
    return BASIS_FAIL_CLOSED
