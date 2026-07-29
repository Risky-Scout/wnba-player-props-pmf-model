"""Explicit feature-provenance metadata (replaces regex-only market classification).

The ablation harness historically classified "market-derived" columns with a small
regex (``player_market_.*``, ``blowout_probability`` ...). That regex silently
**failed to catch** the current-game Vegas columns ``game_total``,
``game_spread_home`` and ``implied_team_total``, so those features leaked into the
study that was reported as *market-excluded / pure*. This module makes provenance
**explicit and enumerated** so a "pure" study can never silently ingest a
current-game market feature again.

Every model feature is assigned exactly one :class:`Provenance` label:

``PURE_LAGGED``                strictly-lagged, causally-valid, non-market signal.
``IDENTITY_ONLY``             identity / fixed pregame facts (ids, position, home).
``INTERNAL_GAME_MODEL``       game context from an INTERNAL net-rating model (NOT Vegas).
``EXTERNAL_MARKET_CURRENT_GAME``  tonight's Vegas total / spread and everything derived.
``EXTERNAL_MARKET_LAGGED``    prior-game closing line / movement (a separate market family).
``FORWARD_PREGAME_CONTEXT``   requires tonight's confirmed lineup / injuries / game script.
``TARGET_GAME_OUTCOME``       same-game box outcome (leakage; never a feature).

The canonical current-game and lagged market sets are sourced from
``feature_contract`` so there is a single source of truth.
"""
from __future__ import annotations

import re
from enum import Enum

from wnba_props_model.features.feature_contract import (
    LAGGED_MARKET_FEATURES,
    SAME_GAME_MARKET_FEATURES,
)


class Provenance(str, Enum):
    PURE_LAGGED = "PURE_LAGGED"
    IDENTITY_ONLY = "IDENTITY_ONLY"
    INTERNAL_GAME_MODEL = "INTERNAL_GAME_MODEL"
    EXTERNAL_MARKET_CURRENT_GAME = "EXTERNAL_MARKET_CURRENT_GAME"
    EXTERNAL_MARKET_LAGGED = "EXTERNAL_MARKET_LAGGED"
    FORWARD_PREGAME_CONTEXT = "FORWARD_PREGAME_CONTEXT"
    TARGET_GAME_OUTCOME = "TARGET_GAME_OUTCOME"


# --- explicit, enumerated membership -------------------------------------------------
CURRENT_GAME_MARKET: frozenset[str] = frozenset(SAME_GAME_MARKET_FEATURES)
LAGGED_MARKET: frozenset[str] = frozenset(LAGGED_MARKET_FEATURES)

IDENTITY_ONLY_FEATURES: frozenset[str] = frozenset({
    "player_id_code", "team_id_code", "opponent_team_id_code",
    "position_G", "position_F", "position_C", "is_home",
})

# Game context produced by an INTERNAL net-rating / pace model (NOT the sportsbook).
# feature_contract documents that pregame_win_probability comes from a net-rating spread,
# not Vegas; game_pace_predicted / expected_minutes_given_script are internal projections.
INTERNAL_GAME_MODEL_FEATURES: frozenset[str] = frozenset({
    "pregame_win_probability", "blowout_probability", "close_game_probability",
    "game_pace_predicted", "expected_minutes_given_script", "minutes_upside",
})

# Requires tonight's confirmed lineup / injury / vacated-role forecast. Not historically
# constructible from strictly-lagged data alone; must flow through the availability /
# starting-role / minutes subsystem rather than being null-filled into a stat model.
FORWARD_PREGAME_CONTEXT_FEATURES: frozenset[str] = frozenset({
    "lineup_confirmed", "confirmed_starter", "expected_starter", "expected_bench",
    "team_expected_starters_count", "player_is_confirmed_starter",
    "team_out_count", "team_questionable_count", "teammate_out_count",
    "teammate_questionable_count", "team_total_usage_of_out_players",
    "team_top3_scorers_available", "player_role_elevation",
    "usage_vacated_proxy", "rebound_vacated_proxy", "assist_vacated_proxy",
    "projected_usage_given_absences", "usage_transfer_delta",
})

# Prefix / regex rules for dynamically-named columns (only used as a last resort, after
# the explicit sets above). Applied to the WIDE ablation matrix which contains more
# columns than the production MODEL_FEATURES allowlist.
_FORWARD_PREFIX_RULES = (
    re.compile(r"^teammate_\d+_"),
    re.compile(r"^without_\d+_"),
    re.compile(r"^vacated_(minutes|pts|usage)_l\d"),
    re.compile(r"_given_absences$"),
)
_LAGGED_MARKET_PREFIX_RULES = (
    re.compile(r"^player_market_"),
    re.compile(r"^player_line_movement"),
)
# Same-game outcome / box columns that must never be a feature.
_OUTCOME_RULES = (
    re.compile(r"^actual_"),
)


def classify(name: str) -> Provenance:
    """Return the single provenance label for a feature column name."""
    if name in CURRENT_GAME_MARKET:
        return Provenance.EXTERNAL_MARKET_CURRENT_GAME
    if name in LAGGED_MARKET or any(r.search(name) for r in _LAGGED_MARKET_PREFIX_RULES):
        return Provenance.EXTERNAL_MARKET_LAGGED
    if name in IDENTITY_ONLY_FEATURES:
        return Provenance.IDENTITY_ONLY
    if name in INTERNAL_GAME_MODEL_FEATURES:
        return Provenance.INTERNAL_GAME_MODEL
    if name in FORWARD_PREGAME_CONTEXT_FEATURES or any(r.search(name) for r in _FORWARD_PREFIX_RULES):
        return Provenance.FORWARD_PREGAME_CONTEXT
    if any(r.search(name) for r in _OUTCOME_RULES):
        return Provenance.TARGET_GAME_OUTCOME
    return Provenance.PURE_LAGGED


def classify_many(names) -> dict[str, str]:
    return {n: classify(n).value for n in names}


def is_current_game_market(name: str) -> bool:
    return classify(name) is Provenance.EXTERNAL_MARKET_CURRENT_GAME


def is_market_derived(name: str) -> bool:
    return classify(name) in (
        Provenance.EXTERNAL_MARKET_CURRENT_GAME,
        Provenance.EXTERNAL_MARKET_LAGGED,
    )


def pure_compact_eligible(name: str) -> bool:
    """A PURE_COMPACT study may use only strictly-lagged non-market signal plus fixed
    identity/pregame facts. It excludes any market feature, internal game-model context
    (that is a separate INTERNAL_GAME_CONTEXT candidate), forward availability context
    (handled by the minutes subsystem), and any same-game outcome."""
    return classify(name) in (Provenance.PURE_LAGGED, Provenance.IDENTITY_ONLY)


def partition(names) -> dict[str, list[str]]:
    """Group column names by provenance label (sorted within each group)."""
    out: dict[str, list[str]] = {p.value: [] for p in Provenance}
    for n in names:
        out[classify(n).value].append(n)
    return {k: sorted(v) for k, v in out.items()}
