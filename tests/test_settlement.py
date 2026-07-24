"""A7: explicit settlement statuses; appearance settles regardless of minutes; DNP != Under
unless the frozen rule says so; unknown rules fail closed."""
from __future__ import annotations

import pandas as pd

from wnba_props_model.data.settlement import (
    CANCELED,
    OVER_WIN,
    PUSH,
    SPORTSBOOK_SETTLEMENT_RULES,
    SportsbookSettlementRule,
    UNDER_WIN,
    UNRESOLVED,
    VOID_DNP,
    settle_frame,
    settle_one,
)

RULE = SPORTSBOOK_SETTLEMENT_RULES["wnba_player_prop_standard_v1"]


def test_over_win_and_binary_eligible():
    r = settle_one(rule=RULE, line=18.5, appeared=True, actual_outcome=22)
    assert r["settlement_status"] == OVER_WIN
    assert r["binary_score_eligible"] is True and r["binary_target_over"] == 1
    assert r["pmf_score_eligible"] is True


def test_under_win():
    r = settle_one(rule=RULE, line=18.5, appeared=True, actual_outcome=10)
    assert r["settlement_status"] == UNDER_WIN and r["binary_target_over"] == 0


def test_push_is_not_binary_eligible_but_pmf_eligible():
    r = settle_one(rule=RULE, line=18.0, appeared=True, actual_outcome=18)
    assert r["settlement_status"] == PUSH
    assert r["binary_score_eligible"] is False and r["pmf_score_eligible"] is True


def test_appearance_settles_regardless_of_minutes():
    # A player who appeared for 1 minute and scored 0 still settles (Under), not a void.
    r = settle_one(rule=RULE, line=0.5, appeared=True, actual_outcome=0)
    assert r["settlement_status"] == UNDER_WIN


def test_dnp_is_void_not_under_by_default():
    r = settle_one(rule=RULE, line=18.5, appeared=False, actual_outcome=None)
    assert r["settlement_status"] == VOID_DNP
    assert r["binary_score_eligible"] is False and r["did_book_void"] is True


def test_dnp_is_under_only_when_rule_says_so():
    rule = SportsbookSettlementRule("book_x_dnp_under", dnp_is_under=True, void_on_dnp=False)
    r = settle_one(rule=rule, line=18.5, appeared=False, actual_outcome=None)
    assert r["settlement_status"] == UNDER_WIN and r["binary_target_over"] == 0


def test_unknown_rule_fails_closed():
    r = settle_one(rule=None, line=18.5, appeared=True, actual_outcome=22)
    assert r["settlement_status"] == UNRESOLVED and r["binary_score_eligible"] is False


def test_canceled_game_voids():
    r = settle_one(rule=RULE, line=18.5, appeared=True, actual_outcome=22, canceled=True)
    assert r["settlement_status"] == CANCELED and r["did_book_void"] is True


def test_settle_frame_uses_per_row_rule_id():
    df = pd.DataFrame([
        {"sportsbook_rule_id": "wnba_player_prop_standard_v1", "line": 18.5, "appeared": True,
         "actual_outcome": 22},
        {"sportsbook_rule_id": "unknown_book", "line": 18.5, "appeared": True, "actual_outcome": 22},
    ])
    out = settle_frame(df)
    assert out.iloc[0]["settlement_status"] == OVER_WIN
    assert out.iloc[1]["settlement_status"] == UNRESOLVED     # unknown rule -> fail closed
