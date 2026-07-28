"""Unit tests for the vacated-opportunity feature builder (Path A).

Synthetic prior (as-of-before-tip) aggregates -> deterministic redistribution of an OUT
player's minutes / usage / 3PA / possessions to available teammates, proportional to their
own prior share, with minutes capped.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.data.vacated_opportunity import (
    VacatedConfig,
    build_vacated_features_for_slate,
    redistribute_vacated_opportunity,
)


def _prior():
    # Team 1: three players. Player 10 is the star we'll mark OUT.
    # Team 2: one player (control — should never be touched).
    return pd.DataFrame([
        {"player_id": 10, "team_id": 1, "minutes": 30.0, "usage": 28.0, "fg3a": 6.0, "possessions": 60.0},
        {"player_id": 11, "team_id": 1, "minutes": 20.0, "usage": 18.0, "fg3a": 2.0, "possessions": 40.0},
        {"player_id": 12, "team_id": 1, "minutes": 10.0, "usage": 12.0, "fg3a": 2.0, "possessions": 20.0},
        {"player_id": 20, "team_id": 2, "minutes": 25.0, "usage": 22.0, "fg3a": 4.0, "possessions": 50.0},
    ])


def test_vacated_redistribution_proportional_to_prior_share():
    out = redistribute_vacated_opportunity(_prior(), out_player_ids=[10], team_id=1)

    # Only the two available teammates remain.
    assert set(out["player_id"]) == {11, 12}
    r11 = out[out["player_id"] == 11].iloc[0]
    r12 = out[out["player_id"] == 12].iloc[0]

    # fg3a: player 10 vacates 6.0; available prior fg3a = 2 + 2 = 4 -> 50/50 split -> +3 each.
    assert r11["vacated_fg3a_added"] == pytest.approx(3.0)
    assert r12["vacated_fg3a_added"] == pytest.approx(3.0)
    assert r11["proj_fg3a"] == pytest.approx(5.0)
    assert r12["proj_fg3a"] == pytest.approx(5.0)

    # possessions: 10 vacates 60; avail prior = 40 + 20 = 60 -> weights 2/3, 1/3 -> +40, +20.
    assert r11["vacated_possessions_added"] == pytest.approx(40.0)
    assert r12["vacated_possessions_added"] == pytest.approx(20.0)

    # Conservation: vacated total is fully redistributed across available teammates.
    for m, vac in (("usage", 28.0), ("fg3a", 6.0), ("possessions", 60.0)):
        assert out[f"vacated_{m}_added"].sum() == pytest.approx(vac)

    assert (out["n_out"] == 1).all()
    assert out["is_beneficiary"].all()


def test_minutes_are_capped():
    # Player 10 vacates 30 min; player 11 has prior 20 and gets 2/3 of 30 = 20 -> 40 (== cap).
    cfg = VacatedConfig(minutes_cap=35.0)
    out = redistribute_vacated_opportunity(_prior(), out_player_ids=[10], team_id=1, config=cfg)
    r11 = out[out["player_id"] == 11].iloc[0]
    # raw projected = 20 + 20 = 40, capped to 35.
    assert r11["vacated_minutes_added"] == pytest.approx(20.0)
    assert r11["proj_minutes"] == pytest.approx(35.0)


def test_no_out_players_is_identity():
    out = redistribute_vacated_opportunity(_prior(), out_player_ids=[], team_id=1)
    for _, r in out.iterrows():
        for m in ("minutes", "usage", "fg3a", "possessions"):
            assert r[f"vacated_{m}_added"] == pytest.approx(0.0)
            assert r[f"proj_{m}"] == pytest.approx(r[f"prior_{m}"])
    assert (~out["is_beneficiary"]).all()


def test_other_team_is_untouched():
    out = redistribute_vacated_opportunity(_prior(), out_player_ids=[10], team_id=1)
    assert 20 not in set(out["player_id"])  # team 2 player never appears


def test_even_split_when_available_have_zero_prior_mass():
    prior = pd.DataFrame([
        {"player_id": 1, "team_id": 9, "minutes": 30.0, "usage": 0.0, "fg3a": 8.0, "possessions": 0.0},
        {"player_id": 2, "team_id": 9, "minutes": 0.0, "usage": 0.0, "fg3a": 0.0, "possessions": 0.0},
        {"player_id": 3, "team_id": 9, "minutes": 0.0, "usage": 0.0, "fg3a": 0.0, "possessions": 0.0},
    ])
    out = redistribute_vacated_opportunity(prior, out_player_ids=[1], team_id=9)
    # fg3a: player 1 vacates 8; available have zero prior fg3a -> even split -> 4 each.
    assert out["vacated_fg3a_added"].tolist() == pytest.approx([4.0, 4.0])


def test_all_players_out_returns_empty():
    # Marking every player on the team OUT leaves no available teammate -> empty frame.
    out_all = redistribute_vacated_opportunity(_prior(), out_player_ids=[10, 11, 12], team_id=1)
    assert out_all.empty


def test_build_slate_concatenates_teams():
    prior = _prior()
    slate = build_vacated_features_for_slate(prior, {1: [10], 2: []})
    # Team 1 contributes 2 beneficiaries; team 2 has no absences -> its players pass through
    # with zero added. Both teams present.
    assert set(slate["team_id"]) == {1, 2}
    assert (slate[slate["team_id"] == 2]["vacated_minutes_added"] == 0.0).all()


def test_empty_prior_returns_typed_frame():
    out = redistribute_vacated_opportunity(pd.DataFrame(), out_player_ids=[1], team_id=1)
    assert out.empty
    assert "proj_minutes" in out.columns
