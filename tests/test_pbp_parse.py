"""Parser reconciliation tests for the PBP player-attribution parser.

These use hand-built plays whose per-player counts are known by construction, exercising every
attribution rule (3PT vs 2PT makes/misses, assisted makes, off/def rebounds, steals, blocks,
turnovers, free throws) and the game-scoped name resolution.
"""
from __future__ import annotations

import pandas as pd

from wnba_props_model.data.pbp_parse import (
    GameRoster,
    parse_game_plays,
    parse_plays_to_player_game,
    reconcile_against_box,
)

# canonical roster: two teams, ids arbitrary but stable.
BOX = pd.DataFrame([
    {"game_id": 1, "player_id": 10, "player_name": "Breanna Stewart", "team_id": 100,
     "game_date": "2026-05-08", "did_play": True},
    {"game_id": 1, "player_id": 11, "player_name": "Marine Johannes", "team_id": 100,
     "game_date": "2026-05-08", "did_play": True},
    {"game_id": 1, "player_id": 20, "player_name": "Aneesah Morrow", "team_id": 200,
     "game_date": "2026-05-08", "did_play": True},
    {"game_id": 1, "player_id": 21, "player_name": "Charlisse Leger-Walker", "team_id": 200,
     "game_date": "2026-05-08", "did_play": True},
])


def _play(order, etype, text, scoring=False, sv=0):
    return {"game_id": 1, "order": order, "event_type": etype, "text": text,
            "scoring_play": scoring, "score_value": sv}


PLAYS = pd.DataFrame([
    # Stewart makes a 2 assisted by Johannes
    _play(1, "Turnaround Jump Shot", "Breanna Stewart makes 7-foot turnaround jump shot (Marine Johannes assists)", True, 2),
    # Johannes makes a 3 (step back)
    _play(2, "Step Back Jump Shot", "Marine Johannes makes 26-foot three point step back jumpshot", True, 3),
    # Morrow misses a 3
    _play(3, "Jump Shot", "Aneesah Morrow misses 27-foot three point jumper", False, 0),
    # Johannes defensive rebound
    _play(4, "Defensive Rebound", "Marine Johannes defensive rebound", False, 0),
    # Morrow bad pass turnover, Stewart steal
    _play(5, "Bad Pass\nTurnover", "Aneesah Morrow bad pass\nturnover (Breanna Stewart steals)", False, 0),
    # Leger-Walker driving layup blocked by Stewart (miss, 2PA for Leger-Walker, BLK for Stewart)
    _play(6, "Layup Shot", "Breanna Stewart blocks Charlisse Leger-Walker 's 5-foot driving layup", False, 0),
    # Leger-Walker offensive rebound
    _play(7, "Offensive Rebound", "Charlisse Leger-Walker offensive rebound", False, 0),
    # Stewart makes 2 free throws
    _play(8, "Free Throw - 1 of 2", "Breanna Stewart makes free throw 1 of 2", True, 1),
    _play(9, "Free Throw - 2 of 2", "Breanna Stewart makes free throw 2 of 2", True, 1),
    # team rebound (no rostered leading name) -> dropped
    _play(10, "Defensive Rebound", "Sun defensive rebound", False, 0),
    # non-stat plays
    _play(11, "Substitution", "Rebekah Gardner enters the game for Aneesah Morrow", False, 0),
])


def test_parse_counts_exact():
    roster = GameRoster.from_box(BOX[BOX.game_id == 1])
    counts = parse_game_plays(PLAYS, roster)

    stew = counts[10]
    assert stew["fg2m"] == 1 and stew["fg2a"] == 1
    assert stew["ftm"] == 2 and stew["fta"] == 2
    assert stew["stl"] == 1
    assert stew["blk"] == 1
    assert stew["pts"] == 2 + 2  # one 2pt FG + two FTs

    joh = counts[11]
    assert joh["fg3m"] == 1 and joh["fg3a"] == 1
    assert joh["ast"] == 1          # assisted Stewart's make
    assert joh["dreb"] == 1 and joh["reb"] == 1
    assert joh["pts"] == 3

    mor = counts[20]
    assert mor["fg3a"] == 1 and mor["fg3m"] == 0
    assert mor["tov"] == 1

    lw = counts[21]
    assert lw["fg2a"] == 1 and lw["fg2m"] == 0   # blocked layup counts as a missed 2PA
    assert lw["oreb"] == 1 and lw["reb"] == 1


def test_team_rebound_not_attributed_to_player():
    roster = GameRoster.from_box(BOX[BOX.game_id == 1])
    counts = parse_game_plays(PLAYS, roster)
    # total DREB attributed to players should be exactly 1 (Johannes); the "Sun defensive rebound"
    # team rebound must NOT be attributed to any rostered player.
    total_dreb = sum(c["dreb"] for c in counts.values())
    assert total_dreb == 1


def test_reconciliation_matches_synthetic_box():
    parsed, stats = parse_plays_to_player_game(PLAYS, BOX)
    # build a synthetic box with the by-construction truth to reconcile against
    truth = pd.DataFrame([
        {"game_id": 1, "player_id": 10, "pts": 4, "reb": 0, "ast": 0, "stl": 1, "blk": 1,
         "turnover": 0, "fg3m": 0, "did_play": True},
        {"game_id": 1, "player_id": 11, "pts": 3, "reb": 1, "ast": 1, "stl": 0, "blk": 0,
         "turnover": 0, "fg3m": 1, "did_play": True},
        {"game_id": 1, "player_id": 20, "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0,
         "turnover": 1, "fg3m": 0, "did_play": True},
        {"game_id": 1, "player_id": 21, "pts": 0, "reb": 1, "ast": 0, "stl": 0, "blk": 0,
         "turnover": 0, "fg3m": 0, "did_play": True},
    ])
    report = reconcile_against_box(parsed, truth)
    for stat in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m"):
        assert report["per_stat"][stat]["exact_match_rate"] == 1.0, (stat, report["per_stat"][stat])
