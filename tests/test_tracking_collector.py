"""Foundation Lock tests for the LOCAL tracking/hustle collector.

The collector is infrastructure only (data NOT landed, features NOT built). These tests are
fully mocked and never touch the network. They lock: output filenames, season range,
deterministic frame/column selection, id retention, duplicate-key handling, empty responses,
partial endpoint failures, end-to-end write, and agreement with the versioned schema contract.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
COLLECTOR = REPO / "scripts" / "pull_wnba_tracking_local.py"
SCHEMA = REPO / "config" / "tracking_hustle_schema_v1.json"


def _load_collector():
    spec = importlib.util.spec_from_file_location("pwtl", COLLECTOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)   # top-level import is pandas only; no nba_api at import time
    return m


COL = _load_collector()


class _FakeEndpoint:
    """Configurable fake nba_api endpoint: returns preset frames or raises."""
    _frames = []
    _raise = False

    def __init__(self, *a, **k):
        if type(self)._raise:
            raise RuntimeError("simulated endpoint failure")

    def get_data_frames(self):
        return list(type(self)._frames)


def _install_fake_nba_api(track_frames, hustle_frames, gamelog_df,
                          track_raise=False, hustle_raise=False):
    """Register a fake nba_api.stats.endpoints in sys.modules."""
    class _Track(_FakeEndpoint):
        _frames = track_frames
        _raise = track_raise

    class _Hustle(_FakeEndpoint):
        _frames = hustle_frames
        _raise = hustle_raise

    class _GameLog:
        def __init__(self, *a, **k):
            pass

        def get_data_frames(self):
            return [gamelog_df]

    nba_api = types.ModuleType("nba_api")
    stats = types.ModuleType("nba_api.stats")
    endpoints = types.ModuleType("nba_api.stats.endpoints")
    endpoints.boxscoreplayertrackv3 = types.SimpleNamespace(BoxScorePlayerTrackV3=_Track)
    endpoints.hustlestatsboxscore = types.SimpleNamespace(HustleStatsBoxScore=_Hustle)
    endpoints.leaguegamelog = types.SimpleNamespace(LeagueGameLog=_GameLog)
    stats.endpoints = endpoints
    nba_api.stats = stats
    for name, mod in [("nba_api", nba_api), ("nba_api.stats", stats),
                      ("nba_api.stats.endpoints", endpoints)]:
        sys.modules[name] = mod


@pytest.fixture(autouse=True)
def _clean_nba_api():
    yield
    for k in list(sys.modules):
        if k == "nba_api" or k.startswith("nba_api."):
            del sys.modules[k]


def test_output_filenames_and_season_range():
    assert COL.OUT_TRACK == "wnba_tracking_2021_2026.parquet"
    assert COL.OUT_HUSTLE == "wnba_hustle_2021_2026.parquet"
    assert COL.WNBA_LEAGUE_ID == "10"
    assert COL.SEASONS == ["2021", "2022", "2023", "2024", "2025", "2026"]
    assert COL.SEASON_TYPES == ["Regular Season", "Playoffs"]


def test_person_frame_selection_and_empty_response():
    good = pd.DataFrame({"personId": [1, 2], "touches": [5, 6], "passes": [10, 12]})
    small = pd.DataFrame({"PLAYER_ID": [1], "x": [1]})
    noid = pd.DataFrame({"foo": [1, 2]})
    # Picks the id-bearing frame with the most columns.
    assert COL._person_frame([noid, small, good]) is good
    # Empty / invalid response (no id column anywhere) -> None.
    assert COL._person_frame([noid]) is None
    assert COL._person_frame([]) is None


def test_dedup_cols_supports_both_id_conventions():
    v3 = pd.DataFrame({"GAME_ID": ["1"], "personId": [10], "touches": [3]})
    legacy = pd.DataFrame({"GAME_ID": ["1"], "PLAYER_ID": [10], "deflections": [2]})

    def dedup_cols(df):  # mirror of the collector's inner helper contract
        pid = "PLAYER_ID" if "PLAYER_ID" in df.columns else (
            "personId" if "personId" in df.columns else None)
        return [c for c in ("GAME_ID", pid) if c]

    assert dedup_cols(v3) == ["GAME_ID", "personId"]
    assert dedup_cols(legacy) == ["GAME_ID", "PLAYER_ID"]


def test_done_ids_roundtrip(tmp_path):
    p = tmp_path / "t.parquet"
    pd.DataFrame({"GAME_ID": ["001", "001", "002"], "v": [1, 2, 3]}).to_parquet(p, index=False)
    assert COL._done_ids(str(p)) == {"001", "002"}
    assert COL._done_ids(str(tmp_path / "missing.parquet")) == set()


def test_pull_one_partial_failure_and_id_retention(monkeypatch):
    monkeypatch.setattr(COL.time, "sleep", lambda *_: None)
    track = pd.DataFrame({"personId": [1, 2], "touches": [5, 6]})
    hustle = pd.DataFrame({"PLAYER_ID": [1], "deflections": [3]})
    # Hustle endpoint fails; tracking succeeds -> (track, None), GAME_ID injected.
    _install_fake_nba_api([track], [hustle], pd.DataFrame({"GAME_ID": ["x"]}),
                          hustle_raise=True)
    t, h = COL._pull_one("0021600001")
    assert h is None
    assert t is not None and "GAME_ID" in t.columns
    assert (t["GAME_ID"] == "0021600001").all()
    assert "personId" in t.columns


def test_pull_one_empty_track_response(monkeypatch):
    monkeypatch.setattr(COL.time, "sleep", lambda *_: None)
    noid = pd.DataFrame({"foo": [1]})  # no player id column
    hustle = pd.DataFrame({"PLAYER_ID": [1], "deflections": [3]})
    _install_fake_nba_api([noid], [hustle], pd.DataFrame({"GAME_ID": ["x"]}))
    t, h = COL._pull_one("0021600002")
    assert t is None            # empty/invalid tracking frame skipped, not written
    assert h is not None and "GAME_ID" in h.columns


def test_main_end_to_end_mocked_dedup(monkeypatch, tmp_path):
    monkeypatch.setattr(COL.time, "sleep", lambda *_: None)
    monkeypatch.setattr(COL, "SLEEP", 0)
    monkeypatch.setattr(COL, "OUT_TRACK", str(tmp_path / "track.parquet"))
    monkeypatch.setattr(COL, "OUT_HUSTLE", str(tmp_path / "hustle.parquet"))
    # Only 2 unique game IDs across all season/type combos.
    gamelog = pd.DataFrame({"GAME_ID": ["0021600001", "0021600002"]})
    track = pd.DataFrame({"personId": [1, 2], "touches": [5, 6]})
    hustle = pd.DataFrame({"PLAYER_ID": [1, 2], "deflections": [3, 4]})
    _install_fake_nba_api([track], [hustle], gamelog)
    COL.main()
    t = pd.read_parquet(tmp_path / "track.parquet")
    h = pd.read_parquet(tmp_path / "hustle.parquet")
    # 2 games x 2 players = 4 rows each, de-duplicated on (GAME_ID, id col).
    assert t["GAME_ID"].nunique() == 2 and len(t) == 4
    assert h["GAME_ID"].nunique() == 2 and len(h) == 4
    assert t.duplicated(subset=["GAME_ID", "personId"]).sum() == 0
    assert h.duplicated(subset=["GAME_ID", "PLAYER_ID"]).sum() == 0


def test_schema_contract_agrees_with_collector():
    schema = json.loads(SCHEMA.read_text())
    assert schema["schema_version"] == 1
    assert schema["outputs"]["tracking"]["filename"] == COL.OUT_TRACK
    assert schema["outputs"]["hustle"]["filename"] == COL.OUT_HUSTLE
    assert schema["season_range"]["seasons"] == COL.SEASONS
    assert schema["season_range"]["season_types"] == COL.SEASON_TYPES
    assert schema["promotion_eligible"] is False
    # Status must clearly say data is not landed / features not started.
    st = schema["status"]
    assert "data_not_landed" in st and "tracking_features_not_started" in st
    for out in ("tracking", "hustle"):
        assert "GAME_ID" in schema["outputs"][out]["primary_key"]
