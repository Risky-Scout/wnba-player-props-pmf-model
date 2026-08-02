"""Phase-2 canonical repair + participation-label contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wnba_props_model.data.injury_workbook import (
    LEAKAGE_PROHIBITED_FEATURES,
    assert_no_onset_leakage,
    eligibility_evidence_from_injury_events,
    load_injury_events_from_rows,
    match_athlete_exact,
)
from wnba_props_model.data.normalize import (
    flatten_player_stat_row,
    normalize_player_stats,
    shooting_identity_violations,
)
from wnba_props_model.data.participation_labels import (
    CONFIRMED_ACTIVE,
    CONFIRMED_INACTIVE,
    INFERRED_ELIGIBLE_DNP,
    UNKNOWN_ROSTER_ELIGIBILITY,
    build_conditional_minutes_training_table,
    build_participation_labels,
    classify_box_score_row,
)


def _box_row(**overrides):
    base = {
        "player": {"id": 1, "first_name": "A", "last_name": "Player", "position": "G"},
        "team": {"id": 10, "abbreviation": "NYL"},
        "game": {"id": 100, "date": "2025-06-01", "season": 2025},
        "min": "28",
        "pts": 17,
        "reb": 5,
        "ast": 3,
        "stl": 1,
        "blk": 0,
        "turnover": 2,
        "fgm": 6,
        "fga": 12,
        "fg3m": 1,
        "fg3a": 4,
        "ftm": 4,
        "fta": 4,
        "oreb": 1,
        "dreb": 4,
        "pf": 2,
        "plus_minus": 3,
    }
    base.update(overrides)
    return base


def test_fgm_ftm_survive_normalization():
    flat = flatten_player_stat_row(_box_row())
    assert flat["fgm"] == 6
    assert flat["ftm"] == 4
    assert flat["fg2m"] == 5
    assert flat["fg2a"] == 8
    df = normalize_player_stats([_box_row()])
    assert int(df.loc[0, "fgm"]) == 6
    assert int(df.loc[0, "ftm"]) == 4


def test_shooting_identities_reconcile():
    df = normalize_player_stats([_box_row()])
    viol = shooting_identity_violations(df)
    assert viol["rows_evaluated"] == 1
    assert viol["pts_identity_violation"] == 0
    assert viol["fg2m_gt_fg2a"] == 0
    assert viol["ftm_gt_fta"] == 0


def test_official_reb_remains_primary_and_discrepancy_preserved():
    # Official reb=7 but oreb+dreb=5 → do not rewrite reb
    flat = flatten_player_stat_row(_box_row(reb=7, oreb=1, dreb=4))
    assert flat["reb"] == 7
    assert flat["reb_oreb_dreb_sum"] == 5
    assert flat["reb_reconcile_flag"] == "provider_or_team_reb_discrepancy"
    flat_ok = flatten_player_stat_row(_box_row(reb=5, oreb=1, dreb=4))
    assert flat_ok["reb_reconcile_flag"] == "match"


def test_confirmed_active_classification():
    packed = classify_box_score_row(minutes=22.0)
    assert packed["participation_label_class"] == CONFIRMED_ACTIVE
    assert packed["participation_binary_label"] == 1
    assert packed["training_eligible"] is True
    assert packed["training_weight"] == 1.0


def test_confirmed_inactive_requires_eligibility_evidence():
    no_evid = classify_box_score_row(minutes=0.0, minutes_flag="non_playing")
    assert no_evid["participation_label_class"] == INFERRED_ELIGIBLE_DNP
    assert no_evid["training_eligible"] is False

    with_evid = classify_box_score_row(
        minutes=0.0,
        minutes_flag="non_playing",
        eligibility_evidence={
            "on_eligible_roster": True,
            "injury_interval": True,
            "evidence_timestamp": "2025-05-01",
            "label_source": "injury_workbook_exact_identity",
        },
    )
    assert with_evid["participation_label_class"] == CONFIRMED_INACTIVE
    assert with_evid["participation_binary_label"] == 0
    assert with_evid["training_eligible"] is True


def test_absent_box_score_row_alone_cannot_create_confirmed_inactive():
    packed = classify_box_score_row(
        minutes=0.0,
        eligibility_evidence={"box_score_row": False},
    )
    assert packed["participation_label_class"] == UNKNOWN_ROSTER_ELIGIBILITY
    assert packed["participation_binary_label"] is None
    assert packed["training_eligible"] is False


def test_inferred_dnp_excluded_from_training_by_default():
    packed = classify_box_score_row(minutes=0.0)
    assert packed["participation_label_class"] == INFERRED_ELIGIBLE_DNP
    assert packed["training_eligible"] is False
    assert packed["training_weight"] == 0.0


def test_unknown_eligibility_has_no_binary_training_label():
    packed = classify_box_score_row(
        minutes=0.0,
        eligibility_evidence={"box_score_row": False},
    )
    assert packed["participation_label_class"] == UNKNOWN_ROSTER_ELIGIBILITY
    assert packed["participation_binary_label"] is None


def test_injury_return_and_games_missed_prohibited_as_onset_features():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_onset_leakage(["player_minutes_mean_l5", "date_returned"])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_onset_leakage(["total_games_missed"])
    assert LEAKAGE_PROHIBITED_FEATURES == {"date_returned", "total_games_missed"}
    assert_no_onset_leakage(["player_minutes_mean_l5", "player_rest_days"])


def test_unresolved_workbook_identities_cannot_create_labels():
    events = load_injury_events_from_rows(
        [
            {
                "athlete": "Unknown Person",
                "team": "NYL",
                "date_injured": "2025-06-01",
                "date_returned": "2025-06-20",
                "total_games_missed": 5,
                "season_sheet": 2025,
            }
        ],
        roster_name_to_ids={"a player": [1]},
    )
    assert events.loc[0, "identity_status"] == "unresolved"
    panel = pd.DataFrame(
        [
            {
                "game_id": 100,
                "player_id": 1,
                "game_date": "2025-06-05",
                "minutes": 0.0,
            }
        ]
    )
    evid = eligibility_evidence_from_injury_events(events, panel)
    assert evid.empty


def test_ambiguous_name_cannot_confirm_inactive():
    match = match_athlete_exact("Jane Doe", {"jane doe": [1, 2]})
    assert match.status == "ambiguous_name"
    assert match.player_id is None


def test_conditional_minutes_table_active_only_and_no_target_leakage():
    labels = build_participation_labels(
        pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "game_date": "2025-06-01",
                    "season": 2025,
                    "player_id": 1,
                    "team_id": 10,
                    "minutes": 30.0,
                    "minutes_flag": None,
                },
                {
                    "game_id": 1,
                    "game_date": "2025-06-01",
                    "season": 2025,
                    "player_id": 2,
                    "team_id": 10,
                    "minutes": 0.0,
                    "minutes_flag": None,
                },
            ]
        )
    )
    feats = pd.DataFrame(
        [
            {"game_id": 1, "player_id": 1, "player_minutes_mean_l5": 28.0},
            {"game_id": 1, "player_id": 2, "player_minutes_mean_l5": 10.0},
        ]
    )
    with pytest.raises(ValueError, match="leakage"):
        build_conditional_minutes_training_table(
            labels,
            feats,
            feature_cols=["player_minutes_mean_l5", "actual_minutes"],
            feature_cutoff="t",
            data_hash="x",
            feature_contract_hash="y",
        )
    table = build_conditional_minutes_training_table(
        labels,
        feats,
        feature_cols=["player_minutes_mean_l5"],
        feature_cutoff="prior_game_date_shift1",
        data_hash="abc",
        feature_contract_hash="def",
    )
    assert len(table) == 1
    assert int(table.iloc[0]["player_id"]) == 1
    assert float(table.iloc[0]["actual_minutes"]) == 30.0


def test_prospective_snapshots_append_only(tmp_path, monkeypatch):
    # Fail-open without API key still writes manifest; second run keeps append-only contract.
    import scripts.collect_roster_injury_snapshots as mod
    from wnba_props_model.data.bdl_client import BDLAPIError

    out_dir = tmp_path / "snaps"
    man = tmp_path / "man.json"

    class _NoClient:
        def __init__(self, *args, **kwargs):
            raise BDLAPIError("BDL_API_KEY is required")

    monkeypatch.setattr(mod, "BDLClient", _NoClient)
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_roster_injury_snapshots.py",
            "--out-dir",
            str(out_dir),
            "--manifest-out",
            str(man),
            "--date",
            "2026-08-02",
        ],
    )
    rc = mod.main()
    assert rc == 2
    assert man.exists()
    payload = json.loads(man.read_text())
    assert payload["append_only"] is True
    assert payload["fail_closed"] is True
    assert payload["fail_open"] is False
    assert payload["private_payloads_committed_to_git"] is False
    assert payload["status"] == "AUTHENTICATION_FAILURE"
    # Failed auth must never be treated as healthy-empty injury/roster data.
    for src in payload["sources"].values():
        assert src.get("healthy_empty") is False
        assert src.get("status") == "AUTHENTICATION_FAILURE"

    # Second invocation must not rewrite prior contract fields to allow commits
    man.write_text(json.dumps({**payload, "run": 1}))
    mod.main()
    payload2 = json.loads(man.read_text())
    assert payload2["append_only"] is True
    assert payload2["private_payloads_committed_to_git"] is False


def test_no_private_injury_rows_or_api_payloads_in_git_paths():
    """Guard: phase2 artifacts must not embed private injury event rows."""
    root = Path("artifacts/phase2_repair")
    if not root.exists():
        pytest.skip("phase2 artifacts not built yet")
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".json", ".csv", ".md"}:
            continue
        text = p.read_text(errors="ignore")
        # Aggregate audits may mention column names; ban private workbook path / row dumps
        assert "Desktop/Technical Portfolio" not in text
        assert "WNBA Injuries .xlsx" not in text
        if p.name.startswith("INJURY_WORKBOOK_NORMALIZED"):
            pytest.fail(f"private normalized injury artifact committed: {p}")


def test_build_participation_labels_training_policy():
    df = pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date": "2025-06-01",
                "season": 2025,
                "player_id": 1,
                "team_id": 10,
                "minutes": 12.0,
                "minutes_flag": None,
            },
            {
                "game_id": 1,
                "game_date": "2025-06-01",
                "season": 2025,
                "player_id": 2,
                "team_id": 10,
                "minutes": 0.0,
                "minutes_flag": None,
            },
        ]
    )
    labels = build_participation_labels(df)
    active = labels[labels["participation_label_class"] == CONFIRMED_ACTIVE].iloc[0]
    inferred = labels[labels["participation_label_class"] == INFERRED_ELIGIBLE_DNP].iloc[0]
    assert bool(active["training_eligible"]) is True
    assert bool(inferred["training_eligible"]) is False
