"""Fast synthetic proof points for Phase-3 availability, labels, and minutes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.sharp_v6.availability_policy import (
    AvailabilityStatus,
    decide_availability,
    normalize_status,
)
from wnba_props_model.sharp_v6.bundle import load_bundle
from wnba_props_model.sharp_v6.phase3_labels import (
    LabelRevalidationError,
    injury_conditioned_training_cohort,
    revalidate_participation_labels,
)
from wnba_props_model.sharp_v6.phase3_minutes import (
    ROLE_STATES,
    active_minutes_cohort,
    enforce_monotone_survival,
    fit_minutes_candidate,
    minute_features,
    role_state_from_minutes,
    survival_to_pmf,
)
from wnba_props_model.sharp_v6.phase3_participation import (
    PROHIBITED_FEATURES,
    chronological_folds,
    evaluate_chronological,
)

REPO = Path(__file__).resolve().parents[1]
PHASE2_PY = [
    "scripts/build_canonical_tables.py",
    "scripts/collect_roster_injury_snapshots.py",
    "scripts/phase2_rebuild_canonical_and_labels.py",
    "src/wnba_props_model/constants.py",
    "src/wnba_props_model/data/injury_workbook.py",
    "src/wnba_props_model/data/normalize.py",
    "src/wnba_props_model/data/participation_labels.py",
    "src/wnba_props_model/data/schema.py",
    "tests/test_phase2_canonical_participation_repair.py",
]


def _load_collector():
    path = REPO / "scripts/collect_roster_injury_snapshots.py"
    spec = importlib.util.spec_from_file_location("collect_roster_injury_snapshots", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_phase2_files_are_ruff_clean():
    import subprocess

    files = [str(REPO / p) for p in PHASE2_PY]
    fmt = subprocess.run(
        ["python3", "-m", "ruff", "format", "--check", *files],
        capture_output=True,
        text=True,
        check=False,
    )
    chk = subprocess.run(
        ["python3", "-m", "ruff", "check", *files], capture_output=True, text=True, check=False
    )
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr
    assert chk.returncode == 0, chk.stdout + chk.stderr


def test_snapshot_failure_is_not_healthy_empty():
    mod = _load_collector()
    assert mod.classify_payload([], status_hint=mod.ENDPOINT_FAILURE) == mod.ENDPOINT_FAILURE
    assert mod.classify_bdl_error(Exception("401 unauthorized")) == mod.AUTHENTICATION_FAILURE
    assert normalize_status(None, snapshot_success=False) == AvailabilityStatus.UNKNOWN_SOURCE
    assert decide_availability(None, snapshot_success=False).should_abstain


@pytest.mark.parametrize("status", [AvailabilityStatus.OUT, AvailabilityStatus.SUSPENDED])
def test_out_is_not_priced(status):
    got = decide_availability(status, snapshot_success=True)
    assert got.should_abstain and got.p_active == 0 and got.reason == "ABSTAIN_PLAYER_OUT"


def test_not_listed_is_conditional_active_without_historical_calibration():
    got = decide_availability("NOT_LISTED", snapshot_success=True)
    assert (got.action, got.p_active, got.dnp_mass) == ("ACTIVE_CONDITIONAL", 1.0, 0.0)
    assert got.historically_calibrated is False


def test_unspecified_does_not_override_model():
    got = decide_availability(None, snapshot_success=True)
    assert got.action == "MODEL_DEFAULT"
    assert got.p_active is None


def test_doubtful_requires_domain_support():
    assert decide_availability("DOUBTFUL", snapshot_success=True).should_abstain
    assert (
        decide_availability("DOUBTFUL", snapshot_success=True, injury_model_in_domain=True).action
        == "INJURY_MODEL"
    )


def test_confirmed_inactive_requires_evidence_and_box_absence_alone_is_not_negative():
    labels = pd.DataFrame(
        {
            "game_id": [1, 2],
            "player_id": [10, 11],
            "game_date": ["2025-01-01", "2025-01-02"],
            "season": [2025, 2025],
            "team_id": [1, 1],
            "participation_label_class": ["CONFIRMED_INACTIVE", "UNKNOWN_ROSTER_ELIGIBILITY"],
            "label_source": ["injury_workbook_exact_identity", "box_score_player_stats"],
            "evidence_timestamp": ["2024-12-20", None],
            "training_eligible": [True, False],
            "training_weight": [1.0, 0.0],
            "participation_binary_label": [0, None],
        }
    )
    stats = pd.DataFrame(
        {
            "game_id": [1, 2],
            "player_id": [10, 99],
            "minutes": [0.0, 0.0],
            "team_id": [1, 1],
        }
    )
    out, summary, rejects = revalidate_participation_labels(labels, stats)
    assert out.loc[out.player_id == 10, "participation_label_class"].iloc[0] == "CONFIRMED_INACTIVE"
    assert (
        out.loc[out.player_id == 11, "participation_label_class"].iloc[0]
        == "UNKNOWN_ROSTER_ELIGIBILITY"
    )
    assert out.loc[out.player_id == 11, "training_weight"].iloc[0] == 0
    assert summary["rejected_inactive_labels"] == 0
    assert rejects.empty or "conflicting_active_appearance" not in set(rejects.reject_reason)


def test_inferred_and_unknown_never_enter_fitting_cohort():
    labels = pd.DataFrame(
        {
            "participation_label_class": [
                "CONFIRMED_ACTIVE",
                "CONFIRMED_INACTIVE",
                "INFERRED_ELIGIBLE_DNP",
                "UNKNOWN_ROSTER_ELIGIBILITY",
            ],
            "label_source": [
                "injury_workbook_exact_identity",
                "injury_workbook_exact_identity",
                "box_score_player_stats",
                "box_score_player_stats",
            ],
            "training_eligible": [True, True, False, False],
            "training_weight": [1.0, 1.0, 0.0, 0.0],
            "actual_minutes": [20, 0, 0, 0],
        }
    )
    cohort = injury_conditioned_training_cohort(labels)
    assert set(cohort.participation_label_class) <= {"CONFIRMED_ACTIVE", "CONFIRMED_INACTIVE"}
    assert len(active_minutes_cohort(labels)) == 1


@pytest.mark.parametrize("bad", sorted(PROHIBITED_FEATURES)[:4])
def test_return_date_and_games_missed_blocked_from_participation_features(bad):
    from wnba_props_model.sharp_v6.phase3_participation import _features

    with pytest.raises(ValueError):
        _features(pd.DataFrame({bad: [1.0], "player_minutes_last1": [1.0]}), [bad])


@pytest.mark.parametrize(
    "bad", ["date_returned", "total_games_missed", "actual_minutes", "is_starter", "market_odds"]
)
def test_minutes_leakage_features_are_blocked(bad):
    with pytest.raises(ValueError):
        minute_features(pd.DataFrame({bad: [1, 2], "player_minutes_last1": [1, 2]}), [bad])


def test_chronological_splits_keep_dates_together_and_no_self_calibration_path():
    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "participation_label_class": ["CONFIRMED_ACTIVE"] * 40,
            "participation_binary_label": [1] * 30 + [0] * 10,
            "label_source": ["injury_workbook_exact_identity"] * 40,
            "training_eligible": [True] * 40,
            "player_minutes_last1": np.linspace(5, 30, 40),
            "game_id": range(40),
            "player_id": [1] * 40,
        }
    )
    folds = list(chronological_folds(df))
    assert folds
    for _, train, test in folds:
        assert df.loc[train, "game_date"].max() < df.loc[test, "game_date"].min()
    metrics, oof = evaluate_chronological(df, feature_cols=["player_minutes_last1"], max_year=2025)
    assert not metrics.empty
    # Each OOF row belongs to exactly one fold for a candidate — no self-calibration identity.
    assert oof.groupby(["candidate", "row"]).size().max() == 1


def test_minutes_training_active_only_and_ot_not_clipped():
    df = pd.DataFrame(
        {
            "participation_label_class": [
                "CONFIRMED_ACTIVE",
                "CONFIRMED_INACTIVE",
                "CONFIRMED_ACTIVE",
            ],
            "training_eligible": [True, True, True],
            "actual_minutes": [47, 0, 12],
        }
    )
    cohort = active_minutes_cohort(df)
    assert set(cohort.actual_minutes) == {47, 12}
    assert cohort.actual_minutes.max() == 47


def test_ordinal_monotone_normalized_and_role_mixture_sums():
    raw = np.array([[0.9, 0.8, 0.2] + [0.1] * 37])
    surv = enforce_monotone_survival(raw)
    pmf = survival_to_pmf(surv)
    assert np.all(np.diff(surv, axis=1) <= 1e-12)
    assert np.all(pmf >= 0) and np.allclose(pmf.sum(axis=1), 1)
    # Role mixture probabilities
    roles = role_state_from_minutes(np.array([32, 24, 14, 8, 20, 2]))
    assert roles.min() >= 0 and roles.max() < len(ROLE_STATES)
    train = pd.DataFrame(
        {
            "participation_label_class": ["CONFIRMED_ACTIVE"] * 60,
            "training_eligible": [True] * 60,
            "actual_minutes": np.clip(np.linspace(4, 38, 60) + np.sin(np.arange(60)), 1, 45),
            "player_minutes_last1": np.linspace(4, 36, 60),
            "is_home": np.resize([0.0, 1.0], 60),
            "game_date": pd.date_range("2025-05-01", periods=60, freq="D"),
            "game_id": range(60),
            "player_id": np.resize(np.arange(10), 60),
        }
    )
    model = fit_minutes_candidate(
        train, "role_mixture", feature_cols=["player_minutes_last1", "is_home"]
    )
    probs = model.models["classifier"].predict_proba(train[["player_minutes_last1", "is_home"]])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    pmf2 = model.pmf(train.head(5))
    assert np.all(pmf2 >= 0) and np.allclose(pmf2.sum(axis=1), 1.0, atol=1e-6)
    assert pmf2.shape[1] > 40  # OT support retained


def test_v1_1_remains_loadable():
    path = REPO / "artifacts/releases/wnba-pmf-production-v1.1"
    if not path.exists():
        pytest.skip("v1.1 bundle not present in workspace")
    bundle = load_bundle(path)
    assert bundle.minutes is not None
    assert bundle.participation is not None


def test_no_private_workbook_or_api_payloads_in_git_artifacts():
    # Guardrail: sharp_v6_phase3 committed artifacts must not look like raw API dumps.
    root = REPO / "artifacts/sharp_v6_phase3"
    if not root.exists():
        return
    banned_substrings = ('"payload":', "BDL_API_KEY", "WNBA Injuries")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".csv", ".md"}:
            text = p.read_text(errors="ignore")[:200_000]
            for bad in banned_substrings:
                assert bad not in text, f"{p} contains banned private content marker {bad}"


def test_inactive_reject_gate():
    labels = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "player_id": [1, 2, 3],
            "game_date": ["2025-01-01"] * 3,
            "season": [2025] * 3,
            "team_id": [1] * 3,
            "participation_label_class": ["CONFIRMED_INACTIVE"] * 3,
            "label_source": ["injury_workbook_exact_identity"] * 3,
            "evidence_timestamp": ["2024-12-01"] * 3,
            "training_eligible": [True] * 3,
            "training_weight": [1.0] * 3,
            "participation_binary_label": [0] * 3,
        }
    )
    # All three have positive minutes → 100% reject > 2%
    stats = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "player_id": [1, 2, 3],
            "minutes": [20, 21, 22],
            "team_id": [1, 1, 1],
        }
    )
    with pytest.raises(LabelRevalidationError):
        revalidate_participation_labels(labels, stats)


def test_downstream_tolerances_frozen_before_comparison():
    path = REPO / "artifacts/sharp_v6_phase3/DOWNSTREAM_TOLERANCES.json"
    assert path.exists()
    tol = json.loads(path.read_text())
    assert tol.get("frozen_before_comparison") is True
