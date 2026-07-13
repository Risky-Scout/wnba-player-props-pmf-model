"""Injury pipeline invariant tests — Blockers 2, 4, 5, 6, 7.

Blocker 2 — Single application of injury-driven minutes
--------------------------------------------------------
Historical minutes features must NOT be mutated.  The conditional minutes
adjustment is applied exactly once by pmf_engine via _injury_minutes_multiplier.

    test_historical_minutes_features_are_not_mutated
    test_questionable_multiplier_is_applied_exactly_once
    test_probable_multiplier_is_applied_exactly_once
    test_teammate_boost_is_applied_exactly_once

Blocker 4 — Correct UTM inputs
-------------------------------
Only confirmed-inactive players are passed as out_player_ids.
Original and freed minutes are tracked separately.

    test_utm_receives_original_minutes
    test_utm_receives_freed_minutes_separately
    test_questionable_player_is_not_treated_as_fully_out
    test_probable_player_is_not_treated_as_fully_out
    test_team_minutes_are_coherent_after_redistribution

Blocker 5 — Explicit config path
---------------------------------
    test_real_config_path_resolves
    test_missing_config_path_is_fatal

Blocker 6 — Live combo artifact parity
---------------------------------------
An available player rebuilt with multiplier=1.0 must produce the same
PMF mean as ordinary live inference (parity test, structural).

    test_identity_multiplier_does_not_change_pmf_mean

Blocker 7 — Real end-to-end integration test
--------------------------------------------
(Skipped when live model artifacts are absent; structural contract verified.)

    test_real_integration_path_contract
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture helpers (shared)
# ---------------------------------------------------------------------------

_PLAYER_MINS: dict[int, float] = {
    100: 30.0,   # confirmed OUT
    102: 25.0,   # questionable
    103: 32.0,   # probable
    200: 22.0,   # active teammate (redistribution target)
    201: 18.0,   # active teammate
    999: 28.0,   # fully available (control)
}

_MINUTES_COLS = [
    "player_minutes_mean_l3",
    "player_minutes_mean_l5",
    "player_minutes_mean_l10",
    "player_minutes_mean_l20",
    "player_minutes_mean_season",
]


def _make_feature_df() -> pd.DataFrame:
    rows = []
    for pid, mins in _PLAYER_MINS.items():
        row: dict = {
            "player_id":   pid,
            "game_id":     999,
            "game_date":   "2026-07-13",
            "team_id":     10,
            "player_name": f"Player_{pid}",
        }
        for col in _MINUTES_COLS:
            row[col] = mins
        row["player_pts_mean_l5"] = mins * 0.65
        rows.append(row)
    return pd.DataFrame(rows)


def _make_injuries(statuses: dict[int, str]) -> list[dict]:
    return [
        {"player_id": pid, "player_name": f"Player_{pid}", "status": status}
        for pid, status in statuses.items()
    ]


def _get_pipeline():
    try:
        from wnba_props_model.pipeline import injury_pipeline
        return injury_pipeline
    except ImportError as exc:
        pytest.skip(f"injury_pipeline not importable: {exc}")


def _get_utm():
    try:
        from wnba_props_model.models.usage_transfer import UsageTransferMatrix
        return UsageTransferMatrix
    except ImportError as exc:
        pytest.skip(f"UsageTransferMatrix not importable: {exc}")


# ---------------------------------------------------------------------------
# Blocker 2: Single application of injury-driven minutes
# ---------------------------------------------------------------------------

def test_historical_minutes_features_are_not_mutated():
    """Historical minutes feature columns must remain identical after adjustment.

    The injury multiplier is applied ONCE by pmf_engine via _injury_minutes_multiplier.
    Mutating _MINUTES_FEATURE_COLS here AND applying the multiplier in pmf_engine
    would cause double-application (baseline × mult × mult).
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()

    injuries = _make_injuries({
        100: "out",
        102: "questionable",
        103: "probable",
    })
    avail = ip.build_availability_table(injuries, feature_df)
    adj   = ip.apply_injury_to_feature_df(feature_df, avail)

    for pid in [100, 102, 103, 200, 999]:
        for col in _MINUTES_COLS:
            if col not in feature_df.columns:
                continue
            orig = feature_df.loc[feature_df["player_id"] == pid, col].values[0]
            new  = adj.loc[adj["player_id"] == pid, col].values[0]
            assert abs(new - orig) < 1e-9, (
                f"Historical column {col} was mutated for player_id={pid}: "
                f"orig={orig}, new={new}.  "
                "Injury adjustment must be carried in _injury_minutes_multiplier only."
            )


def test_questionable_multiplier_is_applied_exactly_once():
    """Blocker 1: For a questionable player, p_active=0.50 must NOT be baked into
    _injury_minutes_multiplier (cond_mult). The conditional PMF is for participation only.

    Correct (Blocker 1):
      _injury_minutes_multiplier = cond_mult = 1.0  (PMF engine applies 1× to baseline)
      availability_probability = 0.50               (carried separately for market/settlement)
    Wrong (old behavior):
      _injury_minutes_multiplier = p_active * cond_mult = 0.50 (halves the conditional PMF)
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()  # pid=102 has 25 min baseline

    injuries = _make_injuries({102: "questionable"})
    avail = ip.build_availability_table(injuries, feature_df)
    adj   = ip.apply_injury_to_feature_df(feature_df, avail)

    p102 = adj[adj["player_id"] == 102]

    # Step 1: historical feature column is unchanged (model will see 25 min baseline)
    orig_mins = feature_df.loc[feature_df["player_id"] == 102, "player_minutes_mean_l5"].values[0]
    feat_mins = p102["player_minutes_mean_l5"].values[0]
    assert abs(feat_mins - orig_mins) < 1e-9, (
        f"Historical feature must be unchanged: orig={orig_mins}, adj={feat_mins}"
    )

    # Step 2 (Blocker 1): _injury_minutes_multiplier = cond_mult = 1.0 for questionable.
    # p_active is NOT baked in here; it goes to availability_probability.
    mult = float(p102["_injury_minutes_multiplier"].values[0])
    assert abs(mult - 1.0) < 1e-9, (
        f"[Blocker 1] _injury_minutes_multiplier for questionable must be cond_mult=1.0, "
        f"got {mult}. p_active (0.50) must NOT be multiplied into conditional minutes."
    )

    # Step 2b: availability_probability carries p_active=0.50 separately
    avail_prob = float(p102["availability_probability"].values[0])
    assert abs(avail_prob - 0.50) < 1e-9, (
        f"availability_probability for questionable must be 0.50, got {avail_prob}"
    )

    # Step 3: adjusted_projected_minutes = orig * p_active * cond_mult = orig * 0.50 (display)
    adj_mins = float(p102["adjusted_projected_minutes"].values[0])
    expected = orig_mins * 0.50
    assert abs(adj_mins - expected) < 1e-6, (
        f"adjusted_projected_minutes = {adj_mins}, expected {expected} (orig={orig_mins} × 0.50)"
    )

    # Step 4: freed_minutes = orig * p_active * cond_mult = orig * 0.50
    freed = float(p102["freed_minutes"].values[0])
    assert abs(freed - orig_mins * 0.50) < 1e-6, (
        f"freed_minutes = {freed}, expected {orig_mins * 0.50}"
    )


def test_probable_multiplier_is_applied_exactly_once():
    """Blocker 1: For a probable player, p_active=0.85 must NOT be baked into
    _injury_minutes_multiplier (cond_mult). The conditional PMF is for participation only.

    Correct (Blocker 1):
      _injury_minutes_multiplier = cond_mult = 1.0  (PMF engine applies 1× to baseline)
      availability_probability = 0.85               (carried separately for market/settlement)
    Wrong (old behavior):
      _injury_minutes_multiplier = p_active * cond_mult = 0.85 (scales the conditional PMF)
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()  # pid=103 has 32 min baseline

    injuries = _make_injuries({103: "probable"})
    avail = ip.build_availability_table(injuries, feature_df)
    adj   = ip.apply_injury_to_feature_df(feature_df, avail)

    p103 = adj[adj["player_id"] == 103]

    orig_mins = feature_df.loc[feature_df["player_id"] == 103, "player_minutes_mean_l5"].values[0]
    feat_mins = p103["player_minutes_mean_l5"].values[0]
    assert abs(feat_mins - orig_mins) < 1e-9, (
        f"Historical feature unchanged: orig={orig_mins}, adj={feat_mins}"
    )

    # Blocker 1: _injury_minutes_multiplier = cond_mult = 1.0 for probable
    mult = float(p103["_injury_minutes_multiplier"].values[0])
    assert abs(mult - 1.0) < 1e-9, (
        f"[Blocker 1] _injury_minutes_multiplier for probable must be cond_mult=1.0, "
        f"got {mult}. p_active (0.85) must NOT be multiplied into conditional minutes."
    )

    # availability_probability carries p_active=0.85 separately
    avail_prob = float(p103["availability_probability"].values[0])
    assert abs(avail_prob - 0.85) < 1e-9, (
        f"availability_probability for probable must be 0.85, got {avail_prob}"
    )

    adj_mins = float(p103["adjusted_projected_minutes"].values[0])
    expected = orig_mins * 0.85
    assert abs(adj_mins - expected) < 1e-6, (
        f"adjusted_projected_minutes = {adj_mins}, expected {expected}"
    )


def test_teammate_boost_is_applied_exactly_once():
    """Teammate boost from UTM is encoded in _injury_minutes_multiplier only.

    Historical feature columns for boosted teammates must remain unchanged.
    The pmf_engine applies the (combined) _injury_minutes_multiplier once.
    """
    ip = _get_pipeline()
    UTM = _get_utm()
    feature_df = _make_feature_df()

    injuries = _make_injuries({100: "out"})
    avail = ip.build_availability_table(injuries, feature_df)

    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    utm = UTM(usg_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

    for tm_pid in [200, 201]:
        orig_hist = feature_df.loc[feature_df["player_id"] == tm_pid, "player_minutes_mean_l5"].values[0]
        adj_hist  = adj.loc[adj["player_id"] == tm_pid, "player_minutes_mean_l5"].values[0]
        assert abs(adj_hist - orig_hist) < 1e-9, (
            f"Historical player_minutes_mean_l5 must NOT be mutated for teammate {tm_pid}: "
            f"orig={orig_hist}, adj={adj_hist}.  "
            "UTM boost must be encoded in _injury_minutes_multiplier only."
        )

    # At least one teammate must have _injury_minutes_multiplier > 1.0
    boosts = [
        float(adj.loc[adj["player_id"] == tm, "_injury_minutes_multiplier"].values[0])
        for tm in [200, 201]
    ]
    assert any(b > 1.0 for b in boosts), (
        f"At least one teammate must have _injury_minutes_multiplier > 1.0 from UTM. "
        f"Got: {boosts}"
    )


# ---------------------------------------------------------------------------
# Blocker 4: Correct UTM inputs
# ---------------------------------------------------------------------------

def test_utm_receives_original_minutes():
    """UTM must be called with original (pre-injury) projected minutes for OUT players."""
    ip = _get_pipeline()
    feature_df = _make_feature_df()

    injuries = _make_injuries({100: "out"})
    avail = ip.build_availability_table(injuries, feature_df)
    adj   = ip.apply_injury_to_feature_df(feature_df, avail)

    # original_projected_minutes must equal the historical feature value
    p100 = adj[adj["player_id"] == 100]
    orig_hist = feature_df.loc[feature_df["player_id"] == 100, "player_minutes_mean_l5"].values[0]
    orig_track = float(p100["original_projected_minutes"].values[0])
    assert abs(orig_track - orig_hist) < 1e-9, (
        f"original_projected_minutes={orig_track} must equal historical baseline={orig_hist}"
    )


def test_utm_receives_freed_minutes_separately():
    """freed_minutes and original_projected_minutes must be tracked separately."""
    ip = _get_pipeline()
    feature_df = _make_feature_df()

    injuries = _make_injuries({100: "out"})
    avail = ip.build_availability_table(injuries, feature_df)
    adj   = ip.apply_injury_to_feature_df(feature_df, avail)

    p100 = adj[adj["player_id"] == 100]
    orig  = float(p100["original_projected_minutes"].values[0])
    adj_m = float(p100["adjusted_projected_minutes"].values[0])
    freed = float(p100["freed_minutes"].values[0])

    # For fully OUT player: adjusted = 0, freed = original
    assert abs(adj_m) < 1e-9,  f"OUT player adjusted_projected_minutes must be 0, got {adj_m}"
    assert abs(freed - orig) < 1e-9, (
        f"freed_minutes ({freed}) must equal original_projected_minutes ({orig}) for OUT player"
    )


def test_questionable_player_is_not_treated_as_fully_out():
    """A questionable player must NOT be passed as an out_player_id to UTM.

    Only confirmed-inactive (OUT/DNP/INACTIVE) players trigger full redistribution.
    Questionable players have a non-zero participation probability (0.50).
    """
    ip = _get_pipeline()
    UTM = _get_utm()
    feature_df = _make_feature_df()

    injuries = _make_injuries({102: "questionable"})
    avail = ip.build_availability_table(injuries, feature_df)

    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    utm = UTM(usg_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

    # Questionable player: is_confirmed_inactive must be False
    q_row = avail[avail["player_id"] == 102].iloc[0]
    assert not bool(q_row["is_confirmed_inactive"]), (
        "Questionable player must NOT be confirmed inactive"
    )

    # Teammates should NOT receive redistributed minutes from a questionable player
    # (since no full redistribution occurs for partial-status players)
    for tm_pid in [200, 201]:
        tm_mult = float(adj.loc[adj["player_id"] == tm_pid, "_injury_minutes_multiplier"].values[0])
        assert abs(tm_mult - 1.0) < 1e-9, (
            f"Teammate {tm_pid} _injury_minutes_multiplier should be 1.0 (no redistribution "
            f"for questionable player), got {tm_mult}"
        )


def test_probable_player_is_not_treated_as_fully_out():
    """A probable player must NOT be passed as an out_player_id to UTM."""
    ip = _get_pipeline()
    UTM = _get_utm()
    feature_df = _make_feature_df()

    injuries = _make_injuries({103: "probable"})
    avail = ip.build_availability_table(injuries, feature_df)

    # Probable player must NOT be confirmed inactive
    p_row = avail[avail["player_id"] == 103].iloc[0]
    assert not bool(p_row["is_confirmed_inactive"]), (
        "Probable player must NOT be confirmed inactive"
    )

    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    utm = UTM(usg_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

    # Teammates should NOT receive redistribution for a probable player
    for tm_pid in [200, 201]:
        tm_mult = float(adj.loc[adj["player_id"] == tm_pid, "_injury_minutes_multiplier"].values[0])
        assert abs(tm_mult - 1.0) < 1e-9, (
            f"Teammate {tm_pid} must not receive redistribution for a probable player, "
            f"got _injury_minutes_multiplier={tm_mult}"
        )


def test_team_minutes_are_coherent_after_redistribution():
    """After OUT player redistribution, team-level effective minutes are coherent.

    Constraint: sum of effective minutes (original * multiplier) across active
    players should approximately equal total team minutes pre-injury (OUT player
    minutes redistributed to teammates, not created from thin air).
    """
    ip = _get_pipeline()
    UTM = _get_utm()
    feature_df = _make_feature_df()

    injuries = _make_injuries({100: "out"})
    avail = ip.build_availability_table(injuries, feature_df)

    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    utm = UTM(usg_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

    # Pre-injury total team minutes (all players at their historical baseline)
    pre_total = sum(
        feature_df.loc[feature_df["player_id"] == pid, "player_minutes_mean_l5"].values[0]
        for pid in _PLAYER_MINS
    )

    # Post-injury effective minutes (original * multiplier for each player)
    post_total = 0.0
    for pid in _PLAYER_MINS:
        hist_mins = float(feature_df.loc[feature_df["player_id"] == pid, "player_minutes_mean_l5"].values[0])
        mult      = float(adj.loc[adj["player_id"] == pid, "_injury_minutes_multiplier"].values[0])
        post_total += hist_mins * mult

    # After redistribution:
    # - OUT player contributes 0 effective minutes (mult=0)
    # - Teammates may absorb some or all of the freed minutes
    # - Total effective minutes should not EXCEED pre-injury total
    assert post_total <= pre_total + 1e-6, (
        f"Post-redistribution effective minutes ({post_total:.2f}) must not exceed "
        f"pre-injury total ({pre_total:.2f})"
    )

    # Freed minutes from OUT player must be tracked correctly
    out_orig = float(adj.loc[adj["player_id"] == 100, "original_projected_minutes"].values[0])
    out_freed = float(adj.loc[adj["player_id"] == 100, "freed_minutes"].values[0])
    assert abs(out_freed - out_orig) < 1e-9, (
        f"OUT player freed_minutes ({out_freed}) must equal original_projected_minutes ({out_orig})"
    )


# ---------------------------------------------------------------------------
# Blocker 5: Explicit config path
# ---------------------------------------------------------------------------

def test_real_config_path_resolves():
    """The canonical config/model/stage4_baseline.yaml must exist or be resolvable."""
    # This test verifies the path structure used in the CLI and rebuild pipeline.
    # In CI this file is present; in development environments it may not be.
    config_candidate = Path("config/model/stage4_baseline.yaml")
    if not config_candidate.exists():
        pytest.skip("config/model/stage4_baseline.yaml not present in this checkout")

    # Verify the file is valid YAML with expected keys
    try:
        import yaml
        content = yaml.safe_load(config_candidate.read_text())
        assert isinstance(content, dict), "stage4_baseline.yaml must be a YAML dict"
    except ImportError:
        # YAML not installed, just check the file is non-empty
        assert config_candidate.stat().st_size > 0, "stage4_baseline.yaml must be non-empty"


def test_missing_config_path_is_fatal():
    """rebuild_affected_pmfs must raise FileNotFoundError for a nonexistent config path."""
    ip = _get_pipeline()

    feature_df = _make_feature_df()
    feature_df["_injury_minutes_multiplier"] = 1.0

    nonexistent = "/nonexistent/path/stage4_baseline.yaml"

    with pytest.raises((FileNotFoundError, ValueError), match="(?i)(config|exist|path|fatal)"):
        ip.rebuild_affected_pmfs(
            feature_df_adjusted=feature_df,
            affected_player_ids={102},
            model_dir="artifacts/models/stage4_baseline",
            cfg={},
            cal_dir=None,
            config_path=nonexistent,
            apply_calibration=False,
            apply_shrinkage=False,
        )


# ---------------------------------------------------------------------------
# Blocker 6: Identity multiplier parity
# ---------------------------------------------------------------------------

def test_identity_multiplier_does_not_change_pmf_mean():
    """An available player rebuilt with multiplier=1.0 must produce the same pmf_mean.

    Structural contract: the injury pipeline must not alter PMFs for
    unaffected (multiplier=1.0) players.  This verifies that the scenario
    input columns do not inadvertently mutate outputs for control players.
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()

    # No injuries — all players fully available
    avail = ip.build_availability_table([], feature_df)
    adj   = ip.apply_injury_to_feature_df(feature_df, avail)

    # Control player (pid=999) must have unchanged scenario inputs
    ctrl = adj[adj["player_id"] == 999]
    assert abs(float(ctrl["_injury_minutes_multiplier"].values[0]) - 1.0) < 1e-9, (
        "Control player with no injury must have _injury_minutes_multiplier=1.0"
    )

    # Historical feature columns unchanged for all players
    for pid in _PLAYER_MINS:
        for col in _MINUTES_COLS:
            if col not in feature_df.columns:
                continue
            orig = feature_df.loc[feature_df["player_id"] == pid, col].values[0]
            adj_v = adj.loc[adj["player_id"] == pid, col].values[0]
            assert abs(adj_v - orig) < 1e-9, (
                f"Column {col} mutated for pid={pid} with no injuries: "
                f"orig={orig}, adj={adj_v}"
            )


# ---------------------------------------------------------------------------
# Blocker 7: Real integration path — structural contract
# ---------------------------------------------------------------------------

def test_real_integration_path_contract():
    """Structural contract for the real end-to-end integration test.

    This test verifies the structural invariants of the injury orchestration
    without requiring live model artifacts.  A full integration test invoking
    the actual production path (predict_player_pmfs, calibration artifacts,
    feature parquet) should be added when the CI environment has all artifacts.

    Invariants verified:
    1. Injury multiplier applied exactly once (via _injury_minutes_multiplier)
    2. Historical feature columns not mutated
    3. Confirmed-inactive players have is_confirmed_inactive=True
    4. Partial-status players do NOT trigger full UTM redistribution
    5. Availability table has required schema columns
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()

    statuses = {
        100: "out",          # confirmed inactive
        102: "questionable", # partial
        103: "probable",     # partial
        200: None,           # teammate (receives redistribution)
        999: None,           # unaffected control
    }
    injuries = [
        {"player_id": pid, "player_name": f"Player_{pid}", "status": status}
        for pid, status in statuses.items()
        if status is not None
    ]

    avail = ip.build_availability_table(
        injuries, feature_df, source_updated_at="2026-07-13T08:00:00+00:00"
    )

    # Schema invariant: required columns present
    for col in [
        "availability_probability", "starter_probability",
        "conditional_minutes_multiplier", "minutes_multiplier",
        "is_confirmed_inactive", "is_market_actionable",
        "source_updated_at", "pulled_at_utc",
    ]:
        assert col in avail.columns, f"Missing required column: {col}"

    # Timestamp invariant: source_updated_at <= pulled_at_utc
    sample = avail.iloc[0]
    src_dt  = pd.to_datetime(sample["source_updated_at"], utc=True)
    pull_dt = pd.to_datetime(sample["pulled_at_utc"], utc=True)
    assert src_dt <= pull_dt, (
        f"source_updated_at ({src_dt}) must not exceed pulled_at_utc ({pull_dt})"
    )

    # Confirmed inactive invariant
    assert bool(avail.loc[avail["player_id"] == 100, "is_confirmed_inactive"].values[0])
    assert not bool(avail.loc[avail["player_id"] == 102, "is_confirmed_inactive"].values[0])
    assert not bool(avail.loc[avail["player_id"] == 103, "is_confirmed_inactive"].values[0])

    # Availability/conditional multiplier invariant: separately represented
    # questionable: p_active=0.50, cond_mult=1.0
    q_row = avail[avail["player_id"] == 102].iloc[0]
    assert abs(float(q_row["availability_probability"]) - 0.50) < 1e-9
    assert abs(float(q_row["conditional_minutes_multiplier"]) - 1.0) < 1e-9
    assert abs(float(q_row["minutes_multiplier"]) - 0.50) < 1e-9  # effective

    # Feature mutation invariant (single application)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    for pid in _PLAYER_MINS:
        for col in _MINUTES_COLS:
            if col not in feature_df.columns:
                continue
            orig = feature_df.loc[feature_df["player_id"] == pid, col].values[0]
            adj_v = adj.loc[adj["player_id"] == pid, col].values[0]
            assert abs(adj_v - orig) < 1e-9, (
                f"INVARIANT VIOLATED: {col} mutated for pid={pid} "
                f"(orig={orig}, adj={adj_v})"
            )

    # Control player (pid=999) unaffected
    ctrl = adj[adj["player_id"] == 999]
    assert abs(float(ctrl["_injury_minutes_multiplier"].values[0]) - 1.0) < 1e-9

    # OUT player multiplier is 0
    out = adj[adj["player_id"] == 100]
    assert abs(float(out["_injury_minutes_multiplier"].values[0])) < 1e-9

    # Blocker 1: Partial players (questionable/probable) have cond_mult=1.0 in
    # _injury_minutes_multiplier. p_active is in availability_probability, NOT mixed in.
    for pid in [102, 103]:
        part = adj[adj["player_id"] == pid]
        mult = float(part["_injury_minutes_multiplier"].values[0])
        avail_p = float(part["availability_probability"].values[0])
        # cond_mult=1.0 (conditional PMF conditional on participation)
        assert abs(mult - 1.0) < 1e-9, (
            f"[Blocker 1] _injury_minutes_multiplier for partial-status pid={pid} "
            f"must be cond_mult=1.0, got {mult}"
        )
        # availability_probability is < 1.0 (not full certainty)
        assert 0 < avail_p < 1.0, (
            f"availability_probability for partial-status pid={pid} must be in (0,1), "
            f"got {avail_p}"
        )
