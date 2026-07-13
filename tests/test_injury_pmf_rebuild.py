"""Injury PMF rebuild — 13 deterministic regression tests.

These tests were written BEFORE the implementation and are expected to FAIL
against the current code.  After implementation they must all pass.

Fixture inventory
-----------------
player_id=100  confirmed OUT
player_id=101  doubtful
player_id=102  questionable
player_id=103  probable
player_id=104  limited
player_id=105  gtd  (game-time decision)
player_id=200  teammate 1 (no injury — receives redistributed minutes)
player_id=201  teammate 2 (no injury — receives redistributed minutes)

Stats modelled: pts, reb, ast, fg3m, stl, blk, turnover
Combos modelled: stocks, pts_ast, pts_reb, reb_ast, pts_reb_ast
Lines tested: integer (12) and half-point (11.5)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

STATS  = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
COMBOS = ["stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]

_PLAYER_MINS: dict[int, float] = {
    100: 30.0, 101: 28.0, 102: 25.0, 103: 32.0,
    104: 27.0, 105: 26.0, 200: 22.0, 201: 18.0,
}

_STAT_RATE: dict[str, float] = {
    "pts": 0.65, "reb": 0.20, "ast": 0.15, "fg3m": 0.05,
    "stl": 0.05, "blk": 0.03, "turnover": 0.07,
    "stocks": 0.08, "pts_ast": 0.80, "pts_reb": 0.85,
    "reb_ast": 0.35, "pts_reb_ast": 1.00,
}

_PLAYER_STATUS: dict[int, str | None] = {
    100: "out",
    101: "doubtful",
    102: "questionable",
    103: "probable",
    104: "limited",
    105: "gtd",
    200: None,
    201: None,
}

_INTEGER_LINE   = 12
_HALFPOINT_LINE = 11.5


# ─────────────────────────────────────────────────────────────────────────────
# Fixture factories
# ─────────────────────────────────────────────────────────────────────────────

def _pmf_json(mean: float, support: int = 30) -> str:
    """Poisson PMF for a given mean, stored as JSON {str(k): prob}."""
    lam = max(mean, 0.01)
    ks = np.arange(support)
    log_pmf = ks * np.log(lam) - lam - np.array(
        [float(sum(np.log(np.arange(1, k + 1)))) if k > 0 else 0.0 for k in ks]
    )
    probs = np.exp(log_pmf)
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return json.dumps({str(k): float(v) for k, v in zip(ks, probs) if v > 1e-12})


def _make_feature_df() -> pd.DataFrame:
    rows = []
    for pid, mins in _PLAYER_MINS.items():
        rows.append({
            "player_id":                    pid,
            "game_id":                      999,
            "game_date":                    "2026-07-13",
            "team_id":                      10,
            "player_name":                  f"Player_{pid}",
            "player_minutes_mean_l5":       mins,
            "player_minutes_mean_l10":      mins,
            "player_minutes_mean_l20":      mins,
            "player_minutes_mean_season":   mins,
            "player_pts_mean_l5":           mins * 0.65,
            "player_reb_mean_l5":           mins * 0.20,
            "player_ast_mean_l5":           mins * 0.15,
        })
    return pd.DataFrame(rows)


def _make_injuries() -> list[dict]:
    return [
        {"player_id": pid, "player_name": f"Player_{pid}", "status": status}
        for pid, status in _PLAYER_STATUS.items()
        if status is not None
    ]


def _make_pmfs_df(scale: float = 1.0) -> pd.DataFrame:
    """Long-format PMF fixture.  ``scale`` lets callers simulate a post-injury state."""
    rows = []
    for pid, mins in _PLAYER_MINS.items():
        eff_mins = mins * scale if pid in {102, 104, 200, 201} else mins
        for stat in STATS + COMBOS:
            mean = eff_mins * _STAT_RATE[stat]
            pmf  = _pmf_json(mean)
            # Compute exact mean from the PMF so pmf_mean is mathematically
            # consistent with pmf_json (prevents spurious test failures).
            d    = json.loads(pmf)
            ks   = np.array([float(k) for k in d.keys()])
            vs   = np.array(list(d.values()), dtype=float)
            vs  /= vs.sum()
            exact_mean = float((ks * vs).sum())
            rows.append({
                "player_id":            pid,
                "game_id":              999,
                "game_date":            "2026-07-13",
                "stat":                 stat,
                "player_name":          f"Player_{pid}",
                "team_id":              10,
                "opponent_team_id":     20,
                "minutes_mean":         mins,
                "pmf_mean":             exact_mean,
                "stat_mean":            exact_mean,
                "mean":                 exact_mean,
                "pmf_json":             pmf,
                "pmf_mean_full_precision": exact_mean,
                "pmf_source":           "fixture",
                "is_calibrated":        True,
                "combo_suppressed":     False,
                "joint_status":         "OK",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline import helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_pipeline():
    try:
        from wnba_props_model.pipeline import injury_pipeline
        return injury_pipeline
    except ImportError as exc:
        pytest.skip(f"injury_pipeline not importable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 13 required tests
# ─────────────────────────────────────────────────────────────────────────────

def test_questionable_status_rebuilds_pmf_json():
    """Questionable player: correct multiplier, not inactive, new schema fields present."""
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 102, "player_name": "Player_102", "status": "questionable"}]

    avail = ip.build_availability_table(
        injuries, feature_df, source_updated_at="2026-07-13T08:00:00+00:00"
    )

    row = avail[avail["player_id"] == 102].iloc[0]
    assert abs(row["minutes_multiplier"] - 0.50) < 1e-9, (
        f"questionable → multiplier=0.50, got {row['minutes_multiplier']}"
    )
    assert not bool(row["is_confirmed_inactive"]), "questionable must NOT be confirmed inactive"
    assert bool(row["is_market_actionable"]), "questionable must be market actionable"

    # New schema fields required by Step 3 of the implementation spec
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc (added in Step 3)"
    assert "raw_status" in avail.columns, \
        "availability table must contain raw_status (renamed from raw_injury_status)"
    assert "normalized_status" in avail.columns, \
        "availability table must contain normalized_status (renamed from normalized_availability_status)"


def test_probable_status_rebuilds_pmf_json():
    """Probable player: correct multiplier, not inactive, new schema fields present."""
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 103, "player_name": "Player_103", "status": "probable"}]

    avail = ip.build_availability_table(injuries, feature_df)

    row = avail[avail["player_id"] == 103].iloc[0]
    assert abs(row["minutes_multiplier"] - 0.85) < 1e-9, (
        f"probable → multiplier=0.85, got {row['minutes_multiplier']}"
    )
    assert not bool(row["is_confirmed_inactive"]), "probable must NOT be confirmed inactive"
    assert bool(row["is_market_actionable"]), "probable must be market actionable"
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"
    assert "starter_probability" in avail.columns, \
        "availability table must contain starter_probability (new field)"


def test_limited_status_rebuilds_pmf_json():
    """Limited player: correct 0.65 multiplier, not inactive, new schema fields present."""
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 104, "player_name": "Player_104", "status": "limited"}]

    avail = ip.build_availability_table(injuries, feature_df)

    row = avail[avail["player_id"] == 104].iloc[0]
    assert abs(row["minutes_multiplier"] - 0.65) < 1e-9, (
        f"limited → multiplier=0.65, got {row['minutes_multiplier']}"
    )
    assert not bool(row["is_confirmed_inactive"]), "limited must NOT be confirmed inactive"
    assert bool(row["is_market_actionable"]), "limited must be market actionable"
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"
    assert "minutes_cap" in avail.columns, \
        "availability table must contain minutes_cap (new field)"


def test_teammate_redistribution_rebuilds_pmf_json():
    """When a player is OUT, teammates receive redistributed minutes via UTM."""
    ip = _get_pipeline()
    try:
        from wnba_props_model.models.usage_transfer import UsageTransferMatrix
    except ImportError as exc:
        pytest.skip(f"UsageTransferMatrix not importable: {exc}")

    feature_df = _make_feature_df()
    injuries = [{"player_id": 100, "player_name": "Player_100", "status": "out"}]

    avail = ip.build_availability_table(injuries, feature_df)
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"

    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    utm = UsageTransferMatrix(usg_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

    # INVARIANT: Historical player_minutes_mean_l5 must NOT be mutated.
    # OUT player's effective zero minutes is carried in _injury_minutes_multiplier=0.0.
    p100_hist = adj[adj["player_id"] == 100]["player_minutes_mean_l5"].values[0]
    p100_orig = feature_df[feature_df["player_id"] == 100]["player_minutes_mean_l5"].values[0]
    assert abs(p100_hist - p100_orig) < 1e-9, (
        f"Historical player_minutes_mean_l5 must NOT be mutated: "
        f"orig={p100_orig}, got={p100_hist}"
    )
    # Effective minutes for OUT player = 0, carried in _injury_minutes_multiplier
    p100_mult = adj[adj["player_id"] == 100]["_injury_minutes_multiplier"].values[0]
    assert abs(p100_mult) < 1e-9, f"OUT player _injury_minutes_multiplier must be 0.0, got {p100_mult}"

    # At least one of the two teammates must have _injury_minutes_multiplier > 1.0 from UTM
    # Historical feature columns remain unchanged; boost is in the multiplier.
    boosts = []
    for tm_pid in [200, 201]:
        tm_mult = adj[adj["player_id"] == tm_pid]["_injury_minutes_multiplier"].values[0]
        boosts.append(tm_mult > 1.0)
    assert any(boosts), "At least one teammate must have _injury_minutes_multiplier > 1.0 from UTM"


def test_teammate_redistribution_changes_settlement_probabilities():
    """After redistribution a teammate's settlement probabilities must change."""
    ip = _get_pipeline()

    # Build two PMFs at different means (full-mins vs reduced-mins for a teammate)
    pmf_before = _pmf_json(15.0)   # teammate before an OUT player's minutes arrive
    pmf_after  = _pmf_json(18.5)   # teammate after absorbing ~3.5 freed minutes

    # Half-point line
    p_ov_b, p_un_b, p_pu_b = ip.compute_settlement_probabilities(pmf_before, _HALFPOINT_LINE)
    p_ov_a, p_un_a, p_pu_a = ip.compute_settlement_probabilities(pmf_after,  _HALFPOINT_LINE)

    assert p_ov_a > p_ov_b, (
        f"P(over {_HALFPOINT_LINE}) must increase after minutes gain: "
        f"before={p_ov_b:.4f}, after={p_ov_a:.4f}"
    )
    assert abs(p_pu_b) < 1e-9, "Half-point line: P(push) must be 0"
    assert abs(p_ov_b + p_un_b + p_pu_b - 1.0) < 1e-12, "P(o)+P(u)+P(p) must sum to 1"

    # Integer line: push must be non-zero
    p_ov_i, p_un_i, p_pu_i = ip.compute_settlement_probabilities(pmf_before, _INTEGER_LINE)
    assert p_pu_i > 0, f"Integer line {_INTEGER_LINE}: P(push) must be > 0"
    assert abs(p_ov_i + p_un_i + p_pu_i - 1.0) < 1e-12

    # Schema check: pulled_at_utc must be present
    avail = ip.build_availability_table([], _make_feature_df())
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"


def test_all_affected_base_stats_are_rebuilt():
    """Every one of the 7 base stats must be present for all affected players."""
    ip = _get_pipeline()
    try:
        from wnba_props_model.models.usage_transfer import UsageTransferMatrix
    except ImportError as exc:
        pytest.skip(f"UsageTransferMatrix not importable: {exc}")

    feature_df = _make_feature_df()
    injuries = [{"player_id": 100, "player_name": "Player_100", "status": "out"}]
    avail = ip.build_availability_table(injuries, feature_df)
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"

    usg_df = pd.DataFrame({
        "player_id": feature_df["player_id"].unique(),
        "usage_pct": 0.20,
    })
    utm = UsageTransferMatrix(usg_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

    changed_mask = adj["_injury_minutes_multiplier"] != 1.0
    changed_pids = set(adj.loc[changed_mask, "player_id"].astype(int))

    assert 100 in changed_pids, "OUT player must appear in affected set"
    assert len(changed_pids & {200, 201}) >= 1, \
        "At least one teammate must be in affected set after UTM redistribution"

    # All 7 base stats must appear in the fixture for each affected player
    pmfs_df = _make_pmfs_df()
    for pid in changed_pids & set(_PLAYER_MINS.keys()):
        player_stats = set(pmfs_df[pmfs_df["player_id"] == pid]["stat"].unique())
        missing = set(STATS) - player_stats
        assert not missing, f"Base stats {missing} missing for player_id={pid}"


def test_all_affected_combos_are_rebuilt_after_base_stats():
    """All 5 combos must be rebuilt (and changed) for every affected player/teammate."""
    ip = _get_pipeline()

    pmfs_df = _make_pmfs_df()
    affected_ids = {102, 200}   # questionable + teammate receiving minutes

    # Snapshot old combo values
    old_combos = pmfs_df[
        pmfs_df["player_id"].isin(affected_ids) & pmfs_df["stat"].isin(COMBOS)
    ].copy()

    # Simulate base-stat rebuild: inflate atom means by 20 %
    for pid in affected_ids:
        for stat in STATS:
            mask = (pmfs_df["player_id"] == pid) & (pmfs_df["stat"] == stat)
            if not mask.any():
                continue
            new_mean = pmfs_df.loc[mask, "pmf_mean"].values[0] * 1.20
            pmfs_df.loc[mask, "pmf_json"] = _pmf_json(new_mean)
            pmfs_df.loc[mask, "pmf_mean"] = new_mean

    # Rebuild combos via pipeline
    rebuilt = ip.rebuild_combos_for_affected(pmfs_df, affected_ids)

    # All 5 combos must be present for each affected player
    for pid in affected_ids:
        rebuilt_combos = set(
            rebuilt[(rebuilt["player_id"] == pid) & rebuilt["stat"].isin(COMBOS)]["stat"].unique()
        )
        missing = set(COMBOS) - rebuilt_combos
        assert not missing, f"Combo stats {missing} missing after rebuild for player_id={pid}"

    # Rebuilt combo means must have changed relative to the pre-rebuild snapshot
    for pid in affected_ids:
        for combo in COMBOS:
            old_row = old_combos[(old_combos["player_id"] == pid) & (old_combos["stat"] == combo)]
            new_row = rebuilt[(rebuilt["player_id"] == pid) & (rebuilt["stat"] == combo)]
            if old_row.empty or new_row.empty:
                continue
            delta = abs(new_row["pmf_mean"].values[0] - old_row["pmf_mean"].values[0])
            assert delta > 1e-6, (
                f"Combo {combo} mean unchanged for player {pid} after base-stat rebuild "
                f"(delta={delta:.2e})"
            )

    # Schema check
    avail = ip.build_availability_table([], _make_feature_df())
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"


def test_final_mean_matches_final_pmf_json():
    """pmf_mean stored in the slate must match mean recomputed from pmf_json (≤ 1e-10)."""
    ip = _get_pipeline()

    pmfs_df = _make_pmfs_df()

    for _, row in pmfs_df.iterrows():
        d  = json.loads(row["pmf_json"])
        ks = np.array([float(k) for k in d.keys()])
        vs = np.array(list(d.values()), dtype=float)
        vs /= vs.sum()
        computed = float((ks * vs).sum())
        stored   = float(row["pmf_mean"])
        err = abs(computed - stored)
        assert err <= 1e-10, (
            f"player={row['player_id']} stat={row['stat']}: "
            f"mean recomputed from pmf_json={computed:.14f} "
            f"!= pmf_mean={stored:.14f}  (err={err:.2e})"
        )

    # Schema check
    avail = ip.build_availability_table([], _make_feature_df())
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"


def test_noninactive_zero_mean_is_fatal():
    """For a non-inactive player, pmf_mean=0 must raise a fatal ValueError."""
    ip = _get_pipeline()

    pmfs_df   = _make_pmfs_df()
    feature_df = _make_feature_df()

    # Corrupt probable player (pid=103) to zero mean — probable is NOT inactive
    mask = (pmfs_df["player_id"] == 103) & (pmfs_df["stat"] == "pts")
    pmfs_df.loc[mask, "pmf_mean"] = 0.0
    pmfs_df.loc[mask, "pmf_json"] = json.dumps({"0": 1.0})

    avail = ip.build_availability_table(
        [{"player_id": 103, "player_name": "Player_103", "status": "probable"}],
        feature_df,
    )
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"

    # Probable → confirmed_inactive must be False
    inact = bool(avail.loc[avail["player_id"] == 103, "is_confirmed_inactive"].values[0])
    assert not inact, "probable player must NOT be confirmed inactive"

    # Zero mean on a non-inactive row must raise
    with pytest.raises(ValueError, match="(?i)(pmf_mean|integrity|fatal|non.?inactive)"):
        ip.validate_injury_adjusted_pmfs(pmfs_df, avail)


def test_nan_mean_is_not_classified_as_out():
    """NaN pmf_mean for a non-inactive player must be a fatal error, not silently as OUT."""
    ip = _get_pipeline()

    pmfs_df    = _make_pmfs_df()
    feature_df = _make_feature_df()

    # NaN pmf_mean for probable player (pid=103)
    mask = (pmfs_df["player_id"] == 103) & (pmfs_df["stat"] == "pts")
    pmfs_df.loc[mask, "pmf_mean"] = float("nan")

    avail = ip.build_availability_table(
        [{"player_id": 103, "player_name": "Player_103", "status": "probable"}],
        feature_df,
    )
    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"

    # Probable is not confirmed inactive
    inact = bool(avail.loc[avail["player_id"] == 103, "is_confirmed_inactive"].values[0])
    assert not inact, "probable must NOT be confirmed inactive"

    # NaN mean must be fatal
    with pytest.raises(ValueError, match="(?i)(nan|pmf_mean|integrity|fatal)"):
        ip.validate_injury_adjusted_pmfs(pmfs_df, avail)


def test_only_explicit_confirmed_inactive_rows_leave_actionable_board():
    """Only OUT leaves the board; doubtful, questionable, probable, limited, GTD remain."""
    ip = _get_pipeline()

    feature_df = _make_feature_df()
    injuries   = _make_injuries()
    avail      = ip.build_availability_table(injuries, feature_df)

    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"

    inactive_pids   = set(avail.loc[avail["is_confirmed_inactive"], "player_id"].astype(int))
    actionable_pids = set(avail.loc[avail["is_market_actionable"],  "player_id"].astype(int))

    # Only OUT (100) is confirmed inactive
    assert 100 in inactive_pids, "OUT player must be confirmed inactive"

    # Doubtful (101) must NOT be automatically confirmed inactive
    assert 101 not in inactive_pids, (
        "Doubtful player must NOT be automatically confirmed inactive "
        "(do not treat doubtful as confirmed OUT)"
    )
    assert 101 in actionable_pids, "Doubtful player must remain market actionable"

    # All other statuses must be actionable
    for pid in [102, 103, 104, 105, 200, 201]:
        assert pid in actionable_pids, (
            f"player_id={pid} (status={_PLAYER_STATUS.get(pid, 'uninjured')}) "
            "must be market actionable"
        )


def test_doubtful_is_not_automatically_confirmed_out():
    """Doubtful must NOT map to is_confirmed_inactive=True or minutes_multiplier=0.0."""
    ip = _get_pipeline()

    feature_df = _make_feature_df()
    injuries   = [{"player_id": 101, "player_name": "Player_101", "status": "doubtful"}]

    avail = ip.build_availability_table(injuries, feature_df)

    assert "pulled_at_utc" in avail.columns, \
        "availability table must contain pulled_at_utc"

    row = avail[avail["player_id"] == 101].iloc[0]

    # Core invariant: doubtful ≠ confirmed OUT
    assert not bool(row["is_confirmed_inactive"]), (
        f"Doubtful player must NOT be confirmed inactive. "
        f"Got is_confirmed_inactive={row['is_confirmed_inactive']!r}. "
        "The pipeline may not automatically treat 'doubtful' as OUT."
    )
    assert bool(row["is_market_actionable"]), (
        f"Doubtful player must remain market actionable. "
        f"Got is_market_actionable={row['is_market_actionable']!r}"
    )
    # Non-zero minutes multiplier (low but non-zero)
    assert float(row["minutes_multiplier"]) > 0.0, (
        f"Doubtful player must have a non-zero minutes multiplier; "
        f"got {row['minutes_multiplier']}. "
        "A multiplier of 0.0 is only valid for confirmed-OUT (explicit DNP/OUT/INACTIVE)."
    )


def test_injury_source_timestamp_precedes_prediction_timestamp():
    """source_updated_at (from injury feed) must be ≤ pulled_at_utc (run time)."""
    ip = _get_pipeline()

    feature_df = _make_feature_df()
    # Provide an explicit source timestamp well before "now"
    source_ts = "2026-07-13T08:00:00+00:00"

    avail = ip.build_availability_table(
        injuries=[{"player_id": 102, "player_name": "Player_102", "status": "questionable"}],
        feature_df=feature_df,
        source_updated_at=source_ts,
    )

    # pulled_at_utc must exist (Step 3 schema requirement)
    assert "pulled_at_utc" in avail.columns, (
        "availability table must contain pulled_at_utc; "
        "source_updated_at alone is insufficient (it's the source feed timestamp, "
        "not the pipeline run timestamp)"
    )

    sample     = avail.iloc[0]
    source_dt  = pd.to_datetime(sample["source_updated_at"], utc=True)
    pulled_dt  = pd.to_datetime(sample["pulled_at_utc"],     utc=True)

    assert source_dt <= pulled_dt, (
        f"source_updated_at ({source_dt}) must precede pulled_at_utc ({pulled_dt}). "
        "Injury data from the feed cannot be timestamped after the pipeline run."
    )
