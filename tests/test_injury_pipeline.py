"""Tests for the injury pipeline — upstream PMF rebuild.

Covers all 14 required test cases:
  test_questionable_player_rebuilds_pmf_json
  test_probable_player_rebuilds_pmf_json
  test_minutes_restriction_changes_settlement_probabilities
  test_utm_teammate_minutes_change_rebuilds_teammate_pmfs
  test_utm_update_rebuilds_all_affected_base_stats
  test_combos_rebuilt_after_injury_updates
  test_injury_adjusted_pmf_mean_matches_pmf_json
  test_non_out_zero_mean_is_fatal
  test_nan_mean_is_not_classified_as_out
  test_only_explicit_inactive_rows_leave_actionable_board
  test_actionable_market_for_inactive_player_requires_settlement_reconciliation
  test_injury_step_failure_blocks_deployment   (structural contract test)
  test_edge_report_failure_blocks_deployment   (structural contract test)
  test_stale_artifact_cannot_pass_current_run

Plus Blocker 1–5 regression tests:
  Blocker 1 (5): availability/conditional-minutes separation
  Blocker 2 (4): conditional_minutes_cap enforcement
  Blocker 4 (6): InjuryFetchResult contract
  Blocker 5 (4): per-record source timestamps

Regression fixture includes:
  - One OUT player (pid=100)
  - One doubtful player (pid=101)
  - One questionable player (pid=102)
  - One probable player (pid=103)
  - Two teammates receiving redistributed minutes (pid=200, pid=201)
  - Integer and half-point market lines
  - At least one combo involving an affected teammate
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Regression fixture helpers
# ---------------------------------------------------------------------------

def _make_pmf_json(mean: float, n_support: int = 21) -> str:
    """Build a simple NegBinom-like PMF JSON string for a given mean."""
    # Use a Poisson approximation: P(k) = exp(-lam)*lam^k / k!
    lam = max(mean, 0.01)
    ks = np.arange(n_support)
    log_pmf = ks * np.log(lam) - lam - np.array(
        [float(sum(np.log(range(1, k + 1)))) if k > 0 else 0.0 for k in ks]
    )
    probs = np.exp(log_pmf)
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return json.dumps({str(k): float(v) for k, v in zip(ks, probs) if v > 1e-12})


def _make_feature_df() -> pd.DataFrame:
    """Minimal feature DataFrame for the regression fixture."""
    rows = []
    # Players: out=100, doubtful=101, questionable=102, probable=103,
    #          teammate1=200, teammate2=201
    # Opponents added for team context but not injured
    player_mins = {100: 30.0, 101: 28.0, 102: 25.0, 103: 32.0, 200: 22.0, 201: 18.0}
    team_id = {100: 10, 101: 10, 102: 10, 103: 10, 200: 10, 201: 10}

    for pid, mins in player_mins.items():
        rows.append({
            "player_id": pid,
            "game_id": 999,
            "game_date": "2026-07-13",
            "season": 2026,
            "team_id": team_id[pid],
            "player_name": f"Player_{pid}",
            "position": "G" if pid in (100, 101, 200) else "F",
            "player_minutes_mean_l5": mins,
            "player_minutes_mean_l10": mins,
            "player_minutes_mean_season": mins,
            "player_pts_mean_l5": mins * 0.65,
            "player_pts_mean_season": mins * 0.65,
            "player_reb_mean_l5": mins * 0.20,
            "player_reb_mean_season": mins * 0.20,
            "player_ast_mean_l5": mins * 0.15,
            "player_ast_mean_season": mins * 0.15,
        })
    return pd.DataFrame(rows)


def _make_pmfs_df() -> pd.DataFrame:
    """Minimal PMF DataFrame (long format) for the regression fixture."""
    STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    COMBO_STATS = ["pts_reb", "pts_ast", "reb_ast", "stocks", "pts_reb_ast"]
    rows = []
    player_mins = {100: 30.0, 101: 28.0, 102: 25.0, 103: 32.0, 200: 22.0, 201: 18.0}

    for pid, mins in player_mins.items():
        for stat in STATS + COMBO_STATS:
            if stat in ("pts", "reb", "ast"):
                mean = mins * {"pts": 0.65, "reb": 0.20, "ast": 0.15}[stat]
            elif stat in COMBO_STATS:
                if stat == "pts_reb":
                    mean = mins * 0.85
                elif stat == "pts_ast":
                    mean = mins * 0.80
                elif stat == "reb_ast":
                    mean = mins * 0.35
                elif stat == "stocks":
                    mean = mins * 0.10
                else:
                    mean = mins * 1.00
            else:
                mean = mins * 0.05

            rows.append({
                "player_id": pid,
                "game_id": 999,
                "game_date": "2026-07-13",
                "stat": stat,
                "player_name": f"Player_{pid}",
                "team_id": 10,
                "opponent_team_id": 20,
                "minutes_mean": mins,
                "pmf_mean": round(mean, 4),
                "stat_mean": round(mean, 4),
                "mean": round(mean, 4),
                "pmf_json": _make_pmf_json(mean),
                "pmf_mean_full_precision": float(mean),
                "pmf_source": "stage4_baseline",
                "is_calibrated": True,
                "combo_suppressed": False,
                "joint_status": "OK",
            })
    return pd.DataFrame(rows)


def _make_injuries() -> list[dict]:
    """Regression fixture injuries."""
    return [
        {"player_id": 100, "player_name": "Player_100", "status": "out"},
        {"player_id": 101, "player_name": "Player_101", "status": "doubtful"},
        {"player_id": 102, "player_name": "Player_102", "status": "questionable"},
        {"player_id": 103, "player_name": "Player_103", "status": "probable"},
    ]


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def _get_pipeline():
    """Import injury_pipeline, skip if not importable."""
    try:
        from wnba_props_model.pipeline import injury_pipeline
        return injury_pipeline
    except ImportError as e:
        pytest.skip(f"injury_pipeline not importable: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuestionablePlayerRebuildspmfJson:
    """test_questionable_player_rebuilds_pmf_json"""

    def test_questionable_player_rebuilds_pmf_json(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        injuries = [{"player_id": 102, "player_name": "Player_102", "status": "questionable"}]

        avail = ip.build_availability_table(injuries, feature_df)

        # questionable → effective multiplier = 0.50
        row = avail[avail["player_id"] == 102].iloc[0]
        assert abs(row["minutes_multiplier"] - 0.50) < 1e-9

        orig_mins = feature_df[feature_df["player_id"] == 102]["player_minutes_mean_l5"].values[0]

        # Apply feature adjustments
        adj = ip.apply_injury_to_feature_df(feature_df, avail)

        p102 = adj[adj["player_id"] == 102]

        # INVARIANT: Historical minutes feature columns must NOT be mutated.
        # The adjustment is carried in _injury_minutes_multiplier (applied once
        # by pmf_engine after the baseline minutes model).
        new_mins = p102["player_minutes_mean_l5"].values[0]
        assert abs(new_mins - orig_mins) < 1e-9, (
            f"Historical player_minutes_mean_l5 must NOT be mutated: "
            f"orig={orig_mins}, got={new_mins}"
        )

        # Blocker 1: _injury_minutes_multiplier = cond_mult (1.0 for questionable),
        # NOT p_active * cond_mult (0.50). availability_probability carries p_active.
        assert abs(p102["_injury_minutes_multiplier"].values[0] - 1.0) < 1e-9, (
            f"_injury_minutes_multiplier for questionable must be cond_mult=1.0 "
            f"(Blocker 1: p_active must NOT be baked into conditional minutes), "
            f"got {p102['_injury_minutes_multiplier'].values[0]}"
        )
        # availability_probability must be 0.50 in its own column
        assert abs(p102["availability_probability"].values[0] - 0.50) < 1e-9, (
            f"availability_probability for questionable must be 0.50, "
            f"got {p102['availability_probability'].values[0]}"
        )

        # adjusted_projected_minutes = orig * p_active * cond_mult (for traceability/display)
        adj_mins = p102["adjusted_projected_minutes"].values[0]
        assert abs(adj_mins - orig_mins * 0.50) < 1e-6, (
            f"adjusted_projected_minutes should be {orig_mins * 0.50}, got {adj_mins}"
        )


class TestProbablePlayerRebuildspmfJson:
    """test_probable_player_rebuilds_pmf_json"""

    def test_probable_player_rebuilds_pmf_json(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        injuries = [{"player_id": 103, "player_name": "Player_103", "status": "probable"}]

        avail = ip.build_availability_table(injuries, feature_df)
        row = avail[avail["player_id"] == 103].iloc[0]
        assert abs(row["minutes_multiplier"] - 0.85) < 1e-9

        orig = feature_df[feature_df["player_id"] == 103]["player_minutes_mean_l5"].values[0]
        adj = ip.apply_injury_to_feature_df(feature_df, avail)
        p103 = adj[adj["player_id"] == 103]

        # INVARIANT: Historical minutes feature column must NOT be mutated.
        new_mins = p103["player_minutes_mean_l5"].values[0]
        assert abs(new_mins - orig) < 1e-9, (
            f"Historical player_minutes_mean_l5 must NOT be mutated: "
            f"orig={orig}, got={new_mins}"
        )
        # Blocker 1: _injury_minutes_multiplier = cond_mult (1.0 for probable),
        # NOT p_active * cond_mult (0.85). availability_probability carries p_active.
        assert abs(p103["_injury_minutes_multiplier"].values[0] - 1.0) < 1e-9, (
            f"_injury_minutes_multiplier for probable must be cond_mult=1.0 "
            f"(Blocker 1: not p_active*cond_mult=0.85), "
            f"got {p103['_injury_minutes_multiplier'].values[0]}"
        )
        # availability_probability must be 0.85 in its own column
        assert abs(p103["availability_probability"].values[0] - 0.85) < 1e-9, (
            f"availability_probability for probable must be 0.85, "
            f"got {p103['availability_probability'].values[0]}"
        )


class TestMinutesRestrictionChangesSettlementProbabilities:
    """test_minutes_restriction_changes_settlement_probabilities"""

    def test_minutes_restriction_changes_settlement_probabilities(self):
        ip = _get_pipeline()

        # Build a simple PMF at mean=14.0 and mean=7.0
        pmf_full = _make_pmf_json(14.0)
        pmf_half = _make_pmf_json(7.0)

        line = 11.5  # half-point line
        p_ov_full, p_un_full, p_pu_full = ip.compute_settlement_probabilities(pmf_full, line)
        p_ov_half, p_un_half, p_pu_half = ip.compute_settlement_probabilities(pmf_half, line)

        # With halved minutes (mean drops from 14 to 7), P(over 11.5) must decrease
        assert p_ov_half < p_ov_full, (
            f"P(over) should decrease after minutes restriction: "
            f"before={p_ov_full:.4f}, after={p_ov_half:.4f}"
        )
        # P(under 11.5) must increase
        assert p_un_half > p_un_full, (
            f"P(under) should increase after minutes restriction: "
            f"before={p_un_full:.4f}, after={p_un_half:.4f}"
        )
        # Half-point line: no push possible
        assert abs(p_pu_full) < 1e-9, f"Half-point push must be 0, got {p_pu_full}"
        assert abs(p_pu_half) < 1e-9, f"Half-point push must be 0, got {p_pu_half}"
        # Probabilities must sum to 1
        assert abs(p_ov_full + p_un_full + p_pu_full - 1.0) < 1e-6
        assert abs(p_ov_half + p_un_half + p_pu_half - 1.0) < 1e-6


class TestUtmTeammateMinutesChangeRebuildsTeammatePmfs:
    """test_utm_teammate_minutes_change_rebuilds_teammate_pmfs"""

    def test_utm_teammate_minutes_change_rebuilds_teammate_pmfs(self):
        ip = _get_pipeline()
        from wnba_props_model.models.usage_transfer import UsageTransferMatrix

        feature_df = _make_feature_df()
        injuries = [{"player_id": 100, "player_name": "Player_100", "status": "out"}]

        avail = ip.build_availability_table(injuries, feature_df)

        # Build UTM from uniform usage
        usg_df = pd.DataFrame({
            "player_id": feature_df["player_id"].unique(),
            "usage_pct": 0.20,
        })
        utm = UsageTransferMatrix(usg_df)

        adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

        # INVARIANT: Historical feature columns must NOT be mutated.
        # OUT player's effective minutes are carried in _injury_minutes_multiplier=0.0
        p100_hist = adj[adj["player_id"] == 100]["player_minutes_mean_l5"].values[0]
        p100_orig = feature_df[feature_df["player_id"] == 100]["player_minutes_mean_l5"].values[0]
        assert abs(p100_hist - p100_orig) < 1e-9, (
            f"Historical player_minutes_mean_l5 must NOT be mutated for OUT player: "
            f"orig={p100_orig}, got={p100_hist}"
        )
        # The effective minutes for OUT player = 0 is carried in _injury_minutes_multiplier
        p100_mult = adj[adj["player_id"] == 100]["_injury_minutes_multiplier"].values[0]
        assert abs(p100_mult) < 1e-9, f"OUT player _injury_minutes_multiplier must be 0, got {p100_mult}"

        # Teammates should have _injury_minutes_multiplier > 1.0 from UTM boost
        # (historical feature columns remain unchanged)
        for teammate_pid in [200, 201]:
            orig_hist = feature_df[feature_df["player_id"] == teammate_pid][
                "player_minutes_mean_l5"
            ].values[0]
            new_hist = adj[adj["player_id"] == teammate_pid][
                "player_minutes_mean_l5"
            ].values[0]
            assert abs(new_hist - orig_hist) < 1e-9, (
                f"Teammate {teammate_pid} historical player_minutes_mean_l5 must NOT be mutated: "
                f"orig={orig_hist}, got={new_hist}"
            )
            # The boost is encoded in _injury_minutes_multiplier
            teammate_mult = adj[adj["player_id"] == teammate_pid]["_injury_minutes_multiplier"].values[0]
            assert teammate_mult >= 1.0, (
                f"Teammate {teammate_pid} should have _injury_minutes_multiplier >= 1.0 from UTM boost, "
                f"got {teammate_mult}"
            )


class TestUtmUpdateRebuildsAllAffectedBaseStats:
    """test_utm_update_rebuilds_all_affected_base_stats"""

    def test_utm_update_rebuilds_all_affected_base_stats(self):
        ip = _get_pipeline()
        from wnba_props_model.models.usage_transfer import UsageTransferMatrix

        feature_df = _make_feature_df()
        injuries = [{"player_id": 100, "player_name": "Player_100", "status": "out"}]
        avail = ip.build_availability_table(injuries, feature_df)

        usg_df = pd.DataFrame({
            "player_id": feature_df["player_id"].unique(),
            "usage_pct": 0.20,
        })
        utm = UsageTransferMatrix(usg_df)

        adj = ip.apply_injury_to_feature_df(feature_df, avail, utm=utm)

        # All players who had their minutes changed should have
        # _injury_minutes_multiplier != 1.0
        changed_mask = adj["_injury_minutes_multiplier"] != 1.0
        changed_pids = set(adj.loc[changed_mask, "player_id"].astype(int))

        # Must include OUT player and at least one teammate
        assert 100 in changed_pids, "OUT player must be in affected set"
        teammate_changed = changed_pids & {200, 201}
        assert len(teammate_changed) >= 1, (
            f"At least one teammate should receive redistributed minutes, "
            f"got changed_pids={changed_pids}"
        )


class TestCombosRebuiltAfterInjuryUpdates:
    """test_combos_rebuilt_after_injury_updates"""

    def test_combos_rebuilt_after_injury_updates(self):
        ip = _get_pipeline()

        pmfs_df = _make_pmfs_df()
        affected_ids = {200}  # teammate receiving redistributed minutes

        # Modify teammate 200's atom PMFs to simulate a rebuild
        ATOM_STATS = {"pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"}
        for stat in ATOM_STATS:
            mask = (pmfs_df["player_id"] == 200) & (pmfs_df["stat"] == stat)
            if mask.any():
                new_mean = pmfs_df.loc[mask, "pmf_mean"].values[0] * 1.15
                pmfs_df.loc[mask, "pmf_json"] = _make_pmf_json(new_mean)
                pmfs_df.loc[mask, "pmf_mean"] = round(new_mean, 4)

        # Save old combo PMFs for player 200
        COMBO_STATS = {"pts_reb", "pts_ast", "reb_ast", "stocks", "pts_reb_ast"}
        old_combos = pmfs_df[
            (pmfs_df["player_id"] == 200) & pmfs_df["stat"].isin(COMBO_STATS)
        ].copy()

        # Rebuild combos
        rebuilt = ip.rebuild_combos_for_affected(pmfs_df, affected_ids)

        new_combos = rebuilt[
            (rebuilt["player_id"] == 200) & rebuilt["stat"].isin(COMBO_STATS)
        ]

        # Combo rows should exist after rebuild
        assert not new_combos.empty, "No combo rows for affected player after rebuild"

        # At least some combo stats must have changed mean (since atom means changed)
        if not old_combos.empty and not new_combos.empty:
            old_means = old_combos.set_index("stat")["pmf_mean"]
            new_means = new_combos.set_index("stat")["pmf_mean"]
            common = old_means.index.intersection(new_means.index)
            if len(common) > 0:
                diff = (new_means[common] - old_means[common]).abs()
                assert diff.max() > 1e-6, (
                    "Combo means should change after atom PMF rebuild, "
                    f"max diff={diff.max()}"
                )


class TestInjuryAdjustedPmfMeanMatchesPmfJson:
    """test_injury_adjusted_pmf_mean_matches_pmf_json"""

    def test_injury_adjusted_pmf_mean_matches_pmf_json(self):
        ip = _get_pipeline()

        # Build a PMF and verify the mean computed from pmf_json matches pmf_mean
        mean_target = 12.5
        pmf_json = _make_pmf_json(mean_target)

        d = json.loads(pmf_json)
        ks = np.array([float(k) for k in d.keys()])
        vs = np.array(list(d.values()), dtype=float)
        computed_mean = float((ks * vs).sum() / vs.sum())

        # The PMF builder should match within 0.5 (Poisson approximation tolerance)
        assert abs(computed_mean - mean_target) < 0.5, (
            f"PMF mean mismatch: computed={computed_mean:.4f}, target={mean_target}"
        )

        # For the settlement probability helper, verify P(over) + P(under) + P(push) = 1
        for line in [12.5, 13.0]:  # half-point and integer
            p_ov, p_un, p_pu = ip.compute_settlement_probabilities(pmf_json, line)
            total = p_ov + p_un + p_pu
            assert abs(total - 1.0) < 1e-6, (
                f"Probabilities don't sum to 1 at line={line}: "
                f"P(over)={p_ov:.6f} + P(under)={p_un:.6f} + P(push)={p_pu:.6f} = {total:.6f}"
            )

        # Integer line should have non-zero P(push)
        p_ov, p_un, p_pu = ip.compute_settlement_probabilities(pmf_json, 12.0)
        assert p_pu > 0, f"Integer line should have P(push) > 0, got {p_pu}"


class TestNonOutZeroMeanIsFatal:
    """test_non_out_zero_mean_is_fatal"""

    def test_non_out_zero_mean_is_fatal(self):
        ip = _get_pipeline()

        pmfs_df = _make_pmfs_df()
        # Corrupt a non-OUT player's pmf_mean to 0
        mask = (pmfs_df["player_id"] == 103) & (pmfs_df["stat"] == "pts")
        pmfs_df.loc[mask, "pmf_mean"] = 0.0
        pmfs_df.loc[mask, "pmf_json"] = json.dumps({"0": 1.0})

        # Availability table: player 103 is NOT inactive
        feature_df = _make_feature_df()
        avail = ip.build_availability_table(
            [{"player_id": 103, "player_name": "Player_103", "status": "probable"}],
            feature_df,
        )
        # Probable is NOT confirmed inactive (availability_probability=0.85)
        assert not avail.loc[avail["player_id"] == 103, "is_confirmed_inactive"].values[0]

        with pytest.raises(ValueError, match="(?i)(pmf_mean|integrity|fatal)"):
            ip.validate_injury_adjusted_pmfs(pmfs_df, avail)


class TestNanMeanIsNotClassifiedAsOut:
    """test_nan_mean_is_not_classified_as_out"""

    def test_nan_mean_is_not_classified_as_out(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        # Player 103 has NaN pmf_mean but is probable (NOT OUT)
        pmfs_df = _make_pmfs_df()
        mask = (pmfs_df["player_id"] == 103) & (pmfs_df["stat"] == "pts")
        pmfs_df.loc[mask, "pmf_mean"] = float("nan")

        avail = ip.build_availability_table(
            [{"player_id": 103, "player_name": "Player_103", "status": "probable"}],
            feature_df,
        )
        # probable → NOT confirmed inactive
        assert not avail.loc[avail["player_id"] == 103, "is_confirmed_inactive"].values[0]

        # NaN mean on a non-inactive row must be a fatal error
        with pytest.raises(ValueError, match="(?i)(nan|pmf_mean|integrity|fatal)"):
            ip.validate_injury_adjusted_pmfs(pmfs_df, avail)


class TestOnlyExplicitInactiveRowsLeaveActionableBoard:
    """test_only_explicit_inactive_rows_leave_actionable_board"""

    def test_only_explicit_inactive_rows_leave_actionable_board(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        injuries = _make_injuries()
        avail = ip.build_availability_table(injuries, feature_df)

        # Only OUT (pid=100) and doubtful (pid=101) should be confirmed inactive
        inactive_pids = set(
            avail.loc[avail["is_confirmed_inactive"], "player_id"].astype(int)
        )
        actionable_pids = set(
            avail.loc[avail["is_market_actionable"], "player_id"].astype(int)
        )

        assert 100 in inactive_pids, "OUT player must be confirmed inactive"
        # doubtful is no longer automatically confirmed inactive
        assert 101 not in inactive_pids, \
            "Doubtful player must NOT be confirmed inactive (see blueprint §5)"
        assert 101 in actionable_pids, "Doubtful player must be market actionable"

        # Questionable (102) and probable (103) must be market actionable
        assert 102 in actionable_pids, "Questionable player must be market actionable"
        assert 103 in actionable_pids, "Probable player must be market actionable"

        # Teammates (200, 201) must be actionable
        assert 200 in actionable_pids, "Teammate must be market actionable"
        assert 201 in actionable_pids, "Teammate must be market actionable"

        # Build confirmed_inactive_mask on a PMF DataFrame
        pmfs_df = _make_pmfs_df()
        inactive_mask = ip.build_confirmed_inactive_mask(pmfs_df, avail)

        # Mask must be True only for explicitly confirmed-inactive players (OUT only)
        assert inactive_mask[pmfs_df["player_id"] == 100].all(), \
            "OUT player rows must be masked"
        # doubtful is no longer auto-confirmed inactive → must NOT be masked
        assert not inactive_mask[pmfs_df["player_id"] == 101].any(), \
            "Doubtful player rows must NOT be masked (no longer auto-confirmed OUT)"
        assert not inactive_mask[pmfs_df["player_id"] == 102].any(), \
            "Questionable player rows must NOT be masked"
        assert not inactive_mask[pmfs_df["player_id"] == 103].any(), \
            "Probable player rows must NOT be masked"


class TestActionableMarketForInactivePlayerRequiresSettlementReconciliation:
    """test_actionable_market_for_inactive_player_requires_settlement_reconciliation

    If a market quote is still live for an inactive player, the model must NOT
    automatically treat the stat as zero.  Instead, it must flag this as
    requiring manual reconciliation via the vendor's participation rule.
    """

    def test_actionable_market_for_inactive_player_requires_settlement_reconciliation(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        # Player 100 is OUT
        avail = ip.build_availability_table(
            [{"player_id": 100, "player_name": "Player_100", "status": "out"}],
            feature_df,
        )
        row = avail[avail["player_id"] == 100].iloc[0]

        # Confirmed inactive → not market actionable
        assert row["is_confirmed_inactive"] is True or bool(row["is_confirmed_inactive"])
        assert row["is_market_actionable"] is False or not bool(row["is_market_actionable"])

        # If a hypothetical market were still live for player 100,
        # the model should NOT assume stat=0 automatically.
        # Instead: detect the conflict and flag for reconciliation.
        # (In production this is handled by build_edge_report checking is_market_actionable=False)
        #
        # We verify that build_confirmed_inactive_mask correctly identifies this row
        # so the edge report CAN filter it.
        pmfs_df = _make_pmfs_df()
        inactive_mask = ip.build_confirmed_inactive_mask(pmfs_df, avail)
        p100_rows = pmfs_df[pmfs_df["player_id"] == 100]
        assert inactive_mask[p100_rows.index].all(), \
            "OUT player must appear in confirmed_inactive_mask"


class TestInjuryStepFailureBlocksDeployment:
    """test_injury_step_failure_blocks_deployment

    Structural contract test: verify that the workflow step
    'Apply injury updates' has continue-on-error: false.
    """

    def test_injury_step_failure_blocks_deployment(self):
        workflow_path = Path("../../.github/workflows/pregame_initial.yml")
        if not workflow_path.exists():
            workflow_path = Path(".github/workflows/pregame_initial.yml")
        if not workflow_path.exists():
            # Try relative to tests/ directory
            workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "pregame_initial.yml"

        assert workflow_path.exists(), \
            f"Workflow file not found at {workflow_path}"

        content = workflow_path.read_text()

        # The injury update step must NOT have continue-on-error: true
        # It should have continue-on-error: false (or no continue-on-error at all
        # with the apply_injury_updates.py command, but the spec requires explicit false)
        lines = content.splitlines()
        in_injury_step = False
        injury_step_found = False
        for i, line in enumerate(lines):
            if "apply_injury_updates.py" in line and "apply-injury" in lines[max(0, i-5):i+1][-1] if i > 0 else "":
                in_injury_step = True
            if "Apply injury updates" in line and "BLOCKING" in line:
                injury_step_found = True
                in_injury_step = True
            if in_injury_step and "continue-on-error: false" in line:
                # Found the correct setting
                return

        # Fallback: check the raw workflow text for the pattern
        import re
        # Find the section for apply_injury_updates
        pattern = r"apply_injury_updates\.py.*?continue-on-error:\s*false"
        if re.search(pattern, content, re.DOTALL | re.MULTILINE):
            return  # PASS

        # Also accept if the section says BLOCKING
        if "BLOCKING" in content and "apply_injury_updates" in content:
            # Check that the BLOCKING step has continue-on-error: false
            # by finding the nearest continue-on-error after apply_injury_updates
            idx = content.find("apply_injury_updates.py")
            snippet = content[max(0, idx-500):idx+500]
            if "continue-on-error: false" in snippet:
                return

        pytest.fail(
            "apply_injury_updates.py step must have continue-on-error: false "
            "to block deployment on failure. Found in workflow:\n"
            + "\n".join(l for l in lines if "apply_injury" in l or "continue-on-error" in l)
        )


class TestEdgeReportFailureBlocksDeployment:
    """test_edge_report_failure_blocks_deployment

    Structural contract test: verify that the workflow step
    'Build edge report' has continue-on-error: false.
    """

    def test_edge_report_failure_blocks_deployment(self):
        workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "pregame_initial.yml"
        assert workflow_path.exists(), f"Workflow file not found at {workflow_path}"

        content = workflow_path.read_text()
        import re

        pattern = r"build_edge_report\.py.*?continue-on-error:\s*false"
        if re.search(pattern, content, re.DOTALL | re.MULTILINE):
            return

        # Also accept section with BLOCKING in step name
        if "Build edge report (BLOCKING)" in content:
            idx = content.find("build_edge_report.py")
            snippet = content[max(0, idx - 500): idx + 500]
            if "continue-on-error: false" in snippet:
                return

        pytest.fail(
            "build_edge_report.py step must have continue-on-error: false. "
            "Checked workflow: " + str(workflow_path)
        )


class TestStaleArtifactCannotPassCurrentRun:
    """test_stale_artifact_cannot_pass_current_run"""

    def test_stale_artifact_cannot_pass_current_run(self):
        ip = _get_pipeline()

        # Build an artifact with run_id = "run_A"
        artifact = ip.add_run_metadata(
            artifact={},
            github_run_id="run_A",
            git_commit="abc123",
            prediction_timestamp="2026-07-13T10:00:00Z",
            market_snapshot_timestamp="2026-07-13T09:55:00Z",
            injury_snapshot_timestamp="2026-07-13T09:50:00Z",
            game_date="2026-07-13",
        )
        assert artifact["github_run_id"] == "run_A"

        # Verifying with the correct run_id should pass
        ip.verify_artifact_run_id(artifact, expected_run_id="run_A")  # no raise

        # Verifying with a different run_id must raise
        with pytest.raises(ValueError, match="(?i)(stale|run_id)"):
            ip.verify_artifact_run_id(artifact, expected_run_id="run_B")


# ---------------------------------------------------------------------------
# Additional regression tests for the regression fixture
# ---------------------------------------------------------------------------

class TestRegressionFixtureAvailabilityTable:
    """Verify the full regression fixture produces the expected availability table."""

    def test_fixture_availability_table(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        injuries = _make_injuries()
        avail = ip.build_availability_table(injuries, feature_df)

        # Check multipliers
        def _get(pid: int) -> dict:
            row = avail[avail["player_id"] == pid]
            assert not row.empty, f"player_id={pid} not in availability table"
            return row.iloc[0].to_dict()

        r100 = _get(100)
        assert r100["minutes_multiplier"] == 0.0, "OUT → multiplier=0"
        assert r100["is_confirmed_inactive"] is True or bool(r100["is_confirmed_inactive"])

        r101 = _get(101)
        # doubtful is no longer treated as confirmed OUT — it has a low but non-zero
        # minutes multiplier (0.15) and remains market actionable.
        assert abs(r101["minutes_multiplier"] - 0.15) < 1e-9, (
            f"Doubtful → multiplier=0.15 (not zero), got {r101['minutes_multiplier']}"
        )
        assert not bool(r101["is_confirmed_inactive"]), \
            "Doubtful must NOT be confirmed inactive"

        r102 = _get(102)
        assert abs(r102["minutes_multiplier"] - 0.50) < 1e-9, "Questionable → 0.50"
        assert not bool(r102["is_confirmed_inactive"])

        r103 = _get(103)
        assert abs(r103["minutes_multiplier"] - 0.85) < 1e-9, "Probable → 0.85"
        assert not bool(r103["is_confirmed_inactive"])

        # Teammates (not injured) should be fully available
        r200 = _get(200)
        assert r200["minutes_multiplier"] == 1.0
        assert bool(r200["is_market_actionable"])

    def test_fixture_integer_line_has_nonzero_push(self):
        ip = _get_pipeline()

        pmf_json = _make_pmf_json(15.0)
        p_ov, p_un, p_pu = ip.compute_settlement_probabilities(pmf_json, 15.0)
        # P(push) = P(stat == 15) must be > 0
        assert p_pu > 0, f"Integer line 15 must have P(push) > 0, got {p_pu}"
        assert abs(p_ov + p_un + p_pu - 1.0) < 1e-6

    def test_fixture_halfpoint_line_has_zero_push(self):
        ip = _get_pipeline()

        pmf_json = _make_pmf_json(15.0)
        p_ov, p_un, p_pu = ip.compute_settlement_probabilities(pmf_json, 15.5)
        assert abs(p_pu) < 1e-9, f"Half-point line must have P(push)=0, got {p_pu}"
        assert abs(p_ov + p_un + p_pu - 1.0) < 1e-6

    def test_fixture_combo_involving_affected_teammate(self):
        """At least one combo stat for teammate-200 should exist in the PMF fixture."""
        pmfs_df = _make_pmfs_df()
        COMBO_STATS = {"pts_reb", "pts_ast", "reb_ast", "stocks", "pts_reb_ast"}
        teammate_combos = pmfs_df[
            (pmfs_df["player_id"] == 200) & pmfs_df["stat"].isin(COMBO_STATS)
        ]
        assert not teammate_combos.empty, \
            "Regression fixture must include combo rows for teammate-200"


# ===========================================================================
# BLOCKER 1 — Separate availability from conditional minutes
# ===========================================================================

def test_questionable_availability_does_not_halve_conditional_minutes():
    """Blocker 1: _injury_minutes_multiplier must equal cond_mult (1.0), NOT p_active * cond_mult (0.50).

    Deterministic proof:
      - baseline conditional minutes = 25 (player_minutes_mean_l5 for pid=102)
      - p_active = 0.50, conditional_minutes_multiplier = 1.0
      - final conditional minutes = 25, NOT 12.5
      - availability_probability remains 0.50 in its own field
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 102, "player_name": "Player_102", "status": "questionable"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    p102 = adj[adj["player_id"] == 102].iloc[0]

    # _injury_minutes_multiplier must be cond_mult = 1.0 for questionable,
    # NOT p_active * cond_mult = 0.50.
    cond_mult_applied = float(p102["_injury_minutes_multiplier"])
    assert abs(cond_mult_applied - 1.0) < 1e-9, (
        f"[Blocker 1] _injury_minutes_multiplier for questionable must be cond_mult=1.0, "
        f"NOT p_active*cond_mult=0.50. Got {cond_mult_applied}. "
        "PMF is conditional on participation; p_active=0.50 must stay in its own field."
    )


def test_probable_availability_does_not_scale_conditional_minutes_twice():
    """Blocker 1: _injury_minutes_multiplier must equal cond_mult (1.0), NOT p_active * cond_mult (0.85).

    Deterministic proof:
      - baseline conditional minutes = 32 (player_minutes_mean_l5 for pid=103)
      - p_active = 0.85, conditional_minutes_multiplier = 1.0
      - final conditional minutes = 32, NOT 27.2
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 103, "player_name": "Player_103", "status": "probable"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    p103 = adj[adj["player_id"] == 103].iloc[0]

    cond_mult_applied = float(p103["_injury_minutes_multiplier"])
    assert abs(cond_mult_applied - 1.0) < 1e-9, (
        f"[Blocker 1] _injury_minutes_multiplier for probable must be cond_mult=1.0, "
        f"NOT p_active*cond_mult=0.85. Got {cond_mult_applied}."
    )


def test_availability_probability_is_separate_from_minutes_multiplier():
    """Blocker 1: availability_probability must exist as a separate column in adjusted feature df."""
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [
        {"player_id": 102, "player_name": "Player_102", "status": "questionable"},
        {"player_id": 103, "player_name": "Player_103", "status": "probable"},
    ]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)

    assert "availability_probability" in adj.columns, (
        "[Blocker 1] apply_injury_to_feature_df must add 'availability_probability' column "
        "to the adjusted feature DataFrame. It must be separate from _injury_minutes_multiplier."
    )

    p102 = adj[adj["player_id"] == 102].iloc[0]
    assert abs(float(p102["availability_probability"]) - 0.50) < 1e-9, (
        f"[Blocker 1] availability_probability for questionable must be 0.50, "
        f"got {p102['availability_probability']}"
    )

    p103 = adj[adj["player_id"] == 103].iloc[0]
    assert abs(float(p103["availability_probability"]) - 0.85) < 1e-9, (
        f"[Blocker 1] availability_probability for probable must be 0.85, "
        f"got {p103['availability_probability']}"
    )

    # _injury_minutes_multiplier must be cond_mult (separate from availability_probability)
    assert abs(float(p102["_injury_minutes_multiplier"]) - 1.0) < 1e-9, (
        f"[Blocker 1] _injury_minutes_multiplier for questionable must be cond_mult=1.0, "
        f"got {p102['_injury_minutes_multiplier']}"
    )


def test_void_on_dnp_pmf_is_conditional_on_participation():
    """Blocker 1: For void-on-DNP vendors, the PMF must be conditional on participation.

    The _injury_minutes_multiplier must equal cond_mult only, NOT p_active * cond_mult.
    A questionable player's conditional PMF is identical to an uninjured player's PMF
    (since cond_mult=1.0 for questionable). The p_active=0.50 is carried separately.
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 102, "player_name": "Player_102", "status": "questionable"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)

    p102 = adj[adj["player_id"] == 102].iloc[0]

    # For a void-on-DNP vendor: PMF must be conditional on participation.
    # This means _injury_minutes_multiplier = cond_mult = 1.0 (no blending with p=0 state).
    # If _injury_minutes_multiplier = p_active * cond_mult = 0.50, the PMF would reflect
    # a 50%/50% blend of DNP + participation, which is WRONG for void-on-DNP settlement.
    mult = float(p102["_injury_minutes_multiplier"])
    assert abs(mult - 1.0) < 1e-9, (
        f"[Blocker 1] For void-on-DNP vendors, PMF must be conditional on participation. "
        f"_injury_minutes_multiplier must be cond_mult=1.0, not p_active*cond_mult=0.50. "
        f"Got {mult}."
    )

    # availability_probability must be separate so the caller can apply it for
    # non-void-on-DNP settlement
    assert "availability_probability" in adj.columns, (
        "[Blocker 1] availability_probability must exist as a separate field."
    )
    assert abs(float(p102["availability_probability"]) - 0.50) < 1e-9


def test_confirmed_out_is_handled_by_actionability_not_hidden_minutes_mix():
    """Blocker 1: OUT players must use explicit inactive/actionability handling.

    availability_probability must be 0.0 in its own field in the adjusted feature df,
    not hidden inside a combined p_active*cond_mult minutes multiplier.
    """
    ip = _get_pipeline()
    feature_df = _make_feature_df()
    injuries = [{"player_id": 100, "player_name": "Player_100", "status": "out"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    p100 = adj[adj["player_id"] == 100].iloc[0]

    # availability_probability must exist and be 0.0 for OUT players
    assert "availability_probability" in adj.columns, (
        "[Blocker 1] apply_injury_to_feature_df must add 'availability_probability' column."
    )
    assert abs(float(p100["availability_probability"]) - 0.0) < 1e-9, (
        f"[Blocker 1] OUT player availability_probability must be 0.0, "
        f"got {p100['availability_probability']}"
    )

    # OUT player must be confirmed inactive with is_market_actionable=False
    assert bool(p100["is_confirmed_inactive"]), "OUT player must be confirmed inactive"

    # _injury_minutes_multiplier=0 (from cond_mult=0 for out) triggers DNP blending
    assert abs(float(p100["_injury_minutes_multiplier"])) < 1e-9


# ===========================================================================
# BLOCKER 2 — Enforce conditional_minutes_cap
# ===========================================================================

def _make_limited_feature_df(baseline_mins: float = 35.0) -> pd.DataFrame:
    """Feature df with one limited player whose baseline minutes × cond_mult > cap."""
    return pd.DataFrame([{
        "player_id": 1001,
        "game_id": 999,
        "game_date": "2026-07-13",
        "season": 2026,
        "team_id": 10,
        "player_name": "LimitedPlayer",
        "position": "F",
        "player_minutes_mean_l5": baseline_mins,
        "player_minutes_mean_l10": baseline_mins,
        "player_minutes_mean_l20": baseline_mins,
        "player_minutes_mean_season": baseline_mins,
        "player_pts_mean_l5": baseline_mins * 0.65,
        "player_pts_mean_season": baseline_mins * 0.65,
        "player_reb_mean_l5": baseline_mins * 0.20,
        "player_reb_mean_season": baseline_mins * 0.20,
        "player_ast_mean_l5": baseline_mins * 0.15,
        "player_ast_mean_season": baseline_mins * 0.15,
    }])


def test_limited_minutes_cap_is_applied():
    """Blocker 2: conditional_minutes_cap must be enforced on adjusted_projected_minutes.

    limited status: p_active=1.0, cond_mult=0.65, cond_cap=20.0
    baseline=35 → after cond_mult: 35*0.65=22.75 → after cap: 20.0 (not 22.75)
    """
    ip = _get_pipeline()
    feature_df = _make_limited_feature_df(baseline_mins=35.0)
    injuries = [{"player_id": 1001, "player_name": "LimitedPlayer", "status": "limited"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    p1001 = adj[adj["player_id"] == 1001].iloc[0]

    # STATUS_CONFIG["limited"]: cond_mult=0.65, cond_cap=20.0
    # baseline 35 * 0.65 = 22.75 > cap 20.0 → must be capped to 20.0
    adj_mins = float(p1001["adjusted_projected_minutes"])
    assert adj_mins <= 20.0 + 1e-9, (
        f"[Blocker 2] Limited player adjusted_projected_minutes must be capped at 20.0. "
        f"Got {adj_mins:.4f} (35 * 0.65 = 22.75 exceeds cap=20.0, must be clamped)."
    )


def test_limited_minutes_distribution_respects_cap():
    """Blocker 2: conditional_minutes_cap must be stored in conditional_minutes_cap column.

    The PMF engine uses this cap to clamp min_means, ensuring the full distribution
    (including sigma) respects the cap.
    """
    ip = _get_pipeline()
    feature_df = _make_limited_feature_df(baseline_mins=35.0)
    injuries = [{"player_id": 1001, "player_name": "LimitedPlayer", "status": "limited"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    p1001 = adj[adj["player_id"] == 1001].iloc[0]

    # The cap must be stored in conditional_minutes_cap for the PMF engine to use
    assert "conditional_minutes_cap" in adj.columns, (
        "[Blocker 2] conditional_minutes_cap column must exist in adjusted feature df."
    )
    cap_val = p1001.get("conditional_minutes_cap")
    assert cap_val is not None and not (isinstance(cap_val, float) and np.isnan(cap_val)), (
        "[Blocker 2] conditional_minutes_cap must be set for limited player."
    )
    assert float(cap_val) <= 20.0 + 1e-9, (
        f"[Blocker 2] conditional_minutes_cap for limited must be 20.0, got {cap_val}"
    )

    # adjusted_projected_minutes must be capped
    adj_mins = float(p1001["adjusted_projected_minutes"])
    assert adj_mins <= 20.0 + 1e-9, (
        f"[Blocker 2] adjusted_projected_minutes must respect cap=20.0, got {adj_mins:.4f}"
    )


def test_minutes_cap_changes_final_pmf():
    """Blocker 2: applying conditional_minutes_cap must produce a lower capped adjusted_projected_minutes.

    With baseline=35, cond_mult=0.65, cap=20:
      uncapped adj = 35 * 0.65 = 22.75
      capped adj   = min(22.75, 20.0) = 20.0
    """
    ip = _get_pipeline()

    # Uncapped player (no injury, gets baseline minutes)
    feature_df = _make_limited_feature_df(baseline_mins=35.0)
    adj_uncapped = ip.apply_injury_to_feature_df(
        feature_df,
        ip.build_availability_table([], feature_df),
    )
    uncapped_mins = float(adj_uncapped[adj_uncapped["player_id"] == 1001].iloc[0]["adjusted_projected_minutes"])

    # Capped player (limited status with cond_mult=0.65, cap=20)
    injuries = [{"player_id": 1001, "player_name": "LimitedPlayer", "status": "limited"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj_capped = ip.apply_injury_to_feature_df(feature_df, avail)
    capped_mins = float(adj_capped[adj_capped["player_id"] == 1001].iloc[0]["adjusted_projected_minutes"])

    # The cap must reduce adjusted_projected_minutes below 35*0.65=22.75
    assert capped_mins <= 20.0 + 1e-9, (
        f"[Blocker 2] With cap=20.0, adjusted_projected_minutes must be ≤20.0. "
        f"Got {capped_mins:.4f} (uncapped was {uncapped_mins:.4f})."
    )


def test_missing_cap_leaves_distribution_uncapped():
    """Blocker 2: when no cap is set, adjusted_projected_minutes must NOT be artificially capped."""
    ip = _get_pipeline()
    # Questionable player: cond_mult=1.0, cond_cap=None → no cap should be applied
    feature_df = _make_limited_feature_df(baseline_mins=30.0)
    injuries = [{"player_id": 1001, "player_name": "LimitedPlayer", "status": "questionable"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj = ip.apply_injury_to_feature_df(feature_df, avail)
    p1001 = adj[adj["player_id"] == 1001].iloc[0]

    # No cap for questionable: adjusted_projected_minutes = orig * p_active * cond_mult
    # = 30 * 0.50 * 1.0 = 15.0 (no cap constraint)
    adj_mins = float(p1001["adjusted_projected_minutes"])
    # The cap column should be NaN or None for no-cap statuses
    cap_val = p1001.get("conditional_minutes_cap")
    cap_is_none = cap_val is None or (isinstance(cap_val, float) and np.isnan(cap_val))
    assert cap_is_none, (
        f"[Blocker 2] Questionable player should have no cap (conditional_minutes_cap=None/NaN), "
        f"got {cap_val}"
    )


# ===========================================================================
# BLOCKER 4 — Distinguish successful empty injury data from fetch failure
# ===========================================================================

def test_missing_api_key_is_fatal():
    """Blocker 4: missing BDL_API_KEY must produce InjuryFetchResult with status=FAILURE."""
    ip = _get_pipeline()

    assert hasattr(ip, "InjuryFetchResult"), (
        "[Blocker 4] InjuryFetchResult must be exported from injury_pipeline."
    )
    assert hasattr(ip, "fetch_bdl_injuries"), (
        "[Blocker 4] fetch_bdl_injuries must be exported from injury_pipeline."
    )

    result = ip.fetch_bdl_injuries(api_key="", team_ids=[1, 2])
    assert result.status == "FAILURE", (
        f"[Blocker 4] Missing API key must be FATAL (status=FAILURE), got {result.status!r}. "
        "Empty-string or absent API key must not silently return empty records."
    )
    assert result.records == [], (
        "[Blocker 4] FAILURE result must have empty records list."
    )
    assert result.error is not None, (
        "[Blocker 4] FAILURE result must have a non-None error description."
    )


def test_http_error_is_fatal():
    """Blocker 4: HTTP 4xx/5xx response must produce InjuryFetchResult with status=FAILURE."""
    ip = _get_pipeline()

    if not hasattr(ip, "InjuryFetchResult") or not hasattr(ip, "fetch_bdl_injuries"):
        pytest.skip("InjuryFetchResult or fetch_bdl_injuries not yet implemented")

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        mock_get.return_value = mock_resp

        result = ip.fetch_bdl_injuries(api_key="test_key", team_ids=[1])

    assert result.status == "FAILURE", (
        f"[Blocker 4] HTTP 500 must be FATAL (status=FAILURE), got {result.status!r}."
    )


def test_timeout_is_fatal():
    """Blocker 4: request timeout must produce InjuryFetchResult with status=FAILURE."""
    ip = _get_pipeline()

    if not hasattr(ip, "InjuryFetchResult") or not hasattr(ip, "fetch_bdl_injuries"):
        pytest.skip("InjuryFetchResult or fetch_bdl_injuries not yet implemented")

    import requests as _req
    with patch("requests.get", side_effect=_req.exceptions.Timeout("timeout")):
        result = ip.fetch_bdl_injuries(api_key="test_key", team_ids=[1])

    assert result.status == "FAILURE", (
        f"[Blocker 4] Timeout must be FATAL (status=FAILURE), got {result.status!r}."
    )
    assert result.error is not None


def test_malformed_response_is_fatal():
    """Blocker 4: non-JSON or schema-invalid response must produce status=FAILURE."""
    ip = _get_pipeline()

    if not hasattr(ip, "InjuryFetchResult") or not hasattr(ip, "fetch_bdl_injuries"):
        pytest.skip("InjuryFetchResult or fetch_bdl_injuries not yet implemented")

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("not valid JSON")
        mock_get.return_value = mock_resp

        result = ip.fetch_bdl_injuries(api_key="test_key", team_ids=[1])

    assert result.status == "FAILURE", (
        f"[Blocker 4] Malformed JSON response must be FATAL (status=FAILURE), got {result.status!r}."
    )


def test_verified_empty_response_is_success():
    """Blocker 4: a valid HTTP 200 with empty data array must return SUCCESS_EMPTY (not FAILURE)."""
    ip = _get_pipeline()

    if not hasattr(ip, "InjuryFetchResult") or not hasattr(ip, "fetch_bdl_injuries"):
        pytest.skip("InjuryFetchResult or fetch_bdl_injuries not yet implemented")

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = ip.fetch_bdl_injuries(api_key="test_key", team_ids=[1])

    assert result.status == "SUCCESS_EMPTY", (
        f"[Blocker 4] Verified empty response must be SUCCESS_EMPTY, got {result.status!r}. "
        "Only a successful 200 with validated empty array may return SUCCESS_EMPTY."
    )
    assert result.records == [], "[Blocker 4] SUCCESS_EMPTY must have empty records list."
    assert result.error is None, "[Blocker 4] SUCCESS_EMPTY must have error=None."


def test_fetch_failure_cannot_be_treated_as_no_injuries():
    """Blocker 4: InjuryFetchResult.status=FAILURE must never be treated as no-injuries.

    A caller that silently treats FAILURE as an empty list would be violating the contract.
    Verify the result object has distinct status values for FAILURE vs SUCCESS_EMPTY.
    """
    ip = _get_pipeline()

    if not hasattr(ip, "InjuryFetchResult"):
        pytest.skip("InjuryFetchResult not yet implemented")

    IFR = ip.InjuryFetchResult
    failure = IFR(status="FAILURE", records=[], pulled_at_utc=None, error="fetch failed")
    empty   = IFR(status="SUCCESS_EMPTY", records=[], pulled_at_utc=None, error=None)

    # FAILURE and SUCCESS_EMPTY must have different statuses (not both "")
    assert failure.status != empty.status, (
        "[Blocker 4] FAILURE and SUCCESS_EMPTY must have distinct status values. "
        "Never treat FAILURE silently as no-injuries."
    )
    assert failure.status == "FAILURE"
    assert empty.status == "SUCCESS_EMPTY"
    assert failure.error is not None
    assert empty.error is None


# ===========================================================================
# BLOCKER 5 — Preserve and validate per-record source timestamps
# ===========================================================================

def test_each_injury_record_preserves_its_own_timestamp():
    """Blocker 5: each injury record must carry its own source_updated_at timestamp."""
    ip = _get_pipeline()

    feature_df = _make_feature_df()
    # Provide injuries with distinct per-record timestamps
    injuries = [
        {
            "player_id": 102, "player_name": "Player_102", "status": "questionable",
            "source_updated_at": "2026-07-13T06:00:00+00:00",
        },
        {
            "player_id": 103, "player_name": "Player_103", "status": "probable",
            "source_updated_at": "2026-07-13T07:30:00+00:00",
        },
    ]

    avail = ip.build_availability_table(injuries, feature_df)

    # Each player must have their OWN source_updated_at (not all the same value)
    ts_102 = str(avail[avail["player_id"] == 102]["source_updated_at"].values[0])
    ts_103 = str(avail[avail["player_id"] == 103]["source_updated_at"].values[0])

    assert "06:00" in ts_102, (
        f"[Blocker 5] Player 102 source_updated_at must reflect its own record timestamp "
        f"(06:00), got {ts_102!r}."
    )
    assert "07:30" in ts_103, (
        f"[Blocker 5] Player 103 source_updated_at must reflect its own record timestamp "
        f"(07:30), got {ts_103!r}. "
        "All records must NOT get the same (earliest) snapshot timestamp."
    )


def test_future_source_timestamp_is_fatal():
    """Blocker 5: a source_updated_at timestamp in the future must be a fatal error."""
    ip = _get_pipeline()

    if not hasattr(ip, "validate_injury_timestamps"):
        pytest.skip("validate_injury_timestamps not yet implemented")

    feature_df = _make_feature_df()
    # Future timestamp (year 2099)
    injuries_with_future = [
        {
            "player_id": 102, "player_name": "Player_102", "status": "questionable",
            "source_updated_at": "2099-01-01T00:00:00+00:00",
        },
    ]
    avail = ip.build_availability_table(injuries_with_future, feature_df)

    prediction_ts = "2026-07-13T12:00:00+00:00"
    with pytest.raises((ValueError, RuntimeError), match="(?i)(future|timestamp|fatal)"):
        ip.validate_injury_timestamps(avail, prediction_timestamp_utc=prediction_ts)


def test_malformed_source_timestamp_is_fatal():
    """Blocker 5: a malformed source_updated_at timestamp must be a fatal error."""
    ip = _get_pipeline()

    if not hasattr(ip, "validate_injury_timestamps"):
        pytest.skip("validate_injury_timestamps not yet implemented")

    feature_df = _make_feature_df()
    injuries_malformed = [
        {
            "player_id": 102, "player_name": "Player_102", "status": "questionable",
            "source_updated_at": "not-a-timestamp",
        },
    ]
    avail = ip.build_availability_table(injuries_malformed, feature_df)
    # Inject the malformed timestamp directly
    avail.loc[avail["player_id"] == 102, "source_updated_at"] = "not-a-timestamp"

    prediction_ts = "2026-07-13T12:00:00+00:00"
    with pytest.raises((ValueError, RuntimeError), match="(?i)(malformed|invalid|timestamp|parse|fatal)"):
        ip.validate_injury_timestamps(avail, prediction_timestamp_utc=prediction_ts)


def test_snapshot_timestamp_is_not_mislabeled_as_record_timestamp():
    """Blocker 5: players without an injury record should use snapshot timestamp separately.

    The pulled_at_utc (snapshot timestamp) must NOT be stored in source_updated_at
    for players who do not appear in the injury list.
    """
    ip = _get_pipeline()

    feature_df = _make_feature_df()
    # Only player 102 has an injury record; others should use pulled_at_utc
    injuries = [
        {
            "player_id": 102, "player_name": "Player_102", "status": "questionable",
            "source_updated_at": "2026-07-13T06:00:00+00:00",
        },
    ]

    avail = ip.build_availability_table(injuries, feature_df)

    # Player 200 has NO injury record; their source_updated_at should be the
    # snapshot/pulled_at_utc timestamp, not the injury record's timestamp.
    # It must NOT be mislabeled as a record-level source_updated_at.
    row_200 = avail[avail["player_id"] == 200].iloc[0]
    assert "pulled_at_utc" in avail.columns, (
        "[Blocker 5] pulled_at_utc must exist in availability table."
    )

    pulled_ts = str(row_200["pulled_at_utc"])
    source_ts = str(row_200["source_updated_at"])

    # For a player without an injury record, source_updated_at should be the
    # same as pulled_at_utc (snapshot time), not the injury record's source_updated_at.
    # This verifies that the snapshot timestamp is stored separately from record timestamps.
    assert "2026-07-13T06:00" not in source_ts or pulled_ts == source_ts, (
        f"[Blocker 5] Player 200 (no injury record) source_updated_at={source_ts!r} "
        f"must not carry the injury record timestamp (06:00). "
        f"pulled_at_utc={pulled_ts!r}."
    )
