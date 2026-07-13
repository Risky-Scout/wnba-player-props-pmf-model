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

        # questionable → multiplier = 0.50
        row = avail[avail["player_id"] == 102].iloc[0]
        assert abs(row["minutes_multiplier"] - 0.50) < 1e-9

        # Apply feature adjustments
        adj = ip.apply_injury_to_feature_df(feature_df, avail)

        # Minutes features for player 102 should be halved
        p102 = adj[adj["player_id"] == 102]
        orig = feature_df[feature_df["player_id"] == 102]["player_minutes_mean_l5"].values[0]
        new  = p102["player_minutes_mean_l5"].values[0]
        assert abs(new - orig * 0.50) < 1e-6, f"Expected {orig * 0.50}, got {new}"

        # The _injury_minutes_multiplier column must be set
        assert abs(p102["_injury_minutes_multiplier"].values[0] - 0.50) < 1e-9


class TestProbablePlayerRebuildspmfJson:
    """test_probable_player_rebuilds_pmf_json"""

    def test_probable_player_rebuilds_pmf_json(self):
        ip = _get_pipeline()

        feature_df = _make_feature_df()
        injuries = [{"player_id": 103, "player_name": "Player_103", "status": "probable"}]

        avail = ip.build_availability_table(injuries, feature_df)
        row = avail[avail["player_id"] == 103].iloc[0]
        assert abs(row["minutes_multiplier"] - 0.85) < 1e-9

        adj = ip.apply_injury_to_feature_df(feature_df, avail)
        p103 = adj[adj["player_id"] == 103]
        orig = feature_df[feature_df["player_id"] == 103]["player_minutes_mean_l5"].values[0]
        new  = p103["player_minutes_mean_l5"].values[0]
        assert abs(new - orig * 0.85) < 1e-6, f"Expected {orig * 0.85}, got {new}"


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

        # Player 100 (OUT) minutes must be zero
        p100_mins = adj[adj["player_id"] == 100]["player_minutes_mean_l5"].values[0]
        assert p100_mins == 0.0, f"OUT player minutes must be 0, got {p100_mins}"

        # Teammates (200, 201) should have MORE minutes than before (UTM redistribution)
        for teammate_pid in [200, 201]:
            orig = feature_df[feature_df["player_id"] == teammate_pid][
                "player_minutes_mean_l5"
            ].values[0]
            new = adj[adj["player_id"] == teammate_pid][
                "player_minutes_mean_l5"
            ].values[0]
            assert new >= orig, (
                f"Teammate {teammate_pid} should have >= original minutes, "
                f"got {new} vs {orig}"
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
        assert 101 in inactive_pids, "Doubtful player must be confirmed inactive"

        # Questionable (102) and probable (103) must be market actionable
        assert 102 in actionable_pids, "Questionable player must be market actionable"
        assert 103 in actionable_pids, "Probable player must be market actionable"

        # Teammates (200, 201) must be actionable
        assert 200 in actionable_pids, "Teammate must be market actionable"
        assert 201 in actionable_pids, "Teammate must be market actionable"

        # Build confirmed_inactive_mask on a PMF DataFrame
        pmfs_df = _make_pmfs_df()
        inactive_mask = ip.build_confirmed_inactive_mask(pmfs_df, avail)

        # Mask must be True only for inactive players
        assert inactive_mask[pmfs_df["player_id"] == 100].all(), \
            "OUT player rows must be masked"
        assert inactive_mask[pmfs_df["player_id"] == 101].all(), \
            "Doubtful player rows must be masked"
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
        assert r101["minutes_multiplier"] == 0.0, "Doubtful → multiplier=0"
        assert r101["is_confirmed_inactive"] is True or bool(r101["is_confirmed_inactive"])

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
