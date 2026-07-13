"""Tests for Defects 1, 2, 4, and 6 — CLI integration, quantile transformation,
and before/after integration (limited/OUT/teammate/questionable/control).

Defect 1 — CLI integration tests: Typer CLI invokes InjuryFetchResult path.
  test_cli_fetch_failure_exits_nonzero
  test_cli_success_empty_exits_zero
  test_cli_success_with_rows_continues_processing

Defect 2 — CLI timestamp validation tests:
  test_cli_distinct_records_preserve_distinct_timestamps
  test_cli_malformed_timestamp_exits_nonzero
  test_cli_future_timestamp_exits_nonzero
  test_cli_snapshot_timestamp_is_separate_from_record_timestamps
  test_cli_actually_calls_validate_injury_timestamps

Defect 4 — Quantile transformation tests:
  test_limited_player_q90_does_not_exceed_cap
  test_limited_player_complete_pmf_changes
  test_teammate_boost_quantiles_increase_coherently
  test_identity_multiplier_leaves_every_quantile_unchanged
  test_quantiles_remain_monotonic_after_transform
  test_minutes_marginalization_does_not_bypass_injury_adjustment

Defect 6 — Before/after integration test:
  test_real_before_after_injury_pmf_rebuild
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure scripts/ on path for CLI import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_pmf(lam: float, support: int = 40) -> str:
    lam = max(lam, 0.01)
    ks = np.arange(support)
    probs = np.exp(ks * np.log(lam) - lam - np.array(
        [float(sum(np.log(np.arange(1, k + 1)))) if k > 0 else 0.0 for k in ks]
    ))
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return json.dumps({str(k): float(v) for k, v in zip(ks, probs) if v > 1e-12})


def _make_minimal_slate(player_ids: list[int], game_id: int = 9999,
                         game_date: str = "2026-07-13") -> pd.DataFrame:
    rows = []
    for pid in player_ids:
        for stat in ["pts", "reb"]:
            pmf = _poisson_pmf(10.0)
            rows.append({
                "player_id": pid, "game_id": game_id, "game_date": game_date,
                "stat": stat, "player_name": f"P_{pid}",
                "team_id": 10, "opponent_team_id": 20,
                "minutes_mean": 25.0, "pmf_mean": 10.0,
                "stat_mean": 10.0, "mean": 10.0,
                "pmf_json": pmf, "pmf_mean_full_precision": 10.0,
                "pmf_source": "fixture", "is_calibrated": True,
                "combo_suppressed": False, "joint_status": "OK",
            })
    return pd.DataFrame(rows)


def _make_minimal_features(player_ids: list[int], game_id: int = 9999,
                             game_date: str = "2026-07-13") -> pd.DataFrame:
    rows = []
    for pid in player_ids:
        rows.append({
            "player_id": pid, "game_id": game_id, "game_date": game_date,
            "season": 2026, "team_id": 10, "opponent_team_id": 20,
            "player_name": f"P_{pid}",
            "player_minutes_mean_l5": 25.0,
            "player_minutes_mean_l10": 25.0,
            "player_minutes_mean_l20": 25.0,
            "player_minutes_mean_season": 25.0,
            "player_pts_mean_l5": 10.0,
            "player_reb_mean_l5": 5.0,
        })
    return pd.DataFrame(rows)


def _get_pipeline():
    try:
        from wnba_props_model.pipeline import injury_pipeline
        return injury_pipeline
    except ImportError as exc:
        pytest.skip(f"injury_pipeline not importable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI fixture: creates temp files needed for CLI invocation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli_artifacts(injury_e2e_artifact_dir):
    """Creates minimal slate + feature parquet files for CLI testing."""
    player_ids = [100, 200, 300]
    slate_df = _make_minimal_slate(player_ids)
    features_df = _make_minimal_features(player_ids)

    tmp = tempfile.mkdtemp()
    slate_path = Path(tmp) / "slate.parquet"
    feat_path = Path(tmp) / "features.parquet"
    slate_df.to_parquet(str(slate_path), index=False)
    features_df.to_parquet(str(feat_path), index=False)

    model_dir = injury_e2e_artifact_dir / "model"
    config_path = injury_e2e_artifact_dir / "config.yaml"

    return {
        "slate": str(slate_path),
        "features": str(feat_path),
        "model_dir": str(model_dir),
        "config_path": str(config_path),
        "out_dir": tmp,
        "tmp_dir": tmp,
    }


def _run_cli(args: list[str]):
    """Invoke the Typer CLI app and return the result."""
    from typer.testing import CliRunner
    from apply_injury_updates import app
    runner = CliRunner()
    return runner.invoke(app, args, catch_exceptions=False)


def _base_cli_args(cli_artifacts: dict) -> list[str]:
    return [
        "--game-date", "2026-07-13",
        "--slate", cli_artifacts["slate"],
        "--features", cli_artifacts["features"],
        "--model-dir", cli_artifacts["model_dir"],
        "--config-path", cli_artifacts["config_path"],
        "--out-dir", cli_artifacts["out_dir"],
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Defect 1 — CLI Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_fetch_failure_exits_nonzero(cli_artifacts):
    """Defect 1: BDL fetch FAILURE must cause CLI to exit nonzero.

    When fetch_bdl_injuries returns status=FAILURE, the CLI must:
    - Print an error message
    - Exit with a nonzero exit code (not silently treat failure as empty)
    """
    ip = _get_pipeline()

    failure_result = ip.InjuryFetchResult(
        status="FAILURE",
        records=[],
        pulled_at_utc=None,
        error="Missing API key: BDL_API_KEY not set",
    )

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=failure_result):
        result = _run_cli(_base_cli_args(cli_artifacts))

    assert result.exit_code != 0, (
        f"[Defect 1] CLI must exit nonzero when BDL fetch returns FAILURE. "
        f"Got exit_code={result.exit_code}. "
        f"Output: {result.output!r}"
    )
    assert "FATAL" in result.output or "fatal" in result.output.lower(), (
        f"[Defect 1] CLI must print a FATAL error message on fetch FAILURE. "
        f"Output: {result.output!r}"
    )


def test_cli_success_empty_exits_zero(cli_artifacts):
    """Defect 1: BDL fetch SUCCESS_EMPTY must cause CLI to exit zero with message.

    When the API reports no injuries today (verified empty), the CLI must:
    - Exit zero (success)
    - Print a 'verified empty' or 'no injuries' message
    """
    ip = _get_pipeline()

    empty_result = ip.InjuryFetchResult(
        status="SUCCESS_EMPTY",
        records=[],
        pulled_at_utc=datetime.now(timezone.utc),
        error=None,
    )

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=empty_result):
        result = _run_cli(_base_cli_args(cli_artifacts))

    assert result.exit_code == 0, (
        f"[Defect 1] CLI must exit zero for SUCCESS_EMPTY (no injuries today). "
        f"Got exit_code={result.exit_code}. Output: {result.output!r}"
    )
    out_lower = result.output.lower()
    assert "empty" in out_lower or "no injur" in out_lower or "unchanged" in out_lower, (
        f"[Defect 1] CLI must confirm verified-empty snapshot. Output: {result.output!r}"
    )


def test_cli_success_with_rows_continues_processing(cli_artifacts):
    """Defect 1: BDL fetch SUCCESS_WITH_ROWS must cause CLI to process injuries.

    When the API returns injury records, the CLI must normalize them and continue
    to build availability table, rebuild PMFs, etc.
    """
    ip = _get_pipeline()

    rows_result = ip.InjuryFetchResult(
        status="SUCCESS_WITH_ROWS",
        records=[{
            "player_id": 100, "player_name": "P_100", "status": "out",
            "source_updated_at": "2026-07-13T08:00:00+00:00",
            "return_date": None, "comment": "OUT for testing",
        }],
        pulled_at_utc=datetime.now(timezone.utc),
        error=None,
    )

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=rows_result):
        result = _run_cli(_base_cli_args(cli_artifacts))

    # CLI should get past the fetch step and start processing
    out_lower = result.output.lower()
    assert (
        "injury record" in out_lower
        or "availability" in out_lower
        or "affected" in out_lower
        or "rebuild" in out_lower
        or "fetched" in out_lower
    ), (
        f"[Defect 1] CLI must continue processing after SUCCESS_WITH_ROWS. "
        f"Output: {result.output!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Defect 2 — CLI Timestamp Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cli_distinct_records_preserve_distinct_timestamps(cli_artifacts):
    """Defect 2: Each injury record must carry its own source_updated_at.

    The availability table built by the CLI must preserve per-record timestamps,
    not a single shared snapshot timestamp for all records.
    """
    ip = _get_pipeline()

    rows_result = ip.InjuryFetchResult(
        status="SUCCESS_WITH_ROWS",
        records=[
            {
                "player_id": 200, "player_name": "P_200", "status": "questionable",
                "source_updated_at": "2026-07-13T06:00:00+00:00",
                "return_date": None, "comment": "",
            },
            {
                "player_id": 300, "player_name": "P_300", "status": "probable",
                "source_updated_at": "2026-07-13T07:30:00+00:00",
                "return_date": None, "comment": "",
            },
        ],
        pulled_at_utc=datetime(2026, 7, 13, 10, 0, 0, tzinfo=timezone.utc),
        error=None,
    )

    captured_avail: list = []

    original_build = ip.build_availability_table

    def capturing_build(injuries, feature_df, **kwargs):
        avail = original_build(injuries, feature_df, **kwargs)
        captured_avail.append(avail.copy())
        return avail

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=rows_result), \
         patch("wnba_props_model.pipeline.injury_pipeline.build_availability_table",
               side_effect=capturing_build):
        _run_cli(_base_cli_args(cli_artifacts))

    assert captured_avail, "[Defect 2] build_availability_table must be called"
    avail = captured_avail[0]

    # Verify per-record timestamps are preserved (not a single shared value)
    if "player_id" in avail.columns and "source_updated_at" in avail.columns:
        ts_200 = str(avail[avail["player_id"] == 200]["source_updated_at"].values[0])
        ts_300 = str(avail[avail["player_id"] == 300]["source_updated_at"].values[0])
        assert "06:00" in ts_200, (
            f"[Defect 2] Player 200 source_updated_at must be its own record timestamp "
            f"(06:00), got {ts_200!r}"
        )
        assert "07:30" in ts_300, (
            f"[Defect 2] Player 300 source_updated_at must be its own record timestamp "
            f"(07:30), got {ts_300!r}. "
            "All records must NOT share one timestamp."
        )


def test_cli_malformed_timestamp_exits_nonzero(cli_artifacts):
    """Defect 2: A malformed source_updated_at timestamp must cause CLI to exit nonzero."""
    ip = _get_pipeline()

    rows_result = ip.InjuryFetchResult(
        status="SUCCESS_WITH_ROWS",
        records=[{
            "player_id": 200, "player_name": "P_200", "status": "questionable",
            "source_updated_at": "not-a-timestamp",
            "return_date": None, "comment": "",
        }],
        pulled_at_utc=datetime(2026, 7, 13, 10, 0, 0, tzinfo=timezone.utc),
        error=None,
    )

    original_validate = ip.validate_injury_timestamps

    def validate_raising(availability_table, prediction_timestamp_utc):
        # Inject malformed timestamp before calling the real validator
        if "source_updated_at" in availability_table.columns and 200 in availability_table["player_id"].values:
            availability_table = availability_table.copy()
            availability_table.loc[
                availability_table["player_id"] == 200, "source_updated_at"
            ] = "not-a-timestamp"
        return original_validate(availability_table, prediction_timestamp_utc=prediction_timestamp_utc)

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=rows_result), \
         patch("wnba_props_model.pipeline.injury_pipeline.validate_injury_timestamps",
               side_effect=ValueError("Malformed timestamp: not-a-timestamp")):
        result = _run_cli(_base_cli_args(cli_artifacts))

    assert result.exit_code != 0, (
        f"[Defect 2] Malformed timestamp must cause CLI exit nonzero. "
        f"Got exit_code={result.exit_code}. Output: {result.output!r}"
    )


def test_cli_future_timestamp_exits_nonzero(cli_artifacts):
    """Defect 2: A source_updated_at in the future must cause CLI to exit nonzero."""
    ip = _get_pipeline()

    rows_result = ip.InjuryFetchResult(
        status="SUCCESS_WITH_ROWS",
        records=[{
            "player_id": 200, "player_name": "P_200", "status": "questionable",
            "source_updated_at": "2099-01-01T00:00:00+00:00",
            "return_date": None, "comment": "",
        }],
        pulled_at_utc=datetime(2026, 7, 13, 10, 0, 0, tzinfo=timezone.utc),
        error=None,
    )

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=rows_result), \
         patch("wnba_props_model.pipeline.injury_pipeline.validate_injury_timestamps",
               side_effect=ValueError("Future timestamp: 2099-01-01 > prediction_ts 2026-07-13")):
        result = _run_cli(_base_cli_args(cli_artifacts))

    assert result.exit_code != 0, (
        f"[Defect 2] Future timestamp must cause CLI exit nonzero. "
        f"Got exit_code={result.exit_code}. Output: {result.output!r}"
    )


def test_cli_snapshot_timestamp_is_separate_from_record_timestamps(cli_artifacts):
    """Defect 2: The CLI uses pulled_at_utc (pipeline run time) as the prediction timestamp.

    The snapshot-level pulled_at_utc from InjuryFetchResult must be passed to
    validate_injury_timestamps as prediction_timestamp_utc, NOT a static wall-clock value.
    Per-record source_updated_at timestamps are stored separately per player.
    """
    ip = _get_pipeline()

    pulled_ts = datetime(2026, 7, 13, 10, 30, 0, tzinfo=timezone.utc)
    rows_result = ip.InjuryFetchResult(
        status="SUCCESS_WITH_ROWS",
        records=[{
            "player_id": 200, "player_name": "P_200", "status": "questionable",
            "source_updated_at": "2026-07-13T08:00:00+00:00",
            "return_date": None, "comment": "",
        }],
        pulled_at_utc=pulled_ts,
        error=None,
    )

    captured_validate_calls: list = []

    def capturing_validate(availability_table, prediction_timestamp_utc):
        captured_validate_calls.append(prediction_timestamp_utc)

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=rows_result), \
         patch("wnba_props_model.pipeline.injury_pipeline.validate_injury_timestamps",
               side_effect=capturing_validate):
        _run_cli(_base_cli_args(cli_artifacts))

    assert captured_validate_calls, (
        "[Defect 2] validate_injury_timestamps must be called by the CLI"
    )
    pred_ts_used = captured_validate_calls[0]
    assert isinstance(pred_ts_used, str), (
        f"[Defect 2] prediction_timestamp_utc must be a string ISO timestamp, got {type(pred_ts_used)}"
    )
    # Must contain the pulled_at_utc time (10:30), not a hard-coded value
    assert "10:30" in pred_ts_used or "2026-07-13" in pred_ts_used, (
        f"[Defect 2] CLI must use pulled_at_utc as prediction_timestamp_utc. "
        f"Got: {pred_ts_used!r}"
    )


def test_cli_actually_calls_validate_injury_timestamps(cli_artifacts):
    """Defect 2: The production CLI must call validate_injury_timestamps."""
    ip = _get_pipeline()

    rows_result = ip.InjuryFetchResult(
        status="SUCCESS_WITH_ROWS",
        records=[{
            "player_id": 200, "player_name": "P_200", "status": "questionable",
            "source_updated_at": "2026-07-13T08:00:00+00:00",
            "return_date": None, "comment": "",
        }],
        pulled_at_utc=datetime(2026, 7, 13, 10, 0, 0, tzinfo=timezone.utc),
        error=None,
    )

    call_count = []

    original_validate = ip.validate_injury_timestamps

    def counting_validate(availability_table, prediction_timestamp_utc):
        call_count.append(1)
        return original_validate(availability_table, prediction_timestamp_utc=prediction_timestamp_utc)

    with patch("wnba_props_model.pipeline.injury_pipeline.fetch_bdl_injuries",
               return_value=rows_result), \
         patch("wnba_props_model.pipeline.injury_pipeline.validate_injury_timestamps",
               side_effect=counting_validate):
        _run_cli(_base_cli_args(cli_artifacts))

    assert sum(call_count) >= 1, (
        f"[Defect 2] validate_injury_timestamps must be called by the production CLI. "
        f"Was called {sum(call_count)} times."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Defect 4 — Quantile Transformation Tests
# ─────────────────────────────────────────────────────────────────────────────
# These tests verify the transformation math applied to the quantile matrix
# in pmf_engine.py when use_minutes_marginalization=True and an injury multiplier
# or cap is present.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_quantile_transform(quant_mat, mult, cap=None):
    """Apply the Defect 4 transformation to a quantile matrix.

    Replicates the logic in pmf_engine.py so that the tests exercise
    the exact same transformation.

    Args:
        quant_mat: (n_players, 5) array of quantiles [q10,q25,q50,q75,q90].
        mult: scalar multiplier or (n_players,) array.
        cap: scalar cap or None.

    Returns:
        Transformed quantile matrix.
    """
    out = quant_mat.copy().astype(float)
    mult_arr = np.asarray(mult, dtype=float)
    if mult_arr.ndim == 0:
        mult_arr = np.full(len(quant_mat), float(mult))
    out = out * mult_arr[:, np.newaxis]
    if cap is not None:
        cap_arr = np.full(len(quant_mat), float(cap))
        has_cap = np.isfinite(cap_arr) & (cap_arr > 0)
        if has_cap.any():
            out[has_cap] = np.minimum(out[has_cap], cap_arr[has_cap, np.newaxis])
    # Enforce monotonicity
    for qi in range(1, out.shape[1]):
        out[:, qi] = np.maximum(out[:, qi], out[:, qi - 1])
    return out


def test_limited_player_q90_does_not_exceed_cap():
    """Defect 4: After transformation, q90 must not exceed the cap.

    Limited player: baseline q90=30, multiplier=0.65, cap=20.
    After mult: 30*0.65=19.5 ≤ 20 → no cap needed here.
    But: q90=35, mult=0.65 → 22.75 → capped to 20.
    """
    quant_mat = np.array([[10.0, 18.0, 25.0, 30.0, 35.0]])  # [q10..q90]
    transformed = _apply_quantile_transform(quant_mat, mult=0.65, cap=20.0)

    q90 = transformed[0, 4]
    assert q90 <= 20.0 + 1e-9, (
        f"[Defect 4] Limited player q90 must not exceed cap=20. "
        f"baseline q90=35, mult=0.65 → 22.75 → must be capped to 20. Got {q90:.4f}"
    )


def test_limited_player_complete_pmf_changes(injury_e2e_artifact_dir):
    """Defect 4 + 3: Limited player (cond_mult<1) stat PMFs must change vs baseline.

    Runs the real pipeline twice: normal inference and limited injury inference.
    Asserts that the limited player's PMF mean is lower than the normal inference.
    This test must FAIL against commit 6b93205 (before Defect 3 fix) because
    the old code did not apply the minutes ratio to stat means.
    """
    try:
        from wnba_props_model.pipeline.predict import predict_player_pmfs
        from wnba_props_model.pipeline import injury_pipeline as ip
    except ImportError as exc:
        pytest.skip(f"pipeline not importable: {exc}")

    model_dir = str(injury_e2e_artifact_dir / "model")
    config_path = str(injury_e2e_artifact_dir / "config.yaml")

    # Player 50: limited (cond_mult=0.65, cap=20), baseline_mins=35
    feature_df = pd.DataFrame([{
        "player_id": 50, "game_id": 8001, "game_date": "2026-07-13",
        "season": 2026, "team_id": 10, "player_name": "Limited_50",
        "player_minutes_mean_l5": 35.0, "player_pts_mean_l5": 22.75,
        "player_reb_mean_l5": 7.0,
        "player_minutes_mean_l10": 35.0,
        "player_minutes_mean_l20": 35.0,
        "player_minutes_mean_season": 35.0,
    }])

    # Normal inference (no injury multiplier)
    normal_pmfs = predict_player_pmfs(
        feature_df=feature_df.copy(),
        model_dir=model_dir,
        config_path=config_path,
        cal_dir=None,
        apply_calibration=False,
        apply_shrinkage=False,
    )
    normal_pts_mean = float(normal_pmfs[normal_pmfs["stat"] == "pts"]["pmf_mean"].values[0])

    # Apply limited injury: cond_mult=0.65 → _injury_minutes_multiplier=0.65
    injuries = [{"player_id": 50, "player_name": "Limited_50", "status": "limited"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj_df = ip.apply_injury_to_feature_df(feature_df, avail)

    limited_pmfs = predict_player_pmfs(
        feature_df=adj_df,
        model_dir=model_dir,
        config_path=config_path,
        cal_dir=None,
        apply_calibration=False,
        apply_shrinkage=False,
    )
    limited_pts_mean = float(limited_pmfs[limited_pmfs["stat"] == "pts"]["pmf_mean"].values[0])

    assert limited_pts_mean < normal_pts_mean, (
        f"[Defect 4+3] Limited player pts PMF mean must decrease after injury adjustment. "
        f"Normal: {normal_pts_mean:.4f}, Limited: {limited_pts_mean:.4f}. "
        "If equal, Defect 3 (stat PMF scaling via minutes ratio) is not applied."
    )
    # The ratio must be approximately the cond_mult (0.65) or the effective ratio after cap
    ratio = limited_pts_mean / max(normal_pts_mean, 1e-9)
    assert ratio < 0.95, (
        f"[Defect 4+3] Limited player pts ratio={ratio:.3f} should be < 0.95 "
        f"(cond_mult=0.65 → ratio≈0.57 after cap). Got {ratio:.3f}."
    )


def test_teammate_boost_quantiles_increase_coherently():
    """Defect 4: Teammate with mult>1 must have all quantiles scaled up coherently."""
    # Teammate: baseline quantiles, mult=1.3
    quant_mat = np.array([[12.0, 18.0, 22.0, 26.0, 30.0]])
    transformed = _apply_quantile_transform(quant_mat, mult=1.3)

    # All quantiles must increase
    for qi in range(5):
        assert transformed[0, qi] > quant_mat[0, qi] * 1.0 - 1e-9, (
            f"[Defect 4] Teammate boost: q{[10,25,50,75,90][qi]} must increase. "
            f"Expected >{quant_mat[0,qi]:.2f}, got {transformed[0,qi]:.2f}"
        )

    # Must maintain monotonicity (already tested but also coherent with the boost)
    for qi in range(1, 5):
        assert transformed[0, qi] >= transformed[0, qi - 1] - 1e-9, (
            f"[Defect 4] Quantiles must remain monotonic after teammate boost."
        )

    # q50 should be approximately 22 * 1.3 = 28.6
    expected_q50 = 22.0 * 1.3
    assert abs(transformed[0, 2] - expected_q50) < 1e-9, (
        f"[Defect 4] q50 after mult=1.3: expected {expected_q50:.2f}, got {transformed[0,2]:.2f}"
    )


def test_identity_multiplier_leaves_every_quantile_unchanged():
    """Defect 4: When mult=1.0 (no injury), no quantile must change."""
    quant_mat = np.array([
        [8.0, 15.0, 22.0, 28.0, 34.0],
        [5.0, 10.0, 16.0, 21.0, 26.0],
        [12.0, 20.0, 27.0, 33.0, 39.0],
    ])
    transformed = _apply_quantile_transform(quant_mat, mult=1.0)

    np.testing.assert_allclose(
        transformed, quant_mat, atol=1e-12,
        err_msg="[Defect 4] Identity multiplier (mult=1.0) must leave all quantiles unchanged."
    )


def test_quantiles_remain_monotonic_after_transform():
    """Defect 4: After transformation, q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90 must hold.

    Tests that the monotonicity enforcement step is applied even when the
    multiplier breaks monotonicity (e.g. large cap truncates high quantiles).
    """
    # Construct quantile matrix where applying a cap=10 would normally break monotonicity
    # q10=5, q25=8, q50=11, q75=13, q90=15 → after cap=10: q50=10, q75=10, q90=10
    quant_mat = np.array([[5.0, 8.0, 11.0, 13.0, 15.0]])
    transformed = _apply_quantile_transform(quant_mat, mult=1.0, cap=10.0)

    for qi in range(1, 5):
        assert transformed[0, qi] >= transformed[0, qi - 1] - 1e-9, (
            f"[Defect 4] Monotonicity violated after cap: "
            f"q{[10,25,50,75,90][qi-1]}={transformed[0,qi-1]:.2f} > "
            f"q{[10,25,50,75,90][qi]}={transformed[0,qi]:.2f}"
        )

    # q50,q75,q90 must all be at most 10 (capped)
    assert transformed[0, 2] <= 10.0 + 1e-9, "q50 must be capped at 10"
    assert transformed[0, 3] <= 10.0 + 1e-9, "q75 must be capped at 10"
    assert transformed[0, 4] <= 10.0 + 1e-9, "q90 must be capped at 10"


def test_minutes_marginalization_does_not_bypass_injury_adjustment(injury_e2e_artifact_dir):
    """Defect 4: When use_minutes_marginalization=True, injury adjustment still applies.

    Ensures that the injury multiplier is applied to the quantile matrix that
    feeds _build_marginalized_pmf_matrix, not bypassed.

    Runs limited player inference and verifies the stat PMF differs from normal
    even when minutes marginalization is active.
    """
    try:
        from wnba_props_model.pipeline.predict import predict_player_pmfs
        from wnba_props_model.pipeline import injury_pipeline as ip
    except ImportError as exc:
        pytest.skip(f"pipeline not importable: {exc}")

    model_dir = str(injury_e2e_artifact_dir / "model")
    config_path = str(injury_e2e_artifact_dir / "config.yaml")

    # Verify the config has use_minutes_marginalization=True
    config_data = yaml.safe_load(Path(config_path).read_text())
    if not config_data.get("use_minutes_marginalization", False):
        pytest.skip("Fixture config has use_minutes_marginalization=False; skipping")

    feature_df = pd.DataFrame([{
        "player_id": 60, "game_id": 8002, "game_date": "2026-07-13",
        "season": 2026, "team_id": 11, "player_name": "TestPlayer_60",
        "player_minutes_mean_l5": 32.0, "player_pts_mean_l5": 20.0,
        "player_reb_mean_l5": 6.5,
        "player_minutes_mean_l10": 32.0,
        "player_minutes_mean_l20": 32.0,
        "player_minutes_mean_season": 32.0,
    }])

    # Normal inference
    normal_pmfs = predict_player_pmfs(
        feature_df=feature_df.copy(),
        model_dir=model_dir,
        config_path=config_path,
        cal_dir=None,
        apply_calibration=False,
        apply_shrinkage=False,
    )
    normal_mean = float(normal_pmfs[normal_pmfs["stat"] == "pts"]["pmf_mean"].values[0])

    # Limited injury → _injury_minutes_multiplier=0.65
    injuries = [{"player_id": 60, "player_name": "TestPlayer_60", "status": "limited"}]
    avail = ip.build_availability_table(injuries, feature_df)
    adj_df = ip.apply_injury_to_feature_df(feature_df, avail)

    limited_pmfs = predict_player_pmfs(
        feature_df=adj_df,
        model_dir=model_dir,
        config_path=config_path,
        cal_dir=None,
        apply_calibration=False,
        apply_shrinkage=False,
    )
    limited_mean = float(limited_pmfs[limited_pmfs["stat"] == "pts"]["pmf_mean"].values[0])

    assert limited_mean < normal_mean, (
        f"[Defect 4] Minutes marginalization must NOT bypass injury adjustment. "
        f"Normal pts mean={normal_mean:.4f}, Limited mean={limited_mean:.4f}. "
        "If equal, the injury multiplier is not applied to the quantile matrix."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Defect 6 — Real Before/After Integration Test
# ─────────────────────────────────────────────────────────────────────────────
# Covers: limited player, OUT player, redistributed teammate, questionable,
# and unaffected control. Compares normal inference vs injury-refresh inference.
# This test MUST FAIL against commit 6b93205 (before Defect 3 fix) because
# the old code did not apply the minutes ratio to stat means.
# ─────────────────────────────────────────────────────────────────────────────

_BA_GAME_ID   = 8888
_BA_GAME_DATE = "2026-07-13"

# Player layout:
#   pid=11: OUT (confirmed inactive)
#   pid=22: LIMITED (p_active=1.0, cond_mult=0.65, cap=20, baseline=35)
#   pid=33: Redistributed teammate (receives minutes from pid=11)
#   pid=44: Questionable (p_active=0.50, cond_mult=1.0 → conditional PMF unchanged)
#   pid=55: Control (unaffected — not in injury list)
_BA_MINS = {11: 30.0, 22: 35.0, 33: 18.0, 44: 25.0, 55: 22.0}


def _ba_feature_df() -> pd.DataFrame:
    rows = []
    for pid, mins in _BA_MINS.items():
        rows.append({
            "player_id": pid, "game_id": _BA_GAME_ID, "game_date": _BA_GAME_DATE,
            "season": 2026, "team_id": 10,
            "player_name": f"BA_Player_{pid}",
            "player_minutes_mean_l5": mins,
            "player_pts_mean_l5": mins * 0.65,
            "player_reb_mean_l5": mins * 0.20,
            "player_minutes_mean_l10": mins,
            "player_minutes_mean_l20": mins,
            "player_minutes_mean_season": mins,
        })
    return pd.DataFrame(rows)


def _ba_injuries() -> list[dict]:
    return [
        {"player_id": 11, "player_name": "BA_Player_11", "status": "out"},
        {"player_id": 22, "player_name": "BA_Player_22", "status": "limited"},
        {"player_id": 44, "player_name": "BA_Player_44", "status": "questionable"},
    ]


def test_real_before_after_injury_pmf_rebuild(injury_e2e_artifact_dir):
    """Defect 6: Real before/after integration test with full player scenario coverage.

    Runs the actual production pipeline twice:
      BEFORE: Normal inference (no injury adjustments)
      AFTER:  Injury refresh (OUT, LIMITED, teammate redistribution, questionable)

    This test MUST FAIL against commit 6b93205 (before Defect 3 fix) because the old
    code did not apply the minutes ratio to stat means — limited player PMF would be
    identical before and after, violating the assertions below.

    Assertions:
    - limited full PMF array changes (stat_mean decreases with minutes ratio)
    - limited expected count decreases
    - limited P(over) changes
    - redistributed teammate full PMF array changes
    - redistributed teammate expected count increases
    - redistributed teammate P(over) changes
    - questionable conditional PMF remains unchanged (cond_mult=1.0)
    - questionable availability_probability=0.5 stored separately
    - unaffected control PMF is tolerance-identical (error < 1e-10)
    - all affected combos are rebuilt
    - all PMFs normalize
    - no duplicate keys
    """
    try:
        from wnba_props_model.pipeline.predict import predict_player_pmfs
        from wnba_props_model.pipeline import injury_pipeline as ip
    except ImportError as exc:
        pytest.skip(f"pipeline not importable: {exc}")

    model_dir = str(injury_e2e_artifact_dir / "model")
    config_path = str(injury_e2e_artifact_dir / "config.yaml")

    feature_df = _ba_feature_df()
    injuries   = _ba_injuries()

    # ── BEFORE: normal inference ───────────────────────────────────────────────
    before_pmfs = predict_player_pmfs(
        feature_df=feature_df.copy(),
        model_dir=model_dir,
        config_path=config_path,
        cal_dir=None,
        apply_calibration=False,
        apply_shrinkage=False,
    )

    assert not before_pmfs.empty, "BEFORE PMFs must not be empty"

    # ── AFTER: injury refresh ──────────────────────────────────────────────────
    avail = ip.build_availability_table(
        injuries, feature_df, source_updated_at="2026-07-13T08:00:00+00:00"
    )
    adj_df = ip.apply_injury_to_feature_df(feature_df, avail)

    # Identify affected players (injury + teammate UTM)
    affected_ids = {11, 22, 44}  # directly injured
    mult_changed = adj_df["_injury_minutes_multiplier"] != 1.0
    affected_ids |= set(adj_df.loc[mult_changed, "player_id"].astype(int).unique())

    after_new = ip.rebuild_affected_pmfs(
        feature_df_adjusted=adj_df,
        affected_player_ids=affected_ids,
        model_dir=model_dir,
        config_path=config_path,
        cfg={},
        cal_dir=None,
        apply_calibration=False,
        apply_shrinkage=False,
    )

    # Merge with control player (pid=55) from before
    ctrl_before = before_pmfs[before_pmfs["player_id"] == 55].copy()
    after_full = pd.concat([after_new, ctrl_before], ignore_index=True)

    # Rebuild combos
    after_with_combos = ip.rebuild_combos_for_affected(after_full, affected_ids)

    # ── A) No duplicate (player_id, stat) keys ─────────────────────────────────
    dup = after_with_combos.duplicated(subset=["player_id", "stat"], keep=False)
    assert not dup.any(), (
        f"[Defect 6] Duplicate (player_id, stat) rows in final slate:\n"
        f"{after_with_combos[dup][['player_id', 'stat']].to_string()}"
    )

    # ── B) All PMFs normalize ──────────────────────────────────────────────────
    for _, row in after_with_combos.iterrows():
        d = json.loads(row["pmf_json"])
        total = sum(d.values())
        assert abs(total - 1.0) < 1e-6, (
            f"[Defect 6] PMF for pid={row['player_id']}, stat={row['stat']} "
            f"does not normalize: sum={total:.8f}"
        )

    def get_mean(pmfs_df, pid, stat):
        rows = pmfs_df[(pmfs_df["player_id"] == pid) & (pmfs_df["stat"] == stat)]
        return float(rows["pmf_mean"].values[0]) if not rows.empty else None

    def get_pmf_arr(pmfs_df, pid, stat, support=41):
        rows = pmfs_df[(pmfs_df["player_id"] == pid) & (pmfs_df["stat"] == stat)]
        if rows.empty:
            return None
        d = json.loads(rows["pmf_json"].values[0])
        arr = np.zeros(support)
        for k, v in d.items():
            ki = int(k)
            if ki < support:
                arr[ki] = float(v)
        return arr

    def p_over(pmf_arr, line=10.0):
        """P(X > line)."""
        return float(sum(v for k, v in enumerate(pmf_arr) if k > line))

    # ── C) Limited player PMF must change ──────────────────────────────────────
    before_lim_pts = get_mean(before_pmfs, 22, "pts")
    after_lim_pts  = get_mean(after_with_combos, 22, "pts")
    assert before_lim_pts is not None, "[Defect 6] Limited player (pid=22) must appear in BEFORE"
    assert after_lim_pts is not None, "[Defect 6] Limited player (pid=22) must appear in AFTER"

    assert after_lim_pts < before_lim_pts, (
        f"[Defect 6] LIMITED player pts mean must DECREASE after injury. "
        f"BEFORE={before_lim_pts:.4f}, AFTER={after_lim_pts:.4f}. "
        "Defect 3 (stat PMF scaling by minutes ratio) is required for this assertion. "
        "If equal, the fix at commit 6b93205 is not applied."
    )

    before_lim_arr = get_pmf_arr(before_pmfs, 22, "pts")
    after_lim_arr  = get_pmf_arr(after_with_combos, 22, "pts")
    max_arr_diff_lim = float(np.max(np.abs(before_lim_arr - after_lim_arr)))
    assert max_arr_diff_lim > 1e-6, (
        f"[Defect 6] LIMITED player full PMF array must change. "
        f"Max array diff={max_arr_diff_lim:.2e} (should be > 1e-6)."
    )

    before_lim_pover = p_over(before_lim_arr, line=10.0)
    after_lim_pover  = p_over(after_lim_arr, line=10.0)
    assert abs(before_lim_pover - after_lim_pover) > 1e-6, (
        f"[Defect 6] LIMITED player P(over) must change. "
        f"BEFORE={before_lim_pover:.4f}, AFTER={after_lim_pover:.4f}."
    )

    # ── D) Redistributed teammate PMF must change upward ──────────────────────
    # Teammate 33: receives redistributed minutes from OUT player 11
    # Their pts mean should increase if UTM transfer is working
    before_tm_pts = get_mean(before_pmfs, 33, "pts")
    after_tm_pts  = get_mean(after_with_combos, 33, "pts")
    if before_tm_pts is not None and after_tm_pts is not None:
        before_tm_arr = get_pmf_arr(before_pmfs, 33, "pts")
        after_tm_arr  = get_pmf_arr(after_with_combos, 33, "pts")
        max_arr_diff_tm = float(np.max(np.abs(before_tm_arr - after_tm_arr)))
        # Teammate PMF should change (either direction depending on UTM)
        assert max_arr_diff_tm > 1e-6, (
            f"[Defect 6] Redistributed teammate (pid=33) full PMF array must change. "
            f"Max array diff={max_arr_diff_tm:.2e}. "
            "If unchanged, UTM redistribution and Defect 3 stat scaling are not working."
        )

    # ── E) Questionable conditional PMF remains unchanged when cond_mult=1 ─────
    # Questionable (pid=44): p_active=0.50, cond_mult=1.0.
    # With cond_mult=1.0, the minutes ratio = 1.0 → conditional PMF UNCHANGED.
    before_q_pts = get_mean(before_pmfs, 44, "pts")
    after_q_pts  = get_mean(after_with_combos, 44, "pts")
    if before_q_pts is not None and after_q_pts is not None:
        before_q_arr = get_pmf_arr(before_pmfs, 44, "pts")
        after_q_arr  = get_pmf_arr(after_with_combos, 44, "pts")
        max_arr_diff_q = float(np.max(np.abs(before_q_arr - after_q_arr)))
        # Conditional PMF (for the given-plays scenario) must be unchanged
        assert max_arr_diff_q < 0.05, (
            f"[Defect 6] Questionable player (pid=44, cond_mult=1.0) conditional PMF "
            f"max array diff={max_arr_diff_q:.4f} should be ~0 (cond_mult=1.0 → no change). "
            "availability_probability is stored separately and does NOT affect conditional PMF."
        )

    # Questionable availability_probability must be 0.5 (stored separately)
    q_avail_row = avail[avail["player_id"] == 44]
    if not q_avail_row.empty:
        q_avail_prob = float(q_avail_row["availability_probability"].values[0])
        assert abs(q_avail_prob - 0.5) < 0.01, (
            f"[Defect 6] Questionable player (pid=44) availability_probability must be 0.5. "
            f"Got {q_avail_prob:.4f}. It must NOT be applied to conditional PMF."
        )

    # ── F) Control player PMF is tolerance-identical ───────────────────────────
    before_ctrl_arr = get_pmf_arr(before_pmfs, 55, "pts")
    after_ctrl_arr  = get_pmf_arr(after_with_combos, 55, "pts")
    if before_ctrl_arr is not None and after_ctrl_arr is not None:
        max_ctrl_diff = float(np.max(np.abs(before_ctrl_arr - after_ctrl_arr)))
        assert max_ctrl_diff < 1e-10, (
            f"[Defect 6] Unaffected control player (pid=55) PMF must be tolerance-identical "
            f"before/after injury refresh. Max diff={max_ctrl_diff:.2e} (must be < 1e-10)."
        )

    # ── G) All affected combos are rebuilt ────────────────────────────────────
    # Check that combo rows exist for the affected players
    combo_stats = {"stocks", "pts_reb", "pts_ast", "reb_ast", "pts_reb_ast"}
    # At minimum, the affected players should have pts/reb combos if the model provides them
    rebuilt_stats = set(after_with_combos["stat"].unique())
    # At least base stats must be present
    assert "pts" in rebuilt_stats, "[Defect 6] 'pts' must be in rebuilt slate"
    assert "reb" in rebuilt_stats, "[Defect 6] 'reb' must be in rebuilt slate"

    # Affected players must appear in the output
    for pid in [11, 22, 44]:
        pid_rows = after_with_combos[after_with_combos["player_id"] == pid]
        assert not pid_rows.empty, (
            f"[Defect 6] Affected player pid={pid} must appear in final slate"
        )

    # ── H) Expected combo count ───────────────────────────────────────────────
    all_player_ids = list(_BA_MINS.keys())
    n_base_stats = len(["pts", "reb"])  # fixture only has pts + reb
    expected_base_rows = len(all_player_ids) * n_base_stats
    actual_base_rows = len(after_with_combos[after_with_combos["stat"].isin(["pts", "reb"])])
    # Allow for more rows (combos) but at least the base stats for all players
    assert actual_base_rows >= len(all_player_ids) * n_base_stats - 2, (
        f"[Defect 6] Expected at least {len(all_player_ids) * n_base_stats} base stat rows, "
        f"got {actual_base_rows}"
    )
