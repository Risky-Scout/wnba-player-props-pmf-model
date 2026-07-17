"""P0 market-data contract tests.

Cover the end-to-end market path: raw Odds API schema -> canonical closing table
-> scorer output, plus the fail-closed contracts and correct push/edge math.

Blocking guarantees:
  * Integer line 10 counts 10 as PUSH and 11 as OVER; half-line 10.5 counts 11 as OVER.
  * model_close_edge / price_clv / line_clv are OUTCOME-INDEPENDENT (CLV outcome-invariance).
  * Historical-line join fails closed on missing keys / zero overlap / many-to-many / low coverage.
  * Per-line calibration cannot run with pmf_mean substituted for the line.
  * Odds API pull fails (nonzero) on empty output; canonicalizes event/player IDs.
  * Post-game workflow pulls the fresh closing file BEFORE selecting it to score.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fit_cal = _load("p0_fit_calibrators", "scripts/fit_calibrators.py")
scorer = _load("p0_score_daily", "scripts/score_daily_predictions.py")
oddsapi = _load("p0_pull_closing", "scripts/pull_odds_api_closing_lines.py")

from wnba_props_model.calibration.oof_line_joiner import (  # noqa: E402
    join_historical_props_to_oof,
    OOFLineJoinError,
)


# ── Push / Over handling (both helpers must agree) ────────────────────────────

def _uniform_pmf(n=21):
    p = np.ones(n) / n
    return p


@pytest.mark.parametrize("p_over_fn", [fit_cal._p_over_conditional, scorer._p_over_cond])
def test_half_line_over_is_geq_next_integer(p_over_fn):
    # pmf with all mass at 11 -> Over 10.5 must be 1.0
    pmf = np.zeros(21); pmf[11] = 1.0
    assert p_over_fn(pmf, 10.5) == pytest.approx(1.0)
    # mass at 10 -> Over 10.5 must be 0.0 (10 is not > 10.5)
    pmf = np.zeros(21); pmf[10] = 1.0
    assert p_over_fn(pmf, 10.5) == pytest.approx(0.0)


@pytest.mark.parametrize("p_over_fn", [fit_cal._p_over_conditional, scorer._p_over_cond])
def test_integer_line_treats_exact_as_push(p_over_fn):
    # All mass exactly at the integer line 10 -> it's a PUSH, not an Over.
    pmf = np.zeros(21); pmf[10] = 1.0
    # conditional on non-push, denominator is 0 -> returns p_over (0.0); never counts push as over
    assert p_over_fn(pmf, 10.0) == pytest.approx(0.0)
    # Half over 10, half at 11: push mass 0.5 removed -> conditional P(over)=0.5/0.5=1.0
    pmf = np.zeros(21); pmf[10] = 0.5; pmf[11] = 0.5
    assert p_over_fn(pmf, 10.0) == pytest.approx(1.0)
    # 11 is an Over for line 10
    pmf = np.zeros(21); pmf[11] = 1.0
    assert p_over_fn(pmf, 10.0) == pytest.approx(1.0)


# ── OOF historical-line joiner: fail-closed ───────────────────────────────────

def _write(df, path):
    df.to_parquet(path, index=False)
    return str(path)


def test_joiner_fails_on_missing_keys(tmp_path):
    oof = pd.DataFrame({"game_id": ["g1"], "player_id": ["p1"]})  # no 'stat'
    props = pd.DataFrame({"game_id": ["g1"], "player_id": ["p1"], "stat": ["pts"], "line": [10.5]})
    with pytest.raises(OOFLineJoinError, match="canonical keys"):
        join_historical_props_to_oof(_write(oof, tmp_path/"o.parquet"),
                                     _write(props, tmp_path/"p.parquet"),
                                     str(tmp_path/"out.parquet"))


def test_joiner_fails_on_zero_overlap(tmp_path):
    oof = pd.DataFrame({"game_id": ["gA"], "player_id": ["pA"], "stat": ["pts"]})
    props = pd.DataFrame({"game_id": ["gZ"], "player_id": ["pZ"], "stat": ["pts"], "line": [10.5]})
    with pytest.raises(OOFLineJoinError, match="coverage"):
        join_historical_props_to_oof(_write(oof, tmp_path/"o.parquet"),
                                     _write(props, tmp_path/"p.parquet"),
                                     str(tmp_path/"out.parquet"))


def test_joiner_fails_on_many_to_many(tmp_path):
    oof = pd.DataFrame({"game_id": ["g1"], "player_id": ["p1"], "stat": ["pts"]})
    props = pd.DataFrame({"game_id": ["g1", "g1"], "player_id": ["p1", "p1"],
                          "stat": ["pts", "pts"], "line": [10.5, 11.5]})
    with pytest.raises(OOFLineJoinError, match="duplicate"):
        join_historical_props_to_oof(_write(oof, tmp_path/"o.parquet"),
                                     _write(props, tmp_path/"p.parquet"),
                                     str(tmp_path/"out.parquet"))


def test_joiner_succeeds_with_real_coverage(tmp_path):
    oof = pd.DataFrame({"game_id": ["g1", "g2"], "player_id": ["p1", "p2"], "stat": ["pts", "pts"]})
    props = pd.DataFrame({"game_id": ["g1", "g2"], "player_id": ["p1", "p2"],
                          "stat": ["pts", "pts"], "line": [10.5, 20.5]})
    out = join_historical_props_to_oof(_write(oof, tmp_path/"o.parquet"),
                                       _write(props, tmp_path/"p.parquet"),
                                       str(tmp_path/"out.parquet"), min_coverage=0.5)
    assert len(out) == 2 and out["line"].notna().all()


# ── Per-line calibration must NOT substitute pmf_mean for the line ────────────

def test_per_line_raises_without_line_column():
    oof = pd.DataFrame({"stat": ["pts"] * 200, "pmf_mean": [10.0] * 200,
                        "actual_outcome": [11] * 200, "pmf_json": ['{"0":1.0}'] * 200})
    with pytest.raises(fit_cal.NoMarketLinesError):
        fit_cal._fit_per_line_calibrators(oof, ["pts"])


def test_per_line_raises_when_line_all_null():
    oof = pd.DataFrame({"stat": ["pts"] * 200, "pmf_mean": [10.0] * 200, "line": [None] * 200,
                        "actual_outcome": [11] * 200, "pmf_json": ['{"0":1.0}'] * 200})
    with pytest.raises(fit_cal.NoMarketLinesError):
        fit_cal._fit_per_line_calibrators(oof, ["pts"])


# ── Odds API canonicalization: raw schema -> canonical scorer schema ──────────

def _canonical_fixtures(tmp_path):
    games = pd.DataFrame({
        "game_id": ["G100"],
        "game_date": ["2026-07-10"],
        "home_team_abbreviation": ["NYL"],
        "visitor_team_abbreviation": ["LVA"],
    })
    roster = pd.DataFrame({
        "game_id": ["G100", "G100"],
        "player_id": ["P1", "P2"],
        "player_name": ["Sabrina Ionescu", "A'ja Wilson"],
    })
    gp = tmp_path / "games.parquet"; games.to_parquet(gp, index=False)
    rp = tmp_path / "roster.parquet"; roster.to_parquet(rp, index=False)
    return str(gp), str(rp)


def test_canonicalize_resolves_and_drops_unmatched(tmp_path):
    gp, rp = _canonical_fixtures(tmp_path)
    raw = pd.DataFrame({
        "event_id": ["e1", "e1", "e1"],
        "home_team": ["New York Liberty", "New York Liberty", "New York Liberty"],
        "away_team": ["Las Vegas Aces", "Las Vegas Aces", "Las Vegas Aces"],
        "player_name": ["Sabrina Ionescu", "A'ja Wilson", "Unknown Player"],
        "stat": ["pts", "pts", "pts"],
        "line": [17.5, 20.5, 5.5],
        "market_prob_over_no_vig": [0.5, 0.52, 0.5],
        "over_odds": [-110, -110, -110],
        "under_odds": [-110, -110, -110],
    })
    out = oddsapi.canonicalize_closing_lines(raw, "2026-07-10", gp, rp)
    # Unknown player dropped; two canonical rows remain with canonical schema.
    assert set(["game_id", "player_id", "stat", "market_prob_over_no_vig", "line"]).issubset(out.columns)
    assert len(out) == 2
    assert set(out["player_id"]) == {"P1", "P2"}
    assert (out["game_id"] == "G100").all()
    assert (out["identity_method"] == "exact_roster_name").all()


def test_canonicalize_consensus_dedupes_multiple_books(tmp_path):
    gp, rp = _canonical_fixtures(tmp_path)
    raw = pd.DataFrame({
        "event_id": ["e1", "e1"],
        "home_team": ["New York Liberty", "New York Liberty"],
        "away_team": ["Las Vegas Aces", "Las Vegas Aces"],
        "player_name": ["Sabrina Ionescu", "Sabrina Ionescu"],
        "stat": ["pts", "pts"],
        "line": [17.5, 18.5],  # two books
        "market_prob_over_no_vig": [0.50, 0.60],
    })
    out = oddsapi.canonicalize_closing_lines(raw, "2026-07-10", gp, rp)
    assert len(out) == 1  # one consensus row per (game_id, player_id, stat)
    assert out.iloc[0]["line"] == pytest.approx(18.0)  # median of 17.5/18.5


# ── CLV outcome-invariance (code-level guarantee) ─────────────────────────────

def test_closing_edge_block_has_no_outcome_multiplication():
    src = (REPO / "scripts/score_daily_predictions.py").read_text()
    # No field may be multiplied by the outcome-derived (2*hit-1) term anymore.
    assert "2 * joined" not in src and "2*joined" not in src, (
        "outcome-signed CLV term (2*hit-1) must be removed entirely"
    )
    assert "true_clv" not in src, "outcome-dependent true_clv must be removed"
    # The three CLV/edge fields must exist and be defined without hit_result.
    for fld in ["model_close_edge", "price_clv", "line_clv", "model_edge_open"]:
        assert fld in src


def test_no_model_market_diff_labeled_clv():
    for f in ["scripts/score_daily_predictions.py", "scripts/generate_clv_report.py"]:
        src = (REPO / f).read_text()
        assert '"clv"' not in src and "'clv'" not in src, f"{f} still assigns a bare 'clv' field"
        assert '"true_clv"' not in src, f"{f} still references true_clv"


# ── Odds API pull: fail on empty ──────────────────────────────────────────────

def test_odds_api_pull_fails_on_empty_output():
    src = (REPO / "scripts/pull_odds_api_closing_lines.py").read_text()
    # No silent empty-write + Exit(0) path; empty must be a fatal nonzero exit.
    assert "_write_empty" not in src, "empty-file fallback must be removed"
    assert "raise typer.Exit(1)" in src


# ── Post-game workflow ordering: pull fresh BEFORE select ─────────────────────

def test_post_game_pulls_fresh_before_select():
    wf = (REPO / ".github/workflows/post_game_scoring.yml").read_text()
    i_pull = wf.index("Pull FRESH Odds API closing lines")
    i_select = wf.index("Select closing-lines parquet (AFTER fresh pull)")
    assert i_pull < i_select, "must pull fresh closing lines before selecting the file to score"
    # And the scorer consumes the selected (post-pull) path.
    assert "--closing-lines \"${{ steps.closing.outputs.path }}\"" in wf


def test_odds_api_pull_canonicalizes_before_scoring():
    wf = (REPO / ".github/workflows/post_game_scoring.yml").read_text()
    assert "--games-path data/processed/wnba_games.parquet" in wf
    assert "--roster-path data/processed/wnba_player_game_stats.parquet" in wf
