"""Tests for the soft-book / MARKET_DISLOCATION CLV backtest and its actionability wiring.

Covers:
  * CLV computation correctness on a hand-checkable fixture (price CLV + same-book CLV).
  * No-vig / LOBO consensus correctness and fail-closed on a missing opposite side.
  * The date-cluster bootstrap + segment significance (CI excludes 0) logic.
  * ``actionable`` flips True ONLY when a segment has positive mean CLV, a 95% CI excluding 0,
    AND N >= min_segment_n — and is fail-closed otherwise (small sample, straddling CI,
    single date cluster, missing table).
  * End-to-end ``run_backtest`` on a synthetic multi-slate panel.
  * The acceptance gate accepts a CLV-validated actionable row and fails closed without the
    embedded CLV evidence.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wnba_props_model.edge.clv_backtest import (
    apply_validation_table_to_board,
    build_novig_tables,
    compute_clv,
    date_cluster_bootstrap_ci,
    ev_bucket_label,
    lookup_actionability,
    normalize_quotes,
    run_backtest,
    summarize_segment,
)
from wnba_props_model.edge.path_b_gate import validate_row


# --------------------------------------------------------------------------- #
# CLV computation correctness (deterministic arithmetic)
# --------------------------------------------------------------------------- #
def _candidate(side="over", ev_frac=0.03, market="player_points", line=18.5, book="fanduel"):
    return pd.DataFrame([{
        "event_id": "E1", "market_key": market, "player_name": "P", "line": line,
        "book": book, "side": side, "ev_frac": ev_frac, "ev_pct": ev_frac * 100.0,
        "game_date": "2026-05-08",
    }])


def test_clv_price_and_same_book_arithmetic():
    dec_per = {("E1", "player_points", "P", 18.5, "fanduel"): {
        "fair_over": 0.55, "fair_under": 0.45, "over_odds": -110, "under_odds": -110}}
    close_per = {("E1", "player_points", "P", 18.5, "fanduel"): {
        "fair_over": 0.58, "fair_under": 0.42, "over_odds": -120, "under_odds": 100}}
    close_cons = {("E1", "player_points", "P", 18.5): {"p_over": 0.60, "n_books": 4}}

    over = compute_clv(_candidate("over"), dec_per, close_per, close_cons).iloc[0]
    # price CLV = closing consensus P(over) - decision book's own no-vig P(over)
    assert over["price_clv"] == pytest.approx(0.60 - 0.55)
    assert over["price_clv_cents"] == pytest.approx(5.0)
    # same-book CLV = same book's closing P(over) - decision P(over)
    assert over["same_book_clv"] == pytest.approx(0.58 - 0.55)
    assert over["beat_close"] is True or bool(over["beat_close"]) is True

    under = compute_clv(_candidate("under"), dec_per, close_per, close_cons).iloc[0]
    # under: p_bet=0.45, closing consensus P(under)=1-0.60=0.40 -> negative CLV
    assert under["price_clv"] == pytest.approx(0.40 - 0.45)
    assert bool(under["beat_close"]) is False


def test_clv_fail_closed_missing_close_reference():
    dec_per = {("E1", "player_points", "P", 18.5, "fanduel"): {
        "fair_over": 0.55, "fair_under": 0.45, "over_odds": -110, "under_odds": -110}}
    # No closing consensus and no same-book close for this prop.
    out = compute_clv(_candidate("over"), dec_per, {}, {}).iloc[0]
    assert out["has_close_consensus"] is False or bool(out["has_close_consensus"]) is False
    assert out["price_clv"] is None
    assert out["same_book_clv"] is None


# --------------------------------------------------------------------------- #
# No-vig / LOBO consensus correctness + fail-closed on missing opposite side
# --------------------------------------------------------------------------- #
def _quote(book, side, odds, *, line=18.5, market="player_points", player="P", pid=1,
           label="decision", ts="2026-05-08T22:00:00Z"):
    return {
        "event_id": "E1", "commence_time": "2026-05-08T23:30:00Z",
        "home_team": "H", "away_team": "A", "book": book, "book_last_update": ts,
        "market_key": market, "stat": "pts", "player_name": player, "player_id": pid,
        "side": side, "line": line, "american_odds": odds, "collected_utc": ts,
        "snapshot_time": ts, "snapshot_label": label, "game_date": "2026-05-08",
    }


def test_novig_consensus_is_median_and_symmetric_is_half():
    rows = []
    for bk in ("draftkings", "fanduel", "betrivers"):
        rows += [_quote(bk, "over", -110), _quote(bk, "under", -110)]
    per, cons = build_novig_tables(pd.DataFrame(rows))
    # -110/-110 is symmetric => each book's no-vig P(over) ~ 0.5.
    for bk in ("draftkings", "fanduel", "betrivers"):
        assert per[("E1", "player_points", "P", 18.5, bk)]["fair_over"] == pytest.approx(0.5, abs=1e-6)
    c = cons[("E1", "player_points", "P", 18.5)]
    assert c["n_books"] == 3
    assert c["p_over"] == pytest.approx(0.5, abs=1e-6)


def test_novig_asymmetric_shifts_prob_and_fail_closed_on_missing_side():
    rows = [
        _quote("draftkings", "over", -200), _quote("draftkings", "under", 170),
        _quote("fanduel", "over", -110), _quote("fanduel", "under", -110),
        _quote("betrivers", "over", -105),  # betrivers has NO under -> fail closed
    ]
    per, cons = build_novig_tables(pd.DataFrame(rows))
    # A heavy over favourite (-200) implies P(over) > 0.5.
    assert per[("E1", "player_points", "P", 18.5, "draftkings")]["fair_over"] > 0.5
    # betrivers is excluded (missing opposite side): never de-vigged.
    assert ("E1", "player_points", "P", 18.5, "betrivers") not in per
    assert cons[("E1", "player_points", "P", 18.5)]["n_books"] == 2


def test_normalize_quotes_renames_event_id():
    raw = pd.DataFrame([{
        "odds_event_id": "E9", "side": "Over", "snapshot_time": "2026-05-08T22:00:00Z",
        "snapshot_label": "decision",
    }])
    out = normalize_quotes(raw)
    assert "event_id" in out.columns and out["event_id"].iloc[0] == "E9"
    assert out["side"].iloc[0] == "over"
    assert out["collected_utc"].iloc[0] == "2026-05-08T22:00:00Z"


# --------------------------------------------------------------------------- #
# Bootstrap + segment significance
# --------------------------------------------------------------------------- #
def _seg_df(values, dates):
    return pd.DataFrame({
        "price_clv": values,
        "game_date": dates,
        "beat_close": [v > 0 for v in values],
    })


def test_bootstrap_ci_excludes_zero_for_all_positive_multi_date():
    df = _seg_df([0.03] * 60, [f"d{i%6}" for i in range(60)])
    s = summarize_segment(df, "price_clv", min_segment_n=50)
    assert s["n"] == 60 and s["n_dates"] == 6
    assert s["ci_low"] > 0 and s["significant"] is True and s["qualifies"] is True


def test_segment_not_qualified_when_sample_too_small():
    df = _seg_df([0.03] * 40, [f"d{i%6}" for i in range(40)])
    s = summarize_segment(df, "price_clv", min_segment_n=50)
    assert s["significant"] is True       # sign/CI fine
    assert s["qualifies"] is False        # but N < min_segment_n -> fail closed


def test_segment_not_significant_when_ci_straddles_zero():
    vals = ([0.05, -0.05] * 30)
    df = _seg_df(vals, [f"d{i%6}" for i in range(60)])
    s = summarize_segment(df, "price_clv", min_segment_n=10)
    assert s["ci_low"] <= 0 <= s["ci_high"]
    assert s["significant"] is False and s["qualifies"] is False


def test_single_date_cluster_is_fail_closed():
    df = _seg_df([0.03] * 60, ["only-one-date"] * 60)
    lo, hi, n_clusters = date_cluster_bootstrap_ci(
        df["price_clv"].to_numpy(), df["game_date"].to_numpy())
    assert n_clusters == 1 and lo is None and hi is None
    s = summarize_segment(df, "price_clv", min_segment_n=10)
    assert s["significant"] is False and s["qualifies"] is False


def test_ev_bucket_label():
    assert ev_bucket_label(0.03) == "2.5-5%"
    assert ev_bucket_label(0.07) == "5-10%"
    assert ev_bucket_label(0.20) == ">10%"
    assert ev_bucket_label(0.01) is None
    assert ev_bucket_label(None) is None


# --------------------------------------------------------------------------- #
# Validation table lookup + board application (actionable flips)
# --------------------------------------------------------------------------- #
def _qual_entry(key, n=60, mean=0.03, lo=0.01, hi=0.05, qualifies=True):
    return {
        "key": key, "segment_type": "market", "n": n, "n_dates": 6, "mean": mean,
        "median": mean, "pct_beat_close": 90.0, "ci_low": lo, "ci_high": hi,
        "significant": lo > 0 and mean > 0, "qualifies": qualifies,
    }


def _table(market=None, market_ev=None, min_n=50):
    return {
        "schema_version": "clv_validation_table_v1", "min_segment_n": min_n,
        "primary_metric": "price_clv",
        "segments": {"market": market or {}, "market_ev_bucket": market_ev or {}},
        "actionable_segments": [],
    }


def test_lookup_actionable_when_market_segment_qualifies():
    table = _table(market={"player_points": _qual_entry("player_points")})
    ok, reason, ev = lookup_actionability("player_points", 0.03, table)
    assert ok is True and ev is not None and "market=player_points" in reason


def test_lookup_prefers_market_ev_bucket_segment():
    table = _table(
        market={"player_points": _qual_entry("player_points", qualifies=False,
                                             lo=-0.01, mean=0.001)},
        market_ev={"player_points|2.5-5%": _qual_entry("player_points|2.5-5%")},
    )
    ok, reason, ev = lookup_actionability("player_points", 0.03, table)
    assert ok is True and "market_ev_bucket=player_points|2.5-5%" in reason


def test_lookup_fail_closed_reasons():
    # small sample
    table = _table(market={"player_points": _qual_entry(
        "player_points", n=10, qualifies=False)}, min_n=50)
    ok, reason, _ = lookup_actionability("player_points", 0.03, table)
    assert ok is False and "insufficient_sample" in reason
    # CI straddles zero
    table2 = _table(market={"player_points": _qual_entry(
        "player_points", n=60, mean=0.0, lo=-0.02, qualifies=False)}, min_n=50)
    ok2, reason2, _ = lookup_actionability("player_points", 0.03, table2)
    assert ok2 is False and "clv_ci_includes_zero_or_nonpositive" in reason2
    # no segment for the market at all
    ok3, reason3, _ = lookup_actionability("player_threes", 0.03, _table(min_n=50))
    assert ok3 is False and "no_validated_segment_for_market" in reason3


def _board(market="player_points", ev=0.03, book="fanduel", side="over", qualified=True):
    return pd.DataFrame([{
        "event_id": "E1", "player_name": "P", "player_id": 1, "player_id_resolved": True,
        "market_key": market, "line": 18.5, "side": side, "book": book, "bookmaker": book,
        "ev_frac": ev, "ev_pct": ev * 100.0, "self_excluded": True, "qualified": qualified,
        "consensus_n_books": 3, "consensus_books": ["a", "b", "c"],
        "validation_status": "PENDING_VALIDATION", "source_type": "MARKET_DISLOCATION",
    }])


def test_apply_validation_table_flips_actionable_and_sets_evidence():
    table = _table(market={"player_points": _qual_entry("player_points")})
    out = apply_validation_table_to_board(_board(), table)
    r = out.iloc[0]
    assert bool(r["actionable"]) is True
    assert bool(r["forward_clv_validated"]) is True
    assert r["validation_status"] == "VALIDATED_EXECUTABLE"
    assert r["clv_ci_low"] == pytest.approx(0.01)
    assert r["clv_segment"] == "player_points"
    assert r["source_type"] == "MARKET_DISLOCATION"    # untouched


def test_apply_validation_table_fail_closed_non_qualifying():
    table = _table(market={"player_assists": _qual_entry(
        "player_assists", n=2, qualifies=False)})
    out = apply_validation_table_to_board(_board(market="player_assists"), table)
    r = out.iloc[0]
    assert bool(r["actionable"]) is False
    assert bool(r["forward_clv_validated"]) is False
    assert "insufficient_sample" in r["actionable_reason"]


def test_apply_non_qualified_row_never_actionable():
    # A qualifying segment must NOT make a non-flagged (not +EV) row actionable.
    table = _table(market={"player_points": _qual_entry("player_points")})
    out = apply_validation_table_to_board(_board(qualified=False), table)
    r = out.iloc[0]
    assert bool(r["actionable"]) is False
    assert r["actionable_reason"] == "not_qualified_ev_candidate"


def test_apply_no_table_is_fail_closed():
    out = apply_validation_table_to_board(_board(), None)
    assert bool(out.iloc[0]["actionable"]) is False
    assert out.iloc[0]["actionable_reason"] == "no_validation_table"


# --------------------------------------------------------------------------- #
# End-to-end run_backtest on a synthetic multi-slate panel
# --------------------------------------------------------------------------- #
def _synthetic_panel(n_dates=8):
    """Each slate has one prop where the soft book (williamhill_us) offers a mispriced OVER
    (+140) vs a ~0.5 consensus (flagged +EV), and the market CLOSES higher for the over
    (consensus books move to -160 over) -> systematically positive price CLV."""
    rows = []
    for i in range(n_dates):
        gd = f"2026-05-{8 + i:02d}"
        ev = f"E{i}"
        base = dict(event_id=ev, commence_time=f"{gd}T23:30:00Z", home_team="H",
                    away_team="A", market_key="player_points", stat="pts",
                    player_name="Test Player", player_id=999, line=18.5, game_date=gd)

        def q(book, side, odds, label, ts):
            r = dict(base)
            r.update(book=book, side=side, american_odds=odds, snapshot_label=label,
                     book_last_update=ts, collected_utc=ts, snapshot_time=ts)
            return r

        dts, cts = f"{gd}T22:00:00Z", f"{gd}T23:20:00Z"
        # Decision: 3 consensus books ~0.5, soft book cheap over (+140) -> +EV over flagged.
        for bk in ("draftkings", "fanduel", "betrivers"):
            rows += [q(bk, "over", -110, "decision", dts), q(bk, "under", -110, "decision", dts)]
        rows += [q("williamhill_us", "over", 140, "decision", dts),
                 q("williamhill_us", "under", -160, "decision", dts)]
        # Close: consensus moves toward the over (books now price over -160) => higher P(over).
        for bk in ("draftkings", "fanduel", "betrivers", "williamhill_us"):
            rows += [q(bk, "over", -160, "close", cts), q(bk, "under", 140, "close", cts)]
    return normalize_quotes(pd.DataFrame(rows))


def test_run_backtest_marks_segment_actionable_with_small_min_n():
    panel = _synthetic_panel(n_dates=8)
    res = run_backtest(panel, min_segment_n=5, min_consensus_books=3, bootstrap_iters=1000)
    cov = res["coverage"]
    assert cov["n_flagged_candidates"] == 8      # one soft-book over per slate
    assert cov["n_with_close_consensus"] == 8
    ov = res["segments_price_clv"]["overall"]
    assert ov["mean"] > 0 and ov["ci_low"] > 0 and ov["significant"] is True
    assert "player_points" in res["validation_table"]["actionable_segments"]


def test_run_backtest_fail_closed_when_min_n_too_high():
    panel = _synthetic_panel(n_dates=8)
    res = run_backtest(panel, min_segment_n=50, min_consensus_books=3, bootstrap_iters=1000)
    # Signal is still significant, but N=8 < 50 -> nothing actionable (fail closed).
    assert res["segments_price_clv"]["overall"]["significant"] is True
    assert res["validation_table"]["actionable_segments"] == []


# --------------------------------------------------------------------------- #
# Gate integration: actionable row must carry qualifying CLV evidence
# --------------------------------------------------------------------------- #
def _gate_row(**over):
    row = {
        "event_id": "E1", "player_name": "P", "player_id": 1, "player_id_resolved": True,
        "market_key": "player_points", "is_alternate_market": False, "line": 18.5,
        "side": "over", "bookmaker": "fanduel", "displayed_odds": 140, "reference_p": 0.5,
        "consensus_p_over": 0.5, "consensus_n_books": 3, "consensus_books": ["a", "b", "c"],
        "consensus_dispersion_stdev": 0.01, "consensus_dispersion_iqr": 0.01,
        "consensus_includes_sharp": True, "self_excluded": True, "theoretical_ev_pct": 3.0,
        "executable_ev_pct": None, "price_survived_30s": None, "price_survived_60s": None,
        "provider_timestamp": "2026-05-08T22:00:00Z", "ingestion_timestamp": "2026-05-08T22:00:00Z",
        "scan_timestamp": "2026-05-08T22:00:00Z", "scheduled_tip": "2026-05-08T23:30:00Z",
        "quote_age_seconds": 10.0, "validation_status": "VALIDATED_EXECUTABLE",
        "actionable": True, "rejection_reason": None, "warning_reason": None,
        "source_type": "MARKET_DISLOCATION", "forward_clv_validated": True,
        "clv_segment_n": 60, "clv_ci_low": 0.01, "clv_mean": 0.03,
    }
    row.update(over)
    return row


def test_gate_accepts_clv_validated_actionable_row():
    viol = validate_row(_gate_row(), "board_rows[0]")
    assert not any(v.code == "PREMATURE_ACTIONABLE" for v in viol), [str(v) for v in viol]


def test_gate_fails_closed_actionable_without_positive_ci():
    viol = validate_row(_gate_row(clv_ci_low=-0.01), "board_rows[0]")
    assert any(v.code == "PREMATURE_ACTIONABLE" for v in viol)


def test_gate_fails_closed_actionable_without_forward_clv_flag():
    viol = validate_row(_gate_row(forward_clv_validated=False), "board_rows[0]")
    assert any(v.code == "PREMATURE_ACTIONABLE" for v in viol)


def test_gate_fails_closed_actionable_without_clv_evidence():
    row = _gate_row()
    del row["clv_ci_low"]
    viol = validate_row(row, "board_rows[0]")
    assert any(v.code == "PREMATURE_ACTIONABLE" for v in viol)
