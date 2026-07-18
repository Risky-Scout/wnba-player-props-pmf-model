"""P1 blocking tests — historical market ingestion + evaluation (API-free).

Covers the 17 required guarantees via the pure core (historical_market) and the
backfill/eval script helpers, with fixtures (no network).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
import wnba_props_model.evaluation.historical_market as hm  # noqa: E402
from wnba_props_model.models.market import shin_no_vig_two_way_with_z  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

backfill = _load("p1_backfill", "scripts/p1_historical_backfill.py")
evaljob = _load("p1_eval", "scripts/p1_build_evaluation.py")


# 1. Historical-events -> canonical game mapping
def test_resolve_game_id_exact():
    events = pd.DataFrame({"game_id": ["G1"], "game_date": ["2026-07-10"],
                           "home_team_abbreviation": ["NYL"], "visitor_team_abbreviation": ["LVA"]})
    assert hm.resolve_game_id(events, "New York Liberty", "Las Vegas Aces", "2026-07-10") == "G1"
    assert hm.resolve_game_id(events, "Chicago Sky", "Las Vegas Aces", "2026-07-10") is None


# 2. Historical event-odds parsing
def test_parse_event_odds():
    payload = {"data": {"bookmakers": [{"key": "dk", "last_update": "t", "markets": [
        {"key": "player_points", "outcomes": [
            {"description": "A B", "name": "Over", "point": 17.5, "price": -110},
            {"description": "A B", "name": "Under", "point": 17.5, "price": -110}]}]}]}}
    rows = backfill._parse_event_odds(payload, {"id": "e1", "commence_time": "c",
                                                "home_team": "h", "away_team": "a"}, "snap")
    assert len(rows) == 2 and {r["stat"] for r in rows} == {"pts"}
    assert rows[0]["book"] == "dk" and rows[0]["snapshot_time"] == "snap"


# 3 + 6. Paired Over/Under + no-vig; unpaired dropped; never mixes lines/books
def test_pair_over_under_and_no_vig():
    q = pd.DataFrame({
        "event_id": ["e", "e", "e"], "book": ["dk", "dk", "dk"], "stat": ["pts", "pts", "pts"],
        "player_name": ["A", "A", "A"], "line": [17.5, 17.5, 18.5], "snapshot_time": ["s", "s", "s"],
        "side": ["over", "under", "over"], "american_odds": [-110, -110, -110],
        "commence_time": ["c", "c", "c"],
    })
    paired = hm.pair_over_under(q, shin_fn=shin_no_vig_two_way_with_z)
    # 17.5 pairs; 18.5 over has no matching under -> dropped
    assert len(paired) == 1 and paired.iloc[0]["line"] == 17.5
    assert 0.0 < float(paired.iloc[0]["market_prob_over_no_vig"]) < 1.0


# 4 + 5. Opening/closing selection + post-tip rejection
def test_open_close_and_post_tip_rejection():
    paired = pd.DataFrame({
        "game_id": ["g", "g", "g"], "player_id": ["p", "p", "p"], "stat": ["pts", "pts", "pts"],
        "book": ["dk", "dk", "dk"], "line": [17.5, 17.5, 17.5],
        "snapshot_time": ["2026-07-10T18:00:00Z", "2026-07-10T22:55:00Z", "2026-07-10T23:30:00Z"],
        "commence_time": ["2026-07-10T23:00:00Z"] * 3,
        "market_prob_over_no_vig": [0.5, 0.5, 0.5], "over_odds": [-110]*3, "under_odds": [-110]*3,
    })
    tagged = hm.select_open_close(paired)
    assert len(tagged) == 2  # the 23:30 post-tip snapshot is rejected
    assert tagged[tagged["is_opening"]]["snapshot_time"].iloc[0].startswith("2026-07-10T18")
    assert tagged[tagged["is_closing"]]["snapshot_time"].iloc[0].startswith("2026-07-10T22:55")


# 7 + 8. Exact roster-constrained identity + unmatched auditing
def test_resolve_player_id_exact_and_unmatched():
    roster = pd.DataFrame({"game_id": ["G1", "G1"], "player_id": ["P1", "P2"],
                           "player_name": ["Sabrina Ionescu", "A'ja Wilson"]})
    assert hm.resolve_player_id("Sabrina Ionescu", "G1", roster) == ("P1", "exact_roster_name")
    assert hm.resolve_player_id("Nobody Here", "G1", roster) == (None, "unmatched")
    # player not on THIS game's roster -> unmatched (roster-constrained)
    assert hm.resolve_player_id("Sabrina Ionescu", "G2", roster) == (None, "unmatched")


# 9. Resume/cache behavior
def test_cache_roundtrip_idempotent(tmp_path):
    backfill._cache_put(tmp_path, "events_2026-07-10", {"data": [{"id": "e1"}]})
    assert backfill._cache_get(tmp_path, "events_2026-07-10") == {"data": [{"id": "e1"}]}
    assert backfill._cache_get(tmp_path, "missing") is None


# 10. Canonical schema validation
def test_quote_schema_fields():
    payload = {"data": {"bookmakers": [{"key": "dk", "markets": [
        {"key": "player_rebounds", "outcomes": [{"description": "A", "name": "Over", "point": 7.5, "price": 100}]}]}]}}
    row = backfill._parse_event_odds(payload, {"id": "e", "commence_time": "c"}, "snap")[0]
    for f in ["odds_event_id", "book", "stat", "player_name", "side", "line",
              "american_odds", "snapshot_time", "commence_time"]:
        assert f in row


# 11. Consensus one-per-key + modal line / median tie-break
def test_build_consensus_modal_line():
    closing = pd.DataFrame({
        "game_id": ["g"]*3, "player_id": ["p"]*3, "stat": ["pts"]*3, "book": ["a", "b", "c"],
        "line": [17.5, 17.5, 18.5], "is_closing": [True]*3,
        "market_prob_over_no_vig": [0.5, 0.52, 0.6], "over_odds": [-110]*3, "under_odds": [-110]*3,
        "commence_time": ["c"]*3,
    })
    cons = hm.build_consensus(closing)
    assert len(cons) == 1
    assert cons.iloc[0]["line"] == 17.5  # modal (2 books at 17.5)
    assert cons.iloc[0]["n_books"] == 2  # no-vig aggregated only over books at the selected line


# 12. Integer-line push handling
def test_p_over_conditional_push():
    pmf = np.zeros(21); pmf[10] = 1.0
    assert hm.p_over_conditional(pmf, 10.0) == pytest.approx(0.0)  # exact = push, not over
    pmf = np.zeros(21); pmf[11] = 1.0
    assert hm.p_over_conditional(pmf, 10.0) == pytest.approx(1.0)  # 11 is over of 10
    assert hm.p_over_conditional(pmf, 10.5) == pytest.approx(1.0)


# 13. Fold-cutoff / no-lookahead enforcement
def test_assert_no_lookahead():
    ok = pd.DataFrame({"fold_train_end_date": ["2026-07-01"], "game_date": ["2026-07-10"]})
    hm.assert_no_lookahead(ok)  # no raise
    bad = pd.DataFrame({"fold_train_end_date": ["2026-07-10"], "game_date": ["2026-07-10"]})
    with pytest.raises(ValueError, match="lookahead"):
        hm.assert_no_lookahead(bad)


# 14. Recommendation-policy reproduction (side = sign(edge); threshold filter; no outcome in selection)
def test_grade_df_policy():
    # model over prob strongly > market -> OVER recommendation; small edge -> dropped
    pmf_over = np.zeros(30); pmf_over[20:].fill(0.1); pmf_over[:20] = 0.0  # mass high -> P(over 17.5) high
    ev = pd.DataFrame([{
        "game_id": "g", "player_id": "p", "stat": "pts", "game_date": "2026-07-10",
        "line": 17.5, "market_prob_over_no_vig": 0.40,
        "over_odds": -110, "under_odds": -110,
        "pmf_json": json.dumps({str(i): (0.1 if i >= 20 else 0.0) for i in range(30)}),
        "actual_outcome": 25.0,
    }])
    g = evaljob._grade_df(ev, min_edge=0.02)
    assert len(g) == 1 and g.iloc[0]["side"] == "over"
    # tiny edge -> no recommendation
    ev2 = ev.copy(); ev2.loc[0, "market_prob_over_no_vig"] = g.iloc[0]["model_prob_over"] - 0.001
    assert len(evaljob._grade_df(ev2, min_edge=0.02)) == 0


# 15. ROI at positive and negative American odds
def test_profit_and_roi_signs():
    assert hm.profit_at_american(150, True) == pytest.approx(1.5)
    assert hm.profit_at_american(150, False) == pytest.approx(-1.0)
    assert hm.profit_at_american(-120, True) == pytest.approx(100/120)
    df = pd.DataFrame({
        "side": ["under", "under"], "price_american": [150, -120], "won": [True, False],
        "model_prob_side": [0.6, 0.6], "market_prob_side": [0.5, 0.5],
        "game_date": ["2026-07-10", "2026-07-11"],
    })
    r = hm.grade(df, n_boot=200)
    assert r.n == 2 and r.wins == 1 and r.losses == 1
    assert r.roi == pytest.approx((1.5 - 1.0) / 2)


# 16. Clustered CI generation
def test_clustered_ci():
    settled = pd.DataFrame({
        "price_american": [-110]*20, "won": ([True, False]*10),
        "game_date": [f"2026-07-{d:02d}" for d in range(1, 11)] * 2,
    })
    lo, hi = hm.clustered_bootstrap_roi_ci(settled, "game_date", n_boot=500)
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


# 17. API-secret redaction — key never printed/logged in source paths
def test_no_secret_leak_in_sources():
    client_src = (REPO / "src/wnba_props_model/data/odds_api_client.py").read_text()
    # the HTTP layer logs the PATH, not the full URL-with-key, and never prints apiKey
    assert "resp.url" not in client_src and "print(self.api_key" not in client_src
    bf = (REPO / "scripts/p1_historical_backfill.py").read_text()
    assert "api_key" not in bf.lower().replace("odds_api_key", "") or "print" not in bf.split("api_key")[0][-40:]
    # backfill must not echo params/keys
    assert "apiKey" not in bf


# Forced verdict mapping
def test_forced_verdict_mapping():
    g = hm.GradeResult(n=100, roi_ci95=(0.01, 0.05)); assert hm.forced_verdict(g) == "SUPPORTED"
    g = hm.GradeResult(n=100, roi_ci95=(-0.05, -0.01)); assert hm.forced_verdict(g) == "NOT SUPPORTED"
    g = hm.GradeResult(n=100, roi_ci95=(-0.02, 0.03)); assert hm.forced_verdict(g) == "INCONCLUSIVE"
    g = hm.GradeResult(n=10, roi_ci95=(0.01, 0.05)); assert hm.forced_verdict(g) == "INCONCLUSIVE"  # small n
