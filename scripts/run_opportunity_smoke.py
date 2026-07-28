#!/usr/bin/env python3
"""Deterministic production-shape smoke for Opportunity V2 (directive section 35).

Runs one complete fold on a deterministic synthetic slate spanning every enabled prop, both roles,
starters and bench, high/low minutes, and a late DNP: feature build -> fit -> bundle save/reload ->
re-predict -> compare PMFs -> run every temporal/PMF/settlement invariant. Fails closed (non-zero
exit) on any violation. Safe to run in CI without private data or API keys.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.audit import audit_temporal_purity
from wnba_props_model.opportunity.bundle import OpportunityModelBundleV2
from wnba_props_model.opportunity.feature_builder import (
    OpportunityFeatureConfig,
    build_opportunity_feature_frame,
)
from wnba_props_model.opportunity.pmf_builders import settled_over_probability


def _slate(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-05-01T23:00:00Z")
    rows, games = [], []
    for g in range(30):
        gid = 5000 + g
        tip = base + pd.Timedelta(days=g)
        games.append({"game_id": gid, "game_date": tip})
        for p in range(1, 13):
            team = 1 if p <= 6 else 2
            starter = p in (1, 2, 7, 8)
            played = (rng.random() > 0.12) and not (p == 1 and g >= 24)  # a high-usage late DNP
            mins = float(rng.uniform(24, 34) if starter else rng.uniform(6, 20)) if played else 0.0
            fg3a = rng.poisson(5 if starter else 2) if played else 0
            rows.append({
                "game_id": gid, "player_id": p, "team_id": team,
                "opponent_team_id": 2 if team == 1 else 1,
                "game_date": tip.tz_convert(None), "minutes": mins, "did_play": played,
                "fga": rng.poisson(12 if starter else 5) if played else 0, "fg3a": fg3a,
                "fta": rng.poisson(3) if played else 0,
                "fg3m": rng.binomial(fg3a, 0.36) if fg3a else 0, "pts": rng.poisson(15 if starter else 7) if played else 0,
                "reb": rng.poisson(5) if played else 0, "ast": rng.poisson(3) if played else 0,
                "turnover": rng.poisson(2) if played else 0, "stl": rng.poisson(1) if played else 0,
                "blk": rng.poisson(1) if played else 0, "oreb": rng.poisson(1) if played else 0,
                "dreb": rng.poisson(4) if played else 0,
                "position": "G" if p % 2 else "F", "started_proxy": bool(played and starter),
            })
    return pd.DataFrame(rows), pd.DataFrame(games)


def main() -> None:
    pg, games = _slate()
    frame, manifest = build_opportunity_feature_frame(
        pg, games, None, None, None, None, None, None, OpportunityFeatureConfig(minimum_history_games=1))
    if manifest["forbidden_market_columns_found"]:
        raise SystemExit(f"SMOKE FAIL: forbidden market features {manifest['forbidden_market_columns_found']}")
    audit = audit_temporal_purity(frame, "prediction_cutoff_utc",
                                  manifest["source_timestamp_columns"],
                                  feature_columns=manifest["model_feature_columns"])
    if not audit.passed:
        raise SystemExit(f"SMOKE FAIL: temporal audit {audit.to_dict()}")

    frame = frame[frame["player_games_played_prior"] >= 3].reset_index(drop=True)
    cut = pd.Timestamp("2026-05-22T00:00:00Z")
    train, val = frame[frame["game_date"] < cut], frame[frame["game_date"] >= cut]
    if len(train) < 100 or len(val) == 0:
        raise SystemExit("SMOKE FAIL: insufficient synthetic rows")

    bundle = OpportunityModelBundleV2().fit(train, train)
    pred = bundle.predict_active_pmfs(val, None, ["fg3m", "pts"])

    with tempfile.TemporaryDirectory() as d:
        bundle.save(Path(d))
        reloaded = OpportunityModelBundleV2.load(Path(d))
        pred2 = reloaded.predict_active_pmfs(val, None, ["fg3m", "pts"])
    if not np.allclose(pred["active_pmf_mean"].to_numpy(), pred2["active_pmf_mean"].to_numpy(), atol=1e-9):
        raise SystemExit("SMOKE FAIL: save/reload prediction mismatch (non-deterministic)")

    for js in pred["active_pmf_json"]:
        arr = np.asarray(json.loads(js), float)
        if abs(arr.sum() - 1.0) > 1e-6 or np.any(arr < -1e-9) or not np.all(np.isfinite(arr)):
            raise SystemExit("SMOKE FAIL: PMF not normalized / invalid")
    for js in pred[pred["stat"] == "fg3m"]["active_pmf_json"].head(20):
        over, under, push = settled_over_probability(np.asarray(json.loads(js), float), 1.5)
        if abs((over + under) - 1.0) > 1e-6 or push != 0.0:
            raise SystemExit("SMOKE FAIL: half-line settlement invariant")
    if set(pred["candidate_id"]) != {"OPP_V2_RAW"}:
        raise SystemExit("SMOKE FAIL: candidate id not fixed")

    print(json.dumps({"smoke": "PASS", "feature_rows": int(manifest["rows"]),
                      "pred_rows": int(len(pred)), "temporal_violations": audit.violation_count,
                      "props": sorted(pred["stat"].unique().tolist())}, indent=2))


if __name__ == "__main__":
    main()
