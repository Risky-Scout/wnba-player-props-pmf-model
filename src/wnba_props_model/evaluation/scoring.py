from __future__ import annotations

from pathlib import Path

import pandas as pd

from wnba_props_model.evaluation.diagnostics import pmf_nll, rps
from wnba_props_model.models.market import brier, binary_logloss, prob_over_from_pmf
from wnba_props_model.models.simulation import json_to_pmf


def score_daily_pmf_delivery_after_game(
    pmfs_path: str | Path,
    outcomes_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    pmfs = pd.read_parquet(pmfs_path)
    outcomes = pd.read_parquet(outcomes_path)
    if "stat" not in outcomes:
        wide = []
        for stat in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "stocks", "pa", "pr", "ra", "pra"):
            if stat in outcomes:
                tmp = outcomes[["game_id", "player_id", stat]].copy()
                tmp["stat"] = stat
                tmp["outcome"] = tmp[stat]
                wide.append(tmp[["game_id", "player_id", "stat", "outcome"]])
        outcomes = pd.concat(wide, ignore_index=True)
    joined = pmfs.merge(outcomes, on=["game_id", "player_id", "stat"], how="inner")
    joined["pmf"] = joined["pmf_json"].map(json_to_pmf)
    joined["pmf_nll"] = [pmf_nll(p, y) for p, y in zip(joined["pmf"], joined["outcome"])]
    joined["rps"] = [rps(p, y) for p, y in zip(joined["pmf"], joined["outcome"])]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joined.drop(columns=["pmf"]).to_parquet(out_path, index=False)
    return joined
