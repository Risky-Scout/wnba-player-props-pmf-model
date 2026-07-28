#!/usr/bin/env python3
"""Phase 8C - cheap box-derived REB opportunity proxy screen (diagnostic; NOT a champion).

All inputs are STRICTLY LAGGED (prior-game) box aggregates -- no target-game info, no market info.
Rebound-opportunity proxies are DERIVED, not measured, and named accordingly:

  team_off_misses_proxy(team, game)  = sum(team OREB) + sum(opponent DREB)   # own missed FGs rebounded
  team_def_misses_proxy(team, game)  = sum(team DREB) + sum(opponent OREB)   # opp missed FGs rebounded
  player_oreb_share_proxy            = player OREB / team_off_misses_proxy   (lagged mean)
  player_dreb_share_proxy            = player DREB / team_def_misses_proxy   (lagged mean)

Candidate expected REB = E[team_off_misses]*oreb_share + E[team_def_misses]*dreb_share  (all lagged).
Baseline (P0-proxy)    = player's lagged expanding-mean REB.

Both means are turned into an NB PMF with ONE shared, data-estimated variance/mean ratio so the screen
isolates the MEAN/share structure. Evaluated on the migrated REB atomic decision-snapshot pairs (settled
binary) via LL / Brier / AUC. STOP RULE: proxy must beat baseline on AUC AND LL AND Brier, else reject.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
BOX = REPO / "data/processed/wnba_player_game_stats.parquet"
PAIRS = REPO / "data/processed/atomic_quotes/atomic_pairs.parquet"
OUT = REPO / "artifacts/pure_model_completion/REB_BOX_PROXY_SCREEN.json"


def _nb_p_over(mu: np.ndarray, line: np.ndarray, phi: float) -> np.ndarray:
    mu = np.clip(mu, 1e-6, None)
    phi = max(phi, 1.0 + 1e-6)
    r = mu / (phi - 1.0)                       # var = mu*phi
    p = r / (r + mu)
    k = np.floor(line).astype(int)            # over line => X >= k+1
    return 1.0 - stats.nbinom.cdf(k, r, p)


def main() -> None:
    box = pd.read_parquet(BOX).sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # ---- per-team, per-game rebound-opportunity proxies (derived from rebound identities) ----
    tm = box.groupby(["game_id", "team_id"], as_index=False).agg(
        team_oreb=("oreb", "sum"), team_dreb=("dreb", "sum"))
    opp = tm[["game_id", "team_id", "team_oreb", "team_dreb"]].rename(
        columns={"team_id": "opponent_team_id", "team_oreb": "opp_oreb", "team_dreb": "opp_dreb"})
    tm = tm.merge(box[["game_id", "team_id", "opponent_team_id"]].drop_duplicates(),
                  on=["game_id", "team_id"], how="left")
    tm = tm.merge(opp, on=["game_id", "opponent_team_id"], how="left")
    tm["team_off_misses_proxy"] = tm["team_oreb"] + tm["opp_dreb"]
    tm["team_def_misses_proxy"] = tm["team_dreb"] + tm["opp_oreb"]
    tm = tm.merge(box[["game_id", "game_date"]].drop_duplicates(), on="game_id", how="left")
    tm = tm.sort_values(["team_id", "game_date", "game_id"])
    # lagged (prior-game) expanding team environment
    tm["team_off_misses_lag"] = tm.groupby("team_id")["team_off_misses_proxy"].transform(
        lambda s: s.shift(1).expanding().mean())
    tm["team_def_misses_lag"] = tm.groupby("team_id")["team_def_misses_proxy"].transform(
        lambda s: s.shift(1).expanding().mean())

    df = box.merge(tm[["game_id", "team_id", "team_off_misses_proxy", "team_def_misses_proxy",
                       "team_off_misses_lag", "team_def_misses_lag"]],
                   on=["game_id", "team_id"], how="left")
    df = df.sort_values(["player_id", "game_date", "game_id"])

    def _lag_mean(s):
        return s.shift(1).expanding().mean()

    g = df.groupby("player_id")
    df["reb_lag"] = g["reb"].transform(_lag_mean)
    # per-game player shares of the derived team misses, then lagged mean of the share
    df["oreb_share_g"] = df["oreb"] / df["team_off_misses_proxy"].replace(0, np.nan)
    df["dreb_share_g"] = df["dreb"] / df["team_def_misses_proxy"].replace(0, np.nan)
    df["oreb_share_lag"] = df.groupby("player_id")["oreb_share_g"].transform(_lag_mean)
    df["dreb_share_lag"] = df.groupby("player_id")["dreb_share_g"].transform(_lag_mean)

    df["mu_baseline"] = df["reb_lag"]
    df["mu_proxy"] = (df["team_off_misses_lag"] * df["oreb_share_lag"]
                      + df["team_def_misses_lag"] * df["dreb_share_lag"])

    # ---- join to migrated REB atomic decision-snapshot settled pairs ----
    pairs = pd.read_parquet(PAIRS)
    reb = pairs[(pairs["prop"] == "reb") & (pairs["snapshot_label"] == "decision")
                & (pairs["binary_settled_eligible"])].copy()
    for c in ("game_id", "player_id"):
        reb[c] = reb[c].astype(str); df[c] = df[c].astype(str)
    dsel = df[["game_id", "player_id", "reb", "mu_baseline", "mu_proxy"]].rename(
        columns={"reb": "reb_actual"})
    m = reb.merge(dsel, on=["game_id", "player_id"], how="left")
    m = m.dropna(subset=["mu_baseline", "mu_proxy"])
    y = (m["outcome"] == "over").astype(int).values
    line = m["line"].values.astype(float)

    # one shared variance/mean ratio from residuals of the baseline mean
    resid_var = float(np.var(m["reb_actual"].values - m["mu_baseline"].values))
    mean_mu = float(np.mean(m["mu_baseline"].values))
    phi = max(1.05, resid_var / max(mean_mu, 1e-6))

    res = {}
    for name, mucol in (("baseline", "mu_baseline"), ("proxy", "mu_proxy")):
        p = np.clip(_nb_p_over(m[mucol].values, line, phi), 1e-6, 1 - 1e-6)
        res[name] = {
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        }

    beats = (res["proxy"]["auc"] > res["baseline"]["auc"]
             and res["proxy"]["log_loss"] < res["baseline"]["log_loss"]
             and res["proxy"]["brier"] < res["baseline"]["brier"])
    verdict = ("PROXY_ADVANCES_TO_FULL_OOF" if beats
               else "REJECT_PROXY_DOES_NOT_IMPROVE_P0 (stop per phase-8C rule)")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "screen": "box-derived REB opportunity proxy (diagnostic, strictly-lagged, no market)",
        "n_rows": int(len(m)), "n_dates": int(m["game_date"].nunique()),
        "nb_variance_mean_ratio_phi": phi,
        "proxies_are_derived_not_measured": True,
        "baseline": res["baseline"], "proxy": res["proxy"], "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
