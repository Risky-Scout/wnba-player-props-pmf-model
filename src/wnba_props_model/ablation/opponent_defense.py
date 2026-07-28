"""Strictly-lagged opponent-defense and team/opponent pace features.

For every (game_id, defending_team_id) we compute how much of each stat the
defending team *allowed* to the opposing team's players in that game (summed
from the box), then attach to each row the EWMA / rolling mean of that allowed
amount over the defending team's games **strictly before** the current game.

A player in game ``g`` on team ``T`` facing opponent ``O`` therefore receives
``oppdef_<stat>_allowed_*`` = O's allowed-rate computed only from O's prior
games (never game ``g``), which is the leakage-safe analogue of the per-player
EWMA in ``data/pbp_features.py``.

The same construction yields a possessions-allowed pace proxy
(``oppdef_poss_allowed_*``) and the team's own prior-game possession proxy
(``oppdef_team_poss_*``).

``assert_no_opponent_defense_leakage`` brute-force recomputes the allowed EWMA
for a sample of rows from the defending team's strictly-prior games and fails on
any same-game / future leakage. This is the explicit leakage guard the study
requires for the opponent-defense group.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# stat -> box column used to sum the opponent-allowed amount
_ALLOWED_STATS = {
    "pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m", "fg3a": "fg3a",
    "stl": "stl", "blk": "blk", "tov": "turnover", "fga": "fga", "fta": "fta",
}


@dataclass
class OppDefConfig:
    ewma_halflife_games: float = 5.0
    roll_window: int = 5
    stats: tuple[str, ...] = field(
        default_factory=lambda: ("pts", "reb", "ast", "fg3m", "fg3a", "stl", "blk", "tov"))


def _poss_proxy(g: pd.DataFrame) -> pd.Series:
    """Team possession proxy = FGA + 0.44*FTA + TOV (per team-game)."""
    fga = pd.to_numeric(g.get("fga"), errors="coerce").fillna(0.0)
    fta = pd.to_numeric(g.get("fta"), errors="coerce").fillna(0.0)
    tov = pd.to_numeric(g.get("turnover"), errors="coerce").fillna(0.0)
    return fga + 0.44 * fta + tov


def _team_game_totals(box: pd.DataFrame, stats) -> pd.DataFrame:
    """Sum each stat over a team's players within a game -> one row per (game_id, team_id)."""
    b = box.copy()
    for c in ("game_id", "player_id", "team_id", "opponent_team_id"):
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["game_date"] = pd.to_datetime(b["game_date"], errors="coerce")
    cols = [s for s in stats if s in b.columns] + [c for c in ("fga", "fta", "turnover") if c in b.columns]
    cols = sorted(set(cols))
    agg = {c: "sum" for c in cols}
    tot = (b.groupby(["game_id", "team_id"], as_index=False)
             .agg({**agg, "game_date": "first", "opponent_team_id": "first"}))
    tot["poss"] = _poss_proxy(tot)
    return tot


def build_opponent_defense_features(box: pd.DataFrame,
                                    config: OppDefConfig | None = None) -> pd.DataFrame:
    """Return per-(game_id, player_id) opponent-defense + pace features (strictly lagged).

    Output is keyed (game_id, player_id) with ``oppdef_*`` columns that a player
    row can join on; the values depend only on the OPPONENT's (and the player's
    own team's) games strictly before the current game.
    """
    cfg = config or OppDefConfig()
    tot = _team_game_totals(box, cfg.stats)
    tot = tot.dropna(subset=["game_id", "team_id", "opponent_team_id", "game_date"])

    # "allowed by team X in game g" = the OPPONENT team's totals in game g.
    opp = tot.rename(columns={"team_id": "def_team_id", "opponent_team_id": "off_team_id"})
    off = tot.rename(columns=lambda c: c)  # offensive totals keyed (game_id, team_id)
    allowed_cols = [s for s in cfg.stats if s in tot.columns] + ["poss"]
    off_small = off[["game_id", "team_id"] + allowed_cols].rename(
        columns={"team_id": "off_team_id", **{c: f"allowed_{c}" for c in allowed_cols}})
    # for each defending team-game, join the offensive totals of the opponent it faced
    dg = opp[["game_id", "def_team_id", "off_team_id", "game_date", "poss"]].merge(
        off_small, on=["game_id", "off_team_id"], how="left")
    dg = dg.rename(columns={"poss": "team_poss"})
    dg = dg.sort_values(["def_team_id", "game_date", "game_id"]).reset_index(drop=True)

    hl, win = cfg.ewma_halflife_games, cfg.roll_window
    out = []
    for tid, g in dg.groupby("def_team_id", sort=False):
        g = g.sort_values(["game_date", "game_id"]).copy()
        g["oppdef_games_prior"] = np.arange(len(g))
        for c in allowed_cols:
            src = pd.to_numeric(g[f"allowed_{c}"], errors="coerce")
            g[f"oppdef_{c}_allowed_ewma"] = src.ewm(halflife=hl, adjust=True).mean().shift(1)
            g[f"oppdef_{c}_allowed_l{win}"] = src.rolling(win, min_periods=1).mean().shift(1)
        # defending team's own possession pace (prior games)
        tp = pd.to_numeric(g["team_poss"], errors="coerce")
        g["oppdef_team_poss_ewma"] = tp.ewm(halflife=hl, adjust=True).mean().shift(1)
        out.append(g)
    feat = pd.concat(out, ignore_index=True)

    feat_cols = [c for c in feat.columns if c.startswith("oppdef_")]
    team_feat = feat[["game_id", "def_team_id"] + feat_cols].copy()

    # attach to each player row via the player's OPPONENT (defending team the player faces)
    b = box[["game_id", "player_id", "team_id", "opponent_team_id", "game_date"]].copy()
    for c in ("game_id", "player_id", "team_id", "opponent_team_id"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    joined = b.merge(team_feat, left_on=["game_id", "opponent_team_id"],
                     right_on=["game_id", "def_team_id"], how="left")
    joined = joined.drop(columns=["def_team_id"])
    return joined.drop_duplicates(["game_id", "player_id"]).reset_index(drop=True)


def assert_no_opponent_defense_leakage(box: pd.DataFrame,
                                       config: OppDefConfig | None = None,
                                       n_spot_checks: int = 150) -> dict:
    """Brute-force leakage guard for the opponent-defense group.

    For a sample of (game, opponent) rows, recompute the opponent's allowed-EWMA
    of a stat from the opponent's games strictly before this game and confirm it
    matches the emitted ``oppdef_<stat>_allowed_ewma``. Raises on any leakage.
    """
    cfg = config or OppDefConfig()
    stat = "pts"
    feats = build_opponent_defense_features(box, cfg)
    tot = _team_game_totals(box, cfg.stats)

    # allowed-by-defending-team series, keyed (game_id, def_team, game_date)
    opp = tot.rename(columns={"team_id": "def_team_id", "opponent_team_id": "off_team_id"})
    off_small = tot[["game_id", "team_id", stat]].rename(
        columns={"team_id": "off_team_id", stat: "allowed"})
    dg = opp[["game_id", "def_team_id", "off_team_id", "game_date"]].merge(
        off_small, on=["game_id", "off_team_id"], how="left")

    rows = feats[feats[f"oppdef_{stat}_allowed_ewma"].notna()]
    if len(rows) == 0:
        return {"rows_checked": 0, "mismatches": 0, "leakage_free": True}
    sample = rows.sample(min(n_spot_checks, len(rows)), random_state=0)
    checked = mismatches = 0
    for _, row in sample.iterrows():
        gid, gdate, opp_team = row["game_id"], row["game_date"], row["opponent_team_id"]
        hist = dg[dg["def_team_id"] == opp_team].copy()
        prior = hist[(hist["game_date"] < gdate) |
                     ((hist["game_date"] == gdate) & (hist["game_id"] < gid))]
        prior = prior.sort_values(["game_date", "game_id"])
        if len(prior) == 0:
            continue
        expected = (pd.to_numeric(prior["allowed"], errors="coerce")
                    .ewm(halflife=cfg.ewma_halflife_games, adjust=True).mean().iloc[-1])
        got = float(row[f"oppdef_{stat}_allowed_ewma"])
        checked += 1
        if not np.isclose(expected, got, rtol=1e-6, atol=1e-9):
            mismatches += 1
    if mismatches:
        raise AssertionError(
            f"opponent-defense leakage guard FAILED: {mismatches}/{checked} rows' "
            f"oppdef_{stat}_allowed_ewma did not match a strictly-prior recomputation")
    return {"rows_checked": checked, "mismatches": mismatches,
            "feature_rows": int(len(feats)), "leakage_free": True}
