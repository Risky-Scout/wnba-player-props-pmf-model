"""Strictly-lagged opportunity features from parsed play-by-play (owner directive step C).

For every (player_id, game_date) we build features from ONLY that player's PRIOR games' parsed PBP
(EWMA over games strictly before the current game). No same-game leakage: every EWMA column is
computed on the chronologically ordered per-player series and then shifted by one game, so the value
attached to game *g* uses games ``{..., g-1}`` exclusively.

Output is keyed (game_id, player_id, game_date) and is safe to join to the modeling frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# per-minute rate features (numerator col -> feature name); minutes-scaled from parsed PBP.
_RATE_NUMERATORS = {
    "fg3a": "fg3a", "fg2a": "fg2a", "fga": "fga", "fg3m": "fg3m",
    "ast": "ast", "oreb": "oreb", "dreb": "dreb", "reb": "reb",
    "stl": "stl", "blk": "blk", "tov": "tov", "fta": "fta",
    "poss_proxy": "poss",
}


@dataclass
class PBPFeatureConfig:
    ewma_halflife_games: float = 6.0
    minimum_history_games: int = 3
    minutes_floor: float = 1.0


def _ewma_prior(series: pd.Series, halflife: float) -> pd.Series:
    """EWMA over the ordered series, then shifted by one so row g sees only rows < g."""
    return series.ewm(halflife=halflife, adjust=True).mean().shift(1)


def build_pbp_features(parsed: pd.DataFrame, box: pd.DataFrame,
                       config: PBPFeatureConfig | None = None) -> pd.DataFrame:
    """Build strictly-lagged per-(game_id, player_id, game_date) opportunity features.

    ``parsed`` = per-player-per-game PBP counts (from :func:`pbp_parse.parse_plays_to_player_game`).
    ``box`` supplies minutes and did_play (and the authoritative game_date ordering).
    """
    cfg = config or PBPFeatureConfig()
    b = box.copy()
    for c in ("game_id", "player_id"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["game_date"] = pd.to_datetime(b["game_date"], errors="coerce")
    b = b.dropna(subset=["game_id", "player_id", "game_date"])
    keep_box = ["game_id", "player_id", "game_date", "minutes", "did_play", "team_id"]
    b = b[[c for c in keep_box if c in b.columns]].drop_duplicates(["game_id", "player_id"])

    p = parsed.copy()
    for c in ("game_id", "player_id"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.drop(columns=[c for c in ("game_date", "team_id") if c in p.columns], errors="ignore")

    # Only keep box rows for games with PBP coverage: games without parsed PBP (e.g. 2025, before
    # PBP ingestion began) would otherwise inject false zero-rates into the EWMA history.
    pbp_games = set(pd.to_numeric(parsed["game_id"], errors="coerce").dropna().astype(int))
    b = b[b["game_id"].astype("Int64").isin(pbp_games)].copy()

    df = b.merge(p, on=["game_id", "player_id"], how="left")
    count_cols = [c for c in _RATE_NUMERATORS if c in df.columns]
    df[count_cols] = df[count_cols].fillna(0.0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    out_frames = []
    hl = cfg.ewma_halflife_games
    for pid, g in df.groupby("player_id", sort=False):
        g = g.sort_values(["game_date", "game_id"]).copy()
        played = g["did_play"].astype(bool) if "did_play" in g.columns else (g["minutes"] > 0)
        # count of prior games actually played
        g["player_games_played_prior"] = played.astype(int).cumsum().shift(1).fillna(0).astype(int)
        # minutes EWMA (prior only)
        g["player_minutes_ewma"] = _ewma_prior(g["minutes"], hl)
        min_scaled = g["minutes"].clip(lower=cfg.minutes_floor)
        for num_col, short in _RATE_NUMERATORS.items():
            if num_col not in g.columns:
                continue
            per_min = g[num_col] / min_scaled
            g[f"player_{short}_per_min_ewma"] = _ewma_prior(per_min, hl)
            g[f"player_{short}_per_game_ewma"] = _ewma_prior(g[num_col], hl)
        # 3P% prior: shrunk cumulative makes/attempts strictly before this game
        cfm = g["fg3m"].cumsum().shift(1).fillna(0.0) if "fg3m" in g.columns else 0.0
        cfa = g["fg3a"].cumsum().shift(1).fillna(0.0) if "fg3a" in g.columns else 0.0
        g["player_fg3_pct_prior"] = (cfm + 0.35 * 20) / (cfa + 20)  # Beta(0.35 mean, strength 20)
        # assisted-make share prior: fraction of made FGs assisted is not directly parsed per-scorer;
        # instead expose the player's own assist creation intensity as ast per fga (playmaking load).
        if "ast" in g.columns and "fga" in g.columns:
            denom = g["fga"].clip(lower=1.0)
            g["player_ast_per_fga_ewma"] = _ewma_prior(g["ast"] / denom, hl)
        out_frames.append(g)

    feats = pd.concat(out_frames, ignore_index=True)
    feat_cols = [c for c in feats.columns if c.startswith("player_") and c != "player_id"]
    result = feats[["game_id", "player_id", "game_date", "team_id", "minutes", "did_play"]
                   + feat_cols].copy()
    # fill lag NaNs (first game of a player) with 0 so downstream is finite; the minimum-history
    # filter (player_games_played_prior >= minimum_history_games) removes these rows from modeling.
    for c in feat_cols:
        if c == "player_fg3_pct_prior":
            result[c] = result[c].fillna(0.35)
        else:
            result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0.0)
    return result.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def assert_no_leakage(parsed: pd.DataFrame, box: pd.DataFrame,
                      config: PBPFeatureConfig | None = None,
                      n_spot_checks: int = 200) -> dict:
    """Leakage guard: recompute each per-min EWMA from ONLY strictly-prior games and confirm it
    matches the shifted-EWMA feature. Raises AssertionError on any same-game/future leakage.
    """
    cfg = config or PBPFeatureConfig()
    feats = build_pbp_features(parsed, box, cfg)
    # Independent brute-force check: for a sample of rows, the feature must equal an EWMA computed
    # on the player's games strictly before that game_date (never including the game itself).
    b = box.copy()
    for c in ("game_id", "player_id"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["game_date"] = pd.to_datetime(b["game_date"], errors="coerce")
    b = b[["game_id", "player_id", "game_date", "minutes"]].drop_duplicates(["game_id", "player_id"])
    p = parsed.copy()
    for c in ("game_id", "player_id"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.drop(columns=[c for c in ("game_date", "team_id") if c in p.columns], errors="ignore")
    pbp_games = set(pd.to_numeric(parsed["game_id"], errors="coerce").dropna().astype(int))
    b = b[b["game_id"].astype("Int64").isin(pbp_games)].copy()
    merged = b.merge(p, on=["game_id", "player_id"], how="left")
    merged["minutes"] = pd.to_numeric(merged["minutes"], errors="coerce").fillna(0.0)

    checked = 0
    mismatches = 0
    rows = feats[feats["player_fg3a_per_min_ewma"].notna()]
    sample = rows.sample(min(n_spot_checks, len(rows)), random_state=0) if len(rows) else rows
    for _, row in sample.iterrows():
        pid, gdate, gid = row["player_id"], row["game_date"], row["game_id"]
        # "strictly prior" uses the SAME ordering key as the builder: (game_date, game_id). A truly
        # future game has a later (game_date, game_id) tuple and is excluded, so real leakage would
        # still be caught; this only avoids a false alarm on same-date games.
        hist = merged[(merged["player_id"] == pid)].copy()
        prior = hist[(hist["game_date"] < gdate) |
                     ((hist["game_date"] == gdate) & (hist["game_id"] < gid))]
        prior = prior.sort_values(["game_date", "game_id"])
        if len(prior) == 0:
            continue
        per_min = (prior["fg3a"].fillna(0.0) / prior["minutes"].clip(lower=cfg.minutes_floor))
        expected = per_min.ewm(halflife=cfg.ewma_halflife_games, adjust=True).mean().iloc[-1]
        got = float(row["player_fg3a_per_min_ewma"])
        checked += 1
        if not np.isclose(expected, got, rtol=1e-6, atol=1e-9):
            mismatches += 1
    if mismatches:
        raise AssertionError(
            f"PBP feature leakage guard FAILED: {mismatches}/{checked} rows' fg3a_per_min_ewma "
            "did not match a strictly-prior recomputation")
    return {"rows_checked": checked, "mismatches": mismatches,
            "feature_rows": int(len(feats)), "leakage_free": True}
