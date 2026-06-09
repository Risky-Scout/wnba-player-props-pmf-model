from __future__ import annotations

import numpy as np
import pandas as pd

from wnba_props_model.constants import DIRECT_STATS
from wnba_props_model.data.normalize import safe_rate
from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket


def _shifted_roll(s: pd.Series, window: int, min_periods: int = 1, agg: str = "mean") -> pd.Series:
    shifted = s.shift(1)
    roller = shifted.rolling(window, min_periods=min_periods)
    return getattr(roller, agg)()


def add_player_rolling_features(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.sort_values(["player_id", "game_date", "game_id"]).copy()
    grp = df.groupby("player_id", group_keys=False)
    df["minutes_lag1"] = grp["minutes"].shift(1)
    for w in (3, 5, 10):
        df[f"minutes_roll{w}"] = grp["minutes"].transform(lambda s, w=w: _shifted_roll(s, w))
    df["minutes_std10"] = grp["minutes"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    df["start_proxy"] = (df["minutes"] >= 24).astype(float)
    df["start_proxy_lag1"] = grp["start_proxy"].shift(1)
    df["recent_starter_rate5"] = grp["start_proxy"].transform(lambda s: _shifted_roll(s, 5))

    for stat in DIRECT_STATS:
        rate = safe_rate(df[stat], df["minutes"].replace(0, np.nan).fillna(0))
        df[f"{stat}_per_min"] = rate.replace([np.inf, -np.inf], np.nan).fillna(0)
        w = 10 if stat in ("stl", "blk") else 5
        df[f"{stat}_per_min_roll{w}"] = grp[f"{stat}_per_min"].transform(lambda s, w=w: _shifted_roll(s, w))

    df["fga_per_min"] = safe_rate(df.get("fga", 0), df["minutes"].replace(0, np.nan).fillna(0)).fillna(0)
    df["fta_per_min"] = safe_rate(df.get("fta", 0), df["minutes"].replace(0, np.nan).fillna(0)).fillna(0)
    df["usage_proxy"] = df["fga_per_min"] + 0.44 * df["fta_per_min"] + df["tov_per_min"]
    for col in ("fga_per_min", "fta_per_min", "usage_proxy"):
        df[f"{col}_roll5"] = grp[col].transform(lambda s: _shifted_roll(s, 5))
    return df


def add_team_context_features(stats: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    df = stats.copy()
    team_game = (
        df.groupby(["game_id", "game_date", "team_id"], as_index=False)
        .agg(team_pts=("pts", "sum"), team_reb=("reb", "sum"), team_ast=("ast", "sum"), team_tov=("tov", "sum"))
    )
    team_game = team_game.sort_values(["team_id", "game_date", "game_id"])
    for c in ("team_pts", "team_reb", "team_ast", "team_tov"):
        team_game[f"{c}_roll5"] = team_game.groupby("team_id")[c].transform(lambda s: _shifted_roll(s, 5))
    team_game["team_pace_proxy_roll5"] = team_game["team_pts_roll5"]  # BDL WNBA PBP can replace this with possession estimate.
    df = df.merge(team_game[["game_id", "team_id", "team_pts_roll5", "team_reb_roll5", "team_ast_roll5", "team_tov_roll5", "team_pace_proxy_roll5"]], on=["game_id", "team_id"], how="left")

    if games is not None and not games.empty:
        game_teams = []
        for _, r in games.iterrows():
            game_teams.append({"game_id": r["game_id"], "team_id": r["home_team_id"], "opponent_team_id": r["away_team_id"], "is_home": 1})
            game_teams.append({"game_id": r["game_id"], "team_id": r["away_team_id"], "opponent_team_id": r["home_team_id"], "is_home": 0})
        gt = pd.DataFrame(game_teams)
        df = df.merge(gt, on=["game_id", "team_id"], how="left")
    else:
        df["opponent_team_id"] = np.nan
        df["is_home"] = 0

    opp_allowed = (
        df.groupby(["game_id", "opponent_team_id"], as_index=False)
        .agg(opp_pts_allowed=("pts", "sum"), opp_reb_allowed=("reb", "sum"), opp_ast_allowed=("ast", "sum"), opp_tov_allowed=("tov", "sum"))
        .rename(columns={"opponent_team_id": "team_id"})
    )
    opp_allowed = opp_allowed.sort_values(["team_id", "game_id"])
    for c in ("opp_pts_allowed", "opp_reb_allowed", "opp_ast_allowed", "opp_tov_allowed"):
        opp_allowed[f"{c}_roll5"] = opp_allowed.groupby("team_id")[c].transform(lambda s: _shifted_roll(s, 5))
    opp_allowed["opp_tov_rate_roll5"] = opp_allowed["opp_tov_allowed_roll5"] / 100.0
    opp_allowed["opp_rim_pressure_proxy_roll5"] = opp_allowed["opp_reb_allowed_roll5"] / 40.0

    df = df.merge(
        opp_allowed[["game_id", "team_id", "opp_pts_allowed_roll5", "opp_reb_allowed_roll5", "opp_ast_allowed_roll5", "opp_tov_rate_roll5", "opp_rim_pressure_proxy_roll5"]],
        left_on=["game_id", "opponent_team_id"],
        right_on=["game_id", "team_id"],
        how="left",
        suffixes=("", "_oppmerge"),
    )
    if "team_id_oppmerge" in df:
        df = df.drop(columns=["team_id_oppmerge"])
    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["team_id", "game_date", "game_id"]).copy()
    out["prev_team_game_date"] = out.groupby("team_id")["game_date"].shift(1)
    out["rest_days"] = (out["game_date"] - out["prev_team_game_date"]).dt.days.fillna(4).clip(0, 10)
    out["is_b2b"] = (out["rest_days"] <= 1).astype(int)
    out["is_3in4"] = (out["rest_days"] <= 2).astype(int)
    out["travel_proxy"] = 0.0
    if "postseason" not in out:
        out["postseason"] = 0
    return out


def finalize_training_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ex-ante minutes proxy: strictly lagged rolling minutes, never actual minutes.
    out["pred_minutes_mean"] = out["minutes_roll5"].fillna(out["minutes_roll10"]).fillna(out["minutes_lag1"]).fillna(0).clip(0, 40)
    out["pred_minutes_q25"] = (out["pred_minutes_mean"] - out["minutes_std10"].fillna(4.0) * 0.675).clip(0, 40)
    out["p_inactive"] = np.where(out["minutes_lag1"].fillna(0) <= 1, 0.15, 0.02)
    out = add_ex_ante_role_bucket(out)
    out["role_bucket_code"] = out["role_bucket"].astype("category").cat.codes.astype(float)
    out["player_id_code"] = out["player_id"].astype("category").cat.codes.astype(float)
    out["team_id_code"] = out["team_id"].astype("category").cat.codes.astype(float)
    out["opponent_team_id_code"] = out.get("opponent_team_id", pd.Series(np.nan, index=out.index)).astype("category").cat.codes.astype(float)
    for pos in ("G", "F", "C"):
        out[f"position_{pos}"] = out.get("position", "").astype(str).str.contains(pos, regex=False).astype(float)
    out["expected_starter"] = (out["pred_minutes_mean"] >= 24).astype(float)
    out["expected_bench"] = 1.0 - out["expected_starter"]
    out["team_expected_starters_count"] = out.groupby(["game_id", "team_id"])["expected_starter"].transform("sum").fillna(5)
    out["lineup_confirmed"] = 0.0
    out["confirmed_starter"] = 0.0
    for c in ("team_out_count", "team_questionable_count", "usage_vacated_proxy", "rebound_vacated_proxy", "assist_vacated_proxy"):
        out[c] = out.get(c, 0.0)
    out["stl_opp_tov_rate"] = out.get("opp_tov_rate_roll5", 0.12)
    out["stl_opp_pass_risk"] = out.get("opp_tov_rate_roll5", 0.12)
    out["blk_opp_rim_att"] = out.get("opp_rim_pressure_proxy_roll5", 0.5)
    out["defender_role_code"] = out["position_C"] * 2 + out["position_F"]
    return out.replace([np.inf, -np.inf], np.nan)


def build_player_training_table(stats: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    df = add_player_rolling_features(stats)
    df = add_team_context_features(df, games)
    df = add_schedule_features(df)
    return finalize_training_features(df)


def build_game_total_training_table(games: pd.DataFrame, team_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    g = games.copy().sort_values(["game_date", "game_id"])
    if g.empty:
        return g
    long_rows = []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        long_rows.append(pd.DataFrame({
            "game_id": g["game_id"],
            "game_date": g["game_date"],
            "team_id": g[f"{side}_team_id"],
            "opponent_team_id": g[f"{opp}_team_id"],
            "is_home": 1 if side == "home" else 0,
            "team_points": g[f"{side}_score"],
            "opp_points": g[f"{opp}_score"],
        }))
    long = pd.concat(long_rows, ignore_index=True).sort_values(["team_id", "game_date", "game_id"])
    for c in ("team_points", "opp_points"):
        for w in (3, 5, 10):
            long[f"{c}_roll{w}"] = long.groupby("team_id")[c].transform(lambda s, w=w: _shifted_roll(s, w))
    long["rest_days"] = long.groupby("team_id")["game_date"].diff().dt.days.fillna(4).clip(0, 10)
    home = long[long["is_home"] == 1].add_prefix("home_").rename(columns={"home_game_id": "game_id", "home_game_date": "game_date"})
    away = long[long["is_home"] == 0].add_prefix("away_").rename(columns={"away_game_id": "game_id", "away_game_date": "game_date"})
    out = g[["game_id", "game_date", "season", "postseason", "game_total", "home_team_total", "away_team_total"]].merge(home, on=["game_id", "game_date"]).merge(away, on=["game_id", "game_date"])
    out["market_total_no_vig"] = np.nan
    return out
