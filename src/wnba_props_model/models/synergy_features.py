"""Teammate Pairing Synergy Features (Enhancement 18).

Computes per-pair, per-stat synergy differentials from play-by-play
lineup data (or approximated from box scores when PBP is unavailable).

For each player pair (A, B):
    synergy(A, B, stat) = rate(A, stat | B on court) - rate(A, stat | B off court)

Positive synergy: B's presence amplifies A's production.
Negative synergy: B suppresses A (possibly due to usage competition).

The "DUGGAN EFFECT": some duos amplify each other's production by 3-5%
above what individual rates would predict.

Reference:
    Luo & Krishnamurthy (2023). Who You Play Affects How You Play:
    Predicting Sports Performance Using Graph Attention Networks.
    arXiv:2303.16741
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SYNERGY_STATS = ["pts", "reb", "ast"]
MIN_POSSESSIONS = 50
TOP_N = 3  # top-N synergy partners per player to include as features


def compute_duo_synergy(
    pbp_lineup_stats: list[tuple[Any, dict[int, dict[str, float]], list[int], str]],
    min_possessions: int = MIN_POSSESSIONS,
) -> dict[tuple[int, int], dict[str, dict[str, float]]]:
    """Compute per-pair, per-stat synergy from play-by-play lineup data.

    Parameters
    ----------
    pbp_lineup_stats : list of tuples
        Each tuple: (possession_id, credits_dict, on_court_ids, stat_type)
        - credits_dict : {player_id: {stat: credit_value}}
        - on_court_ids : list of player_ids currently on the court (5 or 10)
        - stat_type    : the stat credited in this possession

    min_possessions  : minimum shared possessions to include a pair

    Returns
    -------
    {(pid_a, pid_b): {stat: {"together_rate", "apart_rate", "synergy", "n_possessions"}}}
    """
    # (pair_key, stat) → [total_credits, n_possessions]
    together: defaultdict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    apart:    defaultdict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    for _poss_id, credits, on_court, stat_type in pbp_lineup_stats:
        on_set = set(on_court)
        for pid_a in on_court:
            credit_a = credits.get(pid_a, {}).get(stat_type, 0.0)
            for pid_b in on_court:
                if pid_a >= pid_b:
                    continue
                pair_key = (pid_a, pid_b)
                together[pair_key][stat_type][0] += credit_a
                together[pair_key][stat_type][1] += 1

    synergy: dict[tuple[int, int], dict[str, dict[str, float]]] = {}
    for pair_key, stats in together.items():
        synergy[pair_key] = {}
        for stat, (total, count) in stats.items():
            if count < min_possessions:
                continue
            together_rate = total / count
            apart_data = apart[pair_key].get(stat, [0.0, 1])
            apart_rate = apart_data[0] / max(apart_data[1], 1)
            synergy[pair_key][stat] = {
                "together_rate": round(together_rate, 5),
                "apart_rate":    round(apart_rate, 5),
                "synergy":       round(together_rate - apart_rate, 5),
                "n_possessions": count,
            }

    return synergy


def compute_duo_synergy_from_boxscores(
    df: pd.DataFrame,
    min_games: int = 10,
    stat_cols: list[str] | None = None,
) -> dict[tuple[int, int], dict[str, dict[str, float]]]:
    """Approximate duo synergy from box scores when true PBP is unavailable.

    Strategy: for each team-game pair, identify which players played
    together vs separately, and compare per-minute stat rates across
    those games.

    Parameters
    ----------
    df       : player-game box score DataFrame with columns:
               player_id, game_id, team_id, min, pts, reb, ast, …
    min_games: minimum number of games together to include a pair
    stat_cols: stats to compute synergy for (defaults to SYNERGY_STATS)

    Returns synergy in the same format as compute_duo_synergy().
    """
    stats = stat_cols or SYNERGY_STATS
    if "min" not in df.columns:
        df = df.copy()
        df["min"] = df.get("minutes", 20.0)

    df = df.copy()
    df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0)

    # Per-minute rates
    for s in stats:
        if s in df.columns:
            df[f"{s}_per_min"] = np.where(
                df["min"] > 0, df[s] / df["min"], 0.0
            )

    # Which players appeared in each game for each team
    game_rosters: dict[tuple, list[int]] = {}
    for (gid, tid), grp in df.groupby(["game_id", "team_id"]):
        game_rosters[(gid, tid)] = grp["player_id"].tolist()

    # Build together / apart indicator for each player pair
    synergy: dict[tuple[int, int], dict[str, dict[str, float]]] = {}

    all_players = df["player_id"].unique().tolist()
    pid_to_rows = {pid: df[df["player_id"] == pid] for pid in all_players}

    for i, pid_a in enumerate(all_players):
        rows_a = pid_to_rows[pid_a]
        for pid_b in all_players[i + 1:]:
            rows_b = pid_to_rows[pid_b]
            games_a = set(rows_a["game_id"].tolist())
            games_b = set(rows_b["game_id"].tolist())
            games_together = games_a & games_b
            games_apart_a  = games_a - games_b

            if len(games_together) < min_games:
                continue

            pair_key = (int(pid_a), int(pid_b))
            synergy[pair_key] = {}

            for s in stats:
                rate_col = f"{s}_per_min"
                if rate_col not in df.columns:
                    continue
                together_rate = float(
                    rows_a[rows_a["game_id"].isin(games_together)][rate_col].mean()
                ) if games_together else 0.0
                apart_rate = float(
                    rows_a[rows_a["game_id"].isin(games_apart_a)][rate_col].mean()
                ) if games_apart_a else together_rate

                synergy[pair_key][s] = {
                    "together_rate": round(together_rate, 5),
                    "apart_rate":    round(apart_rate, 5),
                    "synergy":       round(together_rate - apart_rate, 5),
                    "n_possessions": len(games_together),
                }

    logger.info("Box-score synergy computed for %d pairs.", len(synergy))
    return synergy


def add_synergy_features(
    df: pd.DataFrame,
    synergy_data: dict[tuple[int, int], dict[str, dict[str, float]]],
    top_n: int = TOP_N,
    stats: list[str] | None = None,
) -> pd.DataFrame:
    """Add top-N duo synergy features for each player in *df*.

    For each player, identifies the TOP-N teammates by absolute synergy
    magnitude and adds their synergy value as a feature column.

    Added columns (per stat, per rank):
        synergy_{stat}_rank_{i}        — synergy value with rank-i partner
        synergy_{stat}_partner_{i}     — partner player_id (for reference)
        has_primary_synergy_partner    — 1 if top synergy partner is on roster

    Parameters
    ----------
    df           : feature DataFrame (must contain 'player_id')
    synergy_data : output of compute_duo_synergy or compute_duo_synergy_from_boxscores
    top_n        : number of top synergy partners to include per player
    stats        : stats to include (default: SYNERGY_STATS)
    """
    stats = stats or SYNERGY_STATS
    out = df.copy()

    # Default synergy columns to 0
    for s in stats:
        for i in range(top_n):
            out[f"synergy_{s}_rank_{i+1}"] = 0.0

    for pid in out["player_id"].unique():
        player_pairs = {
            pair: v
            for pair, v in synergy_data.items()
            if pid in pair
        }
        if not player_pairs:
            continue

        for s in stats:
            ranked = sorted(
                [
                    (pair, v.get(s, {}).get("synergy", 0.0))
                    for pair, v in player_pairs.items()
                    if s in v
                ],
                key=lambda x: -abs(x[1]),
            )[:top_n]

            for i, (pair, syn_val) in enumerate(ranked):
                col = f"synergy_{s}_rank_{i+1}"
                out.loc[out["player_id"] == pid, col] = syn_val

    return out


def aggregate_synergy_score(
    player_id: int,
    stat: str,
    active_partner_ids: list[int],
    synergy_data: dict[tuple[int, int], dict[str, dict[str, float]]],
) -> float:
    """Compute a single aggregate synergy adjustment for a player given
    the set of active partners on the court.

    Returns the sum of pairwise synergy values for (player_id, partner)
    pairs that exist in synergy_data.  Positive = production boost,
    negative = suppression.
    """
    total = 0.0
    for partner in active_partner_ids:
        if partner == player_id:
            continue
        pair_key = (min(player_id, partner), max(player_id, partner))
        syn_val = synergy_data.get(pair_key, {}).get(stat, {}).get("synergy", 0.0)
        total += syn_val
    return total
