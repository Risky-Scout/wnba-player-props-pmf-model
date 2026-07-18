"""P1 historical market evaluation — pure, testable core.

This module holds the leakage-free, API-free logic used by the P1 historical
validation pipeline:
  * canonical event/player identity resolution (exact, roster-constrained),
  * paired Over/Under no-vig probability (same book/snapshot/line only),
  * opening/closing snapshot selection with post-tip rejection,
  * consensus construction (modal closing line; median tie-break),
  * conditional-on-settlement P(over) at an integer/half line (P0 convention),
  * recommendation grading (win/loss/push, ROI at real odds, Brier, log-loss),
  * game-date-clustered bootstrap confidence intervals,
  * the forced SUPPORTED / NOT SUPPORTED / INCONCLUSIVE verdict.

Nothing here calls the network. The downloader/workflow inject raw rows.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# WNBA team bridge for event->game_id resolution, keyed to the BDL canonical
# abbreviations actually produced by build_canonical_tables (GS/LV/LA/NY/PHX/WSH,
# plus 2026 expansion TOR/POR). Match by full name first, then unique nickname
# (robust to "Los Angeles Sparks" vs "LA Sparks" vs "Sparks"). Exact tokens only.
WNBA_TEAM_ABBR: dict[str, str] = {
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "golden state valkyries": "GS", "indiana fever": "IND",
    "las vegas aces": "LV", "los angeles sparks": "LA", "minnesota lynx": "MIN",
    "new york liberty": "NY", "phoenix mercury": "PHX", "seattle storm": "SEA",
    "washington mystics": "WSH", "toronto tempo": "TOR", "portland fire": "POR",
}

# Unique WNBA nickname (last token) -> BDL abbreviation.
WNBA_NICKNAME_ABBR: dict[str, str] = {
    "dream": "ATL", "sky": "CHI", "sun": "CON", "wings": "DAL", "valkyries": "GS",
    "fever": "IND", "aces": "LV", "sparks": "LA", "lynx": "MIN", "liberty": "NY",
    "mercury": "PHX", "storm": "SEA", "mystics": "WSH", "tempo": "TOR", "fire": "POR",
}

# Odds API market key -> model stat (direct + combo).
MARKET_TO_STAT: dict[str, str] = {
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
    "player_threes": "fg3m", "player_blocks": "blk", "player_steals": "stl",
    "player_turnovers": "turnover", "player_blocks_steals": "stocks",
    "player_points_assists": "pts_ast", "player_points_rebounds": "pts_reb",
    "player_rebounds_assists": "reb_ast", "player_points_rebounds_assists": "pts_reb_ast",
}


def norm_name(s: object) -> str:
    """Normalize a player name for EXACT matching (no fuzzy)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def team_abbr(team: object) -> str:
    t = str(team).strip().lower()
    if t in WNBA_TEAM_ABBR:
        return WNBA_TEAM_ABBR[t]
    toks = re.findall(r"[a-z]+", t)
    if toks and toks[-1] in WNBA_NICKNAME_ABBR:
        return WNBA_NICKNAME_ABBR[toks[-1]]
    up = str(team).strip().upper()
    return up if 1 <= len(up) <= 4 else ""


def _valid_american(american: object) -> bool:
    """American odds of 0 (or non-finite) are invalid/placeholder prices."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and a != 0.0


def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        return float("nan")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def american_to_implied(american: float) -> float:
    a = float(american)
    if a == 0:
        return float("nan")
    return (100.0 / (a + 100.0)) if a > 0 else (abs(a) / (abs(a) + 100.0))


def profit_at_american(american: float, won: bool) -> float:
    """Flat 1-unit stake settled profit at the offered American price."""
    a = float(american)
    if a == 0:
        return float("nan")
    if not won:
        return -1.0
    return (a / 100.0) if a > 0 else (100.0 / abs(a))


def p_over_conditional(pmf: np.ndarray, line: float) -> float:
    """P(over | non-push): Over region starts at floor(line)+1; integer-line push
    mass removed from the denominator (P0 convention)."""
    arr = np.asarray(pmf, dtype=float)
    if arr.size == 0:
        return float("nan")
    cutoff = math.floor(line) + 1
    p_over = float(arr[cutoff:].sum()) if cutoff < len(arr) else 0.0
    if float(line).is_integer():
        li = int(round(line))
        p_push = float(arr[li]) if 0 <= li < len(arr) else 0.0
    else:
        p_push = 0.0
    denom = 1.0 - p_push
    return float(min(max(p_over / denom, 0.0), 1.0)) if denom > 1e-9 else float(p_over)


def resolve_game_id(events: pd.DataFrame, home: str, away: str, game_date: str) -> str | None:
    """Resolve a provider event (home/away names + date) to a canonical game_id
    by unordered team-abbrev pair + date. Exact only."""
    if events.empty:
        return None
    pair = frozenset({team_abbr(home), team_abbr(away)}) - {""}
    if len(pair) < 2:
        return None
    e = events.copy()
    if "game_date" in e.columns:
        e = e[pd.to_datetime(e["game_date"]).dt.strftime("%Y-%m-%d") == game_date]
    for _, r in e.iterrows():
        gpair = frozenset({str(r.get("home_team_abbreviation", "")).upper(),
                          str(r.get("visitor_team_abbreviation", "")).upper()}) - {""}
        if gpair == pair:
            return str(r["game_id"])
    return None


def resolve_player_id(name: str, game_id: str, roster: pd.DataFrame,
                      aliases: dict[str, str] | None = None) -> tuple[str | None, str]:
    """Resolve a player name to a canonical player_id within the given game's
    rosters. Returns (player_id, identity_method). EXACT normalized-name match
    or an approved alias only — never fuzzy. identity_method is one of
    'exact_roster_name', 'approved_alias', 'unmatched'."""
    if roster.empty:
        return None, "unmatched"
    sub = roster[roster["game_id"].astype(str) == str(game_id)]
    if sub.empty:
        return None, "unmatched"
    nm = norm_name(name)
    lut = {norm_name(pn): str(pid) for pn, pid in zip(sub["player_name"], sub["player_id"])}
    if nm in lut:
        return lut[nm], "exact_roster_name"
    if aliases:
        alias_nm = norm_name(aliases.get(str(name), aliases.get(nm, "")))
        if alias_nm and alias_nm in lut:
            return lut[alias_nm], "approved_alias"
    return None, "unmatched"


def pair_over_under(quotes: pd.DataFrame, shin_fn=None) -> pd.DataFrame:
    """Pair Over/Under quotes at the SAME (event, book, market/stat, player, line,
    snapshot) and compute no-vig P(over). Never mixes lines or books. Rows without
    a complete pair are dropped.

    Required columns: event_id, book, stat, player_name, line, side, american_odds,
    snapshot_time. Returns one paired row per key with market_prob_over_no_vig.
    """
    if quotes.empty:
        return quotes.iloc[0:0].copy()
    key = ["event_id", "book", "stat", "player_name", "line", "snapshot_time"]
    q = quotes.copy()
    q["side"] = q["side"].astype(str).str.lower()
    over = q[q["side"].str.startswith("over")].rename(columns={"american_odds": "over_odds"})
    under = q[q["side"].str.startswith("under")].rename(columns={"american_odds": "under_odds"})
    paired = over[key + ["over_odds"] + [c for c in ["book_last_update", "commence_time",
              "home_team", "away_team", "game_date", "market_key"] if c in over.columns]].merge(
        under[key + ["under_odds"]], on=key, how="inner")
    if paired.empty:
        return paired
    if shin_fn is None:
        # No-vig by simple implied-prob normalization at the SAME book/line.
        io = paired["over_odds"].map(american_to_implied)
        iu = paired["under_odds"].map(american_to_implied)
        tot = (io + iu).clip(lower=1e-9)
        paired["market_prob_over_no_vig"] = io / tot
    else:
        res = paired.apply(lambda r: shin_fn(r["over_odds"], r["under_odds"]),
                           axis=1, result_type="expand")
        paired["market_prob_over_no_vig"] = res[0]
    return paired


def select_open_close(paired: pd.DataFrame) -> pd.DataFrame:
    """Tag opening (earliest pre-tip) and closing (latest strictly pre-tip) paired
    quotes per (game_id, player_id, stat, book, line). Rejects post-tip snapshots.

    Required: game_id, player_id, stat, book, line, snapshot_time, commence_time.
    """
    if paired.empty:
        return paired.assign(is_opening=[], is_closing=[])
    p = paired.copy()
    p["_snap"] = pd.to_datetime(p["snapshot_time"], utc=True, errors="coerce")
    p["_tip"] = pd.to_datetime(p["commence_time"], utc=True, errors="coerce")
    # Reject snapshots at or after tip (no post-tip information).
    p = p[p["_snap"] < p["_tip"]].copy()
    if p.empty:
        return p.assign(is_opening=[], is_closing=[])
    grp = ["game_id", "player_id", "stat", "book", "line"]
    p["is_opening"] = p["_snap"] == p.groupby(grp)["_snap"].transform("min")
    p["is_closing"] = p["_snap"] == p.groupby(grp)["_snap"].transform("max")
    return p.drop(columns=["_snap", "_tip"])


def build_consensus(closing_paired: pd.DataFrame) -> pd.DataFrame:
    """One consensus observation per (game_id, player_id, stat): the modal closing
    line across books (median line breaks a tie); market no-vig aggregated only
    over books offering the selected line."""
    if closing_paired.empty:
        return closing_paired.iloc[0:0].copy()
    cl = closing_paired[closing_paired.get("is_closing", True) == True].copy()  # noqa: E712
    out = []
    for (gid, pid, stat), g in cl.groupby(["game_id", "player_id", "stat"]):
        lines = g["line"].astype(float)
        counts = lines.value_counts()
        top = sorted(float(x) for x in counts[counts == counts.max()].index.tolist())
        if len(top) == 1:
            sel_line = top[0]
        else:
            # Median tie-break, snapped to an ACTUAL modal line (the median of two
            # half-point lines is not itself a valid line). Deterministic: nearest
            # modal line to the median, lowest on a distance tie.
            med = float(np.median(top))
            sel_line = min(top, key=lambda x: (abs(x - med), x))
        at_line = g[g["line"].astype(float) == sel_line]
        out.append({
            "game_id": str(gid), "player_id": str(pid), "stat": str(stat),
            "line": sel_line,
            "market_prob_over_no_vig": float(at_line["market_prob_over_no_vig"].mean()),
            "over_odds": float(at_line["over_odds"].median()),
            "under_odds": float(at_line["under_odds"].median()),
            "n_books": int(at_line["book"].nunique()),
            "commence_time": at_line["commence_time"].iloc[0] if "commence_time" in at_line else None,
        })
    return pd.DataFrame(out)


@dataclass
class GradeResult:
    n: int = 0
    under_pct: float = float("nan")
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    hit_rate: float = float("nan")
    avg_american_odds: float = float("nan")
    breakeven_rate: float = float("nan")
    roi: float = float("nan")
    profit_units: float = float("nan")
    avg_model_prob: float = float("nan")
    realized_rate: float = float("nan")
    brier: float = float("nan")
    logloss: float = float("nan")
    market_brier: float = float("nan")
    market_logloss: float = float("nan")
    model_minus_market_logloss: float = float("nan")
    roi_ci95: tuple[float, float] = (float("nan"), float("nan"))
    extra: dict[str, Any] = field(default_factory=dict)


def _logloss(p: float, y: int) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def grade(df: pd.DataFrame, cluster_col: str = "game_date", n_boot: int = 2000,
          seed: int = 42) -> GradeResult:
    """Grade a set of SELECTED recommendations.

    Required columns: side ('over'/'under'), price_american (offered price for the
    selected side), won (bool|nan; nan=push), model_prob_side, market_prob_side,
    plus cluster_col for clustered resampling. Pushes have won=nan.
    """
    r = GradeResult(n=int(len(df)))
    if df.empty:
        return r
    r.under_pct = float((df["side"] == "under").mean() * 100)
    push = df["won"].isna()
    r.pushes = int(push.sum())
    settled = df[~push].copy()
    r.wins = int((settled["won"] == True).sum())  # noqa: E712
    r.losses = int((settled["won"] == False).sum())  # noqa: E712
    if len(settled):
        r.hit_rate = float((settled["won"] == True).mean())  # noqa: E712
        r.avg_american_odds = float(settled["price_american"].mean())
        r.breakeven_rate = float(settled["price_american"].map(american_to_implied).mean())
        profits = [profit_at_american(a, bool(w)) for a, w in
                   zip(settled["price_american"], settled["won"])]
        r.profit_units = float(np.sum(profits))
        r.roi = float(np.mean(profits))
        r.avg_model_prob = float(settled["model_prob_side"].mean())
        r.realized_rate = float((settled["won"] == True).mean())  # noqa: E712
        y = (settled["won"] == True).astype(int).values  # noqa: E712
        mp = settled["model_prob_side"].astype(float).clip(1e-9, 1 - 1e-9).values
        kp = settled["market_prob_side"].astype(float).clip(1e-9, 1 - 1e-9).values
        r.brier = float(np.mean((mp - y) ** 2))
        r.market_brier = float(np.mean((kp - y) ** 2))
        r.logloss = float(np.mean([_logloss(p, int(t)) for p, t in zip(mp, y)]))
        r.market_logloss = float(np.mean([_logloss(p, int(t)) for p, t in zip(kp, y)]))
        r.model_minus_market_logloss = r.logloss - r.market_logloss
        r.roi_ci95 = clustered_bootstrap_roi_ci(settled, cluster_col, n_boot, seed)
    return r


def clustered_bootstrap_roi_ci(settled: pd.DataFrame, cluster_col: str,
                               n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% CI for ROI, resampling whole CLUSTERS (games/dates) so multiple props
    from one game are not treated as independent."""
    if settled.empty or cluster_col not in settled.columns:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    clusters = settled[cluster_col].astype(str).values
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    profits = np.array([profit_at_american(a, bool(w)) for a, w in
                        zip(settled["price_american"], settled["won"])])
    by_cluster = {c: profits[clusters == c] for c in uniq}
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by_cluster[c] for c in pick])
        boot.append(vals.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))


def assert_no_lookahead(oof: pd.DataFrame) -> None:
    """Raise if any OOF fold's training window does not strictly precede the
    evaluated game (temporal leakage)."""
    if "fold_train_end_date" not in oof.columns or "game_date" not in oof.columns:
        return
    end = pd.to_datetime(oof["fold_train_end_date"], errors="coerce")
    gd = pd.to_datetime(oof["game_date"], errors="coerce")
    bad = end >= gd
    if bool(bad.fillna(False).any()):
        raise ValueError(
            f"{int(bad.fillna(False).sum())} OOF rows have fold_train_end_date >= game_date (lookahead)."
        )


def forced_verdict(under: GradeResult, n_game_clusters: int | None = None,
                   min_clusters: int = 25) -> str:
    """SUPPORTED / NOT SUPPORTED / INCONCLUSIVE for the Under lean, using the
    game-clustered ROI 95% interval (price-adjusted, not raw hit rate).

    Returns INCONCLUSIVE when coverage is too thin to support any conclusion:
    fewer than 50 Under recommendations, a NaN interval, OR — critically — too
    few independent game clusters (low event coverage makes multiple props from a
    handful of games masquerade as a large, significant sample)."""
    lo, hi = under.roi_ci95
    if under.n < 50 or any(map(lambda x: x != x, [lo, hi])):  # nan check / small n
        return "INCONCLUSIVE"
    if n_game_clusters is not None and n_game_clusters < min_clusters:
        return "INCONCLUSIVE"  # coverage does not support a conclusion
    if lo > 0:
        return "SUPPORTED"
    if hi < 0:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"
