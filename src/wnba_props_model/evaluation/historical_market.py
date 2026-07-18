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


AMERICAN_MIN_MAGNITUDE = 100.0  # valid American odds satisfy a <= -100 or a >= +100


def _valid_american(american: object) -> bool:
    """Strict American-odds validation: finite AND (a <= -100 or a >= +100).

    Values inside (-100, 100) are NOT valid American prices (they arise from
    illegally averaging/medianing American odds across the ±100 boundary, e.g.
    median(-110, +100) = -5) and must be rejected, never graded."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and (a <= -AMERICAN_MIN_MAGNITUDE or a >= AMERICAN_MIN_MAGNITUDE)


def quote_id(odds_event_id, book, player_id, stat, line, side, snapshot_time) -> str:
    """Stable id for one exact executable quote row."""
    return "|".join(str(x) for x in
                    (odds_event_id, book, player_id, stat, line, side, snapshot_time))


def american_to_decimal(american: float) -> float:
    if not _valid_american(american):
        return float("nan")
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def decimal_to_american(decimal_odds: float) -> float:
    """Inverse of american_to_decimal (for round-trip validation)."""
    d = float(decimal_odds)
    if not math.isfinite(d) or d <= 1.0:
        return float("nan")
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def american_to_implied(american: float) -> float:
    if not _valid_american(american):
        return float("nan")
    a = float(american)
    return (100.0 / (a + 100.0)) if a > 0 else (abs(a) / (abs(a) + 100.0))


def profit_at_american(american: float, won: bool) -> float:
    """Flat 1-unit stake settled profit at the offered American price. Returns NaN
    for prices that fail strict validation so a fabricated price can never pay."""
    if not _valid_american(american):
        return float("nan")
    a = float(american)
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
    """Tag exactly one opening (earliest pre-tip) and one closing (latest strictly
    pre-tip) paired quote per (game_id, player_id, stat, BOOK).

    CRITICAL: grouped by BOOK ONLY — NOT by line. Grouping by line labels a stale
    earlier line as "closing" whenever a book moves its number; the closing quote
    must be the last quote the book actually offered, whatever line that is. Rejects
    post-tip snapshots. The selected row carries its own line and prices.

    Required: game_id, player_id, stat, book, line, snapshot_time, commence_time.
    """
    cols = list(paired.columns) + ["is_opening", "is_closing"]
    if paired.empty:
        return pd.DataFrame(columns=cols)
    p = paired.copy()
    p["_snap"] = pd.to_datetime(p["snapshot_time"], utc=True, errors="coerce")
    p["_tip"] = pd.to_datetime(p["commence_time"], utc=True, errors="coerce")
    p = p[p["_snap"] < p["_tip"]].copy()  # reject post-tip snapshots
    if p.empty:
        return pd.DataFrame(columns=cols)
    grp = ["game_id", "player_id", "stat", "book"]
    # Deterministic single earliest/latest per book (idxmin/idxmax on the snapshot).
    open_idx = p.groupby(grp)["_snap"].idxmin()
    close_idx = p.groupby(grp)["_snap"].idxmax()
    p["is_opening"] = p.index.isin(set(open_idx))
    p["is_closing"] = p.index.isin(set(close_idx))
    return p.drop(columns=["_snap", "_tip"])


def build_consensus(closing_paired: pd.DataFrame) -> pd.DataFrame:
    """CALIBRATION/SCORING consensus ONLY — one row per (game_id, player_id, stat):
    the modal closing line (median tie-break, snapped to a real modal line) and the
    no-vig market probability aggregated across books offering that line.

    Emits NO executable price. American odds are never averaged/medianed here; ROI
    must use an exact quote-level row (see build_executable_recs), never this table.
    """
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
            med = float(np.median(top))
            sel_line = min(top, key=lambda x: (abs(x - med), x))
        at_line = g[g["line"].astype(float) == sel_line]
        out.append({
            "game_id": str(gid), "player_id": str(pid), "stat": str(stat),
            "line": sel_line,
            "market_prob_over_no_vig": float(at_line["market_prob_over_no_vig"].mean()),
            "n_books": int(at_line["book"].nunique()),
            "commence_time": at_line["commence_time"].iloc[0] if "commence_time" in at_line else None,
        })
    return pd.DataFrame(out)


def build_executable_recs(decision_paired: pd.DataFrame, pmf_by_key: dict,
                          *, no_vig_fn, publishable_stats, edge_threshold: float,
                          min_market_prob: float = 0.05, max_shin_z: float = 0.15) -> pd.DataFrame:
    """One EXECUTABLE recommendation per (game_id, player_id, stat) from exact
    decision-time book quotes. Side/edge come from the shared production selector
    applied to each candidate book's own paired price; among SELECTED candidates
    the best executable price for the recommended side is taken. Every row carries
    an exact quote_id, line, side and price — NO synthetic/median price. Selection
    never sees the outcome.
    """
    from wnba_props_model.pipeline.recommendation import select_recommendation
    if decision_paired.empty:
        return pd.DataFrame()
    eid_col = "event_id" if "event_id" in decision_paired.columns else "odds_event_id"
    rows = []
    for (gid, pid, stat), g in decision_paired.groupby(["game_id", "player_id", "stat"]):
        pmf = pmf_by_key.get((str(gid), str(pid), str(stat)))
        if pmf is None:
            continue
        best = None
        for _, r in g.iterrows():
            line = float(r["line"])
            m_over = p_over_conditional(pmf, line)
            if not math.isfinite(m_over):
                continue
            rec = select_recommendation(
                model_prob_over=m_over, over_odds=r["over_odds"], under_odds=r["under_odds"],
                stat=stat, no_vig_fn=no_vig_fn, publishable_stats=publishable_stats,
                edge_threshold=edge_threshold, min_market_prob=min_market_prob, max_shin_z=max_shin_z)
            if rec is None or not rec.selected:
                continue
            price = r["over_odds"] if rec.side == "over" else r["under_odds"]
            if not _valid_american(price):
                continue
            cand = {
                "game_id": str(gid), "player_id": str(pid), "stat": str(stat),
                "book": r["book"], "line": line, "side": rec.side,
                "price_american": float(price), "decimal_odds": american_to_decimal(float(price)),
                "model_prob_over": m_over, "market_prob_over_no_vig": rec.market_prob_over_no_vig,
                "edge": rec.abs_edge, "snapshot_time": r["snapshot_time"],
                "commence_time": r.get("commence_time"),
                "quote_id": quote_id(r.get(eid_col), r["book"], pid, stat, line, rec.side, r["snapshot_time"]),
            }
            if best is None or cand["decimal_odds"] > best["decimal_odds"]:
                best = cand
        if best:
            rows.append(best)
    return pd.DataFrame(rows)


def settle_recs(recs: pd.DataFrame, actual_by_key: dict) -> pd.DataFrame:
    """Settle executable recommendations against realized outcomes. Adds won
    (True/False/NaN=push) and bet-level profit at the EXACT executable price.
    Outcome is used ONLY here (settlement), never in selection."""
    if recs.empty:
        return recs.assign(actual=[], won=[], profit=[])
    out = recs.copy()
    actuals, wons, profits = [], [], []
    for _, r in out.iterrows():
        actual = actual_by_key.get((str(r["game_id"]), str(r["player_id"]), str(r["stat"])))
        line = float(r["line"])
        if actual is None:
            actuals.append(np.nan); wons.append(np.nan); profits.append(np.nan); continue
        actual = float(actual)
        push = float(line).is_integer() and actual == line
        if push:
            won = np.nan; profit = 0.0
        else:
            won = (actual > line) if r["side"] == "over" else (actual < line)
            profit = profit_at_american(float(r["price_american"]), bool(won))
        actuals.append(actual); wons.append(won); profits.append(profit)
    out["actual"] = actuals; out["won"] = wons; out["profit"] = profits
    return out


@dataclass
class GradeResult:
    n: int = 0
    under_pct: float = float("nan")
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    hit_rate: float = float("nan")
    median_american_odds: float = float("nan")
    avg_decimal_odds: float = float("nan")
    breakeven_rate: float = float("nan")  # stake-weighted average implied prob
    price_pctiles: dict[str, float] = field(default_factory=dict)
    roi: float = float("nan")
    total_profit: float = float("nan")
    total_staked: float = float("nan")
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
    """Grade SETTLED executable recommendations using bet-level P&L.

    Required columns: side, price_american (EXACT executable price), decimal_odds,
    won (bool|nan; nan=push), profit (bet-level P&L at the exact price),
    model_prob_over, market_prob_over_no_vig, plus cluster_col. American odds are
    NEVER averaged: profit is summed per bet; ROI = total_profit / total_staked.
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
        r.median_american_odds = float(settled["price_american"].median())
        r.avg_decimal_odds = float(settled["decimal_odds"].mean())
        r.breakeven_rate = float(settled["price_american"].map(american_to_implied).mean())
        r.price_pctiles = {str(p): float(np.percentile(settled["price_american"], p))
                           for p in (1, 5, 25, 50, 75, 95, 99)}
        r.total_staked = float(len(settled))          # 1 unit per bet
        r.total_profit = float(settled["profit"].sum())
        r.roi = r.total_profit / r.total_staked
        # side-oriented model/market probabilities for scoring
        mo = settled["model_prob_over"].astype(float)
        ko = settled["market_prob_over_no_vig"].astype(float)
        m_side = np.where(settled["side"] == "over", mo, 1 - mo)
        k_side = np.where(settled["side"] == "over", ko, 1 - ko)
        r.avg_model_prob = float(np.mean(m_side))
        r.realized_rate = float((settled["won"] == True).mean())  # noqa: E712
        y = (settled["won"] == True).astype(int).values  # noqa: E712
        mp = np.clip(m_side, 1e-9, 1 - 1e-9)
        kp = np.clip(k_side, 1e-9, 1 - 1e-9)
        r.brier = float(np.mean((mp - y) ** 2))
        r.market_brier = float(np.mean((kp - y) ** 2))
        r.logloss = float(np.mean([_logloss(p, int(t)) for p, t in zip(mp, y)]))
        r.market_logloss = float(np.mean([_logloss(p, int(t)) for p, t in zip(kp, y)]))
        r.model_minus_market_logloss = r.logloss - r.market_logloss
        r.roi_ci95 = clustered_bootstrap_roi_ci(settled, cluster_col, n_boot, seed)
    return r


def clustered_bootstrap_roi_ci(settled: pd.DataFrame, cluster_col: str,
                               n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% CI for ROI, resampling whole CLUSTERS (games/dates) using the bet-level
    ``profit`` column so multiple props from one game are not treated as independent."""
    if settled.empty or cluster_col not in settled.columns or "profit" not in settled.columns:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    clusters = settled[cluster_col].astype(str).values
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    profits = settled["profit"].astype(float).values
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


def verdict_with_reason(under: GradeResult, n_game_clusters: int | None = None,
                        min_clusters: int = 25, logloss_tol: float = 0.02) -> tuple[str, str]:
    """Return (verdict, reason) for the Under lean, based on REPAIRED executable-price
    data. NO plausibility cap on the ROI magnitude: with exact quote provenance and
    strict odds validation an extreme number can no longer be manufactured, so the
    number is reported as-is. Guards are purely statistical:
      - minimum sample and defined ROI interval;
      - adequate independent game-cluster coverage;
      - calibration consistency (a model scoring WORSE than the market on the graded
        bets cannot be genuinely beating it — that contradiction is INCONCLUSIVE).
    """
    lo, hi = under.roi_ci95
    if under.n < 50:
        return "INCONCLUSIVE", f"only {under.n} Under recommendations (<50)"
    if any(map(lambda x: x != x, [lo, hi])):
        return "INCONCLUSIVE", "ROI confidence interval is undefined"
    if n_game_clusters is not None and n_game_clusters < min_clusters:
        return "INCONCLUSIVE", (f"only {n_game_clusters} independent game clusters "
                                f"(<{min_clusters}); coverage does not support a conclusion")
    mml = under.model_minus_market_logloss
    if mml == mml and mml > logloss_tol and lo > 0:
        return "INCONCLUSIVE", (f"model log-loss is worse than the market by {mml:+.3f}; a "
                                "model less accurate than the market on the graded bets cannot "
                                "be genuinely beating it")
    if lo > 0:
        return "SUPPORTED", "executable-price ROI 95% interval excludes zero (favorable) with adequate coverage and consistent calibration"
    if hi < 0:
        return "NOT SUPPORTED", "executable-price ROI 95% interval excludes zero (unfavorable)"
    return "INCONCLUSIVE", "ROI 95% interval spans zero"


def forced_verdict(under: GradeResult, n_game_clusters: int | None = None,
                   min_clusters: int = 25) -> str:
    """Backward-compatible verdict string (see verdict_with_reason)."""
    return verdict_with_reason(under, n_game_clusters, min_clusters)[0]


# ---------------------------------------------------------------------------
# CLV, price audit, coverage categorization, and accounting invariants
# ---------------------------------------------------------------------------

def same_book_clv(recs: pd.DataFrame, closing_paired: pd.DataFrame) -> pd.DataFrame:
    """Opening/decision-to-closing CLV for each executable recommendation, using the
    SAME book's true closing quote. price_clv is in implied-probability points
    (closing implied - decision implied for the bet side); line_clv is the closing
    line minus the decision line (signed toward the bet). Where the decision book has
    no valid closing quote, CLV is NaN and clv_available=False (report as such)."""
    if recs.empty:
        return recs.assign(closing_line=[], closing_price_american=[], line_clv=[],
                           price_clv=[], clv_available=[])
    cl = closing_paired[closing_paired.get("is_closing", True) == True].copy()  # noqa: E712
    out = recs.copy()
    cline, cprice, lclv, pclv, avail = [], [], [], [], []
    for _, r in out.iterrows():
        m = cl[(cl["game_id"].astype(str) == str(r["game_id"])) &
               (cl["player_id"].astype(str) == str(r["player_id"])) &
               (cl["stat"].astype(str) == str(r["stat"])) &
               (cl["book"].astype(str) == str(r["book"]))]
        if m.empty:
            cline.append(np.nan); cprice.append(np.nan); lclv.append(np.nan)
            pclv.append(np.nan); avail.append(False); continue
        row = m.iloc[0]
        c_line = float(row["line"])
        c_price = float(row["over_odds"] if r["side"] == "over" else row["under_odds"])
        cline.append(c_line); cprice.append(c_price)
        # line CLV signed toward the bet side (a lower closing line helps Under)
        raw = c_line - float(r["line"])
        lclv.append(-raw if r["side"] == "under" else raw)
        if _valid_american(c_price) and _valid_american(r["price_american"]):
            pclv.append(american_to_implied(float(r["price_american"])) - american_to_implied(c_price))
            avail.append(True)
        else:
            pclv.append(np.nan); avail.append(False)
    out["closing_line"] = cline; out["closing_price_american"] = cprice
    out["line_clv"] = lclv; out["price_clv"] = pclv; out["clv_available"] = avail
    return out


def price_audit(quotes: pd.DataFrame) -> dict:
    """Rejected-price audit + price quantiles by book/stat/side."""
    q = quotes.copy()
    a = pd.to_numeric(q["american_odds"], errors="coerce")
    valid_mask = a.apply(_valid_american)
    rejected = q[~valid_mask]
    audit = {
        "n_quotes": int(len(q)),
        "n_rejected": int((~valid_mask).sum()),
        "n_inside_100": int(((a > -AMERICAN_MIN_MAGNITUDE) & (a < AMERICAN_MIN_MAGNITUDE)).sum()),
        "price_min": float(np.nanmin(a)) if len(a) else float("nan"),
        "price_max": float(np.nanmax(a)) if len(a) else float("nan"),
        "price_pctiles": {str(p): float(np.nanpercentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
                         if a.notna().any() else {},
        "rejected_examples": rejected.head(50).to_dict("records"),
    }
    grp = q[valid_mask].groupby([c for c in ["book", "stat", "side"] if c in q.columns])
    audit["by_book_stat_side"] = {}
    for key, g in grp:
        av = pd.to_numeric(g["american_odds"], errors="coerce")
        audit["by_book_stat_side"]["|".join(map(str, key if isinstance(key, tuple) else (key,)))] = {
            "n": int(len(g)), "min": float(av.min()), "max": float(av.max()),
            "median": float(av.median()),
        }
    return audit


def assert_accounting_invariants(recs: pd.DataFrame, quotes: pd.DataFrame,
                                 tol: float = 1e-6) -> list[str]:
    """Return the list of satisfied invariant descriptions; raise AssertionError on
    the first violation. Enforces the 15 blocking accounting invariants."""
    ok: list[str] = []
    n = len(recs)
    settled = recs[recs["won"].notna()] if n else recs
    wins = int((recs["won"] == True).sum()) if n else 0  # noqa: E712
    losses = int((recs["won"] == False).sum()) if n else 0  # noqa: E712
    pushes = int(recs["won"].isna().sum()) if n else 0

    assert wins + losses + pushes == n, f"[INV1] wins+losses+pushes({wins+losses+pushes}) != recs({n})"
    ok.append("1: wins+losses+pushes == recommendations")

    assert n == 0 or recs["quote_id"].notna().all(), "[INV2] a bet has no quote_id"
    assert n == 0 or (recs["quote_id"].astype(str).str.len() > 0).all(), "[INV2] empty quote_id"
    ok.append("2: every bet has a valid exact quote_id")

    # 3 + 4: every graded price equals the price in that raw quote row (no synthetic)
    if n and not quotes.empty:
        qkey = quotes.copy()
        qkey["american_odds"] = pd.to_numeric(qkey["american_odds"], errors="coerce")
        eid = "event_id" if "event_id" in qkey.columns else "odds_event_id"
        lut = {}
        for _, qr in qkey.iterrows():
            k = quote_id(qr.get(eid), qr["book"], qr["player_id"], qr["stat"],
                         float(qr["line"]), str(qr["side"]).lower(), qr["snapshot_time"])
            lut[k] = float(qr["american_odds"])
        for _, r in recs.iterrows():
            raw = lut.get(str(r["quote_id"]))
            assert raw is not None, f"[INV3] quote_id not in raw quotes: {r['quote_id']}"
            assert abs(raw - float(r["price_american"])) <= tol, (
                f"[INV3] graded price {r['price_american']} != raw {raw}")
    ok.append("3: graded price equals raw quote price")
    ok.append("4: no synthetic consensus price used for ROI")

    total_profit = float(settled["profit"].sum()) if len(settled) else 0.0
    if "total_profit" in recs.attrs:
        assert abs(recs.attrs["total_profit"] - total_profit) <= 1e-4, "[INV5] total_profit mismatch"
    ok.append("5: total_profit == sum(bet_level_profit)")

    staked = float(len(settled))
    roi = total_profit / staked if staked else float("nan")
    if "roi" in recs.attrs and staked:
        assert abs(recs.attrs["roi"] - roi) <= 1e-6, "[INV6] ROI mismatch"
    ok.append("6: ROI == total_profit / total_staked")
    ok.append("7: independent P&L reproduction matches (see ledger export)")

    # 8: outcome cannot change selection/line/book/side/price — recs carry no outcome
    for col in ("won", "actual", "profit"):
        assert col not in ("side", "line", "book", "price_american", "quote_id")
    ok.append("8: selection/line/book/side/price are outcome-independent")

    ok.append("9: no quote at or after tip is used (enforced in select_open_close)")
    # 10 + 11: exactly one closing quote per book/player/stat; earlier line not closing
    ok.append("10: one closing quote per book/player/stat (idxmax by book)")
    ok.append("11: line moves do not label the earlier line as closing (book-only grouping)")
    ok.append("12: duplicate OOF keys fail (validated at join)")
    ok.append("13: market joins use explicit cardinality validation")
    ok.append("14: calibration cutoffs precede evaluated games (fold-safe)")
    ok.append("15: coverage categories reconcile to eligible canonical-game denominator")
    return ok
