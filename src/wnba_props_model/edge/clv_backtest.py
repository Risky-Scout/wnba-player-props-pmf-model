"""CLV (Closing Line Value) backtest for the soft-book / MARKET_DISLOCATION scan.

The soft-book scan (``soft_book_scan.scan_soft_book_edges``) flags individual books whose
posted player-prop price beats a leave-one-book-out (LOBO) no-vig consensus. Those flags are
*theoretical* EV. This module answers the only question that determines whether they are
worth acting on: **do the flagged bets beat the CLOSE?**

Closing Line Value is the gold-standard proxy that a betting strategy is +EV. We build it
honestly and fail closed:

1. **Replay the exact same scan at the decision snapshot.** For every prop we take the
   ``decision`` snapshot (falling back to ``open`` if that prop has no decision snapshot) and
   run the unmodified ``scan_soft_book_edges`` with the same rigor the acceptance gate
   requires: exact identity match, atomic line matching (same player/market/side/line only;
   standard never mixed with ``*_alternate``), LOBO consensus with the candidate book
   excluded, Shin no-vig pairing of the same book's over+under (fail closed if a side is
   missing), and a minimum-books consensus-quality floor.

2. **Compute CLV vs the CLOSE for each flagged +EV candidate**, using the *same exact prop*
   (event, market_key, player, line, side, book):
     * ``price_clv``  = closing no-vig **consensus** P(side) − the candidate book's own
       decision no-vig P(side). "Did we get a better number than where the market closed?"
     * ``same_book_clv`` = the same book's closing no-vig P(side) − its decision no-vig
       P(side). Same book only; fails closed (None) if the book is missing a closing side.
   Both are reported in probability terms and in percentage points ("cents"); we also record
   whether each bet "beat the close" (price_clv > 0).

3. **Aggregate** by market, by book, by EV bucket and by market×EV-bucket. Each segment
   reports N, mean, median, %-beating-close and a **date-cluster bootstrap 95% CI**
   (resample the game_date clusters with replacement) plus a significance verdict (CI
   excludes 0). Only segments with N >= ``min_segment_n`` and >= 2 date clusters are eligible.

The persisted validation table (see ``build_validation_table``) is what drives
``actionable`` on the edge board and in the acceptance gate — fail closed: a board row is
actionable only if its segment showed positive mean CLV with a bootstrap 95% CI that
excludes zero on a sufficiently large sample.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wnba_props_model.edge.soft_book_scan import (
    DEFAULT_EV_THRESHOLD,
    DEFAULT_MIN_CONSENSUS_BOOKS,
    american_to_decimal_profit,
    scan_soft_book_edges,
)
from wnba_props_model.models.market import shin_no_vig_two_way

log = logging.getLogger(__name__)

PRIMARY_METRIC = "price_clv"          # actionability is decided on price CLV (vs closing consensus)
DEFAULT_MIN_SEGMENT_N = 50            # min realistic sample per segment (configurable)
DEFAULT_BOOTSTRAP_ITERS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260728
DEFAULT_CI_ALPHA = 0.05              # 95% CI
MIN_DATE_CLUSTERS = 2               # a single date cluster gives no between-cluster variance

# EV buckets (fraction). Half-open [lo, hi).
EV_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.025, 0.05, "2.5-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, math.inf, ">10%"),
)

# Atomic prop key (segregates standard vs alternate via market_key) and the per-book key.
_PROP_KEY = ["event_id", "market_key", "player_name", "line"]
_BOOK_KEY = ["event_id", "market_key", "player_name", "line", "book"]


# --------------------------------------------------------------------------- #
# Loading / snapshot selection
# --------------------------------------------------------------------------- #
def load_quotes(path: str) -> pd.DataFrame:
    """Load the p1 quotes parquet and normalize it to the scan's expected schema."""
    df = pd.read_parquet(path)
    return normalize_quotes(df)


def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw p1 quotes to the columns ``scan_soft_book_edges`` expects.

    ``odds_event_id`` -> ``event_id`` and ``snapshot_time`` -> ``collected_utc`` (the
    ingestion timestamp). Side is lower-cased. Rows missing a snapshot_label are dropped.
    """
    df = df.copy()
    if "event_id" not in df.columns and "odds_event_id" in df.columns:
        df = df.rename(columns={"odds_event_id": "event_id"})
    if "collected_utc" not in df.columns:
        df["collected_utc"] = df.get("snapshot_time")
    df["side"] = df["side"].astype(str).str.lower().str.strip()
    if "snapshot_label" in df.columns:
        df = df[df["snapshot_label"].notna()]
    return df.reset_index(drop=True)


def build_replay_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the decision-time replay frame: the ``decision`` snapshot per prop, falling
    back to ``open`` for any (event, market, player, side, line, book) with no decision row.

    This is what the scan is replayed on: it is the latest pre-close price we could have bet.
    """
    dec = df[df["snapshot_label"] == "decision"]
    opn = df[df["snapshot_label"] == "open"]
    if len(dec) == 0:
        return opn.reset_index(drop=True)
    dec_keys = set(map(tuple, dec[_BOOK_KEY].astype(object).values.tolist()))
    if len(opn):
        opn_mask = [tuple(v) not in dec_keys for v in opn[_BOOK_KEY].astype(object).values.tolist()]
        opn_fb = opn[pd.Series(opn_mask, index=opn.index)]
    else:
        opn_fb = opn
    return pd.concat([dec, opn_fb], ignore_index=True)


def closing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the ``close`` snapshot rows (the closing reference)."""
    return df[df["snapshot_label"] == "close"].reset_index(drop=True)


def build_identity_index(df: pd.DataFrame) -> dict:
    """Build an exact name -> canonical player_id index from the quotes themselves.

    The p1 quotes were identity-resolved upstream (``identity_method == exact_roster_name``),
    so this reproduces the scan's fail-closed exact-identity requirement without a fuzzy step.
    """
    from wnba_props_model.edge.prop_identity import build_name_index  # noqa: PLC0415

    pairs = (
        df[["player_name", "player_id"]]
        .dropna()
        .drop_duplicates()
        .to_dict("records")
    )
    return build_name_index(pairs)


# --------------------------------------------------------------------------- #
# No-vig tables (reuse the scan's Shin de-vig + LOBO/consensus logic)
# --------------------------------------------------------------------------- #
def _latest(rows: pd.DataFrame) -> pd.Series:
    if "collected_utc" in rows.columns:
        return rows.sort_values("collected_utc").iloc[-1]
    return rows.iloc[-1]


def build_novig_tables(snapshot: pd.DataFrame) -> tuple[dict, dict]:
    """Compute per-book and consensus no-vig probabilities for a snapshot.

    Returns ``(per_book, consensus)``:
      * ``per_book[(event, market, player, line, book)]`` =
        ``{fair_over, fair_under, over_odds, under_odds}`` for every book posting BOTH a
        valid over and under at that line (Shin de-vig; fail closed if a side is missing).
      * ``consensus[(event, market, player, line)]`` =
        ``{p_over, n_books, books}`` where ``p_over`` is the MEDIAN of the per-book fair
        P(over) across all books posting a two-sided price (the market's fair number).

    This mirrors ``soft_book_scan._per_book_two_sided`` (same Shin call, same fail-closed
    behaviour) so the CLV reference is computed identically to the scan it validates.
    """
    snap = snapshot.copy()
    snap["side"] = snap["side"].astype(str).str.lower().str.strip()
    snap = snap[snap["side"].isin(["over", "under"])]
    per_book: dict = {}
    consensus: dict = {}
    for (e, m, p, ln), g in snap.groupby(_PROP_KEY, sort=False):
        fair_over_by_book: dict[str, float] = {}
        for book, bg in g.groupby("book", sort=False):
            over_rows = bg[bg["side"] == "over"]
            under_rows = bg[bg["side"] == "under"]
            if over_rows.empty or under_rows.empty:
                continue  # fail closed: never de-vig a one-sided quote
            oo = _latest(over_rows).get("american_odds")
            uu = _latest(under_rows).get("american_odds")
            if american_to_decimal_profit(oo) is None or american_to_decimal_profit(uu) is None:
                continue
            fo, fu = shin_no_vig_two_way(oo, uu)
            if fo is None or fu is None:
                continue
            per_book[(e, m, p, ln, str(book))] = {
                "fair_over": float(fo),
                "fair_under": float(fu),
                "over_odds": float(oo),
                "under_odds": float(uu),
            }
            fair_over_by_book[str(book)] = float(fo)
        if fair_over_by_book:
            consensus[(e, m, p, ln)] = {
                "p_over": float(np.median(list(fair_over_by_book.values()))),
                "n_books": len(fair_over_by_book),
                "books": sorted(fair_over_by_book.keys()),
            }
    return per_book, consensus


# --------------------------------------------------------------------------- #
# Replay + CLV
# --------------------------------------------------------------------------- #
def replay_decision_scan(
    replay_df: pd.DataFrame,
    identity_index: dict,
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    min_consensus_books: int = DEFAULT_MIN_CONSENSUS_BOOKS,
) -> pd.DataFrame:
    """Run ``scan_soft_book_edges`` per game_date on the decision-time replay frame and
    return the qualified (flagged +EV) candidates, tagged with ``game_date``.

    ``drop_stale`` is disabled and the live age gate is off: this is a historical replay, so
    commence-time staleness and provider-quote-age gates (which are about *live* freshness)
    do not apply. Every other rigor guard is unchanged from the production scan.
    """
    out: list[pd.DataFrame] = []
    group_col = "game_date" if "game_date" in replay_df.columns else None
    groups = replay_df.groupby(group_col, sort=True) if group_col else [(None, replay_df)]
    for gd, sub in groups:
        board = scan_soft_book_edges(
            sub,
            ev_threshold=ev_threshold,
            min_consensus_books=min_consensus_books,
            identity_index=identity_index,
            require_identity=True,
            drop_stale=False,
            max_quote_age_seconds=None,
        )
        if len(board):
            board = board.copy()
            board["game_date"] = gd
            out.append(board)
    if not out:
        return pd.DataFrame()
    board = pd.concat(out, ignore_index=True)
    return board[board["qualified"]].reset_index(drop=True)


def ev_bucket_label(ev_frac: float | None) -> str | None:
    if ev_frac is None:
        return None
    try:
        e = float(ev_frac)
    except (TypeError, ValueError):
        return None
    for lo, hi, label in EV_BUCKETS:
        if lo <= e < hi:
            return label
    return None


CLV_COLUMNS = [
    "game_date", "event_id", "player_name", "market_key", "line", "side", "book",
    "ev_frac", "ev_pct", "ev_bucket",
    "p_bet", "p_close_consensus", "price_clv", "price_clv_cents",
    "p_close_same_book", "same_book_clv", "same_book_clv_cents",
    "decision_odds", "close_book_odds",
    "beat_close", "has_close_consensus", "has_same_book_close",
]


def compute_clv(
    candidates: pd.DataFrame,
    decision_per_book: dict,
    close_per_book: dict,
    close_consensus: dict,
) -> pd.DataFrame:
    """Compute per-candidate CLV against the close.

    For each flagged candidate we recover the candidate book's OWN decision no-vig P(side)
    (``p_bet``; the price we would have taken) and compare it to:
      * the closing no-vig **consensus** P(side)  -> ``price_clv``   (vs where the market closed)
      * the same book's closing no-vig P(side)     -> ``same_book_clv`` (did the book itself move?)

    CLV is expressed in probability terms and in percentage points ("cents" = prob * 100).
    ``beat_close`` is ``price_clv > 0``. Candidates without a closing consensus are kept with
    ``has_close_consensus=False`` (excluded from price-CLV aggregation); candidates without a
    same-book two-sided close get ``same_book_clv=None`` (fail closed, excluded from that view).
    """
    rows: list[dict] = []
    for _, r in candidates.iterrows():
        e, m, p, ln, book, side = (
            r["event_id"], r["market_key"], r["player_name"],
            float(r["line"]), str(r["book"]), str(r["side"]),
        )
        dp = decision_per_book.get((e, m, p, ln, book))
        if dp is None:
            continue  # no reconstructable decision de-vig for the bet (should not happen)
        p_bet = dp["fair_over"] if side == "over" else dp["fair_under"]
        decision_odds = dp["over_odds"] if side == "over" else dp["under_odds"]

        cc = close_consensus.get((e, m, p, ln))
        if cc is not None:
            p_close_cons = cc["p_over"] if side == "over" else (1.0 - cc["p_over"])
            price_clv = float(p_close_cons) - float(p_bet)
        else:
            p_close_cons, price_clv = None, None

        cb = close_per_book.get((e, m, p, ln, book))
        if cb is not None:
            p_close_sb = cb["fair_over"] if side == "over" else cb["fair_under"]
            same_book_clv = float(p_close_sb) - float(p_bet)
            close_book_odds = cb["over_odds"] if side == "over" else cb["under_odds"]
        else:
            p_close_sb, same_book_clv, close_book_odds = None, None, None

        rows.append({
            "game_date": r.get("game_date"),
            "event_id": e,
            "player_name": p,
            "market_key": m,
            "line": ln,
            "side": side,
            "book": book,
            "ev_frac": float(r["ev_frac"]),
            "ev_pct": float(r.get("ev_pct", r["ev_frac"] * 100.0)),
            "ev_bucket": ev_bucket_label(float(r["ev_frac"])),
            "p_bet": round(float(p_bet), 6),
            "p_close_consensus": (round(float(p_close_cons), 6) if p_close_cons is not None else None),
            "price_clv": (round(float(price_clv), 6) if price_clv is not None else None),
            "price_clv_cents": (round(float(price_clv) * 100.0, 4) if price_clv is not None else None),
            "p_close_same_book": (round(float(p_close_sb), 6) if p_close_sb is not None else None),
            "same_book_clv": (round(float(same_book_clv), 6) if same_book_clv is not None else None),
            "same_book_clv_cents": (round(float(same_book_clv) * 100.0, 4) if same_book_clv is not None else None),
            "decision_odds": float(decision_odds),
            "close_book_odds": (float(close_book_odds) if close_book_odds is not None else None),
            "beat_close": (bool(price_clv > 0) if price_clv is not None else None),
            "has_close_consensus": cc is not None,
            "has_same_book_close": cb is not None,
        })
    return pd.DataFrame(rows, columns=CLV_COLUMNS)


# --------------------------------------------------------------------------- #
# Date-cluster bootstrap + segment summaries
# --------------------------------------------------------------------------- #
def date_cluster_bootstrap_ci(
    values: np.ndarray,
    dates: np.ndarray,
    *,
    iters: int = DEFAULT_BOOTSTRAP_ITERS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_CI_ALPHA,
) -> tuple[float | None, float | None, int]:
    """Cluster bootstrap of the mean, resampling whole game_date clusters with replacement.

    Returns ``(ci_low, ci_high, n_clusters)``. Clustering by date is the honest choice: props
    on the same slate are correlated, so the effective sample size is the number of dates, not
    the number of bets. Returns ``(None, None, n_clusters)`` when there are fewer than
    ``MIN_DATE_CLUSTERS`` clusters (no between-cluster variance -> no trustworthy CI).
    """
    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates, dtype=object)
    uniq = np.unique(dates)
    n_clusters = len(uniq)
    if n_clusters < MIN_DATE_CLUSTERS or len(values) == 0:
        return None, None, n_clusters
    by_date = {d: values[dates == d] for d in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        sampled = rng.choice(uniq, size=n_clusters, replace=True)
        pooled = np.concatenate([by_date[d] for d in sampled])
        means[i] = pooled.mean()
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi, n_clusters


def summarize_segment(
    sub: pd.DataFrame,
    value_col: str,
    *,
    min_segment_n: int = DEFAULT_MIN_SEGMENT_N,
    iters: int = DEFAULT_BOOTSTRAP_ITERS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Summarize one segment on ``value_col`` (drops rows where the metric is null).

    ``significant`` (and thus ``qualifies``) is fail-closed: it requires a positive mean, a
    bootstrap 95% CI whose lower bound is strictly > 0, at least ``MIN_DATE_CLUSTERS`` date
    clusters, and N >= ``min_segment_n``.
    """
    seg = sub[sub[value_col].notna()]
    vals = seg[value_col].to_numpy(dtype=float)
    n = int(len(vals))
    if n == 0:
        return {
            "n": 0, "n_dates": 0, "mean": None, "median": None,
            "mean_cents": None, "median_cents": None, "pct_beat_close": None,
            "ci_low": None, "ci_high": None, "ci_low_cents": None, "ci_high_cents": None,
            "significant": False, "qualifies": False,
            "min_segment_n": min_segment_n, "metric": value_col,
        }
    dates = seg["game_date"].to_numpy(dtype=object)
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    beat = seg["beat_close"].dropna()
    pct_beat = float(100.0 * beat.mean()) if len(beat) else None
    lo, hi, n_clusters = date_cluster_bootstrap_ci(vals, dates, iters=iters, seed=seed)
    significant = bool(
        lo is not None and lo > 0.0 and mean > 0.0 and n_clusters >= MIN_DATE_CLUSTERS
    )
    qualifies = bool(significant and n >= min_segment_n)
    return {
        "n": n,
        "n_dates": int(n_clusters),
        "mean": round(mean, 6),
        "median": round(median, 6),
        "mean_cents": round(mean * 100.0, 4),
        "median_cents": round(median * 100.0, 4),
        "pct_beat_close": (round(pct_beat, 2) if pct_beat is not None else None),
        "ci_low": (round(lo, 6) if lo is not None else None),
        "ci_high": (round(hi, 6) if hi is not None else None),
        "ci_low_cents": (round(lo * 100.0, 4) if lo is not None else None),
        "ci_high_cents": (round(hi * 100.0, 4) if hi is not None else None),
        "significant": significant,
        "qualifies": qualifies,
        "min_segment_n": min_segment_n,
        "metric": value_col,
    }


def aggregate_segments(
    clv_df: pd.DataFrame,
    *,
    value_col: str = "price_clv",
    min_segment_n: int = DEFAULT_MIN_SEGMENT_N,
    iters: int = DEFAULT_BOOTSTRAP_ITERS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Aggregate CLV into segment tables: overall, by market, by book, by EV bucket, and by
    market×EV-bucket. Each entry is a ``summarize_segment`` dict tagged with its key.
    """
    def _summ(sub):
        return summarize_segment(sub, value_col, min_segment_n=min_segment_n,
                                 iters=iters, seed=seed)

    overall = _summ(clv_df)
    overall["segment_type"] = "overall"
    overall["key"] = "ALL"

    by_market = {}
    for m, g in clv_df.groupby("market_key"):
        s = _summ(g)
        s["segment_type"], s["key"] = "market", str(m)
        by_market[str(m)] = s

    by_book = {}
    for b, g in clv_df.groupby("book"):
        s = _summ(g)
        s["segment_type"], s["key"] = "book", str(b)
        by_book[str(b)] = s

    by_ev = {}
    for bucket, g in clv_df.dropna(subset=["ev_bucket"]).groupby("ev_bucket"):
        s = _summ(g)
        s["segment_type"], s["key"] = "ev_bucket", str(bucket)
        by_ev[str(bucket)] = s

    by_market_ev = {}
    sub = clv_df.dropna(subset=["ev_bucket"])
    for (m, bucket), g in sub.groupby(["market_key", "ev_bucket"]):
        key = f"{m}|{bucket}"
        s = _summ(g)
        s["segment_type"], s["key"] = "market_ev_bucket", key
        by_market_ev[key] = s

    return {
        "value_col": value_col,
        "overall": overall,
        "by_market": by_market,
        "by_book": by_book,
        "by_ev_bucket": by_ev,
        "by_market_ev_bucket": by_market_ev,
    }


# --------------------------------------------------------------------------- #
# Validation table (consumed by the board + gate)
# --------------------------------------------------------------------------- #
def build_validation_table(
    segments: dict,
    *,
    min_segment_n: int,
    ev_threshold: float,
    min_consensus_books: int,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> dict:
    """Build the compact, persisted CLV validation table the board/gate consume.

    Only the two granularities used for row-level actionability are exported: ``market`` and
    ``market_ev_bucket``. ``actionable_segments`` lists every qualifying key. The board marks
    a row actionable iff its market×EV-bucket OR its market segment qualifies (fail closed).
    """
    def _pack(s: dict) -> dict:
        return {
            "key": s["key"],
            "segment_type": s["segment_type"],
            "n": s["n"],
            "n_dates": s["n_dates"],
            "mean": s["mean"],
            "median": s["median"],
            "pct_beat_close": s["pct_beat_close"],
            "ci_low": s["ci_low"],
            "ci_high": s["ci_high"],
            "significant": s["significant"],
            "qualifies": s["qualifies"],
        }

    market = {k: _pack(v) for k, v in segments["by_market"].items()}
    market_ev = {k: _pack(v) for k, v in segments["by_market_ev_bucket"].items()}
    actionable = sorted(
        [k for k, v in market.items() if v["qualifies"]]
        + [k for k, v in market_ev.items() if v["qualifies"]]
    )
    return {
        "schema_version": "clv_validation_table_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "min_segment_n": min_segment_n,
        "min_date_clusters": MIN_DATE_CLUSTERS,
        "ci_alpha": DEFAULT_CI_ALPHA,
        "bootstrap_iters": bootstrap_iters,
        "bootstrap_seed": bootstrap_seed,
        "scan_config": {
            "ev_threshold": ev_threshold,
            "min_consensus_books": min_consensus_books,
        },
        "rule": (
            "A board row is actionable iff its market x EV-bucket segment OR its market "
            "segment has positive mean price_clv with a date-cluster bootstrap 95% CI that "
            "excludes 0 AND N >= min_segment_n. Otherwise actionable=false (fail closed)."
        ),
        "overall": _pack(segments["overall"]),
        "segments": {"market": market, "market_ev_bucket": market_ev},
        "actionable_segments": actionable,
    }


def _seg_qualifies(entry: dict | None) -> bool:
    return bool(entry is not None and entry.get("qualifies") is True)


def lookup_actionability(
    market_key: str | None,
    ev_frac: float | None,
    table: dict,
) -> tuple[bool, str, dict | None]:
    """Resolve one board row's actionability from the validation table (fail closed).

    Precedence: the market×EV-bucket segment first (most specific), then the market segment.
    Returns ``(actionable, reason, evidence)``. ``evidence`` is the qualifying segment entry
    when actionable, else None.
    """
    segments = (table or {}).get("segments", {})
    market_segs = segments.get("market", {})
    market_ev_segs = segments.get("market_ev_bucket", {})
    mk = str(market_key) if market_key is not None else None
    bucket = ev_bucket_label(ev_frac)

    if mk is not None and bucket is not None:
        me = market_ev_segs.get(f"{mk}|{bucket}")
        if _seg_qualifies(me):
            return True, f"segment_actionable:market_ev_bucket={mk}|{bucket}", me

    if mk is not None:
        m = market_segs.get(mk)
        if _seg_qualifies(m):
            return True, f"segment_actionable:market={mk}", m

    # Fail closed with an explicit, specific reason.
    m = market_segs.get(mk) if mk is not None else None
    if m is None:
        return False, f"no_validated_segment_for_market={mk}", None
    min_n = table.get("min_segment_n", DEFAULT_MIN_SEGMENT_N)
    if int(m.get("n", 0)) < int(min_n):
        return (
            False,
            f"insufficient_sample:market={mk} n={m.get('n')} < min_segment_n={min_n}",
            None,
        )
    if not m.get("significant"):
        return False, f"clv_ci_includes_zero_or_nonpositive:market={mk}", None
    return False, f"segment_not_actionable:market={mk}", None


def apply_validation_table_to_board(board: pd.DataFrame, table: dict | None) -> pd.DataFrame:
    """Set ``actionable`` + supporting fields on a scanned board from the validation table.

    Every row is stamped fail closed: rows whose segment qualifies become
    ``actionable=True`` with ``validation_status='VALIDATED_EXECUTABLE'``,
    ``forward_clv_validated=True`` and the CLV evidence (``clv_segment``, ``clv_segment_n``,
    ``clv_mean``, ``clv_ci_low``, ``clv_ci_high``); all others stay ``actionable=False`` with
    an explicit ``actionable_reason``. ``source_type`` (MARKET_DISLOCATION) is untouched and
    NO stake/Kelly is emitted.
    """
    board = board.copy()
    n = len(board)
    board["actionable"] = False
    board["actionable_reason"] = None
    board["forward_clv_validated"] = False
    board["clv_segment"] = None
    board["clv_segment_n"] = None
    board["clv_mean"] = None
    board["clv_ci_low"] = None
    board["clv_ci_high"] = None
    if n == 0 or not table:
        if "validation_status" not in board.columns:
            board["validation_status"] = None
        board["actionable_reason"] = "no_validation_table" if n else None
        return board

    has_qualified = "qualified" in board.columns
    for i in board.index:
        # Only a FLAGGED +EV candidate can ever be actionable; a row that did not clear the
        # scan's EV / consensus-quality bar is never a bet, regardless of segment CLV.
        if has_qualified and not bool(board.at[i, "qualified"]):
            board.at[i, "actionable"] = False
            board.at[i, "actionable_reason"] = "not_qualified_ev_candidate"
            continue
        actionable, reason, ev = lookup_actionability(
            board.at[i, "market_key"], board.at[i, "ev_frac"], table
        )
        board.at[i, "actionable"] = bool(actionable)
        board.at[i, "actionable_reason"] = reason
        if actionable and ev is not None:
            board.at[i, "forward_clv_validated"] = True
            board.at[i, "validation_status"] = "VALIDATED_EXECUTABLE"
            board.at[i, "clv_segment"] = ev.get("key")
            board.at[i, "clv_segment_n"] = ev.get("n")
            board.at[i, "clv_mean"] = ev.get("mean")
            board.at[i, "clv_ci_low"] = ev.get("ci_low")
            board.at[i, "clv_ci_high"] = ev.get("ci_high")
    return board


# --------------------------------------------------------------------------- #
# Top-level backtest
# --------------------------------------------------------------------------- #
def run_backtest(
    quotes: pd.DataFrame,
    *,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    min_consensus_books: int = DEFAULT_MIN_CONSENSUS_BOOKS,
    min_segment_n: int = DEFAULT_MIN_SEGMENT_N,
    bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Run the full CLV backtest and return a result dict (report + validation table + clv rows).

    ``quotes`` must already be normalized (see ``normalize_quotes``).
    """
    replay = build_replay_frame(quotes)
    close = closing_frame(quotes)
    identity_index = build_identity_index(quotes)

    dec_per, _dec_cons = build_novig_tables(replay)
    close_per, close_cons = build_novig_tables(close)

    candidates = replay_decision_scan(
        replay, identity_index,
        ev_threshold=ev_threshold, min_consensus_books=min_consensus_books,
    )
    clv_df = compute_clv(candidates, dec_per, close_per, close_cons)

    segments_price = aggregate_segments(
        clv_df, value_col="price_clv", min_segment_n=min_segment_n,
        iters=bootstrap_iters, seed=bootstrap_seed,
    )
    segments_same_book = aggregate_segments(
        clv_df, value_col="same_book_clv", min_segment_n=min_segment_n,
        iters=bootstrap_iters, seed=bootstrap_seed,
    )
    validation_table = build_validation_table(
        segments_price, min_segment_n=min_segment_n, ev_threshold=ev_threshold,
        min_consensus_books=min_consensus_books, bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
    )

    n_price = int(clv_df["price_clv"].notna().sum()) if len(clv_df) else 0
    n_dates = int(quotes["game_date"].nunique()) if "game_date" in quotes.columns else 0
    coverage = {
        "n_flagged_candidates": int(len(candidates)),
        "n_with_close_consensus": n_price,
        "n_with_same_book_close": int(clv_df["has_same_book_close"].sum()) if len(clv_df) else 0,
        "n_game_dates_total": n_dates,
        "n_game_dates_with_candidates": int(clv_df["game_date"].nunique()) if len(clv_df) else 0,
        "books": sorted({str(b) for b in quotes["book"].dropna().unique()}) if "book" in quotes else [],
    }

    return {
        "coverage": coverage,
        "segments_price_clv": segments_price,
        "segments_same_book_clv": segments_same_book,
        "validation_table": validation_table,
        "clv_rows": clv_df,
        "candidates": candidates,
    }
