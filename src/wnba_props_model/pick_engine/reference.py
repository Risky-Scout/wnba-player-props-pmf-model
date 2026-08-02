"""Same-time external reference market construction (candidate-book excluded)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.models.market import shin_no_vig_two_way
from wnba_props_model.pick_engine.constants import DEFAULT_MIN_REFERENCE_BOOKS
from wnba_props_model.pick_engine.odds_math import american_to_decimal


@dataclass
class ReferenceMarketResult:
    reference_probability: float | None
    consensus_dispersion: float | None
    n_reference_books: int
    reference_books: list[str] = field(default_factory=list)
    book_no_vig: dict[str, float] = field(default_factory=dict)
    quote_ages_hours: dict[str, float | None] = field(default_factory=dict)
    rejected_books: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    has_valid_reference: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_ts(value) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_hours(quote_ts, asof: datetime | None) -> float | None:
    qt = _parse_ts(quote_ts)
    if qt is None or asof is None:
        return None
    return max(0.0, (asof - qt).total_seconds() / 3600.0)


def _huber_logit_aggregate(values: list[float], ages: list[float | None]) -> float:
    """Robust aggregation in logit space with optional quote-age weights."""
    from wnba_props_model.pick_engine.probabilities import inv_logit, logit

    logits = np.asarray([logit(v) for v in values], dtype=float)
    weights = []
    for age in ages:
        if age is None:
            weights.append(1.0)
        else:
            # Half-life ~3h quote-age weighting.
            weights.append(float(math.exp(-math.log(2.0) * max(age, 0.0) / 3.0)))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    # Huber-like: down-weight outliers beyond 1.5 MAD from median.
    med = float(np.median(logits))
    mad = float(np.median(np.abs(logits - med))) + 1e-6
    soft = np.clip(1.5 * mad / np.maximum(np.abs(logits - med), 1e-6), 0.0, 1.0)
    w2 = w * soft
    w2 = w2 / w2.sum()
    return float(inv_logit(float(np.dot(w2, logits))))


def build_reference_probability(
    quotes: pd.DataFrame,
    *,
    event_id: str,
    player_name: str,
    stat: str,
    line: float,
    side: str,
    candidate_book: str,
    asof: datetime | None = None,
    min_books: int = DEFAULT_MIN_REFERENCE_BOOKS,
    outlier_z: float = 3.0,
) -> ReferenceMarketResult:
    """Build a same-time no-vig reference probability excluding the candidate book.

    Steps:
      1. Form same-book Over/Under pairs at the exact line.
      2. Remove vig book-by-book (Shin).
      3. Exclude candidate_book from consensus.
      4. Age-weight + robust logit aggregation with outlier rejection.
    """
    rejected: dict[str, str] = {}
    if quotes is None or quotes.empty:
        return ReferenceMarketResult(
            reference_probability=None,
            consensus_dispersion=None,
            n_reference_books=0,
            has_valid_reference=False,
        )

    q = quotes.copy()
    # Normalize column names used by soft-book snapshots.
    if "sportsbook" in q.columns and "book" not in q.columns:
        q = q.rename(columns={"sportsbook": "book"})
    need = {"event_id", "player_name", "stat", "line", "side", "book", "american_odds"}
    missing = need - set(q.columns)
    if missing:
        raise ValueError(f"quotes missing columns: {sorted(missing)}")

    mask = (
        (q["event_id"].astype(str) == str(event_id))
        & (q["player_name"].astype(str) == str(player_name))
        & (q["stat"].astype(str) == str(stat))
        & (np.isclose(q["line"].astype(float), float(line), atol=1e-9))
    )
    g = q.loc[mask]
    if g.empty:
        return ReferenceMarketResult(
            reference_probability=None,
            consensus_dispersion=None,
            n_reference_books=0,
            has_valid_reference=False,
        )

    book_no_vig: dict[str, float] = {}
    quote_ages: dict[str, float | None] = {}
    for book, bdf in g.groupby("book", sort=False):
        book_s = str(book)
        overs = bdf[bdf["side"].astype(str).str.lower() == "over"]
        unders = bdf[bdf["side"].astype(str).str.lower() == "under"]
        if overs.empty or unders.empty:
            rejected[book_s] = "one_sided"
            continue
        # Most recent side quotes.
        ts_col = "collected_utc" if "collected_utc" in bdf.columns else None
        if ts_col:
            over = overs.sort_values(ts_col).iloc[-1]
            under = unders.sort_values(ts_col).iloc[-1]
        else:
            over, under = overs.iloc[-1], unders.iloc[-1]
        try:
            # Validate executable American odds without averaging.
            american_to_decimal(over["american_odds"])
            american_to_decimal(under["american_odds"])
            fair_over, fair_under = shin_no_vig_two_way(
                over["american_odds"], under["american_odds"]
            )
        except Exception as exc:  # noqa: BLE001
            rejected[book_s] = f"devig_failed:{exc}"
            continue
        if fair_over is None or fair_under is None:
            rejected[book_s] = "devig_null"
            continue
        book_no_vig[book_s] = float(fair_over)
        ts_val = over.get("book_last_update") or over.get("collected_utc")
        quote_ages[book_s] = _age_hours(ts_val, asof)

    cand = str(candidate_book)
    if cand in book_no_vig:
        # Candidate exclusion is mandatory when evaluating sportsbook B.
        del book_no_vig[cand]
        quote_ages.pop(cand, None)
    elif cand:
        rejected.setdefault(cand, "candidate_absent_from_two_sided_set")

    # Outlier rejection on remaining books.
    if len(book_no_vig) >= 3:
        vals = np.asarray(list(book_no_vig.values()), dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-6
        for b, v in list(book_no_vig.items()):
            if abs(v - med) / mad > outlier_z:
                rejected[b] = "outlier_rejected"
                del book_no_vig[b]
                quote_ages.pop(b, None)

    books = sorted(book_no_vig)
    n = len(books)
    if n < int(min_books):
        return ReferenceMarketResult(
            reference_probability=None,
            consensus_dispersion=None,
            n_reference_books=n,
            reference_books=books,
            book_no_vig=book_no_vig,
            quote_ages_hours=quote_ages,
            rejected_books=sorted(rejected),
            rejection_reasons=rejected,
            has_valid_reference=False,
        )

    side_l = str(side).strip().lower()
    over_vals = [book_no_vig[b] for b in books]
    ages = [quote_ages.get(b) for b in books]
    p_over_ref = _huber_logit_aggregate(over_vals, ages)
    p_ref = p_over_ref if side_l == "over" else (1.0 - p_over_ref)
    dispersion = float(np.std(over_vals)) if len(over_vals) > 1 else 0.0
    return ReferenceMarketResult(
        reference_probability=float(p_ref),
        consensus_dispersion=dispersion,
        n_reference_books=n,
        reference_books=books,
        book_no_vig=book_no_vig,
        quote_ages_hours=quote_ages,
        rejected_books=sorted(rejected),
        rejection_reasons=rejected,
        has_valid_reference=True,
    )
