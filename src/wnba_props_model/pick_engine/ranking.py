"""Daily ranked selections and provisional wager labeling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from wnba_props_model.pick_engine.certification import CertificationResult
from wnba_props_model.pick_engine.constants import (
    CERTIFIED_MODEL_PICK,
    DAILY_RANKED_SELECTION,
    DEFAULT_TOP_N,
    NO_POSITIVE_CONSERVATIVE_EV,
    PROVISIONAL_MODEL_PICK,
)


RANK_COLUMNS = [
    "rank",
    "game",
    "scheduled_tip",
    "player",
    "team",
    "opponent",
    "stat",
    "line",
    "side",
    "sportsbook",
    "american_odds",
    "decimal_odds",
    "pure_probability",
    "reference_probability",
    "production_probability",
    "pick_probability",
    "break_even_probability",
    "p_win",
    "p_lose",
    "p_push",
    "raw_probability_edge",
    "shrunken_probability_edge",
    "raw_expected_value",
    "conservative_expected_value",
    "reliability_weight",
    "uncertainty",
    "quote_age",
    "availability_status",
    "selection_status",
    "reason",
    "model_hash",
    "calibrator_hash",
    "feature_hash",
    "data_hash",
    "quote_hash",
    "availability_hash",
    "weights_hash",
    "board_label",
]


def assign_selection_status(
    row: dict[str, Any],
    *,
    certification: CertificationResult | None = None,
) -> tuple[str, str]:
    """Label selection status. Provisional does NOT require certification."""
    raw_ev = float(row.get("raw_expected_value") or 0.0)
    cons_ev = float(row.get("conservative_expected_value") or 0.0)
    w = float(row.get("reliability_weight") or 0.0)
    ood = bool(row.get("ood_warning") or row.get("ood_flag"))
    avail_warn = bool(row.get("availability_warning"))

    certified_ok = bool(certification and certification.certified)
    if (
        cons_ev > 0
        and raw_ev > 0
        and w > 0
        and not ood
        and not avail_warn
        and certified_ok
    ):
        return CERTIFIED_MODEL_PICK, "long_run_certification_gate_passed"

    if cons_ev > 0 and raw_ev > 0 and w > 0 and not ood and not avail_warn:
        return PROVISIONAL_MODEL_PICK, "positive_conservative_ev_provisional"

    if cons_ev <= 0:
        return NO_POSITIVE_CONSERVATIVE_EV, "nonpositive_conservative_ev"

    return DAILY_RANKED_SELECTION, "ranked_model_opinion"


def rank_candidates(
    candidates: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    certification_by_stat: dict[str, CertificationResult] | None = None,
    board_label: str = "",
) -> pd.DataFrame:
    """Rank valid candidate sides; always produce a board when valid sides exist.

    Sort keys:
      1. conservative expected value
      2. pick-probability advantage
      3. reliability
      4. quote freshness (lower age better)
      5. lower uncertainty
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=RANK_COLUMNS)

    df = candidates.copy()
    # Only valid (non-abstaining) rows enter the ranked board.
    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame(columns=RANK_COLUMNS)

    df["pick_probability_advantage"] = df["pick_probability"].astype(float) - df[
        "break_even_probability"
    ].astype(float)
    df["quote_age_sort"] = df["quote_age"].fillna(1e9).astype(float)
    df["uncertainty_sort"] = df["uncertainty"].fillna(1.0).astype(float)
    df = df.sort_values(
        by=[
            "conservative_expected_value",
            "pick_probability_advantage",
            "reliability_weight",
            "quote_age_sort",
            "uncertainty_sort",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Spec: produce at least top_n when >= top_n exist; if fewer, return all.
    ranked = df.copy() if len(df) < int(top_n) else df.iloc[: int(top_n)].copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    statuses = []
    reasons = []
    for _, r in ranked.iterrows():
        cert = None
        if certification_by_stat:
            cert = certification_by_stat.get(str(r.get("stat")))
        status, reason = assign_selection_status(r.to_dict(), certification=cert)
        statuses.append(status)
        reasons.append(reason)
    ranked["selection_status"] = statuses
    ranked["reason"] = reasons
    ranked["board_label"] = board_label

    # Ensure required output columns exist.
    for col in RANK_COLUMNS:
        if col not in ranked.columns:
            ranked[col] = None
    return ranked[RANK_COLUMNS]


def provisional_picks(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked is None or ranked.empty:
        return ranked.iloc[0:0].copy() if ranked is not None else pd.DataFrame(columns=RANK_COLUMNS)
    return ranked[
        ranked["selection_status"].isin([PROVISIONAL_MODEL_PICK, CERTIFIED_MODEL_PICK])
    ].copy()
