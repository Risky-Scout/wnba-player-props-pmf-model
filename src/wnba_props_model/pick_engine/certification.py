"""Long-run prospective certification gate (separate from provisional picks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MIN_SETTLED_ROWS = 300
MIN_GAME_DATES = 30


@dataclass(frozen=True)
class CertificationResult:
    certified: bool
    reason: str
    n_settled: int = 0
    n_dates: int = 0
    logloss_delta: float | None = None
    details: dict[str, Any] | None = None


def evaluate_certification_gate(
    settled: pd.DataFrame,
    *,
    stat: str,
    segment: str | None = None,
    pick_col: str = "pick_probability",
    ref_col: str = "reference_market_probability",
    outcome_col: str = "outcome",
    date_col: str = "game_date",
    max_calibration_ece: float = 0.08,
) -> CertificationResult:
    """Require long-run prospective evidence before CERTIFIED_MODEL_PICK.

    Provisional picks must NOT wait on this gate.
    """
    if settled is None or settled.empty:
        return CertificationResult(False, "insufficient_settled_rows", 0, 0)

    df = settled.copy()
    if "stat" in df.columns:
        df = df[df["stat"].astype(str) == str(stat)]
    if segment and "segment" in df.columns:
        df = df[df["segment"].astype(str) == str(segment)]
    df = df.dropna(subset=[pick_col, outcome_col, date_col])
    n = len(df)
    n_dates = int(df[date_col].nunique()) if n else 0
    if n < MIN_SETTLED_ROWS:
        return CertificationResult(False, "insufficient_settled_rows", n, n_dates)
    if n_dates < MIN_GAME_DATES:
        return CertificationResult(False, "insufficient_game_dates", n, n_dates)

    y = df[outcome_col].to_numpy(float)
    p = np.clip(df[pick_col].to_numpy(float), 1e-6, 1 - 1e-6)
    ll_pick = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    if ref_col in df.columns and df[ref_col].notna().any():
        r = np.clip(df[ref_col].to_numpy(float), 1e-6, 1 - 1e-6)
        ll_ref = float(np.mean(-(y * np.log(r) + (1 - y) * np.log(1 - r))))
        delta = ll_ref - ll_pick  # positive => pick better (lower log loss)
    else:
        ll_ref = None
        delta = None

    # Crude ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(len(bins) - 1):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < len(bins) - 2 else p <= bins[i + 1])
        if not np.any(m):
            continue
        ece += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / len(p))
    if ece > max_calibration_ece:
        return CertificationResult(
            False, "calibration_unacceptable", n, n_dates, delta, {"ece": ece}
        )

    # Catastrophic period: any 10-date block with logloss delta < -0.05
    if date_col in df.columns and delta is not None:
        df = df.sort_values(date_col)
        dates = sorted(df[date_col].unique())
        for i in range(0, max(0, len(dates) - 9)):
            block_dates = set(dates[i : i + 10])
            sub = df[df[date_col].isin(block_dates)]
            if len(sub) < 30:
                continue
            pb = np.clip(sub[pick_col].to_numpy(float), 1e-6, 1 - 1e-6)
            rb = np.clip(sub[ref_col].to_numpy(float), 1e-6, 1 - 1e-6)
            yb = sub[outcome_col].to_numpy(float)
            ll_p = float(np.mean(-(yb * np.log(pb) + (1 - yb) * np.log(1 - pb))))
            ll_r = float(np.mean(-(yb * np.log(rb) + (1 - yb) * np.log(1 - rb))))
            if (ll_r - ll_p) < -0.05:
                return CertificationResult(
                    False, "catastrophic_period", n, n_dates, delta, {"block_start": str(dates[i])}
                )

    if delta is None or delta <= 0:
        return CertificationResult(False, "no_positive_logloss_evidence", n, n_dates, delta)

    # Multiple-testing placeholder: require delta above a simple Bonferroni-ish floor.
    if delta < 0.002:
        return CertificationResult(False, "multiple_testing_correction", n, n_dates, delta)

    return CertificationResult(
        True,
        "certified",
        n,
        n_dates,
        delta,
        {"ece": ece, "logloss_pick": ll_pick, "logloss_ref": ll_ref},
    )
