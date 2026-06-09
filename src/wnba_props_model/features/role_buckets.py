from __future__ import annotations

import numpy as np
import pandas as pd


def role_bucket_from_minutes_dist(mean_minutes: float, q25_minutes: float | None = None, p_inactive: float = 0.0) -> str:
    q25 = mean_minutes if q25_minutes is None else q25_minutes
    if p_inactive >= 0.12 or q25 <= 3:
        return "inactive_risk"
    if mean_minutes < 12:
        return "fringe"
    if mean_minutes < 18:
        return "bench"
    if mean_minutes < 24:
        return "rotation"
    if mean_minutes < 30:
        return "core"
    return "starter"


def add_ex_ante_role_bucket(df: pd.DataFrame, minutes_col: str = "pred_minutes_mean") -> pd.DataFrame:
    out = df.copy()
    if "pred_minutes_q25" not in out:
        out["pred_minutes_q25"] = out[minutes_col]
    if "p_inactive" not in out:
        out["p_inactive"] = 0.0
    out["role_bucket"] = [
        role_bucket_from_minutes_dist(float(m), float(q), float(p))
        for m, q, p in zip(out[minutes_col].fillna(0), out["pred_minutes_q25"].fillna(0), out["p_inactive"].fillna(0))
    ]
    return out
