"""Daily drift correction for PMF predictions.

Compares yesterday's model PMF means to actual outcomes and applies
a small multiplicative correction to today's PMFs. This keeps
calibration fresh between weekly OOF rebuilds.

Design principles:
- Multiplicative only (never shifts a PMF mean by more than 15% per day)
- Per-stat, not per-player (too noisy at player level)
- Decays automatically: if no games played yesterday, correction = 1.0
- Cap applies to the DAILY CHANGE FACTOR (±15%), not the absolute value.
  This preserves large corrections like pts=1.278 — capping the absolute
  value at 1.15 would silently erode them to 1.15 every single day.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.models.simulation import normalize_pmf

logger = logging.getLogger(__name__)

# Maximum daily CHANGE FACTOR (multiplicative), not absolute cap.
# A cap of 0.85-1.15 on the absolute value would silently erode any
# correction above 1.15 (e.g. pts=1.278 → 1.278*0.7+1.0*0.3=1.195 → clipped
# to 1.15 every day). Instead we cap the *per-day change* at ±15%.
DRIFT_CHANGE_FACTOR_LO = 0.85   # max -15% per day from current
DRIFT_CHANGE_FACTOR_HI = 1.15   # max +15% per day from current
DRIFT_BLEND_WEIGHT = 0.3
MIN_GAMES_FOR_DRIFT = 3


def compute_daily_drift(
    yesterday_pmfs: pd.DataFrame,
    yesterday_actuals: pd.DataFrame,
    current_bias_corrections: dict[str, float],
) -> dict[str, float]:
    """Compute per-stat drift multiplier from yesterday's results.

    Parameters
    ----------
    yesterday_pmfs : pd.DataFrame
        Rows from yesterday's prediction with columns: player_id, stat, pmf_mean
    yesterday_actuals : pd.DataFrame
        Yesterday's actual stats with columns: player_id, stat, actual_value
    current_bias_corrections : dict[str, float]
        Current bias_corrections.json values

    Returns
    -------
    dict[str, float]
        Updated bias corrections incorporating daily drift.
    """
    merged = yesterday_pmfs.merge(
        yesterday_actuals[["player_id", "stat", "actual_value"]],
        on=["player_id", "stat"],
        how="inner",
    )
    if len(merged) < MIN_GAMES_FOR_DRIFT:
        logger.info("Daily drift: insufficient data (%d rows), no update", len(merged))
        return dict(current_bias_corrections)

    drift_per_stat: dict[str, float] = {}
    for stat, group in merged.groupby("stat"):
        if len(group) < MIN_GAMES_FOR_DRIFT:
            continue
        ratio = group["actual_value"] / group["pmf_mean"].clip(lower=0.1)
        ratio = ratio.clip(0.1, 10.0)
        drift = float(ratio.median())
        current = current_bias_corrections.get(str(stat), 1.0)
        # drift is actual/pmf_mean where pmf_mean is already calibrated.
        # When predictions are perfect, drift ≈ 1.0 and correction should stay
        # unchanged.  The multiplicative formula:
        #   change_factor = 1 + BLEND_WEIGHT * (drift - 1)
        # produces change_factor=1.0 when drift=1.0 ✓ and ≠0 sensitivity.
        # We cap the *daily change factor*, not the absolute value, so large
        # corrections like pts=1.278 are preserved across days.
        change_factor = 1.0 + DRIFT_BLEND_WEIGHT * (drift - 1.0)
        change_factor = float(np.clip(change_factor, DRIFT_CHANGE_FACTOR_LO, DRIFT_CHANGE_FACTOR_HI))
        blended = current * change_factor
        drift_per_stat[str(stat)] = round(blended, 4)
        logger.info(
            "Daily drift %s: yesterday_ratio=%.3f, current=%.3f, change_factor=%.4f -> updated=%.3f",
            stat, drift, current, change_factor, blended,
        )

    for stat, val in current_bias_corrections.items():
        if stat not in drift_per_stat and not stat.startswith("_"):
            drift_per_stat[stat] = val

    return drift_per_stat


def apply_daily_drift_to_pmfs(
    pmfs_long: pd.DataFrame,
    bias_corrections: dict[str, float],
) -> pd.DataFrame:
    """Apply multiplicative drift correction to PMFs by shifting each PMF.

    Scales each PMF by remapping bin k -> k * correction using linear
    interpolation between adjacent bins, preserving distribution shape.
    """
    out_rows = []
    corrections_applied = 0
    for _, row in pmfs_long.iterrows():
        stat = str(row["stat"])
        correction = bias_corrections.get(stat, 1.0)
        if abs(correction - 1.0) < 0.001:
            out_rows.append(row.to_dict())
            continue
        try:
            pmf_dict = json.loads(row["pmf_json"])
            max_k = max(int(k) for k in pmf_dict)
            pmf = np.array([pmf_dict.get(str(i), 0.0) for i in range(max_k + 1)])
            pmf = normalize_pmf(pmf)
            current_mean = float(np.arange(len(pmf)) @ pmf)
            if current_mean < 0.01:
                out_rows.append(row.to_dict())
                continue
            new_max_k = int(np.ceil(max_k * correction)) + 1
            new_pmf = np.zeros(new_max_k + 1)
            for k in range(len(pmf)):
                if pmf[k] < 1e-10:
                    continue
                target_k = k * correction
                k_lo = int(np.floor(target_k))
                k_hi = k_lo + 1
                frac = target_k - k_lo
                if k_lo < len(new_pmf):
                    new_pmf[k_lo] += pmf[k] * (1.0 - frac)
                if k_hi < len(new_pmf):
                    new_pmf[k_hi] += pmf[k] * frac
            new_pmf = normalize_pmf(new_pmf)
            new_mean_actual = float(np.arange(len(new_pmf)) @ new_pmf)
            new_median = int(np.searchsorted(np.cumsum(new_pmf), 0.5))
            new_p0 = float(new_pmf[0]) if len(new_pmf) > 0 else 0.0
            r = row.to_dict()
            r["pmf_json"] = json.dumps({
                str(i): round(float(v), 6)
                for i, v in enumerate(new_pmf) if v > 1e-8
            })
            r["mean"] = round(new_mean_actual, 4)
            r["pmf_mean"] = round(new_mean_actual, 4)
            r["median"] = new_median
            r["p0"] = round(new_p0, 6)
            r["bias_correction_applied"] = round(correction, 4)
            out_rows.append(r)
            corrections_applied += 1
        except Exception as e:
            logger.warning("Daily drift failed for %s: %s", stat, e)
            out_rows.append(row.to_dict())

    logger.info(
        "Daily drift applied to %d / %d PMF rows",
        corrections_applied, len(pmfs_long),
    )
    return pd.DataFrame(out_rows)


def load_bias_corrections(path: str | Path) -> dict[str, float]:
    """Load bias_corrections.json from disk."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def save_bias_corrections(corrections: dict[str, float], path: str | Path) -> None:
    """Save bias_corrections.json with timestamp."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in corrections.items() if not k.startswith("_")}
    payload["_updated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved bias_corrections to %s", p)
