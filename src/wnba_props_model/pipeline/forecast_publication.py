"""Apply the validated per-market forecast methods to today's slate for publication.

Uses the SAME functions as prequential validation (recalibrate_pmf, hierarchical_empirical_pmf,
build_combo_pmfs) with the champion package's stored artifacts, so evaluated == deployed.
Returns the calibrated Distributions rows restricted to certified markets (+ built combos).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wnba_props_model.evaluation.forecasting import pmf_to_array
from wnba_props_model.evaluation.pmf_recalibration import recalibrate_pmf
from wnba_props_model.evaluation.distribution_calibration import hierarchical_empirical_pmf
from wnba_props_model.models.simulation import build_combo_pmfs

_MINBINS = [-1, 10, 20, 28, 100]
_MINLABELS = ["m0", "m1", "m2", "m3"]
_COMBO_KEY = {"stocks": "stocks", "pts_ast": "pa", "pts_reb": "pr", "pts_reb_ast": "pra"}


def _pmf_mean(arr: np.ndarray) -> float:
    return float((np.arange(len(arr)) * arr).sum())


def _minbucket(minutes: float) -> str:
    for i in range(len(_MINBINS) - 1):
        if _MINBINS[i] < minutes <= _MINBINS[i + 1]:
            return _MINLABELS[i]
    return "m3"


def _cells_from_json(cj: dict) -> dict:
    return {k: (np.array(v["vals"], dtype=int), np.array(v["wts"], dtype=float), float(v["shift"]))
            for k, v in cj.items()}


def _pmf_json(arr: np.ndarray) -> str:
    return json.dumps({str(i): float(round(v, 8)) for i, v in enumerate(arr) if v > 1e-9})


def apply_market(pmf: np.ndarray, point: float, role: str, minutes: float, spec: dict) -> np.ndarray:
    """Apply one market's validated calibration method to a single PMF (location or hierarchical)."""
    method = spec.get("method")
    if method == "location":
        by_role = spec.get("by_role", {})
        d, s = by_role.get(str(role), spec.get("pooled", [0.0, 1.0]))
        return recalibrate_pmf(pmf, float(d), float(s))
    if method == "hierarchical":
        cells = _cells_from_json(spec.get("cells", {}))
        if not cells:
            return pmf
        max_sup = max(len(pmf) - 1, 60)
        hp = hierarchical_empirical_pmf(float(point), f"{role}|{_minbucket(minutes)}", cells, max_sup)
        # Frozen prequential dispersion scale (default 1.0 = identity: existing markets unchanged).
        scale = float(spec.get("scale", 1.0))
        if scale != 1.0:
            hp = recalibrate_pmf(hp, 0.0, scale)
        return hp
    return pmf


def apply_multistat_forecast(proj_df: pd.DataFrame, calib: dict, certified: list) -> pd.DataFrame:
    """Return calibrated Distributions rows for certified markets (direct + built combos).

    Parity: identical transforms to validation. Direct markets are calibrated per row;
    combos are built per player from the calibrated component PMFs using build_combo_pmfs.
    """
    markets = calib.get("markets", {})
    df = proj_df.copy()
    if "role_bucket" not in df.columns:
        df["role_bucket"] = ""
    if "minutes_mean" not in df.columns:
        df["minutes_mean"] = 0.0
    if "pmf_mean" not in df.columns:
        df["pmf_mean"] = df["pmf_json"].map(lambda p: float((np.arange(len(pmf_to_array(p))) * pmf_to_array(p)).sum()))

    # 1) calibrate direct-stat PMFs (hierarchical/location) for every stat that has a spec
    #    (includes blk as a stocks component even though blk direct is not certified).
    calibrated = {}   # (player_id) -> {stat: pmf_array}
    out_rows = []
    for _, r in df.iterrows():
        stat = str(r.get("stat"))
        spec = markets.get(stat)
        if spec is None or spec.get("method") == "combo":
            continue
        arr = apply_market(pmf_to_array(r["pmf_json"]), float(r.get("pmf_mean", 0.0)),
                           str(r.get("role_bucket", "")), float(r.get("minutes_mean", 0.0)), spec)
        calibrated.setdefault(r.get("player_id"), {})[stat] = arr
        if stat in certified:
            rr = r.copy(); rr["pmf_json"] = _pmf_json(arr); rr["forecast_method"] = spec.get("method")
            out_rows.append(rr)

    # 2a) build certified combo-residual markets (hierarchical empirical PMF centered on the
    #     sum of the validated component expectations; frozen prequential dispersion scale).
    for combo in [c for c in certified if markets.get(c, {}).get("method") == "combo_residual"]:
        spec = markets[combo]; parts = spec["parts"]
        cells = _cells_from_json(spec.get("cells", {}))
        scale = float(spec.get("scale", 1.0))
        if not cells:
            continue
        for pid, comps in calibrated.items():
            if not all(p in comps for p in parts):
                continue
            point = float(sum(_pmf_mean(comps[p]) for p in parts))
            meta = df[df["player_id"] == pid].iloc[0]
            max_sup = max(int(point) + 40, 80)
            hp = hierarchical_empirical_pmf(
                point, f"{meta.get('role_bucket', '')}|{_minbucket(float(meta.get('minutes_mean', 0.0)))}",
                cells, max_sup)
            if scale != 1.0:
                hp = recalibrate_pmf(hp, 0.0, scale)
            rr = meta.copy()
            rr["stat"] = combo; rr["pmf_json"] = _pmf_json(np.asarray(hp, float))
            rr["forecast_method"] = "combo_residual"; rr["pmf_mean"] = _pmf_mean(np.asarray(hp, float))
            out_rows.append(rr)

    # 2b) build certified correlated combos per player from calibrated components
    base_row = df.iloc[0] if len(df) else None
    for combo in [c for c in certified if markets.get(c, {}).get("method") == "combo"]:
        spec = markets[combo]; parts = spec["parts"]; key = _COMBO_KEY.get(combo, combo)
        for pid, comps in calibrated.items():
            if not all(p in comps for p in parts):
                continue
            try:
                cpmf = build_combo_pmfs({p: comps[p] for p in parts},
                                        correlations=spec.get("correlations")).get(key)
            except Exception:
                cpmf = None
            if cpmf is None:
                continue
            meta = df[df["player_id"] == pid].iloc[0]
            rr = meta.copy()
            rr["stat"] = combo; rr["pmf_json"] = _pmf_json(np.asarray(cpmf, float))
            rr["forecast_method"] = "combo"
            rr["pmf_mean"] = float((np.arange(len(cpmf)) * np.asarray(cpmf, float)).sum())
            out_rows.append(rr)

    return pd.DataFrame(out_rows) if out_rows else df.iloc[0:0].copy()
