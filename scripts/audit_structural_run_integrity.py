#!/usr/bin/env python3
"""Owner ITEM 1 — structural-run OOF integrity audit (CLI).

Audits EVERY row of a built OOF parquet (``oof_player_stat_pmfs.parquet``) so the owner can run
it locally on the downloaded artifact (the parquet is too large to commit; only the resulting
JSON is committed). It NEVER mutates the parquet and NEVER touches the market — it only reads and
verifies the pure OOF output.

Per-row checks (aggregated into pass/fail counts):
  * ``pmf_json`` parses to a finite, nonnegative distribution summing to 1 (within 1e-6);
  * exported ``pmf_mean`` == mean(pmf_json)   (within tolerance);
  * exported ``pmf_variance`` == var(pmf_json) (within tolerance);
  * ``active_pmf_json`` exists, parses, and is a valid distribution;
  * ``availability_mixture_pmf_json`` exists and is a valid distribution;
  * ``support_tail_warning`` is boolean with no missing values;
  * AST/TOV rows satisfy the minutes-offset self-consistency (stat_mean==pmf_mean==mean(pmf_json));
  * NO ``prior_only`` / ``failed_model_fit`` rows;
  * ``information_contract == 'pure_forecast'`` on every row;
  * ``market_probability_weight == 0`` on every row;
  * NO forbidden market-derived feature column present in the parquet schema;
  * ``fold_train_end_date < fold_validation_start_date`` on every row (no temporal leakage);
  * all seven direct props present;
  * structural candidate fields valid where expected (pts/reb/fg3m rows that carry a structural
    candidate id have a parseable ``structural_active_pmf_json``).

Writes ``artifacts/pure_supremacy/STRUCTURAL_RUN_INTEGRITY_AUDIT.json`` and exits nonzero when any
check fails (so it can gate a workflow), unless ``--no-fail`` is passed.

Usage:
    python scripts/audit_structural_run_integrity.py path/to/oof_player_stat_pmfs.parquet
    python scripts/audit_structural_run_integrity.py <parquet> --out artifacts/.../AUDIT.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from wnba_props_model.models.pure_model_contract import (  # noqa: E402
    forbidden_market_columns,
)
from wnba_props_model.models.structural_pmf import (  # noqa: E402
    SUPPORTED_PROPS as SUPPORTED_STRUCTURAL_PROPS,
)

REQUIRED_PROPS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover")
AST_TOV_PROPS = ("ast", "turnover")

# OOF provenance/metadata columns that legitimately mention "market" but are NOT predictive
# feature inputs — they RECORD that the run carried zero market weight. They must be excluded from
# the forbidden-market-FEATURE scan (which is about model inputs, not provenance).
_PROVENANCE_MARKET_COLS = frozenset({
    "market_probability_weight", "market_prior_lambda", "information_contract",
})
SUM_TOL = 1e-6
MEAN_TOL = 1e-4
VAR_TOL = 1e-3
DEFAULT_OUT = "artifacts/pure_supremacy/STRUCTURAL_RUN_INTEGRITY_AUDIT.json"


def _parse_pmf(payload: Any) -> np.ndarray | None:
    """Parse a PMF JSON payload to a dense array WITHOUT renormalizing (so we can verify the raw
    mass sums to 1). Returns None when the payload is null/unparseable."""
    if payload is None or (isinstance(payload, float) and math.isnan(payload)):
        return None
    try:
        d = json.loads(payload) if isinstance(payload, str) else payload
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(d, list):
        return np.asarray(d, dtype=float)
    if isinstance(d, dict):
        if not d:
            return None
        try:
            kmax = max(int(k) for k in d.keys())
            arr = np.zeros(kmax + 1, dtype=float)
            for k, p in d.items():
                arr[int(k)] = float(p)
            return arr
        except (ValueError, TypeError):
            return None
    return None


def _pmf_mean(arr: np.ndarray) -> float:
    return float(np.dot(np.arange(arr.size), arr))


def _pmf_var(arr: np.ndarray) -> float:
    m = _pmf_mean(arr)
    return float(np.dot((np.arange(arr.size) - m) ** 2, arr))


def _valid_distribution(arr: np.ndarray | None) -> tuple[bool, str]:
    if arr is None:
        return False, "unparseable_or_missing"
    if arr.size == 0:
        return False, "empty"
    if not np.all(np.isfinite(arr)):
        return False, "nonfinite"
    if np.any(arr < -1e-12):
        return False, "negative_mass"
    if abs(float(arr.sum()) - 1.0) > SUM_TOL:
        return False, f"sum_off_by_{abs(float(arr.sum()) - 1.0):.2e}"
    return True, "ok"


def _new_check() -> dict[str, Any]:
    return {"passed": True, "n_fail": 0, "examples": []}


def _record(check: dict[str, Any], ok: bool, ctx: dict[str, Any]) -> None:
    if ok:
        return
    check["passed"] = False
    check["n_fail"] += 1
    if len(check["examples"]) < 10:
        check["examples"].append(ctx)


def audit_oof(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    checks: dict[str, dict[str, Any]] = {name: _new_check() for name in (
        "pmf_parses", "pmf_finite_nonnegative", "pmf_sums_to_one", "pmf_mean_matches",
        "pmf_variance_matches", "active_pmf_valid", "availability_mixture_valid",
        "no_prior_only_or_failed", "information_contract_pure", "market_weight_zero",
        "no_temporal_leakage", "ast_tov_integrity", "structural_fields_valid",
    )}

    def _row_ctx(row: pd.Series, extra: str) -> dict[str, Any]:
        return {
            "game_id": str(row.get("game_id")), "player_id": str(row.get("player_id")),
            "stat": str(row.get("stat")), "reason": extra,
        }

    has_pmf_mean = "pmf_mean" in df.columns
    has_pmf_var = "pmf_variance" in df.columns
    has_active = "active_pmf_json" in df.columns
    has_mixture = "availability_mixture_pmf_json" in df.columns
    has_contract = "information_contract" in df.columns
    has_mkt_w = "market_probability_weight" in df.columns
    has_struct = "structural_active_pmf_json" in df.columns
    has_struct_id = "structural_candidate_id" in df.columns

    for row in df.itertuples(index=False):
        r = pd.Series(row._asdict())
        stat = str(r.get("stat"))
        pmf = _parse_pmf(r.get("pmf_json"))

        _record(checks["pmf_parses"], pmf is not None, _row_ctx(r, "pmf_json_unparseable"))
        if pmf is not None:
            finite_nonneg = bool(np.all(np.isfinite(pmf)) and np.all(pmf >= -1e-12))
            _record(checks["pmf_finite_nonnegative"], finite_nonneg,
                    _row_ctx(r, "nonfinite_or_negative"))
            sum_ok = abs(float(pmf.sum()) - 1.0) <= SUM_TOL
            _record(checks["pmf_sums_to_one"], sum_ok,
                    _row_ctx(r, f"sum={float(pmf.sum()):.6f}"))
            if has_pmf_mean and pd.notna(r.get("pmf_mean")):
                mean_ok = abs(_pmf_mean(pmf) - float(r.get("pmf_mean"))) <= MEAN_TOL
                _record(checks["pmf_mean_matches"], mean_ok,
                        _row_ctx(r, f"exported={float(r.get('pmf_mean')):.5f} "
                                    f"pmf={_pmf_mean(pmf):.5f}"))
            if has_pmf_var and pd.notna(r.get("pmf_variance")):
                var_ok = abs(_pmf_var(pmf) - float(r.get("pmf_variance"))) <= VAR_TOL
                _record(checks["pmf_variance_matches"], var_ok,
                        _row_ctx(r, f"exported={float(r.get('pmf_variance')):.5f} "
                                    f"pmf={_pmf_var(pmf):.5f}"))

        active_ok, active_why = _valid_distribution(_parse_pmf(r.get("active_pmf_json")) if has_active else None)
        _record(checks["active_pmf_valid"], active_ok, _row_ctx(r, f"active:{active_why}"))
        mix_ok, mix_why = _valid_distribution(_parse_pmf(r.get("availability_mixture_pmf_json")) if has_mixture else None)
        _record(checks["availability_mixture_valid"], mix_ok, _row_ctx(r, f"mixture:{mix_why}"))

        oof_type = str(r.get("oof_prediction_type", "model_oof"))
        _record(checks["no_prior_only_or_failed"],
                oof_type not in ("prior_only", "failed_model_fit"),
                _row_ctx(r, f"oof_prediction_type={oof_type}"))

        _record(checks["information_contract_pure"],
                (str(r.get("information_contract")) == "pure_forecast") if has_contract else False,
                _row_ctx(r, f"contract={r.get('information_contract')}"))
        _record(checks["market_weight_zero"],
                (float(r.get("market_probability_weight", 1.0) or 0.0) == 0.0) if has_mkt_w else False,
                _row_ctx(r, f"mkt_w={r.get('market_probability_weight')}"))

        te, vs = r.get("fold_train_end_date"), r.get("fold_validation_start_date")
        try:
            leak_ok = pd.to_datetime(te) < pd.to_datetime(vs)
        except (ValueError, TypeError):
            leak_ok = False
        _record(checks["no_temporal_leakage"], bool(leak_ok),
                _row_ctx(r, f"train_end={te} val_start={vs}"))

        if stat in AST_TOV_PROPS and pmf is not None:
            sm = r.get("stat_mean")
            pm = r.get("pmf_mean")
            ok = True
            if pd.notna(sm) and pd.notna(pm):
                ok = (abs(float(sm) - float(pm)) <= MEAN_TOL
                      and abs(float(pm) - _pmf_mean(pmf)) <= MEAN_TOL)
            _record(checks["ast_tov_integrity"], ok,
                    _row_ctx(r, "stat_mean/pmf_mean/pmf_json mismatch"))

        if stat in SUPPORTED_STRUCTURAL_PROPS and has_struct:
            cid = r.get("structural_candidate_id") if has_struct_id else None
            sjson = r.get("structural_active_pmf_json")
            has_cid = cid is not None and not (isinstance(cid, float) and math.isnan(cid))
            has_sjson = sjson is not None and not (isinstance(sjson, float) and math.isnan(sjson))
            if has_cid or has_sjson:
                s_ok, s_why = _valid_distribution(_parse_pmf(sjson))
                _record(checks["structural_fields_valid"], s_ok,
                        _row_ctx(r, f"structural:{s_why}"))

    # ---- schema/global checks ---------------------------------------------
    props_present = sorted(set(df["stat"].astype(str).unique())) if "stat" in df.columns else []
    missing_props = sorted(set(REQUIRED_PROPS) - set(props_present))
    all_props = {"passed": not missing_props, "props_present": props_present,
                 "missing_props": missing_props}

    _scanned_cols = [c for c in df.columns if c not in _PROVENANCE_MARKET_COLS]
    forbidden = forbidden_market_columns(_scanned_cols)
    no_forbidden = {"passed": not forbidden, "forbidden_columns_present": sorted(forbidden)}

    stw_check: dict[str, Any] = {"passed": True, "reason": "ok"}
    if "support_tail_warning" not in df.columns:
        stw_check = {"passed": False, "reason": "column_absent"}
    else:
        col = df["support_tail_warning"]
        n_missing = int(col.isna().sum())
        is_bool = col.dtype == bool or col.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all()
        stw_check = {"passed": bool(is_bool and n_missing == 0),
                     "dtype": str(col.dtype), "n_missing": n_missing, "is_bool": bool(is_bool)}

    for name, chk in checks.items():
        chk["n_fail"] = int(chk["n_fail"])
        chk["passed"] = bool(chk["passed"])

    overall = (all(c["passed"] for c in checks.values())
               and all_props["passed"] and no_forbidden["passed"] and stw_check["passed"])

    return {
        "overall_passed": bool(overall),
        "n_rows": int(n),
        "row_checks": checks,
        "all_seven_props_present": all_props,
        "no_forbidden_market_feature": no_forbidden,
        "support_tail_warning": stw_check,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Audit OOF parquet integrity (owner item 1).")
    ap.add_argument("oof_parquet", help="Path to oof_player_stat_pmfs.parquet")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path")
    ap.add_argument("--no-fail", action="store_true",
                    help="Always exit 0 (write the report but do not gate on it)")
    args = ap.parse_args()

    path = Path(args.oof_parquet)
    if not path.exists():
        print(f"ERROR: OOF parquet not found: {path}", file=sys.stderr)
        sys.exit(2)

    print(f"[audit] reading {path} ...")
    df = pd.read_parquet(path)
    print(f"[audit] {len(df):,} rows, {len(df.columns)} columns")
    report = audit_oof(df)
    report["source_parquet"] = str(path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[audit] wrote {out_path}")

    print(f"[audit] overall_passed={report['overall_passed']}")
    for name, chk in report["row_checks"].items():
        flag = "PASS" if chk["passed"] else f"FAIL ({chk['n_fail']})"
        print(f"  {name:34s} {flag}")
    print(f"  {'all_seven_props_present':34s} "
          f"{'PASS' if report['all_seven_props_present']['passed'] else 'FAIL'}")
    print(f"  {'no_forbidden_market_feature':34s} "
          f"{'PASS' if report['no_forbidden_market_feature']['passed'] else 'FAIL'}")
    print(f"  {'support_tail_warning':34s} "
          f"{'PASS' if report['support_tail_warning']['passed'] else 'FAIL'}")

    if not report["overall_passed"] and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
