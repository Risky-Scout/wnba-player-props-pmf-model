#!/usr/bin/env python3
"""Post-merge recovery audit for failed OOF run 30236023013 (dad8a01).

Freezes the failed-run artifact (SHA-256 manifest) and runs the six independent
integrity invariants requested by the owner, elementwise, per prop:

  INV1 mixture serialization : mean/var(parse(pmf_json)) vs pmf_mean/pmf_variance
  INV2 active  serialization : mean(parse(active_pmf_json)) vs active_pmf_mean
  INV3 avail-mix serialization: mean(parse(availability_mixture_pmf_json)) vs pmf_mean
  INV4 mixture construction  : expected_mix(active,p_dnp) vs availability_mixture / pmf_json (elementwise)
  INV5 active recovery       : recover_active(mixture,p_dnp) vs active_pmf_json (elementwise)
  INV6 p_dnp==0 identity     : active == availability_mixture == pmf (elementwise)

Read-only against the source artifact. Writes:
  artifacts/pure_supremacy/run_30236023013/ARTIFACT_MANIFEST.json
  artifacts/pure_supremacy/RUN_30236023013_CORRECTED_INTEGRITY_AUDIT.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wnba_props_model.models.pmf_engine import _blend_with_dnp  # production DNP fold (clip 0.99)

ART_SRC = "/tmp/oof_30236023013"
FREEZE_DIR = "artifacts/pure_supremacy/run_30236023013"
AUDIT_OUT = "artifacts/pure_supremacy/RUN_30236023013_CORRECTED_INTEGRITY_AUDIT.json"
EXPECTED_ARTIFACT_DIGEST = "cc8f8688ba7d44e14baf21a2a7df6ee58227dcc3865d74f9491232fd2aed0da2"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pmf(x) -> dict[int, float]:
    """pmf_json -> {int support: mass}. Robust to str/dict/ndarray."""
    if isinstance(x, str):
        d = json.loads(x)
        return {int(k): float(v) for k, v in d.items()}
    if isinstance(x, dict):
        return {int(k): float(v) for k, v in x.items()}
    a = np.asarray(x, dtype=float)
    return {int(i): float(v) for i, v in enumerate(a)}


def dense(d: dict[int, float], kmax: int) -> np.ndarray:
    out = np.zeros(kmax + 1, dtype=float)
    for k, v in d.items():
        if 0 <= k <= kmax:
            out[k] = v
    return out


def mean_var(d: dict[int, float]) -> tuple[float, float]:
    if not d:
        return 0.0, 0.0
    kmax = max(d)
    a = dense(d, kmax)
    s = a.sum()
    if s <= 0:
        return 0.0, 0.0
    a = a / s
    k = np.arange(a.size, dtype=float)
    m = float(np.dot(k, a))
    v = float(np.dot(k * k, a) - m * m)
    return m, v


def expected_mixture(active: dict[int, float], p_dnp: float) -> dict[int, float]:
    """expected_mix[0]=p_dnp+(1-p_dnp)*active[0]; expected_mix[k]=(1-p_dnp)*active[k]."""
    out = {k: (1.0 - p_dnp) * v for k, v in active.items()}
    out[0] = out.get(0, 0.0) + p_dnp
    return out


def recover_active(mix: dict[int, float], p_dnp: float) -> dict[int, float]:
    """Invert the DNP fold. active[k>0]=mix[k]/(1-d); active[0]=max(mix[0]-d,0)/(1-d); renorm."""
    if p_dnp >= 1.0:
        return {0: 1.0}
    denom = 1.0 - p_dnp
    out = {k: v / denom for k, v in mix.items()}
    out[0] = max(mix.get(0, 0.0) - p_dnp, 0.0) / denom
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def elementwise_max_abs(a: dict[int, float], b: dict[int, float]) -> float:
    kmax = max([0] + list(a) + list(b))
    da, db = dense(a, kmax), dense(b, kmax)
    # normalize both for a fair elementwise comparison
    if da.sum() > 0:
        da = da / da.sum()
    if db.sum() > 0:
        db = db / db.sum()
    return float(np.max(np.abs(da - db)))


def pct_bucket(errs: np.ndarray) -> dict:
    errs = np.asarray(errs, dtype=float)
    return {
        "max": float(np.max(errs)) if errs.size else 0.0,
        "p99": float(np.percentile(errs, 99)) if errs.size else 0.0,
        "n_gt_1e-6": int(np.sum(errs > 1e-6)),
        "n_gt_1e-5": int(np.sum(errs > 1e-5)),
        "n_gt_1e-4": int(np.sum(errs > 1e-4)),
    }


def main() -> None:
    os.makedirs(FREEZE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(AUDIT_OUT), exist_ok=True)

    # ---- 1. FREEZE: SHA-256 for every source file --------------------------
    files = sorted(glob.glob(os.path.join(ART_SRC, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    file_records = []
    for f in files:
        rel = os.path.relpath(f, ART_SRC)
        file_records.append({
            "path": rel,
            "bytes": os.path.getsize(f),
            "sha256": sha256_file(f),
        })
    run_manifest = json.load(open(os.path.join(ART_SRC, "artifacts/audits/PURE_OOF_RUN_MANIFEST.json")))
    fold_meta = {}
    for jf in sorted(glob.glob(os.path.join(ART_SRC, "data/oof/checkpoints/fold_*.json"))):
        j = json.load(open(jf))
        fold_meta[os.path.basename(jf)] = {
            "fold_id": j.get("fold_id"),
            "code_sha": j.get("code_sha"),
            "config_hash": j.get("config_hash"),
            "feature_contract_hash": j.get("feature_contract_hash"),
            "encoder_hash": j.get("encoder_hash"),
            "fit_status": j.get("fit_status"),
            "pmf_integrity_ok": j.get("pmf_integrity_ok"),
            "rows_by_prop": j.get("rows_by_prop"),
            "val_date_range": j.get("val_date_range"),
        }
    artifact_manifest = {
        "run_id": 30236023013,
        "run_conclusion": "failure",
        "run_head_sha": "dad8a013ec830e711a93e8d09aa76c67f8fb30e7",
        "expected_artifact_zip_digest": EXPECTED_ARTIFACT_DIGEST,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_location_readonly": ART_SRC,
        "n_files": len(file_records),
        "files": file_records,
        "run_manifest_hashes": {
            "config_sha256": run_manifest.get("config_sha256"),
            "ordered_feature_list_sha256": run_manifest.get("ordered_feature_list_sha256"),
            "pure_feature_count": run_manifest.get("pure_feature_count"),
            "market_probability_weight": run_manifest.get("market_probability_weight"),
            "market_prior_lambda": run_manifest.get("market_prior_lambda"),
            "forbidden_market_columns_present": run_manifest.get("forbidden_market_columns_present"),
        },
        "fold_metadata": fold_meta,
        "candidate_artifacts_present": bool(glob.glob(os.path.join(ART_SRC, "**/candidate*"), recursive=True)),
        "calibration_artifacts_present": bool(glob.glob(os.path.join(ART_SRC, "**/calibrat*"), recursive=True)),
    }
    json.dump(artifact_manifest, open(os.path.join(FREEZE_DIR, "ARTIFACT_MANIFEST.json"), "w"), indent=2)

    # ---- 2-6. LOAD folds + run invariants ----------------------------------
    parquets = sorted(glob.glob(os.path.join(ART_SRC, "data/oof/checkpoints/fold_*.parquet")))
    frames = []
    for p in parquets:
        d = pd.read_parquet(p)
        d["__fold_file"] = os.path.basename(p)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["p_dnp"] = df["p_dnp"].fillna(0.0).astype(float)

    props = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    report = {"n_rows_total": int(len(df)), "n_folds": len(parquets), "by_prop": {}}

    for prop in props:
        sub = df[df["stat"] == prop]
        if sub.empty:
            continue
        inv1_mean, inv1_var = [], []
        inv2_mean = []
        inv3_mean = []
        inv4_mix_avail, inv4_mix_pmf, inv4_prod = [], [], []
        inv5_active = []
        inv6_am_pmf, inv6_act_pmf, inv6_act_am = [], [], []
        n_pdnp0 = 0
        for _, r in sub.iterrows():
            pmf = parse_pmf(r["pmf_json"])
            act = parse_pmf(r["active_pmf_json"])
            am = parse_pmf(r["availability_mixture_pmf_json"])
            d = float(r["p_dnp"])
            m_pmf, v_pmf = mean_var(pmf)
            m_act, _ = mean_var(act)
            m_am, _ = mean_var(am)
            # INV1
            inv1_mean.append(abs(m_pmf - float(r["pmf_mean"])))
            if "pmf_variance" in r and pd.notna(r["pmf_variance"]):
                inv1_var.append(abs(v_pmf - float(r["pmf_variance"])))
            # INV2
            inv2_mean.append(abs(m_act - float(r["active_pmf_mean"])))
            # INV3 (avail mix mean vs pmf_mean, since no separate column and avail==pmf by design)
            inv3_mean.append(abs(m_am - float(r["pmf_mean"])))
            # INV4 construction (standard formula reimplementation, for reference)
            exp_mix = expected_mixture(act, d)
            inv4_mix_avail.append(elementwise_max_abs(exp_mix, am))
            inv4_mix_pmf.append(elementwise_max_abs(exp_mix, pmf))
            # INV4 construction (AUTHORITATIVE: exact production _blend_with_dnp, clip 0.99)
            kmax_a = max(act) if act else 0
            a_dense = dense(act, kmax_a)
            prod_mix = _blend_with_dnp(a_dense[None, :].copy(), np.array([d]))[0]
            prod_d = {i: float(v) for i, v in enumerate(prod_mix)}
            inv4_prod.append(elementwise_max_abs(prod_d, pmf))
            # INV5 recovery
            rec = recover_active(pmf, d)
            inv5_active.append(elementwise_max_abs(rec, act))
            # INV6 p_dnp==0
            if d == 0.0:
                n_pdnp0 += 1
                inv6_am_pmf.append(elementwise_max_abs(am, pmf))
                inv6_act_pmf.append(elementwise_max_abs(act, pmf))
                inv6_act_am.append(elementwise_max_abs(act, am))
        report["by_prop"][prop] = {
            "n_rows": int(len(sub)),
            "n_pdnp0": n_pdnp0,
            "INV1_mixture_serialization_mean": pct_bucket(inv1_mean),
            "INV1_mixture_serialization_var": pct_bucket(inv1_var) if inv1_var else None,
            "INV2_active_serialization_mean": pct_bucket(inv2_mean),
            "INV3_availmix_serialization_mean": pct_bucket(inv3_mean),
            "INV4_construction_expmix_vs_availmix_elementwise": pct_bucket(inv4_mix_avail),
            "INV4_construction_expmix_vs_pmf_elementwise": pct_bucket(inv4_mix_pmf),
            "INV4_construction_PRODUCTION_blend_vs_pmf_elementwise": pct_bucket(inv4_prod),
            "INV5_active_recovery_elementwise": pct_bucket(inv5_active),
            "INV6_pdnp0_active_vs_pmf_elementwise": pct_bucket(inv6_act_pmf) if inv6_act_pmf else None,
            "INV6_pdnp0_availmix_vs_pmf_elementwise": pct_bucket(inv6_am_pmf) if inv6_am_pmf else None,
        }

    # ---- classification -----------------------------------------------------
    verdict = {}
    for prop, r in report["by_prop"].items():
        inv4p = r["INV4_construction_PRODUCTION_blend_vs_pmf_elementwise"]
        inv1 = r["INV1_mixture_serialization_mean"]["max"]
        # AUTHORITATIVE construction test: does the exact production DNP fold of the STORED active
        # reproduce the STORED mixture? >1e-6 for a material row count == stale active_pmf_json.
        construction_defect = int(inv4p["n_gt_1e-6"]) > 10
        serialization_only = (not construction_defect) and (inv1 > 1e-6)
        verdict[prop] = {
            "construction_defect": bool(construction_defect),
            "serialization_rounding_only": bool(serialization_only),
            "prod_blend_vs_pmf_max": inv4p["max"],
            "prod_blend_vs_pmf_rows_gt_1e-6": int(inv4p["n_gt_1e-6"]),
            "inv1_mixture_mean_serialization_max": inv1,
            "classification": (
                "CONSTRUCTION_DEFECT_stale_active_pmf" if construction_defect
                else ("SERIALIZATION_ROUNDING_ONLY" if serialization_only else "CLEAN")
            ),
        }
    report["verdict_by_prop"] = verdict

    json.dump(report, open(AUDIT_OUT, "w"), indent=2)

    # ---- console summary ----------------------------------------------------
    print("ARTIFACT_MANIFEST files:", len(file_records))
    print("code_sha (all folds):", {v["code_sha"] for v in fold_meta.values()})
    print("config_hash (all folds):", {v["config_hash"] for v in fold_meta.values()})
    print("\n=== INVARIANT SUMMARY (max abs error) ===")
    hdr = f"{'prop':<9}{'INV1mean':>11}{'INV2act':>11}{'INV4exp/pmf':>13}{'INV5recov':>11}{'INV6a/pmf':>11}"
    print(hdr)
    for prop, r in report["by_prop"].items():
        i1 = r["INV1_mixture_serialization_mean"]["max"]
        i2 = r["INV2_active_serialization_mean"]["max"]
        i4 = r["INV4_construction_expmix_vs_pmf_elementwise"]["max"]
        i5 = r["INV5_active_recovery_elementwise"]["max"]
        i6 = r["INV6_pdnp0_active_vs_pmf_elementwise"]
        i6m = i6["max"] if i6 else float("nan")
        print(f"{prop:<9}{i1:>11.2e}{i2:>11.2e}{i4:>13.2e}{i5:>11.2e}{i6m:>11.2e}")
    print("\n=== VERDICT (authoritative production-blend construction test) ===")
    for prop, v in verdict.items():
        print(f"  {prop:<9} {v['classification']:<34} "
              f"prodblend_vs_pmf_max={v['prod_blend_vs_pmf_max']:.2e} "
              f"rows>1e-6={v['prod_blend_vs_pmf_rows_gt_1e-6']:<5} "
              f"ser_mean_max={v['inv1_mixture_mean_serialization_max']:.2e}")
    print("\nWrote:", os.path.join(FREEZE_DIR, "ARTIFACT_MANIFEST.json"))
    print("Wrote:", AUDIT_OUT)


if __name__ == "__main__":
    main()
