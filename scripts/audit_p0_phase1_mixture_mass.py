#!/usr/bin/env python3
"""Read-only historical mixture-mass audit: repaired math vs defective legacy mix_atoms.

Does not refit models. Uses v1.1 fitted components + private historical features when present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wnba_props_model.sharp_v6.bundle import load_bundle  # noqa: E402
from wnba_props_model.sharp_v6.contracts import EMERGENCY_CAP  # noqa: E402
from wnba_props_model.sharp_v6.models import (  # noqa: E402
    mix_atoms,
    minutes_pmf_rows,
    predict_stat_atoms,
    predict_stat_distribution,
)


def _legacy_mix_atoms(lam: float, r: float | None, matoms: np.ndarray, K: int):
    """Exact pre-repair mix_atoms (for audit only)."""
    idx = np.where(matoms > 1e-4)[0]
    w = matoms[idx]
    dropped = float(matoms.sum() - w.sum()) if matoms.size else 0.0
    if w.size == 0:
        return np.zeros(K + 1), 1.0, dropped, 0.0
    means = np.clip(lam * idx, 1e-6, None)
    k = np.arange(K + 1)
    if r is None or (isinstance(r, float) and np.isnan(r)):
        comp = poisson.pmf(k[:, None], means[None, :])
    else:
        p = r / (r + means)
        comp = nbinom.pmf(k[:, None], r, p[None, :])
    atoms = comp @ w
    s = float(atoms.sum())
    atoms_renorm = atoms / s if s > 0 else atoms
    ovf = float(max(0.0, 1.0 - s))
    return atoms_renorm, ovf, dropped, s


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default="artifacts/releases/wnba-pmf-production-v1.1")
    ap.add_argument("--features", default="data/recovered_v2/wnba_player_game_features_long.parquet")
    ap.add_argument("--out-dir", default="artifacts/sharp_v6_p0_phase1")
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--dry-slate", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(REPO / args.bundle_dir)
    feat_path = REPO / args.features
    if not feat_path.exists():
        raise SystemExit(f"FAIL: private features missing at {feat_path}")

    df = pd.read_parquet(feat_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Completed historical rows through training cutoff; sample for audit speed.
    cutoff = pd.Timestamp(bundle.meta.get("manifest", {}).get("training_cutoff", "2025-10-31"))
    hist = df[df["game_date"] <= cutoff].copy()
    hist = hist[hist.get("actual_minutes", hist.get("minutes", 1)).fillna(0) >= 0]
    if len(hist) > args.max_rows:
        hist = hist.sort_values("game_date").iloc[:: max(1, len(hist) // args.max_rows)].head(args.max_rows)

    # Need minutes atoms per row — use fitted minutes model.
    # Build a compact slate-like frame.
    required = ["game_id", "player_id", "team_id", "game_date"]
    for c in required:
        if c not in hist.columns:
            raise SystemExit(f"FAIL: missing column {c}")

    matoms = minutes_pmf_rows(bundle.minutes, hist, reconcile_teams=False, mode="research")

    rows = []
    worst_norm = 0.0
    worst_dropped = 0.0
    worst_mean_diff = 0.0
    worst_var_diff = 0.0
    worst_line_diff = 0.0
    n_gt = {0.0001: 0, 0.001: 0, 0.01: 0}

    stats = [s for s in ("pts", "reb", "ast", "fg3m", "stl", "blk", "turnover") if s in bundle.stats]
    for stat in stats:
        model = bundle.stats[stat]
        repaired = predict_stat_atoms(model, hist, matoms)
        # Legacy comparison on the NB2 mixture layer (pre-hurdle) via rate predictions.
        from wnba_props_model.sharp_v6.models import _frame_features, role_band

        X = _frame_features(hist, model.feature_cols, mode="research").to_numpy(float)
        lam = np.clip(model.rate_regressor.predict(X), 1e-6, None)
        bands = role_band(hist)
        K = EMERGENCY_CAP.get(stat, 60)
        for i in range(len(hist)):
            r = model.r_by_band.get(int(bands[i]), model.r_by_band.get("__global__"))
            r_i = None if r is None else float(r)
            old_a, old_ovf, dropped, s_before = _legacy_mix_atoms(float(lam[i]), r_i, matoms[i], K)
            new_a, new_ovf = mix_atoms(float(lam[i]), r_i, matoms[i], K)
            old_mass = float(old_a.sum()) + old_ovf
            new_mass = float(new_a.sum()) + new_ovf
            old_norm_err = abs(old_mass - 1.0)
            new_norm_err = abs(new_mass - 1.0)
            worst_norm = max(worst_norm, new_norm_err)
            worst_dropped = max(worst_dropped, dropped)

            kk = np.arange(K + 1)
            old_mean = float(np.dot(kk, old_a))
            new_mean = float(np.dot(kk, new_a))
            old_var = float(np.dot((kk - old_mean) ** 2, old_a))
            new_var = float(np.dot((kk - new_mean) ** 2, new_a))
            mean_diff = abs(new_mean - old_mean)
            var_diff = abs(new_var - old_var)
            worst_mean_diff = max(worst_mean_diff, mean_diff)
            worst_var_diff = max(worst_var_diff, var_diff)

            # Line prob at mean±0.5
            line = max(0.5, round(new_mean * 2) / 2)
            def _over(a, ovf, L):
                k = np.arange(a.size)
                return float(a[k > L].sum()) + float(ovf)
            line_diff = abs(_over(new_a, new_ovf, line) - _over(old_a, old_ovf, line))
            worst_line_diff = max(worst_line_diff, line_diff)
            max_atom = float(np.max(np.abs(new_a - old_a[: new_a.size] if old_a.size >= new_a.size else np.pad(old_a, (0, new_a.size - old_a.size)))))
            for thr in n_gt:
                if max_atom > thr or line_diff > thr:
                    n_gt[thr] += 1

            # Also check full repaired predict_stat_atoms mass
            ra, ro = repaired[i]
            worst_norm = max(worst_norm, abs(float(ra.sum()) + ro - 1.0))

            if i < 200:  # keep a compact comparison sample
                rows.append({
                    "stat": stat,
                    "game_id": int(hist.iloc[i]["game_id"]),
                    "player_id": int(hist.iloc[i]["player_id"]),
                    "old_norm_error": old_norm_err,
                    "new_norm_error": new_norm_err,
                    "dropped_minutes_mass": dropped,
                    "old_s_before_renorm": s_before,
                    "mean_diff": mean_diff,
                    "var_diff": var_diff,
                    "line": line,
                    "line_prob_diff": line_diff,
                    "max_atom_diff": max_atom,
                })

    cmp = pd.DataFrame(rows)
    cmp_path = out_dir / "OLD_VS_REPAIRED_PROBABILITIES.csv"
    cmp.to_csv(cmp_path, index=False)

    # Tail/moment audit on repaired distributions
    moment_rows = []
    for stat in stats:
        model = bundle.stats[stat]
        dists = predict_stat_distribution(model, hist.head(200), matoms[:200])
        for i, d in enumerate(dists):
            m = d.materialize(required_max=EMERGENCY_CAP.get(stat, 60))
            moment_rows.append({
                "stat": stat,
                "family": m.distribution_family,
                "mean": d.mean(),
                "variance": d.variance(),
                "stored_mass": m.stored_mass,
                "overflow": m.overflow_probability,
                "normalization_error": m.normalization_error,
                "support_max": m.support_max,
            })
    moments = pd.DataFrame(moment_rows)
    moments_path = out_dir / "TAIL_AND_MOMENT_AUDIT.csv"
    moments.to_csv(moments_path, index=False)

    audit = {
        "artifact": "MIXTURE_MASS_AUDIT",
        "phase": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": args.bundle_dir,
        "bundle_model_sha256": bundle.meta.get("manifest", {}).get("model_sha256"),
        "features_path": str(feat_path.relative_to(REPO)),
        "features_sha256": _sha(feat_path),
        "n_rows_audited": int(len(hist) * len(stats)),
        "stats": stats,
        "worst_repaired_normalization_error": worst_norm,
        "worst_legacy_normalization_error_demo": float(cmp["old_norm_error"].max()) if len(cmp) else None,
        "largest_previous_dropped_minutes_mass": worst_dropped,
        "largest_mean_difference": worst_mean_diff,
        "largest_variance_difference": worst_var_diff,
        "largest_line_probability_difference": worst_line_diff,
        "rows_affected_gt": {str(k): int(v) for k, v in n_gt.items()},
        "max_repaired_moment_normalization_error": float(moments["normalization_error"].max()) if len(moments) else None,
        "note": "Parameter estimates unchanged (v1.1). Differences reflect mass/tail math repair only.",
    }
    (out_dir / "MIXTURE_MASS_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

    if args.dry_slate:
        # Lightweight dry inference on a recent historical date cluster (publication disabled).
        day = df["game_date"].max()
        slate = df[df["game_date"] == day].head(40).copy()
        dry: dict = {
            "slate_date": str(pd.Timestamp(day).date()),
            "n_players": int(len(slate)),
            "publication": "disabled",
        }
        try:
            from wnba_props_model.sharp_v6.inference import _core_pmf_delivery

            delivery = _core_pmf_delivery(
                prediction_timestamp=datetime.now(timezone.utc).isoformat(),
                slate=slate,
                bundle=bundle,
                stats=df,
                games_out=[{"game_id": int(g)} for g in slate["game_id"].unique()],
                mode="research",
            )
            dry["n_pmfs"] = int(len(delivery.player_pmfs))
            dry["n_atom_rows"] = int(len(delivery.atoms_frame))
            # Use full in-memory PMF atoms (delivery atom table omits near-zero atoms).
            errs = []
            for pmf in delivery.player_pmfs:
                s = float(np.sum(pmf.active_pmf_atoms))
                ovf = float(pmf.overflow_probability)
                errs.append(abs(s + ovf - 1.0))
            dry["worst_delivery_normalization_error"] = float(max(errs)) if errs else None
            dry["n_pmf_groups"] = int(len(errs))
            dry["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 — audit must record dry-run failure
            # Fallback: direct-stat repaired path only (still proves mass identity).
            dry["core_delivery_error"] = repr(exc)
            slate_matoms = minutes_pmf_rows(
                bundle.minutes, slate, reconcile_teams=False, mode="research"
            )
            errs = []
            n_pmfs = 0
            for stat in stats:
                for a, ovf in predict_stat_atoms(bundle.stats[stat], slate, slate_matoms):
                    errs.append(abs(float(a.sum()) + float(ovf) - 1.0))
                    n_pmfs += 1
            dry["n_pmfs"] = n_pmfs
            dry["worst_delivery_normalization_error"] = float(max(errs)) if errs else None
            dry["status"] = "fallback_direct_stat_path"
        (out_dir / "DRY_SLATE_INFERENCE.json").write_text(json.dumps(dry, indent=2) + "\n")
        print("DRY_SLATE", json.dumps(dry, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
