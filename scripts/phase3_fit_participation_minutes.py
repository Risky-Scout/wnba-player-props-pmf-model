#!/usr/bin/env python3
"""Offline Phase-3 challenger fit; never changes the production pointer."""

# ruff: noqa: B008

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.bundle import load_bundle
from wnba_props_model.sharp_v6.models import minutes_pmf_rows, predict_stat_atoms
from wnba_props_model.sharp_v6.phase3_labels import (
    aggregate_label_audit,
    injury_conditioned_training_cohort,
    revalidate_participation_labels,
)
from wnba_props_model.sharp_v6.phase3_minutes import (
    Phase3MinutesModel,
    persist_minutes_artifacts,
    pmf_metrics,
)
from wnba_props_model.sharp_v6.phase3_minutes import (
    select_and_persist as select_minutes,
)
from wnba_props_model.sharp_v6.phase3_participation import (
    persist_participation_artifacts,
)
from wnba_props_model.sharp_v6.phase3_participation import (
    select_and_persist as select_participation,
)

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]

# Frozen before comparison — do not alter after viewing results.
FROZEN_TOLERANCES = {
    "artifact": "DOWNSTREAM_TOLERANCES",
    "frozen_before_comparison": True,
    "defined_at_utc": "2026-08-02T07:21:00+00:00",
    "aggregate_direct_stat_nll_abs_tolerance": 0.02,
    "per_stat_nll_abs_tolerance": 0.05,
    "subgroup_nll_abs_tolerance": 0.12,
    "pmf_sum_abs_error_max": 1e-6,
    "inactive_label_revalidation_max_reject_rate": 0.02,
    "shadow_expected_minutes_abs_diff": 4.0,
    "shadow_minutes_sd_abs_diff": 3.0,
    "shadow_prop_probability_abs_diff": 0.05,
    "note": "Direct-stat models are loaded from v1.1 and never refit in Phase 3.",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"



@app.command()
def main(
    labels: Path = typer.Option(REPO / "data/processed/phase2/wnba_participation_labels.parquet"),
    features: Path = typer.Option(
        REPO / "data/recovered_v2/modeling/wnba_pregame_features_t12.parquet"
    ),
    stats: Path = typer.Option(REPO / "data/processed/phase2/wnba_player_game_stats.parquet"),
    wide_features: Path = typer.Option(
        REPO / "data/processed/wnba_player_game_features_wide.parquet"
    ),
    targets: Path = typer.Option(
        REPO / "data/recovered_v2/modeling/wnba_player_targets.parquet"
    ),
    cond_minutes: Path = typer.Option(
        REPO / "data/processed/phase2/conditional_minutes_training.parquet"
    ),
    out_dir: Path = typer.Option(REPO / "artifacts/sharp_v6_phase3"),
    bundle_out: Path = typer.Option(REPO / "artifacts/releases/wnba-pmf-production-v1.2-rc1"),
    control_bundle: Path = typer.Option(REPO / "artifacts/releases/wnba-pmf-production-v1.1"),
    max_year: int = typer.Option(2025),
    skip_downstream: bool = typer.Option(False),
) -> None:
    """Fit development data through 2025 and emit a non-production challenger."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve previously frozen tolerances file if present; otherwise write frozen constants.
    tol_path = out_dir / "DOWNSTREAM_TOLERANCES.json"
    if not tol_path.exists():
        _write_json(tol_path, FROZEN_TOLERANCES)
    else:
        # Do not mutate after freeze.
        existing = json.loads(tol_path.read_text())
        if not existing.get("frozen_before_comparison", False):
            _write_json(tol_path, FROZEN_TOLERANCES)
    tolerances = json.loads(tol_path.read_text())

    raw_labels = pd.read_parquet(labels)
    feature_frame = pd.read_parquet(features)
    # Prefer phase2 repaired stats; fall back to wide features minutes.
    if stats.exists():
        stat_frame = pd.read_parquet(stats)
    else:
        stat_frame = pd.read_parquet(wide_features)
    checked, audit, rejects = revalidate_participation_labels(raw_labels, stat_frame)
    # Private row-level outputs stay local / gitignored.
    checked.to_parquet(out_dir / "phase3_revalidated_labels_private.parquet", index=False)
    rejects.to_csv(out_dir / "LABEL_REVALIDATION_REJECTS.csv", index=False)
    _write_json(out_dir / "LABEL_REVALIDATION_AUDIT.json", audit)
    cohort = injury_conditioned_training_cohort(checked)
    aggregate_label_audit(cohort).to_csv(out_dir / "PARTICIPATION_TRAINING_COHORT.csv", index=False)

    train = cohort.merge(
        feature_frame, on=["game_id", "player_id"], how="inner", suffixes=("", "_feature")
    )
    if "game_date" not in train.columns and "game_date_feature" in train.columns:
        train["game_date"] = train["game_date_feature"]

    participation = select_participation(train, max_year=max_year)
    persist_participation_artifacts(participation, out_dir)

    minutes_frame = pd.read_parquet(cond_minutes)
    if "participation_label_class" not in minutes_frame.columns:
        minutes_frame["participation_label_class"] = "CONFIRMED_ACTIVE"
    if "training_eligible" not in minutes_frame.columns:
        minutes_frame["training_eligible"] = True
    minutes_model, minutes_metrics, minutes_oof, minutes_report = select_minutes(
        minutes_frame, max_year=max_year
    )
    persist_minutes_artifacts(minutes_model, minutes_metrics, minutes_oof, minutes_report, out_dir)

    # Downstream PMF comparison using v1.1 direct-stat models (never refit).
    downstream_status = "SKIPPED"
    if not skip_downstream and control_bundle.exists():
        downstream_status = _run_downstream_comparison(
            control_bundle=control_bundle,
            minutes_model=minutes_model,
            minutes_frame=minutes_frame,
            feature_frame=feature_frame,
            wide_features=wide_features,
            targets_path=targets,
            out_dir=out_dir,
            tolerances=tolerances,
            max_year=max_year,
        )

    _package_challenger(
        bundle_out=bundle_out,
        control_bundle=control_bundle,
        out_dir=out_dir,
        participation=participation,
        minutes_model=minutes_model,
        minutes_report=minutes_report,
        downstream_status=downstream_status,
    )
    typer.echo(f"Phase-3 challenger artifacts written to {out_dir} / {bundle_out}")


def _run_downstream_comparison(
    *,
    control_bundle: Path,
    minutes_model: Phase3MinutesModel,
    minutes_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    wide_features: Path,
    targets_path: Path,
    out_dir: Path,
    tolerances: dict,
    max_year: int,
) -> str:
    bundle = load_bundle(control_bundle)
    # Build evaluation slate: confirmed-active rows through max_year with targets.
    mf = minutes_frame.copy()
    mf["game_date"] = pd.to_datetime(mf["game_date"])
    mf = mf[mf["game_date"].dt.year <= max_year].reset_index(drop=True)
    keys = ["game_id", "player_id"]
    # Prefer modeling targets parquet; fall back to wide features.
    target_src = targets_path if targets_path.exists() else wide_features
    if target_src.exists():
        wide = pd.read_parquet(target_src)
        tcols = [
            c
            for c in (
                "pts",
                "reb",
                "ast",
                "fg3m",
                "stl",
                "blk",
                "turnover",
                "actual_minutes",
                "game_date",
                "team_id",
            )
            if c in wide.columns
        ]
        mf = mf.drop(
            columns=[c for c in tcols if c in mf.columns and c not in keys], errors="ignore"
        )
        mf = mf.merge(wide[keys + tcols], on=keys, how="inner", suffixes=("", "_w"))
    # Need feature columns for control minutes + stats.
    feat_cols = sorted(
        set(bundle.minutes.feature_cols)
        | set(minutes_model.feature_cols)
        | {c for s in bundle.stats.values() for c in s.feature_cols}
    )
    present = [c for c in feat_cols if c in feature_frame.columns]
    slate = mf.merge(feature_frame[keys + present], on=keys, how="inner")
    if slate.empty:
        comparison = pd.DataFrame([{"status": "NO_OVERLAP", "direct_stat_models_refit": False}])
        comparison.to_csv(out_dir / "DOWNSTREAM_PMF_COMPARISON.csv", index=False)
        return "NO_OVERLAP"

    # Sample for tractability if huge.
    if len(slate) > 4000:
        slate = slate.sample(4000, random_state=20260730).reset_index(drop=True)

    control_atoms = minutes_pmf_rows(bundle.minutes, slate, reconcile_teams=False, mode="research")
    chall_atoms = list(minutes_model.pmf(slate))

    stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    rows = []
    worst = []
    for stat in stats:
        if stat not in bundle.stats or stat not in slate.columns:
            continue
        y = pd.to_numeric(slate[stat], errors="coerce").fillna(0).to_numpy(float)
        ctrl = predict_stat_atoms(bundle.stats[stat], slate, control_atoms)
        chal = predict_stat_atoms(bundle.stats[stat], slate, chall_atoms)
        # predict_stat_atoms returns list[(atoms, overflow)]
        ctrl_pmf = np.vstack([a for a, _ in ctrl])
        chal_pmf = np.vstack([a for a, _ in chal])
        # Pad to common width
        width = max(ctrl_pmf.shape[1], chal_pmf.shape[1], int(y.max()) + 1)

        def _pad(a, w=width):
            if a.shape[1] >= w:
                return a[:, :w]
            out = np.zeros((a.shape[0], w))
            out[:, : a.shape[1]] = a
            return out

        ctrl_pmf, chal_pmf = _pad(ctrl_pmf), _pad(chal_pmf)
        m_ctrl = pmf_metrics(ctrl_pmf, y)
        m_chal = pmf_metrics(chal_pmf, y)
        rows.append(
            {
                "stat": stat,
                "n": len(y),
                "control_nll": m_ctrl["nll"],
                "challenger_nll": m_chal["nll"],
                "delta_nll": m_chal["nll"] - m_ctrl["nll"],
                "control_crps": m_ctrl["crps"],
                "challenger_crps": m_chal["crps"],
                "control_mae": m_ctrl["mae"],
                "challenger_mae": m_chal["mae"],
                "challenger_variance_bias": m_chal["variance_bias"],
            }
        )
        # Crude worst game-date fold
        dates = pd.to_datetime(slate["game_date"])
        for d, idx in slate.groupby(dates.dt.normalize()).groups.items():
            ii = list(idx)
            if len(ii) < 15:
                continue
            mc = pmf_metrics(chal_pmf[ii], y[ii])
            worst.append({"stat": stat, "game_date": str(d.date()), "n": len(ii), "nll": mc["nll"]})

    by_stat = pd.DataFrame(rows)
    by_stat.to_csv(out_dir / "DOWNSTREAM_METRICS_BY_STAT.csv", index=False)
    worst_df = pd.DataFrame(worst)
    if not worst_df.empty:
        worst_df.sort_values("nll", ascending=False).head(50).to_csv(
            out_dir / "DOWNSTREAM_WORST_FOLD.csv", index=False
        )
    else:
        pd.DataFrame(columns=["stat", "game_date", "n", "nll"]).to_csv(
            out_dir / "DOWNSTREAM_WORST_FOLD.csv", index=False
        )

    agg_delta = float(by_stat["delta_nll"].mean()) if not by_stat.empty else float("nan")
    tol = float(tolerances.get("aggregate_direct_stat_nll_abs_tolerance", 0.02))
    per_tol = float(tolerances.get("per_stat_nll_abs_tolerance", 0.05))
    per_ok = bool((by_stat["delta_nll"] <= per_tol).all()) if not by_stat.empty else False
    agg_ok = bool(agg_delta <= tol) if np.isfinite(agg_delta) else False
    status = "PASS" if agg_ok and per_ok else "FAIL"
    comparison = pd.DataFrame(
        [
            {
                "control_bundle": str(control_bundle),
                "challenger_minutes_family": minutes_model.family,
                "direct_stat_models_refit": False,
                "aggregate_delta_nll": agg_delta,
                "aggregate_tolerance": tol,
                "per_stat_tolerance": per_tol,
                "status": status,
            }
        ]
    )
    comparison.to_csv(out_dir / "DOWNSTREAM_PMF_COMPARISON.csv", index=False)
    by_stat.to_csv(out_dir / "DOWNSTREAM_PMF_COMPARISON_BY_STAT.csv", index=False)
    return status


def _package_challenger(
    *,
    bundle_out: Path,
    control_bundle: Path,
    out_dir: Path,
    participation,
    minutes_model: Phase3MinutesModel,
    minutes_report: dict,
    downstream_status: str,
) -> None:
    if bundle_out.exists():
        shutil.rmtree(bundle_out)
    bundle_out.mkdir(parents=True)

    # Copy feature contracts from control for direct-stat continuity.
    ctrl_contracts = control_bundle / "FEATURE_CONTRACTS.json"
    if ctrl_contracts.exists():
        contracts = json.loads(ctrl_contracts.read_text())
    else:
        contracts = {}
    contracts["participation_phase3"] = participation.feature_cols
    contracts["minutes_phase3"] = minutes_model.feature_cols
    _write_json(bundle_out / "FEATURE_CONTRACTS.json", contracts)

    for name in (
        "PARTICIPATION_APPLICABILITY_CONTRACT.json",
        "MINUTES_MODEL_CONTRACT.json",
        "DOWNSTREAM_PMF_COMPARISON.csv",
    ):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, bundle_out / name)

    with (bundle_out / "PARTICIPATION_MODEL").open("wb") as f:
        pickle.dump(
            {
                "family": participation.family,
                "model": participation.model,
                "feature_cols": participation.feature_cols,
                "feature_hash": participation.feature_hash,
            },
            f,
        )
    with (bundle_out / "PARTICIPATION_CALIBRATOR").open("wb") as f:
        pickle.dump(
            {
                "method": participation.calibration_method,
                "calibrator": participation.calibrator,
                "report": participation.calibration_report,
            },
            f,
        )
    with (bundle_out / "MINUTES_MODEL").open("wb") as f:
        pickle.dump(minutes_model, f)

    manifest = {
        "artifact": "MANIFEST",
        "bundle_id": "wnba-pmf-production-v1.2-rc1",
        "status": "CHALLENGER_NOT_PROMOTED",
        "code_sha": _code_sha(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": "2025-10-31",
        "oof_periods": "chronological_through_2025",
        "participation_model_hash": hashlib.sha256(
            pickle.dumps(participation.model, protocol=4)
        ).hexdigest(),
        "participation_calibrator_hash": hashlib.sha256(
            pickle.dumps(participation.calibrator, protocol=4)
        ).hexdigest(),
        "minutes_model_hash": hashlib.sha256(pickle.dumps(minutes_model, protocol=4)).hexdigest(),
        "feature_hashes": {
            "participation": participation.feature_hash,
            "minutes": hashlib.sha256(",".join(minutes_model.feature_cols).encode()).hexdigest()[
                :16
            ],
        },
        "applicability_limitations": participation.applicability_contract,
        "unsupported_availability_states": participation.applicability_contract.get(
            "unsupported_statuses", []
        ),
        "rollback_bundle": "artifacts/releases/wnba-pmf-production-v1.1",
        "downstream_status": downstream_status,
        "minutes_family": minutes_model.family,
        "minutes_report": minutes_report,
        "production_pointer_updated": False,
        "direct_stat_models": "inherited_from_v1.1_not_refit",
    }
    _write_json(bundle_out / "MANIFEST.json", manifest)
    (bundle_out / "MODEL_CARD.md").write_text(
        "\n".join(
            [
                "# WNBA PMF Production v1.2-rc1 (Phase-3 Challenger)",
                "",
                "Challenger bundle for injury-conditioned participation + conditional-active minutes.",
                "Does **not** claim unconditional P(active | all roster-eligible players).",
                "NOT_LISTED is an operational gate, not a historically calibrated probability.",
                f"Minutes family: `{minutes_model.family}`.",
                f"Participation family: `{participation.family}` / calibrator `{participation.calibration_method}`.",
                f"Downstream gate: `{downstream_status}`.",
                "Rollback: `artifacts/releases/wnba-pmf-production-v1.1`.",
                "Production pointer is not updated by this package step.",
                "",
            ]
        )
    )
    sums = [f"{_sha(p)}  {p.name}" for p in sorted(bundle_out.iterdir()) if p.name != "SHA256SUMS"]
    (bundle_out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    app()
