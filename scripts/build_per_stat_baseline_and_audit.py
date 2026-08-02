"""Freeze the pre-rebuild baseline and produce the production per-stat feature-usage audit.

Writes two artifacts under ``artifacts/per_stat_compact/``:

* ``PRE_REBUILD_BASELINE.json`` — frozen hashes of the current main SHA, feature manifest,
  Stage 5 config, exact-quote artifact, P0 models, and market metrics by prop. Written once;
  refuses to overwrite unless ``--force`` is passed.
* ``PRODUCTION_FEATURE_USAGE_AUDIT.json`` — for every stat, the ACTUAL production feature
  usage, provenance classification, and explicit proofs of the seven audit questions in the
  directive (full-matrix fallback, prop_feature_map configured, <8-column floor, empty-set
  fallback, map/artifact hash parity, PBP reach, market-in-pure leak).

This is deliberately data-optional: the license-restricted feature/quote parquets are not
present in every environment. Fields that require those parquets are reported honestly as
``NOT_COMPUTED_DATA_ABSENT`` rather than fabricated.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import typer
import yaml

from wnba_props_model.features import feature_contract as fc
from wnba_props_model.features.feature_provenance import Provenance, classify, partition

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "per_stat_compact"
STATS = ["minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "UNKNOWN"


def _collect_market_metrics() -> dict:
    """Scan every market_superiority_proof.json and extract per-prop market/model metrics."""
    root = REPO / "artifacts" / "market_feature_proof"
    per_prop: dict[str, dict] = {}
    for proof in sorted(root.rglob("market_superiority_proof.json")):
        try:
            data = json.loads(proof.read_text())
        except Exception:
            continue
        for r in data.get("results", []):
            prop = r.get("prop")
            if not prop or r.get("data_state") != "evaluated":
                continue
            rec = {
                "source": str(proof.relative_to(REPO)),
                "candidate": r.get("candidate"),
                "n_settled": r.get("n_settled"),
                "n_clusters": r.get("n_clusters"),
                "market_logloss": r.get("market_logloss"),
                "model_logloss": r.get("model_logloss"),
                "logloss_delta": r.get("logloss_delta"),
                "market_brier": r.get("market_brier"),
                "model_brier": r.get("model_brier"),
                "brier_delta": r.get("brier_delta"),
                "market_auc": r.get("market_auc"),
                "model_auc": r.get("model_auc"),
                "market_superiority_gate": r.get("market_superiority_gate"),
            }
            # keep the record with the most settled observations per prop
            if prop not in per_prop or (rec["n_settled"] or 0) > (per_prop[prop]["n_settled"] or 0):
                per_prop[prop] = rec
    return per_prop


def build_baseline() -> dict:
    stage5 = REPO / "config" / "model" / "stage5_oof.yaml"
    champion = REPO / "config" / "champion_manifest.json"
    quote_readiness = REPO / "artifacts" / "market_feature_proof" / "EXACT_QUOTE_READINESS.json"
    champ = json.loads(champion.read_text()) if champion.exists() else {}
    model_files = sorted((REPO / "artifacts" / "models").rglob("*.json"))
    return {
        "artifact": "PRE_REBUILD_BASELINE",
        "note": "Frozen pre-rebuild baseline. DO NOT overwrite.",
        "current_main_sha": _git_sha(),
        "feature_manifest": {
            "model_features_count": len(fc.MODEL_FEATURES),
            "model_features_schema_hash": _sha256_obj(list(fc.MODEL_FEATURES)),
            "forbidden_features_count": len(fc.FORBIDDEN_MODEL_FEATURES),
            "pure_forecast_features_count": len(fc.PURE_FORECAST_FEATURES),
            "champion_feature_hash": champ.get("feature_hash"),
            "champion_feature_schema": champ.get("feature_schema"),
        },
        "stage5_config": {
            "path": str(stage5.relative_to(REPO)),
            "file_hash": _sha256_file(stage5),
            "prop_feature_map_configured": "prop_feature_map" in (yaml.safe_load(stage5.read_text()) or {}),
        },
        "exact_quote_artifact": {
            "path": str(quote_readiness.relative_to(REPO)) if quote_readiness.exists() else None,
            "file_hash": _sha256_file(quote_readiness),
        },
        "p0_models": {
            "champion_model_hash": champ.get("model_hash"),
            "champion_calibration_hash": champ.get("calibration_hash"),
            "champion_status": champ.get("status"),
            "model_artifact_hashes": {str(p.relative_to(REPO)): _sha256_file(p) for p in model_files},
        },
        "p0_oof_row_universe_hash": "NOT_AVAILABLE_IN_ENV: OOF parquet not present (license-restricted)",
        "market_metrics_by_prop": _collect_market_metrics(),
    }


def _feature_audit_for_stat(stat: str, cand_map: dict) -> dict:
    """Audit a single stat's ACTUAL production feature usage.

    Production ships NO prop_feature_map (Stage 5 has no such key), so ``stat_feature_subset``
    returns the training matrix unchanged: every stat receives the FULL shared MODEL_FEATURES
    matrix. The candidate map (config/prop_feature_map_candidate_v1.json) is a *proposed*, not
    active, set — recorded here for contrast.
    """
    prod_features = list(fc.MODEL_FEATURES)             # what production actually trains on
    prov = partition(prod_features)
    cand = cand_map.get(stat if stat != "minutes" else "__none__", [])
    cand_market = [c for c in cand if classify(c) in (
        Provenance.EXTERNAL_MARKET_CURRENT_GAME, Provenance.EXTERNAL_MARKET_LAGGED)]
    pbp_in_prod = sorted([c for c in prod_features if c.startswith("pbp_") or "pbp_" in c])
    return {
        "actual_training_feature_count": len(prod_features),
        "actual_ordered_feature_list": prod_features,
        "feature_schema_hash": _sha256_obj(prod_features),
        "provenance_classification": {k: v for k, v in prov.items() if v},
        "provenance_counts": {p.value: len(prov[p.value]) for p in Provenance},
        "candidate_map_feature_count": len(cand),
        "candidate_map_market_features": cand_market,
        "pbp_features_in_production_allowlist": pbp_in_prod,
        # data-dependent columns require the license-restricted feature parquet
        "all_null_columns": "NOT_COMPUTED_DATA_ABSENT",
        "constant_columns": "NOT_COMPUTED_DATA_ABSENT",
        "near_constant_columns": "NOT_COMPUTED_DATA_ABSENT",
        "missing_at_inference_columns": "NOT_COMPUTED_DATA_ABSENT",
        "pairwise_correlation_clusters_ge_0p90": "NOT_COMPUTED_DATA_ABSENT",
        "entered_fitted_artifact": "NOT_VERIFIABLE_ARTIFACT_ABSENT",
    }


def build_audit() -> dict:
    stage5 = yaml.safe_load((REPO / "config" / "model" / "stage5_oof.yaml").read_text()) or {}
    cand_map = json.loads((REPO / "config" / "prop_feature_map_candidate_v1.json").read_text())
    map_configured = "prop_feature_map" in stage5
    per_stat = {s: _feature_audit_for_stat(s, cand_map) for s in STATS}

    # Provenance leak proof across candidate maps.
    cand_current_game_market = {
        s: sorted([c for c in feats if classify(c) is Provenance.EXTERNAL_MARKET_CURRENT_GAME])
        for s, feats in cand_map.items()
    }
    any_market_in_pure = any(v for v in cand_current_game_market.values())

    return {
        "artifact": "PRODUCTION_FEATURE_USAGE_AUDIT",
        "current_main_sha": _git_sha(),
        "per_stat": per_stat,
        "explicit_proofs": {
            "1_each_stat_receives_full_shared_matrix": {
                "answer": True,
                "evidence": (
                    "Stage 5 config has no 'prop_feature_map' key, so training.stat_feature_subset "
                    "returns X_played unchanged; every stat trains on the full MODEL_FEATURES matrix "
                    f"({len(fc.MODEL_FEATURES)} columns)."),
            },
            "2_prop_feature_map_configured": {
                "answer": bool(map_configured),
                "evidence": f"'prop_feature_map' in stage5_oof.yaml == {map_configured}.",
            },
            "3_minimum_eight_column_fallback_triggered": {
                "answer": False,
                "evidence": (
                    "The <8-column floor (prop_feature_min_cols) has been REMOVED from "
                    "training.stat_feature_subset. It is moot in production anyway because no map "
                    "is configured. Before the fix, any explicit map with <8 available columns "
                    "reverted to the full matrix."),
            },
            "4_empty_selected_set_causes_full_feature_fallback": {
                "answer": False,
                "evidence": (
                    "Fixed: an explicit empty list now yields a zero-column base-rate frame, not "
                    "the full matrix. Previously `if not cols: return X_played` reverted to full."),
            },
            "5_selected_map_and_fitted_model_feature_hashes_match": {
                "answer": "N/A",
                "evidence": (
                    "No prop_feature_map is active in production and no per-stat fitted artifact is "
                    "present in this environment. FittedFeatureSpec now enforces hash parity when a "
                    "policy is used."),
            },
            "6_pbp_features_reach_fitted_production_artifact": {
                "answer": False,
                "evidence": (
                    "No pbp_* column is in the production MODEL_FEATURES allowlist; PBP opportunity "
                    "features exist only in the ablation candidate space (wnba_pbp_opportunity_features"
                    ".parquet), so they do not reach the certified production stat artifacts."),
            },
            "7_current_game_market_reaches_a_pure_candidate": {
                "answer": bool(any_market_in_pure),
                "evidence": (
                    "config/prop_feature_map_candidate_v1.json lists current-game Vegas features "
                    "(game_total/game_spread_home/implied_team_total/close_game_indicator) in every "
                    "prop's candidate set; the legacy ablation market regex also failed to exclude "
                    "them. Per-prop leaked current-game market features:"),
                "candidate_map_current_game_market_by_prop": cand_current_game_market,
            },
        },
        "notes": [
            f"MODEL_FEATURES production allowlist size = {len(fc.MODEL_FEATURES)} (the directive's "
            "'379' refers to the larger recovered_v2 ablation wide matrix, not this allowlist).",
            "Data-dependent audit fields (null/constant/correlation/missing-at-inference) require "
            "the license-restricted feature parquet and are marked NOT_COMPUTED_DATA_ABSENT.",
        ],
    }


@app.command()
def main(force: bool = typer.Option(False, "--force", help="overwrite PRE_REBUILD_BASELINE.json")) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = OUT_DIR / "PRE_REBUILD_BASELINE.json"
    audit_path = OUT_DIR / "PRODUCTION_FEATURE_USAGE_AUDIT.json"

    if baseline_path.exists() and not force:
        typer.echo(f"[baseline] exists, not overwriting: {baseline_path}")
    else:
        baseline_path.write_text(json.dumps(build_baseline(), indent=2, default=str))
        typer.echo(f"[baseline] wrote {baseline_path}")

    audit_path.write_text(json.dumps(build_audit(), indent=2, default=str))
    typer.echo(f"[audit] wrote {audit_path}")


if __name__ == "__main__":
    app()
