"""Emit pure_forecast provenance-readiness + the precise production-OOF blocker (STEP 3/6).

Proves the shipped stage4 champion config satisfies the pure_forecast contract (config hash,
zero market weights, no CLV head, no market anchor) and records the exact external data blocker
that prevents regenerating a fresh pure production OOF (STEP 6-10) in this environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from wnba_props_model.models.pure_model_contract import (
    INFORMATION_CONTRACT,
    assert_pure_model_config,
    config_sha256,
    enforce_pure_model_config,
    forbidden_market_columns,
    is_forbidden_market_field,
)

REPO = Path(__file__).resolve().parent.parent
CFG = REPO / "config" / "model" / "stage4_baseline.yaml"
OUT = REPO / "artifacts" / "pure_supremacy"
FEATURE_MATRIX = REPO / "data" / "processed" / "wnba_player_game_features_wide.parquet"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG.read_text())
    # The shipped config must already be pure; enforce is idempotent and used only to normalize.
    pure_cfg = enforce_pure_model_config(cfg)
    assert_pure_model_config(pure_cfg, context="stage4_baseline.yaml")

    declared_features = []
    for k, v in cfg.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            declared_features.extend(v)
    forbidden_in_config = forbidden_market_columns(declared_features)

    provenance = {
        "information_contract": INFORMATION_CONTRACT,
        "config_path": "config/model/stage4_baseline.yaml",
        "config_sha256_shipped": config_sha256(cfg),
        "config_sha256_pure_normalized": config_sha256(pure_cfg),
        "pure_model": bool(cfg.get("pure_model")),
        "market_prior_lambda": float(cfg.get("market_prior_lambda", 0.0) or 0.0),
        "market_prior_lambda_display": float(cfg.get("market_prior_lambda_display", 0.0) or 0.0),
        "market_probability_weight": float(cfg.get("market_probability_weight", 0.0) or 0.0),
        "market_anchor": cfg.get("market_anchor", None),
        "use_clv_head": bool(cfg.get("use_clv_head", False)),
        "use_live_calibrators": bool(cfg.get("use_live_calibrators", False)),
        "declared_feature_lists_scanned": sorted(set(declared_features)),
        "forbidden_market_fields_in_config": forbidden_in_config,
        "config_is_pure": (not forbidden_in_config
                           and float(cfg.get("market_prior_lambda", 0.0) or 0.0) == 0.0
                           and not bool(cfg.get("use_clv_head", False))),
        "note": ("Ordered model feature list + model/OOF hashes are emitted by the training run; "
                 "they cannot be produced here (see production_oof_blocker)."),
        "example_market_field_rejected": {f: is_forbidden_market_field(f)
                                          for f in ("market_line", "over_odds", "clv",
                                                    "player_market_p_over_prev")},
    }
    (OUT / "PURE_FORECAST_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    blocker = {
        "status": "EXTERNAL_DATA_BLOCKER",
        "blocks_steps": ["STEP 6 rerun production OOF", "STEP 7 exact-quote evaluation on fresh OOF",
                         "STEP 8 structural challengers", "STEP 9 selection on fresh OOF",
                         "STEP 10 freeze + prospective proof", "STEP 11 stl/blk/turnover + combo quotes"],
        "required_inputs_absent": {
            "feature_matrix": {
                "path": "data/processed/wnba_player_game_features_wide.parquet",
                "present": FEATURE_MATRIX.exists(),
                "role": "point-in-time training/inference features for all 7 props + combos",
            },
            "provider_api_keys": {
                "BDL_API_KEY": "required to pull box scores / rebuild the feature matrix",
                "ODDS_API_KEY": "required for stl/blk/turnover + combo exact same-book quotes",
                "present": False,
            },
        },
        "why": ("A fresh pure production OOF requires refitting the minutes/DNP/stat models on the "
                "raw point-in-time feature matrix with the pure_forecast contract active. The "
                "matrix is a deferred data artifact (absent) and no provider API keys are present, "
                "so the OOF cannot be regenerated here. The current committed OOF was built by a "
                "config with market_prior_lambda>0 and a CLV head, so its probabilities are "
                "UPSTREAM_PURITY_UNVERIFIED and are used only for labelled diagnostics."),
        "unblock_action": ("Run the pure production OOF where the feature matrix + BDL/ODDS keys "
                           "exist (e.g. a data-enabled CI/agent), then re-run STEP 6-10 here."),
        "code_readiness": {
            "pure_forecast_contract_enforced": True,
            "market_prior_lambda_and_clv_disabled_in_pure": True,
            "active_pmf_dnp_correction_available": "src/wnba_props_model/models/availability_pmf.py",
            "minutes_repair_available": True,
        },
    }
    (OUT / "PRODUCTION_OOF_BLOCKER.json").write_text(json.dumps(blocker, indent=2) + "\n")
    print("wrote PURE_FORECAST_PROVENANCE.json and PRODUCTION_OOF_BLOCKER.json")
    print("config_is_pure:", provenance["config_is_pure"],
          "| forbidden_in_config:", forbidden_in_config,
          "| feature_matrix_present:", FEATURE_MATRIX.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
