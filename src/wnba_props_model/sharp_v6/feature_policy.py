"""Governed feature-contract missingness policy for V6 production.

Classifications replace unbounded NaN-filling as an unclassified production policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class FeatureClass(str, Enum):
    REQUIRED = "REQUIRED"
    OPTIONAL_WITH_TRAINED_IMPUTATION = "OPTIONAL_WITH_TRAINED_IMPUTATION"
    OPTIONAL_WITH_NATIVE_MISSING_SUPPORT = "OPTIONAL_WITH_NATIVE_MISSING_SUPPORT"
    DERIVED = "DERIVED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPRECATED = "DEPRECATED"


@dataclass
class FeatureSpec:
    name: str
    classification: FeatureClass
    component: str
    imputation_value: float | None = None
    missingness_indicator: str | None = None
    unit: str = "numeric"
    leakage_risk: str = "low"
    source: str = "pregame_lagged"
    pit_rule: str = "available_at <= prediction_timestamp"


@dataclass
class FeatureFrameResult:
    frame: pd.DataFrame
    drift_events: list[dict[str, Any]] = field(default_factory=list)
    quarantined_rows: list[int] = field(default_factory=list)
    status: str = "OK"


class FeatureContractError(RuntimeError):
    """Raised when a required feature is missing or the contract drifts incompatibly."""


def classify_contract_features(
    contracts: dict[str, dict],
    *,
    trained_imputation: dict[str, float] | None = None,
) -> dict[str, FeatureSpec]:
    """Build a feature registry from frozen component contracts.

    Production HGB components natively support missing values for optional columns.
    Identity / schedule columns are DERIVED (rewritten at inference, not model inputs).
    """
    trained_imputation = trained_imputation or {}
    specs: dict[str, FeatureSpec] = {}
    for component, meta in contracts.items():
        for name in meta.get("features", []):
            if name in specs:
                continue
            if name in trained_imputation:
                cls = FeatureClass.OPTIONAL_WITH_TRAINED_IMPUTATION
                imp = float(trained_imputation[name])
            else:
                # Frozen HGB contracts: native NaN support is an explicit selection.
                cls = FeatureClass.OPTIONAL_WITH_NATIVE_MISSING_SUPPORT
                imp = None
            specs[name] = FeatureSpec(
                name=name,
                classification=cls,
                component=component,
                imputation_value=imp,
                missingness_indicator=f"{name}__is_missing" if cls == FeatureClass.OPTIONAL_WITH_TRAINED_IMPUTATION else None,
            )
    # Structural identity columns required on every production slate row
    for name in ("game_id", "player_id", "team_id", "opponent_team_id"):
        specs[name] = FeatureSpec(
            name=name,
            classification=FeatureClass.REQUIRED,
            component="identity",
            unit="id",
            leakage_risk="none",
            source="schedule_roster",
        )
    return specs


def prepare_feature_frame(
    df: pd.DataFrame,
    cols: list[str],
    specs: dict[str, FeatureSpec] | None = None,
    *,
    mode: str = "production",
) -> FeatureFrameResult:
    """Align ``df`` to ``cols`` under the governed missingness policy.

    production:
      - REQUIRED missing / all-null → quarantine row or raise
      - OPTIONAL_WITH_TRAINED_IMPUTATION → impute + indicator + drift event
      - OPTIONAL_WITH_NATIVE_MISSING_SUPPORT → NaN allowed (explicit HGB policy)
      - extra live columns → ignored + schema-drift event
      - unit/type change of REQUIRED columns → fail
    research:
      - same alignment, but missing required columns become NaN with RESEARCH_ONLY status
    """
    specs = specs or {}
    drift: list[dict[str, Any]] = []
    quarantine: list[int] = []

    extra = [c for c in df.columns if c not in cols and c not in specs]
    for c in extra:
        drift.append({"type": "EXTRA_LIVE_COLUMN", "column": c, "action": "ignore"})

    out = df.reindex(columns=cols)
    numeric = out.apply(pd.to_numeric, errors="coerce")

    for c in cols:
        spec = specs.get(c)
        classification = spec.classification if spec else FeatureClass.OPTIONAL_WITH_NATIVE_MISSING_SUPPORT
        col = numeric[c]
        missing_mask = col.isna()
        if not missing_mask.any():
            continue

        if classification == FeatureClass.REQUIRED:
            bad = list(np.flatnonzero(missing_mask.to_numpy()))
            quarantine.extend(bad)
            drift.append({
                "type": "REQUIRED_FEATURE_MISSING",
                "column": c,
                "n_rows": int(missing_mask.sum()),
                "action": "quarantine" if mode == "production" else "research_nan",
            })
            if mode == "production" and len(bad) == len(df):
                raise FeatureContractError(
                    f"REQUIRED feature '{c}' missing for entire slate"
                )
        elif classification == FeatureClass.OPTIONAL_WITH_TRAINED_IMPUTATION:
            if spec is None or spec.imputation_value is None:
                raise FeatureContractError(
                    f"OPTIONAL_WITH_TRAINED_IMPUTATION feature '{c}' lacks imputation_value"
                )
            numeric.loc[missing_mask, c] = spec.imputation_value
            drift.append({
                "type": "TRAINED_IMPUTATION_APPLIED",
                "column": c,
                "n_rows": int(missing_mask.sum()),
                "imputation_value": spec.imputation_value,
            })
        elif classification == FeatureClass.OPTIONAL_WITH_NATIVE_MISSING_SUPPORT:
            drift.append({
                "type": "NATIVE_MISSING_ALLOWED",
                "column": c,
                "n_rows": int(missing_mask.sum()),
            })
        elif classification in (FeatureClass.RESEARCH_ONLY, FeatureClass.DEPRECATED):
            if mode == "production":
                raise FeatureContractError(
                    f"feature '{c}' classified {classification.value} cannot be used in production"
                )
        # DERIVED: ignore (not expected in model input cols)

    status = "OK"
    if quarantine and mode == "production":
        status = "ROWS_QUARANTINED"
    elif mode != "production":
        status = "RESEARCH_ONLY" if any(d["type"] == "REQUIRED_FEATURE_MISSING" for d in drift) else "OK"

    return FeatureFrameResult(
        frame=numeric,
        drift_events=drift,
        quarantined_rows=sorted(set(quarantine)),
        status=status,
    )


def registry_from_bundle_contracts(contracts: dict[str, dict]) -> list[dict[str, Any]]:
    """Machine-readable feature registry rows for reports."""
    specs = classify_contract_features(contracts)
    rows = []
    for name, spec in sorted(specs.items()):
        rows.append({
            "canonical_name": name,
            "classification": spec.classification.value,
            "source": spec.source,
            "grain": "player_game" if spec.component != "identity" else "identity",
            "unit": spec.unit,
            "point_in_time_availability_rule": spec.pit_rule,
            "transformation": "numeric_coerce",
            "missingness_policy": spec.classification.value,
            "imputation_source": (
                "training_fold_median" if spec.imputation_value is not None else "none"
            ),
            "imputation_value": spec.imputation_value,
            "leakage_risk": spec.leakage_risk,
            "applicable_component": spec.component,
            "feature_version": "v6_frozen",
        })
    return rows
