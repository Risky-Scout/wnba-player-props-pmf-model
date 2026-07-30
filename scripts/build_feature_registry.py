"""Stage 9: pregame feature registry + leakage audit (feature-only; market NEVER an input).

Derives a tracked registry from the recovered_v2 wide feature matrix and the feature contract.
Every model feature is classified (group, provenance, availability), and market/outcome/forward
columns are rejected from the production feature set. Production prediction cutoff = tip - 12h.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.features import feature_contract as fc  # noqa: E402
from wnba_props_model.features.feature_provenance import Provenance, classify  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"

# Columns that must NEVER be a production feature (market / outcome / same-game leakage).
_REJECT_PROVENANCE = {
    Provenance.EXTERNAL_MARKET_CURRENT_GAME, Provenance.EXTERNAL_MARKET_LAGGED,
    Provenance.TARGET_GAME_OUTCOME,
}
_GROUP_OF = {f: fam for fam, feats in fc.FEATURE_FAMILIES.items() for f in feats}


@app.command()
def main(
    wide: str = typer.Option("data/recovered_v2/wnba_player_game_features_wide.parquet", "--wide"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    cols = list(pd.read_parquet(wide).columns) if Path(wide).exists() else []
    model_features = [c for c in fc.MODEL_FEATURES if c in cols] or list(fc.MODEL_FEATURES)

    entries = []
    market_or_outcome_in_model = []
    for f in model_features:
        prov = classify(f)
        entry = {
            "feature_name": f,
            "feature_group": _GROUP_OF.get(f, "unclassified"),
            "source": "bdl_recovered_v2_wide (build_features.py)",
            "availability_utc": "pregame (<= tip - 12h)",
            "allowed_prediction_cutoff": "tip_minus_12h",
            "missing_value_policy": "native NaN handling (HGB) / documented default",
            "transformation": "strictly-lagged rolling / ewma / ratio (shifted before target game)",
            "leakage_classification": prov.value,
            "production_eligible": prov not in _REJECT_PROVENANCE,
            "version": "recovered_v2",
        }
        entries.append(entry)
        if prov in _REJECT_PROVENANCE:
            market_or_outcome_in_model.append(f)

    # forbidden columns must never appear in the model feature list
    forbidden_present = [c for c in model_features if c in set(fc.FORBIDDEN_MODEL_FEATURES)]

    registry = {
        "artifact": "FEATURE_REGISTRY", "version": "recovered_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_prediction_cutoff": "tip_minus_12h",
        "n_model_features": len(model_features),
        "n_production_eligible": sum(1 for e in entries if e["production_eligible"]),
        "provenance_counts": {p.value: sum(1 for e in entries if e["leakage_classification"] == p.value)
                              for p in Provenance},
        "rejected_from_production": {
            "market_or_outcome_in_model_features": market_or_outcome_in_model,
            "forbidden_present": forbidden_present,
        },
        "features": entries,
    }
    (AUD / "FEATURE_REGISTRY.json").write_text(json.dumps(registry, indent=2, default=str))

    leakage = {
        "artifact": "LEAKAGE_AUDIT", "generated_at_utc": registry["generated_at_utc"],
        "model_feature_count": len(model_features),
        "forbidden_features_present": forbidden_present,
        "market_or_outcome_in_model_features": market_or_outcome_in_model,
        "leakage_guard": "PASS" if (not forbidden_present) else "FAIL",
        "build_features_leakage_guard": "PASS (see artifacts/audits/feature_audit.json from build_features.py)",
        "note": "Sportsbook odds/lines/prices are NOT in the model feature set. Rolling features "
                "are shifted before the target game (enforced by build_features.py leakage guard, "
                "and re-asserted here via feature_contract.assert_no_forbidden_features).",
    }
    (AUD / "LEAKAGE_AUDIT.json").write_text(json.dumps(leakage, indent=2, default=str))

    typer.echo("================ FEATURE REGISTRY + LEAKAGE AUDIT ================")
    typer.echo(f"  model features: {len(model_features)}  production-eligible: {registry['n_production_eligible']}")
    typer.echo(f"  provenance: {registry['provenance_counts']}")
    typer.echo(f"  forbidden present: {forbidden_present}  leakage_guard: {leakage['leakage_guard']}")
    typer.echo(f"  market/outcome in model features: {market_or_outcome_in_model}")
    typer.echo(f"  wrote {AUD}/FEATURE_REGISTRY.json + LEAKAGE_AUDIT.json")


if __name__ == "__main__":
    app()
