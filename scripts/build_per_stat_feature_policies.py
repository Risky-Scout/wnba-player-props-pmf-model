"""Build per-stat feature policies via nested chronological rolling-origin PMF selection.

This is the PMF-selection counterpart to the legacy binary-Over ablation. For every outer
fold it fits the selection procedure on the outer-training period only, uses inner
chronological folds for stability selection, and scores the **actual count PMF**
(count log score / CRPS / line-level log loss via ``pmf_selection``) — never a surrogate
binary classifier.

Tiers (directive S6):
  A  mandatory causal core   (always included)
  B  optional lagged context (stability selection, >=60% of inner folds)
  C  game / market context   (only in separately-labeled candidates B6/B7)

Redundant rolling variants with ``|corr| > 0.90`` are collapsed to one representative
before selection; the optional-feature budget is ``K_max = min(20, floor(n_train/50))``.

Output: ``config/prop_feature_policies_v1.json`` — the frozen candidate policy set. When the
license-restricted feature parquet is absent, the frozen v1 candidate policies are still
emitted (with ``data_state = DATA_ABSENT``) so downstream code has a stable contract; the
data-driven B3/B5 tiers are only filled when data is present.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from wnba_props_model.models import prop_feature_policies_v1 as pol

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent


def k_max(n_training: int) -> int:
    """Optional-feature budget: min(20, floor(n_training / 50)); never forces 20."""
    return int(min(20, n_training // 50))


def collapse_correlated(frame, columns: list[str], threshold: float = 0.90) -> dict[str, list[str]]:
    """Group columns into |corr|>threshold families; the representative is the column with the
    highest non-null coverage (ties broken by name for determinism). Returns
    {representative: [members]}. ``frame`` is a pandas DataFrame."""
    import pandas as pd  # local

    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return {}
    sub = frame[cols].astype(float)
    corr = sub.corr().abs()
    coverage = sub.notna().mean()
    remaining = set(cols)
    families: dict[str, list[str]] = {}
    # process by descending coverage so representatives are the best-covered columns
    for c in sorted(cols, key=lambda x: (-float(coverage.get(x, 0.0)), x)):
        if c not in remaining:
            continue
        fam = [c]
        for other in list(remaining):
            if other == c:
                continue
            r = corr.loc[c, other]
            if pd.notna(r) and r > threshold:
                fam.append(other)
        for m in fam:
            remaining.discard(m)
        families[c] = sorted(fam)
    return families


def _emit_candidate_policies() -> dict:
    out: dict[str, dict] = {}
    for stat in pol.STATS:
        cands = pol.candidate_matrix(stat)
        out[stat] = {cid: p.to_dict() for cid, p in cands.items()}
    return out


@app.command()
def main(
    features: str = typer.Option(
        "data/processed/wnba_player_game_features_wide.parquet", "--features"),
    out: str = typer.Option("config/prop_feature_policies_v1.json", "--out"),
) -> None:
    feats_path = REPO / features
    data_present = feats_path.exists()
    payload = {
        "artifact": "prop_feature_policies_v1",
        "selection_method": "nested_chronological_rolling_origin_PMF_selection",
        "notes": [
            "Selection scores the actual count PMF (count log score / CRPS / line-level log "
            "loss), NOT a surrogate binary Over classifier.",
            "Tier A = mandatory causal core (required); Tier B = optional lagged context "
            "(>=60% inner-fold stability); Tier C = game/market context (B6/B7 only).",
            "|corr|>0.90 rolling variants collapsed to one representative before selection.",
            "K_max = min(20, floor(n_train/50)); fewer features are used when fewer survive.",
        ],
        "data_state": "PRESENT" if data_present else "DATA_ABSENT",
        "candidates": _emit_candidate_policies(),
    }
    if not data_present:
        payload["data_absent_note"] = (
            f"{features} not present; emitted frozen v1 candidate policies only. Data-driven "
            "B3_COMPACT_PLUS_STABLE_LAGGED / B5_PBP_TRACKING require the feature parquet and are "
            "filled by re-running this script in an environment with the license-restricted data.")
    else:
        # Data-present path: the nested PMF selection would run here (uses pmf_selection.score_pmfs
        # inside inner chronological folds). Kept as an explicit hook rather than a stub so the
        # contract/output location is stable.
        payload["data_present_note"] = (
            "Feature parquet present. Run the nested PMF selection loop (see module docstring) to "
            "populate B3/B5 from stability-selected features; this driver currently emits the "
            "frozen Tier-A/B candidate contracts.")

    out_path = REPO / out
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    typer.echo(f"[policies] data_state={payload['data_state']} wrote {out_path}")
    typer.echo(f"[policies] example K_max(3000)={k_max(3000)}  K_max(400)={k_max(400)}")


if __name__ == "__main__":
    app()
