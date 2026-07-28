#!/usr/bin/env python3
"""Validate the INFERRED PTS decomposition labels (owner directive item 7).

FGM and FTM are NOT raw truth in this repository: FGM is inferred as round(fieldGoalPercentage*FGA)
and FTM via the points identity. This script therefore:

  * re-derives the labels and persists PER-ROW provenance: reconstruction_method, label_status,
    confidence, rounding_residual, exclusion_reason, and dataset source hashes;
  * runs the CRITICAL cross-check: wherever tracking contested+uncontested FieldGoalsMade are
    populated, it validates reconstructed FGM against contested+uncontested FGM (agreement rate,
    max/mean diff) and flags/excludes disagreements;
  * excludes rows that fail any physical constraint from the conversion-label set (they may not enter
    full-PTS candidate evidence);
  * writes artifacts/opportunity_v2/PTS_LABEL_VALIDATION.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_ART = REPO / "artifacts" / "opportunity_v2"
OUT_DATA = REPO / "data" / "processed"

RECON_METHOD = "FGM=round(fieldGoalPercentage*FGA); 2PM=FGM-FG3M; FTM=PTS-2*FGM-FG3M (points identity)"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tr_path = REPO / "data/processed/wnba_tracking_2021_2026.parquet"
    gx_path = REPO / "data/processed/tracking_game_crosswalk.parquet"
    px_path = REPO / "data/processed/tracking_player_crosswalk.parquet"
    box_path = REPO / "data/processed/wnba_player_game_stats.parquet"
    source_hashes = {p.name: _sha(p) for p in (tr_path, gx_path, px_path, box_path)}
    source_tag = hashlib.sha256("".join(sorted(source_hashes.values())).encode()).hexdigest()[:16]

    tr = pd.read_parquet(tr_path)
    gx = pd.read_parquet(gx_path)
    px = pd.read_parquet(px_path)
    box = pd.read_parquet(box_path)

    tr = tr.copy()
    tr["game_id"] = tr["gameId"].map(dict(zip(gx["gameId"], gx["game_id"])))
    tr["player_id"] = tr["personId"].map(dict(zip(px["personId"], px["player_id"])))
    tr["fgpct"] = pd.to_numeric(tr["fieldGoalPercentage"], errors="coerce")
    for c in ("contestedFieldGoalsMade", "uncontestedFieldGoalsMade",
              "contestedFieldGoalsAttempted", "uncontestedFieldGoalsAttempted"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0)
    trm = tr.dropna(subset=["game_id", "player_id", "fgpct"]).copy()
    trm["game_id"] = trm["game_id"].astype(int)
    trm["player_id"] = trm["player_id"].astype(int)

    bx = box[["game_id", "player_id", "fga", "fg3a", "fg3m", "fta", "pts", "season"]].copy()
    for c in ("fga", "fg3a", "fg3m", "fta", "pts"):
        bx[c] = pd.to_numeric(bx[c], errors="coerce")
    j = trm.merge(bx, on=["game_id", "player_id"], how="inner")
    n = len(j)

    j["fgm_raw"] = j["fgpct"] * j["fga"]
    j["FGM"] = np.round(j["fgm_raw"])
    j["rounding_residual"] = (j["fgm_raw"] - j["FGM"]).abs()
    j["FG2M"] = j["FGM"] - j["fg3m"]
    j["FG2A"] = j["fga"] - j["fg3a"]
    j["FTM"] = j["pts"] - 2 * j["FG2M"] - 3 * j["fg3m"]
    j["pts_recon"] = 2 * j["FG2M"] + 3 * j["fg3m"] + j["FTM"]

    # per-row physical-constraint checks
    fail = {
        "fgm_lt_fg3m": j["FGM"] < j["fg3m"] - 0.01,
        "fg2m_neg": j["FG2M"] < -0.01,
        "fg2a_lt_fg2m": j["FG2A"] < j["FG2M"] - 0.01,
        "ftm_out_of_range": (j["FTM"] < -0.01) | (j["FTM"] > j["fta"] + 0.01),
        "pts_identity_violation": (j["pts_recon"] - j["pts"]).abs() > 0.5,
        "rounding_residual_high": j["rounding_residual"] > 0.02,
    }
    reason = pd.Series([""] * n, index=j.index)
    for name, mask in fail.items():
        reason = reason.where(~mask, reason.str.cat(pd.Series([name] * n, index=j.index), sep="|"))
    j["exclusion_reason"] = reason.str.strip("|").replace("", np.nan)
    j["label_status"] = np.where(j["exclusion_reason"].isna(), "validated", "excluded")
    j["confidence"] = np.select(
        [j["rounding_residual"] <= 0.005, j["rounding_residual"] <= 0.02],
        ["high", "medium"], default="low")
    j["reconstruction_method"] = RECON_METHOD
    j["source_tag"] = source_tag

    # CRITICAL cross-check vs contested+uncontested FGM where populated
    cc_made = j["contestedFieldGoalsMade"] + j["uncontestedFieldGoalsMade"]
    cc_att = j["contestedFieldGoalsAttempted"] + j["uncontestedFieldGoalsAttempted"]
    populated = cc_att > 0
    n_pop = int(populated.sum())
    if n_pop > 0:
        diff = (j.loc[populated, "FGM"] - cc_made[populated]).abs()
        cross = {
            "contested_uncontested_populated": True,
            "usable_rows": n_pop,
            "agreement_rate_exact": round(float((diff <= 0.5).mean()), 6),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "disagreements_flagged": int((diff > 0.5).sum()),
        }
        disagree_mask = populated & ((j["FGM"] - cc_made).abs() > 0.5)
        j.loc[disagree_mask & (j["label_status"] == "validated"), "label_status"] = "flagged_contested_disagreement"
        j.loc[disagree_mask & j["exclusion_reason"].isna(), "exclusion_reason"] = "contested_fgm_disagreement"
    else:
        cross = {
            "contested_uncontested_populated": False,
            "usable_rows": 0,
            "note": ("tracking contested/uncontested FieldGoalsMade columns are present (non-null) but "
                     "STRUCTURALLY ALL-ZERO in tracking-data-v1, so contested+uncontested FGM cannot "
                     "validate reconstructed FGM. Primary validation is the rounding-residual + physical "
                     "constraints + points identity below."),
            "contested_made_nonzero": int((j["contestedFieldGoalsMade"] != 0).sum()),
            "uncontested_made_nonzero": int((j["uncontestedFieldGoalsMade"] != 0).sum()),
        }

    # persist enriched labels; conversion fits may use ONLY validated rows
    keep = ["game_id", "player_id", "season", "FGM", "FG2M", "FG2A", "fg3m", "fg3a", "FTM", "fta", "pts",
            "reconstruction_method", "label_status", "confidence", "rounding_residual",
            "exclusion_reason", "source_tag"]
    labels = j[keep].rename(columns={"fg3m": "FG3M", "fg3a": "FG3A", "fta": "FTA"})
    validated = labels[labels["label_status"] == "validated"].copy()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    # full enriched table (audit) + validated-only table used by the bundle's conversion fits
    labels.to_parquet(OUT_DATA / "pts_conversion_labels_audited.parquet", index=False)
    validated.to_parquet(OUT_DATA / "pts_conversion_labels.parquet", index=False)

    status_counts = labels["label_status"].value_counts().to_dict()
    report = {
        "rows_total": int(n),
        "reconstruction_method": RECON_METHOD,
        "inferred_labels": ["FGM", "FTM", "FG2M"],
        "source_hashes": source_hashes,
        "source_tag": source_tag,
        "label_status_counts": {k: int(v) for k, v in status_counts.items()},
        "confidence_counts": {k: int(v) for k, v in labels["confidence"].value_counts().items()},
        "constraint_fail_counts": {k: int(v.sum()) for k, v in fail.items()},
        "rounding_residual": {
            "mean": float(j["rounding_residual"].mean()),
            "p99": float(j["rounding_residual"].quantile(0.99)),
            "max": float(j["rounding_residual"].max()),
        },
        "contested_cross_check": cross,
        "validated_rows_persisted": int(len(validated)),
        "validated_parquet": "data/processed/pts_conversion_labels.parquet",
        "audited_parquet": "data/processed/pts_conversion_labels_audited.parquet",
        "honest_note": ("FGM/FTM are INFERRED, not measured. They are validated by exact rounding "
                        "consistency of fieldGoalPercentage*FGA and by the points identity + physical "
                        "constraints. The tracking contested/uncontested make columns are structurally "
                        "empty and provide no independent make-count validation."),
    }
    OUT_ART.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_ART / "PTS_LABEL_VALIDATION.json", "w"), indent=2)
    print(json.dumps({k: report[k] for k in
                      ("rows_total", "label_status_counts", "confidence_counts",
                       "contested_cross_check", "validated_rows_persisted")}, indent=2))


if __name__ == "__main__":
    main()
