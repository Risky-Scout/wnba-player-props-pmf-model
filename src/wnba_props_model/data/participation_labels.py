"""Explicit participation-label classes for historical training panels.

Does not fit models. Inferred / unknown negatives are excluded from training
by default.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CONFIRMED_ACTIVE = "CONFIRMED_ACTIVE"
CONFIRMED_INACTIVE = "CONFIRMED_INACTIVE"
INFERRED_ELIGIBLE_DNP = "INFERRED_ELIGIBLE_DNP"
UNKNOWN_ROSTER_ELIGIBILITY = "UNKNOWN_ROSTER_ELIGIBILITY"

LABEL_CLASSES = (
    CONFIRMED_ACTIVE,
    CONFIRMED_INACTIVE,
    INFERRED_ELIGIBLE_DNP,
    UNKNOWN_ROSTER_ELIGIBILITY,
)

# Columns that must never appear as onset-time features on a minutes panel.
LEAKAGE_PROHIBITED_ONSET_FEATURES = frozenset(
    {
        "date_returned",
        "total_games_missed",
        "actual_minutes",
        "minutes",
        "did_play",
    }
)


def classify_box_score_row(
    *,
    minutes: float | None,
    minutes_flag: str | None = None,
    eligibility_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one box-score / roster panel row.

    Rules:
    - Positive minutes → CONFIRMED_ACTIVE
    - Confirmed inactive requires timestamped eligibility evidence AND
      unavailability evidence (minutes_flag non_playing alone is insufficient
      without roster/injury eligibility context)
    - Zero-minute box row without confirmed eligibility → INFERRED_ELIGIBLE_DNP
    - No reliable eligibility → UNKNOWN_ROSTER_ELIGIBILITY
    - Absent box-score row alone never creates CONFIRMED_INACTIVE
    """
    evid = eligibility_evidence or {}
    has_eligibility = bool(evid.get("on_eligible_roster") or evid.get("roster_snapshot"))
    has_unavailability = bool(
        evid.get("inactive_list")
        or evid.get("injury_interval")
        or evid.get("suspended")
        or evid.get("reviewed_workbook_inactive")
    )
    evidence_ts = evid.get("evidence_timestamp")
    label_source = evid.get("label_source") or "box_score_player_stats"

    if minutes is None or (isinstance(minutes, float) and pd.isna(minutes)):
        mins = 0.0
    else:
        mins = float(minutes)
    if minutes_flag is None or (isinstance(minutes_flag, float) and pd.isna(minutes_flag)):
        flag = None
    else:
        flag = str(minutes_flag).strip().lower() or None

    if mins > 0:
        return _pack(
            CONFIRMED_ACTIVE,
            binary=1,
            confidence="high",
            training_eligible=True,
            weight=1.0,
            source=label_source,
            evidence_timestamp=evidence_ts,
            reason="positive_minutes_appearance",
        )

    if has_eligibility and has_unavailability:
        return _pack(
            CONFIRMED_INACTIVE,
            binary=0,
            confidence="high",
            training_eligible=True,
            weight=1.0,
            source=str(evid.get("label_source") or "timestamped_eligibility_evidence"),
            evidence_timestamp=evidence_ts,
            reason="eligible_and_unavailable",
        )

    # Box association with zero minutes but no full eligibility confirmation.
    if evid.get("box_score_row") is False and not has_eligibility:
        return _pack(
            UNKNOWN_ROSTER_ELIGIBILITY,
            binary=None,
            confidence="none",
            training_eligible=False,
            weight=0.0,
            source=label_source,
            evidence_timestamp=evidence_ts,
            reason="absent_box_score_without_roster_evidence",
        )

    # Zero-minute box row (including minutes_flag=non_playing) without eligibility
    # proof remains inferred — do not promote flag alone to confirmed inactive.
    reason = "zero_minute_box_row"
    if flag == "non_playing":
        reason = "non_playing_flag_without_roster_eligibility_proof"
    return _pack(
        INFERRED_ELIGIBLE_DNP,
        binary=0,
        confidence="low",
        training_eligible=False,
        weight=0.0,
        source=label_source,
        evidence_timestamp=evidence_ts,
        reason=reason,
    )


def _pack(
    label_class: str,
    *,
    binary: int | None,
    confidence: str,
    training_eligible: bool,
    weight: float,
    source: str,
    evidence_timestamp: Any,
    reason: str,
) -> dict[str, Any]:
    return {
        "participation_label_class": label_class,
        "participation_binary_label": binary,
        "label_confidence": confidence,
        "training_eligible": training_eligible,
        "training_weight": weight,
        "label_source": source,
        "evidence_timestamp": evidence_timestamp,
        "label_reason": reason,
    }


def build_participation_labels(
    player_game_stats: pd.DataFrame,
    *,
    eligibility_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build multi-class participation labels from a box-score panel.

    ``eligibility_evidence`` optional columns: game_id, player_id,
    on_eligible_roster, roster_snapshot, inactive_list, injury_interval,
    suspended, reviewed_workbook_inactive, evidence_timestamp, label_source.
    Unresolved / ambiguous workbook identities must not be passed as confirmed.
    """
    if player_game_stats is None or player_game_stats.empty:
        return pd.DataFrame()

    df = player_game_stats.copy()
    evid_map: dict[tuple[Any, Any], dict[str, Any]] = {}
    if eligibility_evidence is not None and not eligibility_evidence.empty:
        for _, r in eligibility_evidence.iterrows():
            key = (r.get("game_id"), r.get("player_id"))
            evid_map[key] = {k: r.get(k) for k in r.index}

    rows = []
    for _, r in df.iterrows():
        key = (r.get("game_id"), r.get("player_id"))
        evid = dict(evid_map.get(key, {}))
        evid.setdefault("box_score_row", True)
        packed = classify_box_score_row(
            minutes=r.get("minutes"),
            minutes_flag=r.get("minutes_flag"),
            eligibility_evidence=evid,
        )
        out = {
            "game_id": r.get("game_id"),
            "game_date": r.get("game_date"),
            "season": r.get("season"),
            "player_id": r.get("player_id"),
            "team_id": r.get("team_id"),
            "minutes": r.get("minutes"),
            "minutes_flag": r.get("minutes_flag"),
        }
        out.update(packed)
        rows.append(out)
    return pd.DataFrame(rows)


def participation_counts_by_season(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=["season", "participation_label_class", "n"])
    g = (
        labels.groupby(["season", "participation_label_class"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["season", "participation_label_class"])
    )
    return g


def build_conditional_minutes_training_table(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_cutoff: str,
    data_hash: str,
    feature_contract_hash: str,
) -> pd.DataFrame:
    """Training-ready conditional minutes table: CONFIRMED_ACTIVE only.

    Rejects leakage columns and non-active label classes.
    """
    leak = [c for c in feature_cols if c in LEAKAGE_PROHIBITED_ONSET_FEATURES]
    if leak:
        raise ValueError(f"target leakage columns in feature list: {leak}")

    active = labels[labels["participation_label_class"] == CONFIRMED_ACTIVE].copy()
    if active.empty:
        return pd.DataFrame()

    keys = ["game_id", "player_id"]
    feat = features.copy()
    for c in keys:
        if c not in feat.columns:
            raise ValueError(f"features missing key {c}")
    merged = active.merge(feat[keys + feature_cols], on=keys, how="inner")
    out = pd.DataFrame(
        {
            "game_id": merged["game_id"],
            "game_date": merged["game_date"],
            "player_id": merged["player_id"],
            "team_id": merged["team_id"],
            "actual_minutes": merged["minutes"],
            "feature_cutoff": feature_cutoff,
            "data_hash": data_hash,
            "feature_contract_hash": feature_contract_hash,
        }
    )
    for c in feature_cols:
        out[c] = merged[c]
    # Safety: never retain non-active classes
    assert len(out) == len(merged)
    return out
