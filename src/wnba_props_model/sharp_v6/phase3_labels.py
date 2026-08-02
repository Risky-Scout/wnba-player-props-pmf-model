"""Phase-3 retrospective label revalidation with conservative negatives."""

from __future__ import annotations

from typing import Any

import pandas as pd

from wnba_props_model.data.participation_labels import (
    CONFIRMED_ACTIVE,
    CONFIRMED_INACTIVE,
    INFERRED_ELIGIBLE_DNP,
    UNKNOWN_ROSTER_ELIGIBILITY,
)

INACTIVE_REJECT_RATE_GATE = 0.02


class LabelRevalidationError(RuntimeError):
    """Hard gate failure during participation-label revalidation."""


def revalidate_participation_labels(
    labels: pd.DataFrame,
    stats: pd.DataFrame,
    games: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    *,
    max_inactive_reject_rate: float = INACTIVE_REJECT_RATE_GATE,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Recheck labels without promoting box-score absence to an inactive negative.

    Returns (revalidated_labels, summary, rejects_frame).
    """
    if labels is None or labels.empty:
        empty = pd.DataFrame()
        return empty, {"rows": 0}, empty

    out = labels.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    keys = ["game_id", "player_id"]
    reject_rows: list[dict[str, Any]] = []

    # --- Active revalidation against official player-stat appearances ---
    stat = stats.copy()
    minutes_col = "minutes" if "minutes" in stat.columns else "actual_minutes"
    keep = [c for c in keys + [minutes_col, "team_id", "game_date"] if c in stat.columns]
    stat = stat[keep].copy()
    stat[minutes_col] = pd.to_numeric(stat[minutes_col], errors="coerce")
    # Duplicate player-game appearances are invalid for CONFIRMED_ACTIVE.
    dup_mask = stat.duplicated(keys, keep=False)
    dup_keys = set(map(tuple, stat.loc[dup_mask, keys].to_numpy()))

    appearance = (
        stat[stat[minutes_col].fillna(0) > 0]
        .drop_duplicates(keys)
        .rename(columns={minutes_col: "_stat_minutes", "team_id": "_stat_team_id"})
    )
    # Keep label game_date authoritative — do not collide on merge.
    appearance = appearance.drop(columns=[c for c in ("game_date",) if c in appearance.columns])
    out = out.merge(appearance, on=keys, how="left")
    if "game_date" not in out.columns and "game_date_x" in out.columns:
        out["game_date"] = out["game_date_x"]

    if games is not None and not games.empty and "game_id" in games.columns:
        gcols = [
            c
            for c in ("game_id", "home_team_id", "away_team_id", "visitor_team_id")
            if c in games.columns
        ]
        g = games[gcols].drop_duplicates("game_id")
        out = out.merge(g, on="game_id", how="left", suffixes=("", "_game"))

    original_class = out["participation_label_class"].astype(str)
    original_inactive = original_class.eq(CONFIRMED_INACTIVE)
    original_active = original_class.eq(CONFIRMED_ACTIVE)

    has_appearance = out["_stat_minutes"].fillna(0) > 0
    missing_ids = out["game_id"].isna() | out["player_id"].isna()
    is_dup = out.apply(lambda r: (r["game_id"], r["player_id"]) in dup_keys, axis=1)

    # Team consistency when both sides present.
    team_ok = pd.Series(True, index=out.index)
    if "_stat_team_id" in out.columns and "team_id" in out.columns:
        both = out["_stat_team_id"].notna() & out["team_id"].notna()
        team_ok = (~both) | (out["_stat_team_id"].astype(float) == out["team_id"].astype(float))

    # Active requires official appearance with minutes > 0, IDs, no dups, team ok.
    active_fail = original_active & ((~has_appearance) | missing_ids | is_dup | (~team_ok))
    for idx in out.index[active_fail]:
        reason = "missing_positive_appearance"
        if missing_ids.loc[idx]:
            reason = "missing_canonical_ids"
        elif is_dup.loc[idx]:
            reason = "duplicate_player_game"
        elif not team_ok.loc[idx]:
            reason = "team_mismatch"
        reject_rows.append(
            {
                "game_id": out.at[idx, "game_id"],
                "player_id": out.at[idx, "player_id"],
                "season": out.at[idx, "season"] if "season" in out.columns else None,
                "prior_class": CONFIRMED_ACTIVE,
                "reject_reason": reason,
            }
        )
        out.at[idx, "participation_label_class"] = UNKNOWN_ROSTER_ELIGIBILITY
        out.at[idx, "training_eligible"] = False
        out.at[idx, "training_weight"] = 0.0
        out.at[idx, "participation_binary_label"] = None

    # Inactive requires timestamped evidence, no appearance, no conflicting active.
    evid_ok = pd.Series(False, index=out.index)
    if "evidence_timestamp" in out.columns:
        evid_ok = out["evidence_timestamp"].notna()
    if "label_source" in out.columns:
        evid_ok = evid_ok & out["label_source"].astype(str).str.contains(
            "injury|eligibility|workbook", case=False, na=False
        )
    if evidence is not None and not evidence.empty:
        ev = evidence.drop_duplicates(keys)
        ev_flag = ev[keys].assign(_has_evidence=True)
        out = out.merge(ev_flag, on=keys, how="left")
        evid_ok = evid_ok | out["_has_evidence"].fillna(False)

    # Evidence must not be after game date when both parseable.
    evid_future = pd.Series(False, index=out.index)
    if "evidence_timestamp" in out.columns:
        ets = pd.to_datetime(out["evidence_timestamp"], errors="coerce")
        evid_future = (
            ets.notna()
            & out["game_date"].notna()
            & (ets.dt.normalize() > out["game_date"].dt.normalize())
        )

    inactive_fail = original_inactive & (
        (~evid_ok) | has_appearance | missing_ids | evid_future | is_dup
    )
    for idx in out.index[inactive_fail]:
        reason = "missing_timestamped_evidence"
        if has_appearance.loc[idx]:
            reason = "conflicting_active_appearance"
        elif missing_ids.loc[idx]:
            reason = "missing_canonical_ids"
        elif evid_future.loc[idx]:
            reason = "evidence_after_game_date"
        elif is_dup.loc[idx]:
            reason = "duplicate_player_game"
        reject_rows.append(
            {
                "game_id": out.at[idx, "game_id"],
                "player_id": out.at[idx, "player_id"],
                "season": out.at[idx, "season"] if "season" in out.columns else None,
                "prior_class": CONFIRMED_INACTIVE,
                "reject_reason": reason,
            }
        )
        # Do not convert to confirmed inactive from absence; demote to unknown/inferred.
        out.at[idx, "participation_label_class"] = (
            INFERRED_ELIGIBLE_DNP
            if (not bool(has_appearance.loc[idx])) and (not bool(missing_ids.loc[idx]))
            else UNKNOWN_ROSTER_ELIGIBILITY
        )
        # Box-score absence alone is never a supervised negative.
        if reason == "conflicting_active_appearance":
            out.at[idx, "participation_label_class"] = CONFIRMED_ACTIVE
            out.at[idx, "training_eligible"] = True
            out.at[idx, "training_weight"] = 1.0
            out.at[idx, "participation_binary_label"] = 1
        else:
            out.at[idx, "training_eligible"] = False
            out.at[idx, "training_weight"] = 0.0
            out.at[idx, "participation_binary_label"] = None

    # Enforce training policy for inferred/unknown.
    inferred = out["participation_label_class"].eq(INFERRED_ELIGIBLE_DNP)
    unknown = out["participation_label_class"].eq(UNKNOWN_ROSTER_ELIGIBILITY)
    out.loc[inferred | unknown, "training_eligible"] = False
    out.loc[inferred | unknown, "training_weight"] = 0.0
    out.loc[unknown, "participation_binary_label"] = None

    # Re-affirm confirmed classes.
    out.loc[out["participation_label_class"].eq(CONFIRMED_ACTIVE), "training_eligible"] = True
    out.loc[out["participation_label_class"].eq(CONFIRMED_ACTIVE), "training_weight"] = 1.0
    out.loc[out["participation_label_class"].eq(CONFIRMED_ACTIVE), "participation_binary_label"] = 1
    out.loc[out["participation_label_class"].eq(CONFIRMED_INACTIVE), "training_eligible"] = True
    out.loc[out["participation_label_class"].eq(CONFIRMED_INACTIVE), "training_weight"] = 1.0
    out.loc[
        out["participation_label_class"].eq(CONFIRMED_INACTIVE), "participation_binary_label"
    ] = 0

    rejects = pd.DataFrame(reject_rows)
    n_prior_inactive = int(original_inactive.sum())
    n_rejected_inactive = int(
        rejects["prior_class"].eq(CONFIRMED_INACTIVE).sum() if not rejects.empty else 0
    )
    reject_rate = (n_rejected_inactive / n_prior_inactive) if n_prior_inactive else 0.0
    if reject_rate > max_inactive_reject_rate:
        reasons = (
            rejects.loc[rejects["prior_class"].eq(CONFIRMED_INACTIVE), "reject_reason"]
            .value_counts()
            .to_dict()
            if not rejects.empty
            else {}
        )
        raise LabelRevalidationError(
            "confirmed inactive revalidation failure exceeds "
            f"{max_inactive_reject_rate:.0%}: rejected={n_rejected_inactive} "
            f"prior={n_prior_inactive} rate={reject_rate:.4f}; reasons={reasons}"
        )

    by_season = []
    for season, g in out.groupby(
        out["season"] if "season" in out.columns else pd.Series([None] * len(out))
    ):
        by_season.append(
            {
                "season": int(season) if season is not None and pd.notna(season) else None,
                "confirmed_active": int(g["participation_label_class"].eq(CONFIRMED_ACTIVE).sum()),
                "confirmed_inactive": int(
                    g["participation_label_class"].eq(CONFIRMED_INACTIVE).sum()
                ),
                "rejected_active_labels": int(
                    rejects.query("season == @season and prior_class == @CONFIRMED_ACTIVE").shape[0]
                    if not rejects.empty and "season" in rejects.columns
                    else 0
                ),
                "rejected_inactive_labels": int(
                    rejects.query("season == @season and prior_class == @CONFIRMED_INACTIVE").shape[
                        0
                    ]
                    if not rejects.empty and "season" in rejects.columns
                    else 0
                ),
                "inferred_dnp_excluded": int(
                    g["participation_label_class"].eq(INFERRED_ELIGIBLE_DNP).sum()
                ),
                "unknown_eligibility_excluded": int(
                    g["participation_label_class"].eq(UNKNOWN_ROSTER_ELIGIBILITY).sum()
                ),
                "evidence_source": (
                    g.loc[g["participation_label_class"].eq(CONFIRMED_INACTIVE), "label_source"]
                    .value_counts()
                    .to_dict()
                    if "label_source" in g.columns
                    else {}
                ),
                "distinct_players": int(g["player_id"].nunique()),
                "distinct_games": int(g["game_id"].nunique()),
                "distinct_game_dates": int(pd.to_datetime(g["game_date"]).dt.normalize().nunique()),
            }
        )

    summary = {
        "rows": len(out),
        "confirmed_active": int(out["participation_label_class"].eq(CONFIRMED_ACTIVE).sum()),
        "confirmed_inactive": int(out["participation_label_class"].eq(CONFIRMED_INACTIVE).sum()),
        "rejected_active_labels": int(
            rejects["prior_class"].eq(CONFIRMED_ACTIVE).sum() if not rejects.empty else 0
        ),
        "rejected_inactive_labels": n_rejected_inactive,
        "inactive_reject_rate": reject_rate,
        "inactive_reject_gate": max_inactive_reject_rate,
        "inferred_dnp_excluded": int(
            out["participation_label_class"].eq(INFERRED_ELIGIBLE_DNP).sum()
        ),
        "unknown_eligibility_excluded": int(
            out["participation_label_class"].eq(UNKNOWN_ROSTER_ELIGIBILITY).sum()
        ),
        "by_season": by_season,
        "reject_reasons": rejects["reject_reason"].value_counts().to_dict()
        if not rejects.empty
        else {},
    }
    drop_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_cols, errors="ignore"), summary, rejects


def injury_conditioned_training_cohort(labels: pd.DataFrame) -> pd.DataFrame:
    """Rows with timestamped injury/workbook evidence only — not global roster DNPs."""
    df = labels.copy()
    src = df.get("label_source", pd.Series("", index=df.index)).astype(str)
    has_src = src.str.contains("injury|workbook|eligibility", case=False, na=False)
    classes = df["participation_label_class"].isin([CONFIRMED_ACTIVE, CONFIRMED_INACTIVE])
    eligible = df.get("training_eligible", True)
    if not isinstance(eligible, pd.Series):
        eligible = pd.Series(bool(eligible), index=df.index)
    out = df[has_src & classes & eligible.fillna(False)].copy()
    out["training_weight"] = out.get("training_weight", 1.0)
    return out


def aggregate_label_audit(labels: pd.DataFrame) -> pd.DataFrame:
    """Commit-safe aggregate-only audit (no private row payloads)."""
    cols = [
        c
        for c in ("season", "participation_label_class", "training_eligible", "label_source")
        if c in labels
    ]
    if not cols:
        return pd.DataFrame({"rows": [len(labels)]})
    g = labels.groupby(cols, dropna=False).size().reset_index(name="rows")
    if "player_id" in labels.columns:
        players = labels.groupby(cols)["player_id"].nunique().reset_index(name="distinct_players")
        g = g.merge(players, on=cols, how="left")
    if "game_id" in labels.columns:
        games = labels.groupby(cols)["game_id"].nunique().reset_index(name="distinct_games")
        g = g.merge(games, on=cols, how="left")
    return g.sort_values(cols)
