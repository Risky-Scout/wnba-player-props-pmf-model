"""Date-effective identity resolution for V6 production slates.

Unresolved identities never receive a silent league-average projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class IdentityStatus(str, Enum):
    OK = "OK"
    IDENTITY_ERROR = "IDENTITY_ERROR"
    ROSTER_UNRESOLVED = "ROSTER_UNRESOLVED"
    GAME_MAPPING_ERROR = "GAME_MAPPING_ERROR"
    TEAM_MISMATCH = "TEAM_MISMATCH"
    DUPLICATE_GAME_ID = "DUPLICATE_GAME_ID"
    POSTPONED_OR_CANCELED = "POSTPONED_OR_CANCELED"
    MISSING_PROVIDER_ID = "MISSING_PROVIDER_ID"


@dataclass
class IdentityAuditResult:
    status: IdentityStatus
    rows: pd.DataFrame
    quarantined: pd.DataFrame
    events: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "ok"  # ok | quarantine | fail_slate


class IdentityResolutionError(RuntimeError):
    """Fatal identity failure in production mode."""


def _normalize_games(scheduled_games: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(scheduled_games, pd.DataFrame):
        g = scheduled_games.copy()
    else:
        records = []
        for gm in scheduled_games:
            home = gm.get("home_team")
            visitor = gm.get("visitor_team")
            records.append({
                "game_id": gm.get("id") if gm.get("id") is not None else gm.get("game_id"),
                "scheduled_tip_utc": gm.get("scheduled_tip_utc") or gm.get("datetime") or gm.get("date") or gm.get("game_date"),
                "home_team_id": home["id"] if isinstance(home, dict) else gm.get("home_team_id"),
                "visitor_team_id": (
                    visitor["id"] if isinstance(visitor, dict)
                    else gm.get("visitor_team_id") or gm.get("away_team_id")
                ),
                "status": str(gm.get("status", "scheduled")).lower(),
            })
        g = pd.DataFrame(records)
    if g.empty:
        return g
    g["game_id"] = pd.to_numeric(g["game_id"], errors="coerce")
    return g


def audit_scheduled_games(
    scheduled_games: list[dict] | pd.DataFrame,
    *,
    mode: str = "production",
) -> IdentityAuditResult:
    """Validate game identity integrity before inference."""
    events: list[dict[str, Any]] = []
    g = _normalize_games(scheduled_games)
    if g.empty:
        return IdentityAuditResult(
            status=IdentityStatus.GAME_MAPPING_ERROR,
            rows=g,
            quarantined=g,
            events=[{"type": "GAME_MAPPING_ERROR", "reason": "empty_schedule"}],
            severity="fail_slate",
        )

    # duplicate game IDs
    if g["game_id"].isna().any():
        events.append({"type": "MISSING_PROVIDER_ID", "entity": "game_id"})
        if mode == "production":
            return IdentityAuditResult(
                IdentityStatus.MISSING_PROVIDER_ID, g, g, events, "fail_slate"
            )

    dup = g["game_id"].duplicated(keep=False)
    if dup.any():
        events.append({
            "type": "DUPLICATE_GAME_ID",
            "game_ids": [int(x) for x in g.loc[dup, "game_id"].tolist()],
        })
        if mode == "production":
            return IdentityAuditResult(
                IdentityStatus.DUPLICATE_GAME_ID, g, g.loc[dup], events, "fail_slate"
            )

    canceled = g["status"].astype(str).str.lower().isin(
        {"postponed", "canceled", "cancelled", "suspended"}
    )
    if canceled.any():
        events.append({
            "type": "POSTPONED_OR_CANCELED",
            "game_ids": [int(x) for x in g.loc[canceled, "game_id"].tolist()],
        })
        g = g.loc[~canceled].copy()
        if g.empty and mode == "production":
            return IdentityAuditResult(
                IdentityStatus.POSTPONED_OR_CANCELED, g, g, events, "fail_slate"
            )

    return IdentityAuditResult(IdentityStatus.OK, g, pd.DataFrame(), events, "ok")


def resolve_roster_identities(
    slate: pd.DataFrame,
    *,
    identity_table: pd.DataFrame | None = None,
    prediction_timestamp: str | None = None,
    mode: str = "production",
) -> IdentityAuditResult:
    """Validate player/team/game identities on a built feature slate.

    ``identity_table`` optional columns:
      player_id, team_id, valid_from, valid_to, roster_status, provider_player_id,
      canonical_player_id, display_name
    """
    events: list[dict[str, Any]] = []
    if slate is None or slate.empty:
        return IdentityAuditResult(
            IdentityStatus.ROSTER_UNRESOLVED,
            slate if slate is not None else pd.DataFrame(),
            pd.DataFrame(),
            [{"type": "ROSTER_UNRESOLVED", "reason": "empty_slate"}],
            "fail_slate" if mode == "production" else "quarantine",
        )

    required = ["game_id", "player_id", "team_id"]
    missing_cols = [c for c in required if c not in slate.columns]
    if missing_cols:
        events.append({"type": "IDENTITY_ERROR", "missing_columns": missing_cols})
        return IdentityAuditResult(
            IdentityStatus.IDENTITY_ERROR, slate, slate, events, "fail_slate"
        )

    work = slate.copy()
    work["_identity_status"] = IdentityStatus.OK.value
    quarantine_idx: list[int] = []

    for col in required:
        bad = work[col].isna() | (pd.to_numeric(work[col], errors="coerce").isna())
        if bad.any():
            for i in work.index[bad]:
                quarantine_idx.append(int(i))
                work.at[i, "_identity_status"] = IdentityStatus.MISSING_PROVIDER_ID.value
            events.append({
                "type": "MISSING_PROVIDER_ID",
                "column": col,
                "n_rows": int(bad.sum()),
            })

    # Duplicate (game_id, player_id) rows
    key_dup = work.duplicated(subset=["game_id", "player_id"], keep=False)
    if key_dup.any():
        for i in work.index[key_dup]:
            quarantine_idx.append(int(i))
            work.at[i, "_identity_status"] = IdentityStatus.IDENTITY_ERROR.value
        events.append({
            "type": "IDENTITY_ERROR",
            "reason": "duplicate_game_player",
            "n_rows": int(key_dup.sum()),
        })

    # Date-effective identity table checks
    if identity_table is not None and len(identity_table):
        idt = identity_table.copy()
        if "canonical_player_id" in idt.columns and "player_id" not in idt.columns:
            idt["player_id"] = idt["canonical_player_id"]
        ts = pd.Timestamp(prediction_timestamp) if prediction_timestamp else None
        if ts is not None and "valid_from" in idt.columns:
            idt["valid_from"] = pd.to_datetime(idt["valid_from"], errors="coerce", utc=True)
            if "valid_to" in idt.columns:
                idt["valid_to"] = pd.to_datetime(idt["valid_to"], errors="coerce", utc=True)
            else:
                idt["valid_to"] = pd.NaT
            active = idt[
                (idt["valid_from"].isna() | (idt["valid_from"] <= ts))
                & (idt["valid_to"].isna() | (idt["valid_to"] >= ts))
            ]
        else:
            active = idt

        by_player = active.groupby("player_id") if "player_id" in active.columns else None
        if by_player is not None:
            for i, row in work.iterrows():
                pid = int(row["player_id"]) if pd.notna(row["player_id"]) else None
                if pid is None:
                    continue
                if pid not in by_player.groups:
                    quarantine_idx.append(int(i))
                    work.at[i, "_identity_status"] = IdentityStatus.ROSTER_UNRESOLVED.value
                    events.append({
                        "type": "ROSTER_UNRESOLVED",
                        "player_id": pid,
                        "game_id": int(row["game_id"]) if pd.notna(row["game_id"]) else None,
                    })
                    continue
                memb = active.loc[by_player.groups[pid]]
                teams = set(int(t) for t in memb["team_id"].dropna().tolist()) if "team_id" in memb.columns else set()
                if teams and int(row["team_id"]) not in teams:
                    quarantine_idx.append(int(i))
                    work.at[i, "_identity_status"] = IdentityStatus.TEAM_MISMATCH.value
                    events.append({
                        "type": "TEAM_MISMATCH",
                        "player_id": pid,
                        "row_team_id": int(row["team_id"]),
                        "expected_team_ids": sorted(teams),
                    })

    quarantine_idx = sorted(set(quarantine_idx))
    quarantined = work.loc[quarantine_idx].copy() if quarantine_idx else pd.DataFrame()
    clean = work.drop(index=quarantine_idx).copy() if quarantine_idx else work

    if quarantine_idx and mode == "production":
        # Partial slate: quarantine bad rows; fail only if nothing remains
        if clean.empty:
            return IdentityAuditResult(
                IdentityStatus.IDENTITY_ERROR,
                clean,
                quarantined,
                events,
                "fail_slate",
            )
        return IdentityAuditResult(
            IdentityStatus.IDENTITY_ERROR if events else IdentityStatus.OK,
            clean,
            quarantined,
            events,
            "quarantine",
        )

    return IdentityAuditResult(
        IdentityStatus.OK if not quarantine_idx else IdentityStatus.IDENTITY_ERROR,
        clean if mode == "production" else work,
        quarantined,
        events,
        "ok" if not quarantine_idx else "quarantine",
    )


def build_date_effective_identity_table(
    stats: pd.DataFrame,
    *,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Construct a minimal date-effective player-team membership table from box history.

    Each contiguous player-team stretch becomes a validity interval. Display names are
    retained as metadata only — production matching uses provider player_id.
    """
    if stats is None or stats.empty:
        return pd.DataFrame(
            columns=[
                "player_id", "team_id", "valid_from", "valid_to",
                "roster_status", "display_name", "provider_player_id",
            ]
        )
    st = stats.copy()
    st["game_date"] = pd.to_datetime(st["game_date"], errors="coerce", utc=True)
    st = st.dropna(subset=["player_id", "team_id", "game_date"]).sort_values(
        ["player_id", "game_date"]
    )
    if as_of is not None:
        st = st[st["game_date"] <= pd.Timestamp(as_of, tz="UTC")]

    rows = []
    for pid, g in st.groupby("player_id"):
        g = g.sort_values("game_date")
        team = None
        start = None
        last = None
        name = None
        for r in g.itertuples():
            tid = int(r.team_id)
            d = pd.Timestamp(r.game_date)
            name = getattr(r, "player_name", None) if hasattr(r, "player_name") else name
            if team is None:
                team, start, last = tid, d, d
                continue
            if tid != team:
                # Close prior membership the day before the new-team observation.
                rows.append({
                    "player_id": int(pid),
                    "provider_player_id": int(pid),
                    "team_id": team,
                    "valid_from": start,
                    "valid_to": d - pd.Timedelta(days=1),
                    "roster_status": "active",
                    "display_name": name,
                })
                team, start, last = tid, d, d
            else:
                last = d
        if team is not None:
            # Current membership remains open-ended so tip-time resolution works
            # after the player's most recent observed game.
            rows.append({
                "player_id": int(pid),
                "provider_player_id": int(pid),
                "team_id": team,
                "valid_from": start,
                "valid_to": pd.NaT,
                "roster_status": "active",
                "display_name": name,
            })
    return pd.DataFrame(rows)
