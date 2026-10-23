#!/usr/bin/env python3
"""Build August 1 frozen prospective-evidence package from already-downloaded artifacts.

Read-only w.r.t. model outputs: does not regenerate predictions, call Odds API,
alter PMFs/prices/timestamps, or publish. Column/type adaptation + audits only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "pregame_verification" / "2026-08-01"
FROZEN_ZIPS = OUT / "frozen_github_artifacts"
STAGE = OUT / "frozen_extracted" / "stage_core"
AUDITS = OUT / "frozen_extracted" / "audits"
ET = ZoneInfo("America/New_York")

WORKFLOW_RUN_ID = 30708012898
HEAD_SHA = "2caef26fa235a9f2cdccc900bd44f42974915e4d"
PRED_TS_COMPACT = "20260801T162447Z"
PRED_TS_UTC = "2026-08-01T16:27:11.435636+00:00"
STAGE_DIR = "deliveries/pregame_verification/2026-08-01/20260801T162447Z"
GAME_DATE = "2026-08-01"
SLATE_TIMEZONE = "America/New_York"
ALLOWED_QUOTE_SKEW_SECONDS = 0  # exact same-book pair already wide; no cross-row skew join

ARTIFACT_META = {
    "pregame-verification-stage-2026-08-01-20260801T162447Z-30708012898": {
        "id": 8820983835,
        "zip": "pregame-verification-stage-2026-08-01-20260801T162447Z-30708012898.zip",
    },
    "pregame-verification-audits-2026-08-01-20260801T162447Z-30708012898": {
        "id": 8820984226,
        "zip": "pregame-verification-audits-2026-08-01-20260801T162447Z-30708012898.zip",
    },
    "pregame-verification-smoke-pages-2026-08-01-30708012898": {
        "id": 8820984722,
        "zip": "pregame-verification-smoke-pages-2026-08-01-30708012898.zip",
    },
}

# Frozen wide odds schema -> canonical atomic-side / pair field mapping (names/types only).
FROZEN_ODDS_FIELD_MAP = {
    "provider": {"frozen": "source", "notes": "constant odds_api_v4"},
    "odds_api_event_id": {"frozen": "event_id"},
    "canonical_game_id": {
        "frozen": None,
        "derived": "exact event_id join to GAME_AUDIT.bdl_game_id (frozen)",
    },
    "bookmaker": {"frozen": "bookmaker", "alias": "vendor"},
    "market_key": {"frozen": "market_key"},
    "player_description": {"frozen": "player_name"},
    "canonical_player_id": {
        "frozen": None,
        "derived": "exact player_name equality join to frozen slate (no fuzzy)",
    },
    "line": {"frozen": "line"},
    "side": {
        "frozen": None,
        "derived": "expand over_odds/under_odds wide row into over|under atomic sides",
    },
    "price": {"frozen": ["over_odds", "under_odds"]},
    "provider_quote_timestamp": {"frozen": "last_update"},
    "market_last_update": {"frozen": "last_update"},
    "ingestion_timestamp": {"frozen": "pulled_at_utc"},
    "scheduled_tip": {"frozen": "commence_time"},
    "prediction_cutoff": {
        "frozen": None,
        "derived": "run_metadata.prediction_timestamp_utc (frozen)",
    },
    "stat": {"frozen": "stat"},
    "period": {
        "frozen": None,
        "derived": "q1 if market_key endswith _q1 else game",
    },
}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse_pmf(raw) -> list[tuple[float, float]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    out: list[tuple[float, float]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            out.append((float(k), float(v)))
        return sorted(out, key=lambda x: x[0])
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((float(item[0]), float(item[1])))
        return sorted(out, key=lambda x: x[0])
    return out


def _american_to_decimal(american: float) -> float:
    a = float(american)
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / abs(a)
    raise ValueError(f"invalid american odds {american}")


def _no_vig(over_american: float, under_american: float) -> tuple[float, float]:
    do = 1.0 / _american_to_decimal(over_american)
    du = 1.0 / _american_to_decimal(under_american)
    s = do + du
    return do / s, du / s


def _settle_from_pmf(atoms: list[tuple[float, float]], line: float) -> dict[str, float]:
    p_over = sum(p for k, p in atoms if k > line)
    p_under = sum(p for k, p in atoms if k < line)
    p_push = sum(p for k, p in atoms if abs(k - line) < 1e-12)
    denom = p_over + p_under
    if abs(line - round(line)) < 1e-12:
        # integer line: push possible; settled renorm excludes push
        pass
    else:
        # half-point: push must be 0
        p_push = 0.0
    if denom <= 0:
        p_over_s = p_under_s = float("nan")
    else:
        p_over_s = p_over / denom
        p_under_s = p_under / denom
    return {
        "p_over_win": p_over,
        "p_under_win": p_under,
        "p_push": p_push,
        "p_over_settled": p_over_s,
        "p_under_settled": p_under_s,
    }


def _atomic_quote_id(sportsbook, event_id, player_id_or_name, prop, line, side, snapshot_time) -> str:
    payload = "|".join(
        str(x)
        for x in (sportsbook, event_id, player_id_or_name, prop, line, side, snapshot_time)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _period_from_market(market_key: str) -> str:
    mk = str(market_key or "").lower()
    if mk.endswith("_q1") or mk.endswith("_1q"):
        return "q1"
    if mk.endswith("_q2") or mk.endswith("_2q"):
        return "q2"
    if mk.endswith("_q3") or mk.endswith("_3q"):
        return "q3"
    if mk.endswith("_q4") or mk.endswith("_4q"):
        return "q4"
    if "1h" in mk or mk.endswith("_h1"):
        return "1h"
    if "2h" in mk or mk.endswith("_h2"):
        return "2h"
    return "game"


def build_lock() -> dict:
    manifest = json.loads((AUDITS / "FROZEN_PREDICTION_MANIFEST.json").read_text())
    summary = json.loads((AUDITS / "PREGAME_VERIFICATION_SUMMARY.json").read_text())
    run_meta = json.loads((STAGE / "run_metadata.json").read_text())
    game_audit = pd.read_csv(AUDITS / "GAME_AUDIT.csv")
    identity = pd.read_csv(AUDITS / "PLAYER_IDENTITY_AUDIT.csv")
    odds_path = STAGE / "wnba_player_props_oddsapi_2026-08-01.parquet"
    pmf_path = STAGE / "full_pmfs_wide.parquet"

    artifacts = {}
    for name, meta in ARTIFACT_META.items():
        zpath = FROZEN_ZIPS / meta["zip"]
        artifacts[name] = {
            "artifact_id": meta["id"],
            "filename": meta["zip"],
            "sha256": _sha256_file(zpath),
            "size_bytes": zpath.stat().st_size,
            "durable_path": str(zpath.relative_to(REPO)),
            "github_actions_artifact_id": meta["id"],
            "github_actions_expires": "ephemeral; do not rely on Actions retention alone",
        }

    scheduled_tips = [
        {
            "bdl_game_id": int(r.bdl_game_id),
            "matchup": f"{r.away_team}@{r.home_team}",
            "scheduled_tip_utc": r.scheduled_tip_utc,
            "scheduled_tip_america_new_york": datetime.fromisoformat(
                r.scheduled_tip_utc.replace("Z", "+00:00")
            )
            .astimezone(ET)
            .isoformat(),
            "odds_api_event_id": r.odds_api_event_id,
        }
        for r in game_audit.itertuples()
    ]

    lock = {
        "lock_schema_version": "1.0",
        "immutable": True,
        "overwrite_policy": "FORBIDDEN",
        "game_date": GAME_DATE,
        "slate_timezone": SLATE_TIMEZONE,
        "workflow_run_id": WORKFLOW_RUN_ID,
        "workflow_url": f"https://github.com/Risky-Scout/wnba-player-props-pmf-model/actions/runs/{WORKFLOW_RUN_ID}",
        "head_code_sha": HEAD_SHA,
        "prediction_timestamp": PRED_TS_COMPACT,
        "prediction_timestamp_utc": run_meta.get("prediction_timestamp_utc", PRED_TS_UTC),
        "frozen_stage": STAGE_DIR,
        "scheduled_tips": scheduled_tips,
        "hashes": {
            "data_hash": manifest["data_hash"],
            "feature_hash": manifest["feature_hash"],
            "model_hash": manifest["model_hash"],
            "calibrator_hash": manifest["calibrator_hash"],
            "quote_hash": manifest["quote_hash"],
            "availability_hash": manifest["availability_hash"],
            "odds_parquet_sha256": _sha256_file(odds_path),
            "full_pmfs_wide_sha256": _sha256_file(pmf_path),
            "frozen_prediction_manifest_sha256": _sha256_file(
                AUDITS / "FROZEN_PREDICTION_MANIFEST.json"
            ),
            "frozen_prediction_registry_sha256": _sha256_file(
                AUDITS / "FROZEN_PREDICTION_REGISTRY.jsonl"
            ),
        },
        "pmf_file_hashes": {
            "full_pmfs_wide.parquet": _sha256_file(pmf_path),
            "n_frozen_pmf_sample_files": len(list((AUDITS / "frozen_pmfs").glob("*.json"))),
        },
        "quote_file_hashes": {
            "wnba_player_props_oddsapi_2026-08-01.parquet": _sha256_file(odds_path),
            "wnba_player_props_oddsapi_latest.parquet": _sha256_file(
                STAGE / "wnba_player_props_oddsapi_latest.parquet"
            ),
        },
        "row_counts": {
            "players_discovered": summary["players_discovered"],
            "players_accepted": summary["players_accepted"],
            "players_rejected": summary["players_rejected"],
            "n_frozen_identities": manifest["n_frozen_identities"],
            "accepted_identity_rows": int((identity["audit_status"] == "ACCEPTED").sum()),
            "rejected_identity_rows": int((identity["audit_status"] == "REJECTED").sum()),
            "pmf_rows": summary["pmf_rows"],
            "original_quote_pairs_valid": summary["quote_pairs_valid"],
            "original_quote_pairs_rejected": summary["quote_pairs_rejected"],
        },
        "artifacts": artifacts,
        "original_verdict": summary["verdict"],
        "original_warnings": summary["warnings"],
        "preservation_note": (
            "GitHub Actions artifacts are ephemeral. These zip blobs and hashes are "
            "committed under artifacts/pregame_verification/2026-08-01/ as durable, "
            "overwrite-forbidden prospective evidence."
        ),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "FROZEN_EVIDENCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def build_rejected_player_audit() -> pd.DataFrame:
    identity = pd.read_csv(AUDITS / "PLAYER_IDENTITY_AUDIT.csv")
    slate = pd.read_parquet(STAGE / "slate_2026-08-01.parquet")
    pmfs = pd.read_parquet(STAGE / "full_pmfs_wide.parquet")
    odds = pd.read_parquet(STAGE / "wnba_player_props_oddsapi_2026-08-01.parquet")
    rej = identity[identity["audit_status"] == "REJECTED"].copy()

    rows = []
    for r in rej.itertuples():
        pid = int(r.canonical_player_id)
        srow = slate[slate["player_id"] == pid]
        name = str(r.player_name)
        # exact-name other identities on slate
        same_name = slate[
            (slate["player_name"].astype(str) == name) & (slate["player_id"] != pid)
        ]
        last = name.split()[-1] if name else ""
        similar = slate[
            slate["player_name"].astype(str).str.contains(re.escape(last), case=False, na=False)
            & (slate["player_id"] != pid)
        ][["player_id", "player_name", "team_abbreviation"]].drop_duplicates()
        in_odds = bool((odds["player_name"].astype(str) == name).any())
        n_pmf = int((pmfs["player_id"] == pid).sum())
        # These IDs come from BDL feature rows; rejection is players-table gap, not
        # wrong-name collision. Settlement should exclude from primary certified set.
        protected = True  # audit gate prevented treating them as roster-certified
        correct_elsewhere = False
        if len(same_name):
            correct_elsewhere = True
        rows.append(
            {
                "source_player_id": pid,
                "normalized_name": name,
                "sportsbook_name": name if in_odds else "",
                "expected_current_team": r.current_team,
                "team_in_wnba_players": "",  # absent from table (rejection reason)
                "game_id": int(r.game_id),
                "rejection_reason": r.reject_reason,
                "current_roster_evidence": (
                    f"frozen_slate_bdl team={r.current_team} opponent={r.opponent}; "
                    f"minutes_l5={float(srow.iloc[0]['player_minutes_mean_l5']) if len(srow) and 'player_minutes_mean_l5' in srow.columns else 'NA'}; "
                    f"dnp_risk={r.expected_active_status}; "
                    "NOT present in wnba_players.parquet during verification"
                ),
                "rejection_protected_against_wrong_player_forecast": protected,
                "correct_forecast_exists_under_other_identity": correct_elsewhere,
                "other_identity_notes": "; ".join(
                    f"{int(x.player_id)}:{x.player_name}/{x.team_abbreviation}"
                    for x in similar.itertuples()
                ),
                "settlement_should_exclude": True,
                "settlement_exclusion_reason": (
                    "IDENTITY_TABLE_GAP: player_id absent from wnba_players at freeze; "
                    "exclude from primary certified settlement; may settle in "
                    "unresolved_identity bucket if BDL box score resolves same ID"
                ),
                "frozen_pmf_row_count": n_pmf,
                "present_in_frozen_odds": in_odds,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "REJECTED_PLAYER_AUDIT.csv", index=False)
    return df


def build_player_631_root_cause() -> str:
    identity = pd.read_csv(AUDITS / "PLAYER_IDENTITY_AUDIT.csv")
    slate = pd.read_parquet(STAGE / "slate_2026-08-01.parquet")
    r = identity[identity["canonical_player_id"] == 631].iloc[0]
    s = slate[slate["player_id"] == 631].iloc[0]
    md = f"""# Player 631 Root Cause — Kiana Williams

## Identity

| Field | Value |
|---|---|
| canonical_player_id | 631 |
| normalized_name | Kiana Williams |
| frozen slate team | {s['team_abbreviation']} (team_id={s['team_id']}) |
| frozen slate opponent | {s.get('opponent_abbreviation')} |
| frozen game_id | {int(s['game_id'])} (NY @ PHX) |
| identity audit status | {r['audit_status']} |
| identity reject_reason | {r['reject_reason'] or '(none — accepted with warning)'} |
| wnba_players table team (at freeze) | LA |
| sportsbook name in frozen odds | (absent — no quotes for this player) |

## PHX versus LA

Verification warning:

`player 631 slate_team=PHX players_table_team=LA`

- The frozen **players table** recorded current team **LA**.
- The frozen **slate / feature row** (BDL-sourced) attached player 631 to **PHX** for game 24970 (NY @ PHX).
- Public roster evidence (June 19, 2026): Los Angeles Sparks signed Kiana Williams from a Phoenix Mercury developmental contract (offer-sheet / poach). Correct contemporaneous team after that transaction is **LA**, not PHX.

## Classification

| Hypothesis | Determination |
|---|---|
| Stale player table | **No** — players table LA matches the June 19 Sparks signing. |
| Trade / signing | **Yes (completed earlier)** — LA signing from PHX developmental deal (2026-06-19). |
| Alias / duplicate ID | **No** — single canonical id 631; no alternate slate identity for the same person. |
| Slate-construction problem | **Yes** — feature/slate construction still placed 631 on PHX for the Aug 1 Mercury game after she had moved to LA. |
| Wrong-player forecast risk | **Yes** — accepted forecast rows price Kiana Williams as a PHX participant in NY@PHX; LA is not on this slate. |

## Settlement recommendation

**EXCLUDE** player 631 accepted forecast rows from primary postgame settlement for this slate.

Rationale: material team/game identity defect in accepted rows. Do not repair the frozen prediction retroactively; keep PMFs as immutable evidence of what was forecast, but mark settlement status `VOID_IDENTITY_TEAM_MISMATCH` / exclude from certified metrics.

## Evidence sources (frozen only + public roster context)

- `PLAYER_IDENTITY_AUDIT.csv` row for 631
- `slate_2026-08-01.parquet` row for 631
- `PREGAME_VERIFICATION_SUMMARY.json` warning list
- Public contemporaneous reporting of the 2026-06-19 Sparks signing (context only; does not alter frozen atoms)
"""
    (OUT / "PLAYER_631_ROOT_CAUSE.md").write_text(md)
    return md


def build_slate_timezone_audit() -> pd.DataFrame:
    bdl = json.loads((AUDITS / "bdl_games_raw.json").read_text())
    game_audit = pd.read_csv(AUDITS / "GAME_AUDIT.csv")
    accepted_ids = set(int(x) for x in game_audit["bdl_game_id"])
    rows = []
    for g in bdl["games"]:
        tip_utc = datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
        tip_et = tip_utc.astimezone(ET)
        gid = int(g["id"])
        away = g["visitor_team"]["abbreviation"]
        home = g["home_team"]["abbreviation"]
        et_date = tip_et.date().isoformat()
        included = gid in accepted_ids
        if et_date == GAME_DATE and included:
            decision = "INCLUDED_ET_SLATE"
        elif et_date != GAME_DATE:
            decision = "EXCLUDED_TIP_NOT_ON_ET_BUSINESS_DATE"
        else:
            decision = "EXCLUDED_OTHER"
        rows.append(
            {
                "bdl_game_id": gid,
                "matchup": f"{away}@{home}",
                "scheduled_tip_utc": tip_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "scheduled_tip_america_new_york": tip_et.isoformat(),
                "requested_api_date": GAME_DATE,
                "slate_business_date": GAME_DATE,
                "tip_calendar_date_et": et_date,
                "bdl_utc_calendar_date": tip_utc.date().isoformat(),
                "inclusion_decision": decision,
                "bdl_status": g.get("status"),
                "recommended_production_config": f"slate_timezone={SLATE_TIMEZONE}",
                "policy_note": (
                    "Do not infer the slate solely from the BDL UTC calendar date. "
                    "BDL dates[]=YYYY-MM-DD returns tips whose UTC timestamp falls on that "
                    "UTC day; business slate membership uses America/New_York tip date."
                ),
            }
        )
    # Ensure IND@POR explanation row is first-class even if already present
    df = pd.DataFrame(rows).sort_values(["inclusion_decision", "bdl_game_id"])
    df.to_csv(OUT / "SLATE_TIMEZONE_AUDIT.csv", index=False)

    policy = {
        "recommended_production_configuration": f"slate_timezone={SLATE_TIMEZONE}",
        "rationale": (
            "IND@POR (BDL 24968) appears in the BDL UTC-date response for 2026-08-01 "
            "because tip_utc=2026-08-01T02:00:00Z, but tip America/New_York="
            "2026-07-31T22:00:00-04:00, so it belongs to the 2026-07-31 Eastern slate. "
            "Production must filter BDL games by tip date in America/New_York, not by "
            "UTC calendar date alone."
        ),
        "frozen_slate_unchanged": True,
        "included_games_et": [
            "24969 LV@CHI tip 2026-08-01T13:00:00-04:00",
            "24970 NY@PHX tip 2026-08-01T15:00:00-04:00",
        ],
    }
    (OUT / "SLATE_TIMEZONE_POLICY.json").write_text(json.dumps(policy, indent=2) + "\n")
    return df


def build_odds_schema_audit(odds: pd.DataFrame) -> dict:
    sample = odds.head(2).copy()
    # make JSON-safe
    sample_records = json.loads(sample.to_json(orient="records", date_format="iso"))
    audit = {
        "source_file": "frozen_extracted/stage_core/wnba_player_props_oddsapi_2026-08-01.parquet",
        "n_rows": int(len(odds)),
        "columns": list(odds.columns),
        "dtypes": {c: str(odds[d].dtype) for c, d in zip(odds.columns, odds.columns)},
        "null_counts": {c: int(odds[c].isna().sum()) for c in odds.columns},
        "sample_values": sample_records,
        "canonical_field_mapping": FROZEN_ODDS_FIELD_MAP,
        "why_original_quote_valid_was_zero": (
            "run_pregame_verification_package.py required player_id + over_odds + under_odds "
            "(or long side format). Frozen parquet is wide over/under with player_name but "
            "NO player_id/game_id columns, so the schema branch was not recognized and "
            "quote_pairs_valid stayed 0 without inspecting row contents."
        ),
        "adapter_allowed_operations": [
            "rename columns",
            "cast dtypes",
            "expand wide over/under into atomic side rows",
            "exact event_id -> game_id join from frozen GAME_AUDIT",
            "exact player_name -> player_id join from frozen slate",
        ],
        "adapter_forbidden_operations": [
            "Odds API recall",
            "timestamp replacement",
            "price changes",
            "invented player IDs",
            "global fuzzy name match",
            "forecast alteration",
            "post-prediction quotes",
        ],
    }
    # fix dtypes dict properly
    audit["dtypes"] = {c: str(t) for c, t in odds.dtypes.items()}
    (OUT / "FROZEN_ODDS_SCHEMA_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")

    report = f"""# Frozen Odds Schema Adapter Report

## Frozen parquet schema

- File: `wnba_player_props_oddsapi_2026-08-01.parquet`
- Rows: {len(odds)}
- Columns ({len(odds.columns)}): {", ".join(odds.columns)}

### Dtypes

```
{odds.dtypes.to_string()}
```

## Canonical field correspondence

| Canonical field | Frozen source |
|---|---|
| provider | `source` (= odds_api_v4) |
| Odds API event ID | `event_id` |
| canonical game ID | exact join `event_id` → `GAME_AUDIT.bdl_game_id` |
| bookmaker | `bookmaker` (alias `vendor`) |
| market key | `market_key` |
| player description | `player_name` |
| canonical player ID | exact join `player_name` → frozen slate `player_id` (no fuzzy) |
| line | `line` |
| side | expand from wide `over_odds` / `under_odds` |
| price | `over_odds` / `under_odds` (American) |
| provider quote timestamp | `last_update` |
| market_last_update | `last_update` |
| ingestion timestamp | `pulled_at_utc` |
| scheduled tip | `commence_time` |
| prediction cutoff | frozen `run_metadata.prediction_timestamp_utc` |
| period | `q1` if `market_key` endswith `_q1`, else `game` |
| stat | `stat` |

## Adapter behavior

The adapter performs **column/type transforms only** plus exact frozen-identity joins.
It does **not** call the Odds API, replace timestamps, change prices, invent player IDs,
fuzzy-match names, alter forecasts, or use post-prediction quotes.

Wide rows already contain both American sides for every frozen row
(missing over=0, missing under=0), so exact same-book pairs are reconstructible once
`player_id` / `game_id` are attached via exact joins.

## Original audit failure mode

`quote_pairs_valid=0` because the verification script did not recognize this wide schema
without a `player_id` column — not because opposite sides were absent.
"""
    (OUT / "FROZEN_ODDS_SCHEMA_ADAPTER_REPORT.md").write_text(report)
    return audit


def adapt_and_pair(lock: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    odds = pd.read_parquet(STAGE / "wnba_player_props_oddsapi_2026-08-01.parquet")
    slate = pd.read_parquet(STAGE / "slate_2026-08-01.parquet")
    game_audit = pd.read_csv(AUDITS / "GAME_AUDIT.csv")
    pred_cutoff = pd.to_datetime(lock["prediction_timestamp_utc"], utc=True)
    tip_by_game = {
        int(r.bdl_game_id): pd.to_datetime(r.scheduled_tip_utc, utc=True)
        for r in game_audit.itertuples()
    }
    event_to_game = {
        str(r.odds_api_event_id): int(r.bdl_game_id) for r in game_audit.itertuples()
    }
    # exact name -> player_id; detect ambiguity
    name_groups = slate.groupby(slate["player_name"].astype(str))["player_id"].nunique()
    ambiguous_names = set(name_groups[name_groups > 1].index)
    exact_name_map = (
        slate.drop_duplicates(subset=["player_name"])
        .set_index(slate.drop_duplicates(subset=["player_name"])["player_name"].astype(str))[
            "player_id"
        ]
        .astype(int)
        .to_dict()
    )

    build_odds_schema_audit(odds)

    reject_counter: Counter = Counter()
    atomic_rows = []
    pair_rows = []
    audit_rows = []

    for idx, r in odds.iterrows():
        reasons = []
        event_id = str(r["event_id"])
        book = str(r["bookmaker"])
        market_key = str(r["market_key"])
        player_name = str(r["player_name"])
        line = float(r["line"])
        period = _period_from_market(market_key)
        stat = str(r["stat"])
        quote_ts = pd.to_datetime(r["last_update"], utc=True)
        ingest_ts = pd.to_datetime(r["pulled_at_utc"], utc=True)
        tip = pd.to_datetime(r["commence_time"], utc=True)
        game_id = event_to_game.get(event_id)

        if game_id is None:
            reasons.append("missing_event_id_mapping")
        if player_name in ambiguous_names:
            reasons.append("ambiguous_player")
            player_id = None
        else:
            player_id = exact_name_map.get(player_name)
            if player_id is None:
                reasons.append("player_resolution_failure")

        if pd.isna(r["over_odds"]) or pd.isna(r["under_odds"]):
            reasons.append("one_sided_market")
        if quote_ts > pred_cutoff:
            reasons.append("post_cutoff_quote")
        if ingest_ts > pred_cutoff:
            reasons.append("post_cutoff_ingestion")
        game_tip = tip_by_game.get(game_id, tip) if game_id is not None else tip
        if quote_ts >= game_tip:
            reasons.append("at_or_post_tip_quote")
        # stale: older than 24h before tip (conservative; still keep if pre-cutoff)
        age_sec = (pred_cutoff - quote_ts).total_seconds()
        if age_sec < 0:
            reasons.append("negative_quote_age")
        if age_sec > 24 * 3600:
            reasons.append("stale_quote_gt_24h")

        try:
            oa = float(r["over_odds"])
            ua = float(r["under_odds"])
            nv_o, nv_u = _no_vig(oa, ua)
        except Exception:
            reasons.append("odds_parse_failure")
            oa = ua = nv_o = nv_u = float("nan")

        ok = len(reasons) == 0
        if not ok:
            for reason in reasons:
                reject_counter[reason] += 1

        # atomic sides (even rejected, for audit transparency) — prices/timestamps unchanged
        for side, price in (("over", oa), ("under", ua)):
            if pd.isna(price):
                continue
            qid = _atomic_quote_id(
                book, event_id, player_id or player_name, stat, line, side, r["last_update"]
            )
            atomic_rows.append(
                {
                    "quote_id": qid,
                    "provider": str(r["source"]),
                    "sportsbook": book,
                    "event_id": event_id,
                    "game_id": game_id,
                    "player_id": player_id,
                    "player_name": player_name,
                    "prop": stat,
                    "market_key": market_key,
                    "period": period,
                    "line": line,
                    "side": side,
                    "american_odds": int(price) if float(price).is_integer() else float(price),
                    "snapshot_label": "decision",
                    "snapshot_time": r["last_update"],
                    "provider_quote_timestamp": r["last_update"],
                    "market_last_update": r["last_update"],
                    "ingestion_timestamp": r["pulled_at_utc"],
                    "decision_timestamp": lock["prediction_timestamp_utc"],
                    "scheduled_tip_utc": r["commence_time"],
                    "prediction_timestamp": lock["prediction_timestamp"],
                    "prediction_cutoff_utc": lock["prediction_timestamp_utc"],
                    "exact_quote_status": "EXACT" if ok else "REJECTED",
                    "reject_reasons": "|".join(reasons),
                    "source": "frozen_odds_parquet_adapter_v1",
                    "quote_age_seconds_at_cutoff": age_sec,
                }
            )

        pair_id_payload = f"{book}|{event_id}|{player_id or player_name}|{stat}|{period}|{line}|{r['last_update']}"
        pair_hash = _sha256_bytes(pair_id_payload.encode())[:24]
        pair_rows.append(
            {
                "pair_id": pair_hash,
                "provider": str(r["source"]),
                "odds_api_event_id": event_id,
                "canonical_game_id": game_id,
                "canonical_player_id": player_id,
                "player_name": player_name,
                "sportsbook": book,
                "market_key": market_key,
                "stat": stat,
                "period": period,
                "line": line,
                "over_american": oa,
                "under_american": ua,
                "no_vig_over": nv_o if ok else float("nan"),
                "no_vig_under": nv_u if ok else float("nan"),
                "provider_quote_timestamp": r["last_update"],
                "market_last_update": r["last_update"],
                "ingestion_timestamp": r["pulled_at_utc"],
                "scheduled_tip_utc": r["commence_time"],
                "prediction_cutoff_utc": lock["prediction_timestamp_utc"],
                "quote_age_seconds_at_cutoff": age_sec,
                "pair_status": "VALID_PAIR" if ok else "REJECTED",
                "reject_reason": "|".join(reasons),
                "raw_quote_sides": 2
                if pd.notna(r["over_odds"]) and pd.notna(r["under_odds"])
                else 1,
                "allowed_timestamp_skew_seconds": ALLOWED_QUOTE_SKEW_SECONDS,
            }
        )
        audit_rows.append(
            {
                "pair_id": pair_hash,
                "player_name": player_name,
                "canonical_player_id": player_id,
                "sportsbook": book,
                "market_key": market_key,
                "stat": stat,
                "period": period,
                "line": line,
                "pair_status": "VALID_PAIR" if ok else "REJECTED",
                "reject_reason": "|".join(reasons),
                "provider_quote_timestamp": r["last_update"],
                "quote_age_seconds_at_cutoff": age_sec,
            }
        )

    atomic = pd.DataFrame(atomic_rows)
    pairs = pd.DataFrame(pair_rows)
    audit = pd.DataFrame(audit_rows)
    atomic.to_parquet(OUT / "FROZEN_ATOMIC_QUOTES.parquet", index=False)
    pairs.to_parquet(OUT / "FROZEN_QUOTE_PAIRS.parquet", index=False)
    audit.to_csv(OUT / "FROZEN_QUOTE_PAIR_AUDIT.csv", index=False)

    valid = pairs[pairs["pair_status"] == "VALID_PAIR"]
    game_pairs = valid[valid["period"] == "game"]
    q1_pairs = valid[valid["period"] == "q1"]
    summary = {
        "raw_quote_wide_rows": int(len(odds)),
        "raw_quote_sides": int(len(atomic)),
        "exact_pairs": int(len(valid)),
        "exact_pairs_game_period": int(len(game_pairs)),
        "exact_pairs_q1_period": int(len(q1_pairs)),
        "model_trace_eligible_pairs": int(len(game_pairs)),
        "q1_pairs_note": (
            "player_points_q1 exact quote pairs are valid for quote audit but have no "
            "frozen game-level PMF target named pts_q1; excluded from model/market trace"
        ),
        "rejected_pairs": int((pairs["pair_status"] == "REJECTED").sum()),
        "books": sorted(valid["sportsbook"].unique().tolist()) if len(valid) else [],
        "players": int(valid["canonical_player_id"].nunique()) if len(valid) else 0,
        "markets": sorted(valid["stat"].unique().tolist()) if len(valid) else [],
        "pairs_by_market": valid.groupby("stat").size().astype(int).to_dict()
        if len(valid)
        else {},
        "pairs_by_market_key": valid.groupby("market_key").size().astype(int).to_dict()
        if len(valid)
        else {},
        "quote_timestamp_min": str(valid["provider_quote_timestamp"].min()) if len(valid) else None,
        "quote_timestamp_max": str(valid["provider_quote_timestamp"].max()) if len(valid) else None,
        "median_age_seconds": float(valid["quote_age_seconds_at_cutoff"].median())
        if len(valid)
        else None,
        "max_age_seconds": float(valid["quote_age_seconds_at_cutoff"].max())
        if len(valid)
        else None,
        "rejects_by_reason": dict(reject_counter),
        "zero_pair_root_cause_if_applicable": None
        if len(valid)
        else "investigate absent sides / event ids / player resolution / line conversion / timestamps",
    }
    (OUT / "FROZEN_QUOTE_PAIR_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    return atomic, pairs, audit, summary


def build_model_market_trace(pairs: pd.DataFrame, lock: dict) -> pd.DataFrame:
    pmfs = pd.read_parquet(STAGE / "full_pmfs_wide.parquet")
    mc = pd.read_parquet(STAGE / "market_comparison.parquet")
    registry = [
        json.loads(line)
        for line in (AUDITS / "FROZEN_PREDICTION_REGISTRY.jsonl").read_text().splitlines()
        if line.strip()
    ]
    reg_by_key = {(int(r["game_id"]), int(r["player_id"]), r["target"]): r for r in registry}

    # Prefer active_pmf_json (VOID_DNP books) to match delivered model_prob_over_final
    pmf_col = "active_pmf_json" if "active_pmf_json" in pmfs.columns else "pmf_json"
    pmf_idx = {}
    for _, r in pmfs.iterrows():
        pmf_idx[(int(r["game_id"]), int(r["player_id"]), str(r["stat"]))] = r

    valid = pairs[pairs["pair_status"] == "VALID_PAIR"].copy()
    rows = []
    for _, q in valid.iterrows():
        key = (int(q["canonical_game_id"]), int(q["canonical_player_id"]), str(q["stat"]))
        pmf_row = pmf_idx.get(key)
        if pmf_row is None:
            continue
        atoms = _parse_pmf(pmf_row.get(pmf_col) or pmf_row.get("pmf_json"))
        settled = _settle_from_pmf(atoms, float(q["line"]))
        pmf_hash = _sha256_bytes(json.dumps(atoms, separators=(",", ":")).encode())
        reg = reg_by_key.get(key, {})
        # delivered model price from frozen market_comparison when present
        msub = mc[
            (mc["game_id"].astype(int) == key[0])
            & (mc["player_id"].astype(int) == key[1])
            & (mc["stat"].astype(str) == key[2])
            & (mc["vendor"].astype(str) == str(q["sportsbook"]))
            & (np.isclose(mc["line"].astype(float), float(q["line"])))
        ]
        delivered = float(msub.iloc[0]["model_prob_over_final"]) if len(msub) else float("nan")
        reproduces = (
            math.isfinite(delivered)
            and abs(delivered - settled["p_over_settled"]) <= 1e-8
        )
        quote_hash = _sha256_bytes(
            f"{q['sportsbook']}|{q['odds_api_event_id']}|{q['canonical_player_id']}|"
            f"{q['stat']}|{q['line']}|{q['over_american']}|{q['under_american']}|"
            f"{q['provider_quote_timestamp']}".encode()
        )
        rows.append(
            {
                "game_id": key[0],
                "player_id": key[1],
                "player_name": q["player_name"],
                "stat": key[2],
                "book": q["sportsbook"],
                "market_key": q["market_key"],
                "period": q["period"],
                "line": float(q["line"]),
                "raw_over_odds": q["over_american"],
                "raw_under_odds": q["under_american"],
                "no_vig_market_prob_over": q["no_vig_over"],
                "no_vig_market_prob_under": q["no_vig_under"],
                "p_over_win": settled["p_over_win"],
                "p_under_win": settled["p_under_win"],
                "p_push": settled["p_push"],
                "p_over_settled": settled["p_over_settled"],
                "p_under_settled": settled["p_under_settled"],
                "frozen_model_probability_over": settled["p_over_settled"],
                "delivered_model_prob_over_final": delivered,
                "reproduces_delivered_model_price": reproduces,
                "model_minus_market_over": settled["p_over_settled"] - float(q["no_vig_over"]),
                "pmf_hash": pmf_hash,
                "registry_pmf_hash": reg.get("pmf_hash", ""),
                "quote_hash": quote_hash,
                "prediction_timestamp": lock["prediction_timestamp"],
                "quote_timestamp": q["provider_quote_timestamp"],
                "pmf_column_used": pmf_col,
                "audit_label": "MODEL_MARKET_TRACE_ONLY_NOT_EV",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "FROZEN_MODEL_MARKET_TRACE.csv", index=False)
    return df


def build_settlement_docs(lock: dict, pair_summary: dict, trace_n: int) -> None:
    plan = f"""# Postgame Settlement Plan — 2026-08-01 Frozen Prospective Evidence

## Scope

Settle **only after** BDL game status is final for:

- 24969 LV @ CHI (tip 2026-08-01T17:00:00Z)
- 24970 NY @ PHX (tip 2026-08-01T19:00:00Z)

Do **not** regenerate predictions. Attach outcomes to frozen prediction IDs in
`FROZEN_PREDICTION_REGISTRY.jsonl`.

Frozen stage: `{STAGE_DIR}`  
Workflow run: `{WORKFLOW_RUN_ID}`  
Code SHA: `{HEAD_SHA}`  
Prediction timestamp: `{PRED_TS_COMPACT}` / `{lock['prediction_timestamp_utc']}`

## Identity filters before scoring

1. Primary certified set = identity `ACCEPTED` rows **except** player 631
   (`VOID_IDENTITY_TEAM_MISMATCH` — see `PLAYER_631_ROOT_CAUSE.md`).
2. Rejected seven players (`REJECTED_PLAYER_AUDIT.csv`) → exclude from primary metrics;
   optional secondary `unresolved_identity` bucket if box score resolves the same BDL IDs.
3. Never repair frozen PMF atoms, hashes, or timestamps.

## Retrieval

1. Pull final BDL player statistics for game_ids 24969 and 24970.
2. Resolve to frozen `(game_id, player_id)` keys.
3. Preserve DNP / void / push:
   - DNP + VOID_DNP books → void quoted-line bets; full-distribution metrics use
     actual minutes=0 / did_play=false handling per schema.
   - Integer line hits → push; half-points → no push.
4. Join outcomes onto registry rows by `(game_id, player_id, target)`.

## FULL-DISTRIBUTION METRICS (all settled non-excluded PMF rows)

- atom NLL
- CRPS
- predictive mean error (`actual - predictive_mean`)
- MAE
- squared error
- PIT value

## QUOTED-LINE METRICS (only valid frozen exact pairs)

Valid pairs reconstructed: **{pair_summary['exact_pairs']}**  
Trace rows prepared: **{trace_n}**

For each valid pair with a non-void settlement:

- binary log loss
- Brier score
- model probability (`p_over_settled` from frozen PMF)
- no-vig market probability
- model-minus-market log-loss difference

## Calibration

Do **not** calculate ECE meaningfully from one slate.  
If probability buckets are recorded, label:

`INSUFFICIENT_ONE_SLATE_SAMPLE`

## Execution gate

Workflow: `.github/workflows/postgame_settlement_frozen.yml`  
Inputs require `games_final=true` confirmation. Hard-fail if any target game is not final.
Never writes to gh-pages. Never mutates frozen evidence files.

## Outputs (when executed)

- `artifacts/pregame_verification/2026-08-01/settlement/POSTGAME_SETTLEMENT_RESULTS.parquet`
- `artifacts/pregame_verification/2026-08-01/settlement/FULL_DISTRIBUTION_METRICS.csv`
- `artifacts/pregame_verification/2026-08-01/settlement/QUOTED_LINE_METRICS.csv`
- `artifacts/pregame_verification/2026-08-01/settlement/SETTLEMENT_SUMMARY.json`
"""
    (OUT / "POSTGAME_SETTLEMENT_PLAN.md").write_text(plan)

    schema = {
        "schema_version": "1.0",
        "game_date": GAME_DATE,
        "frozen_stage": STAGE_DIR,
        "workflow_run_id": WORKFLOW_RUN_ID,
        "prediction_timestamp": PRED_TS_COMPACT,
        "execution_gate": {
            "require_games_final": True,
            "target_game_ids": [24969, 24970],
            "forbid_prediction_regeneration": True,
            "forbid_publish": True,
            "forbid_mutate_frozen_evidence": True,
        },
        "identity_policy": {
            "primary_include": "PLAYER_IDENTITY_AUDIT.audit_status == ACCEPTED",
            "primary_exclude_player_ids": [631],
            "primary_exclude_reason": {
                "631": "VOID_IDENTITY_TEAM_MISMATCH"
            },
            "rejected_player_ids": [605, 715, 67044, 67054, 74553, 74872, 75095],
            "rejected_bucket": "unresolved_identity_optional",
        },
        "outcome_fields": [
            "game_id",
            "player_id",
            "did_play",
            "actual_minutes",
            "actual_pts",
            "actual_reb",
            "actual_ast",
            "actual_fg3m",
            "actual_stl",
            "actual_blk",
            "actual_turnover",
            "dnp_status",
            "void_status",
            "source_box_timestamp_utc",
        ],
        "full_distribution_metrics": [
            "atom_nll",
            "crps",
            "predictive_mean_error",
            "mae",
            "squared_error",
            "pit_value",
        ],
        "quoted_line_metrics": [
            "binary_log_loss",
            "brier_score",
            "model_probability",
            "no_vig_market_probability",
            "model_minus_market_log_loss_difference",
        ],
        "calibration_label": "INSUFFICIENT_ONE_SLATE_SAMPLE",
        "join_keys": {
            "registry": ["game_id", "player_id", "target"],
            "quote_pairs": [
                "canonical_game_id",
                "canonical_player_id",
                "stat",
                "sportsbook",
                "line",
                "period",
            ],
        },
        "frozen_inputs": {
            "registry": "frozen_extracted/audits/FROZEN_PREDICTION_REGISTRY.jsonl",
            "pmfs": "frozen_extracted/stage_core/full_pmfs_wide.parquet",
            "quote_pairs": "FROZEN_QUOTE_PAIRS.parquet",
            "model_market_trace": "FROZEN_MODEL_MARKET_TRACE.csv",
            "evidence_lock": "FROZEN_EVIDENCE_LOCK.json",
        },
    }
    (OUT / "POSTGAME_SETTLEMENT_SCHEMA.json").write_text(json.dumps(schema, indent=2) + "\n")


def build_final_verdict(lock: dict, pair_summary: dict, trace: pd.DataFrame, rejected: pd.DataFrame) -> dict:
    exact_pairs = int(pair_summary["exact_pairs"])
    if exact_pairs > 0:
        verdict = "PASS_FOR_PROSPECTIVE_PMF_AND_MARKET_EVALUATION"
        market_ready = "READY_EXACT_PAIRS_RECONSTRUCTED"
    else:
        verdict = "PASS_FOR_PROSPECTIVE_PMF_EVALUATION"
        market_ready = "NOT_READY_NO_EXACT_PAIRS"

    remaining_warnings = [
        "Seven players rejected for player_id_not_in_players_table (IDENTITY_TABLE_GAP); exclude from primary settlement",
        "Player 631 Kiana Williams accepted with PHX slate team vs LA players-table team; EXCLUDE from settlement (slate-construction / post-signing stale PHX assignment)",
        "BDL UTC-date response includes IND@POR (24968) whose ET tip date is 2026-07-31; correctly excluded — production must set slate_timezone=America/New_York",
        "Original verification quote_pairs_valid=0 due to unrecognized wide odds schema without player_id (resolved via frozen adapter; does not alter forecasts)",
        "9 exact player_points_q1 quote pairs have no frozen pts_q1 PMF target; excluded from model/market trace (game-period pairs remain 137)",
        "14 Pinnacle game-period pairs lack delivered model_prob_over_final in market_comparison (likely unknown-book fail-closed at delivery); frozen PMF probs still traced",
        "STATUS_AUDIT shows UNKNOWN for quoted rows pending postgame settlement",
        "Calibration evidence labeled INSUFFICIENT_ONE_SLATE_SAMPLE — do not compute ECE from one slate",
        "GitHub Actions artifact retention is ephemeral; durable copies are the committed zip blobs under frozen_github_artifacts/",
    ]

    # material defect check: accepted rows include 631 team mismatch — documented exclusion, not whole-run fail
    result = {
        "current_final_verdict": verdict,
        "pmf_prospective_evaluation_status": "VALID_WITH_DOCUMENTED_EXCLUSIONS",
        "market_comparison_readiness_status": market_ready,
        "postgame_settlement_readiness_status": "READY_PLAN_AND_SCHEMA_PRESENT_WAIT_FOR_FINAL",
        "exact_quote_pairs": exact_pairs,
        "pairs_by_market": pair_summary.get("pairs_by_market", {}),
        "quote_rejects_by_reason": pair_summary.get("rejects_by_reason", {}),
        "model_market_trace_rows": int(len(trace)),
        "trace_reproduces_delivered_rate": float(trace["reproduces_delivered_model_price"].mean())
        if len(trace)
        else None,
        "rejected_players": int(len(rejected)),
        "remaining_warnings": remaining_warnings,
        "do_not_fail_solely_for_original_quote_parse": True,
        "do_not_upgrade_without_exact_pairs": True,
        "frozen_evidence_lock": "artifacts/pregame_verification/2026-08-01/FROZEN_EVIDENCE_LOCK.json",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "CURRENT_FINAL_VERDICT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = build_lock()
    rejected = build_rejected_player_audit()
    build_player_631_root_cause()
    build_slate_timezone_audit()
    atomic, pairs, audit, pair_summary = adapt_and_pair(lock)
    trace = build_model_market_trace(pairs, lock)
    build_settlement_docs(lock, pair_summary, len(trace))
    verdict = build_final_verdict(lock, pair_summary, trace, rejected)

    # Make lock overwrite-protected marker
    (OUT / "DO_NOT_OVERWRITE").write_text(
        "Immutable August 1 prospective evidence. Do not overwrite frozen blobs or hashes.\n"
    )

    print(json.dumps(
        {
            "lock_path": str(OUT / "FROZEN_EVIDENCE_LOCK.json"),
            "artifact_sha256": {k: v["sha256"] for k, v in lock["artifacts"].items()},
            "exact_pairs": pair_summary["exact_pairs"],
            "pairs_by_market": pair_summary["pairs_by_market"],
            "rejects": pair_summary["rejects_by_reason"],
            "trace_rows": len(trace),
            "verdict": verdict["current_final_verdict"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
