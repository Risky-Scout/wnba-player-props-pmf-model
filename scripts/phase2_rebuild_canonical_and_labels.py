#!/usr/bin/env python3
"""Phase-2 offline canonical rebuild + participation / conditional-minutes audits.

Uses existing bdl_full caches and recovered shooting labels. Does not fit models.
Does not change the production pointer. Targeted API only for genuinely missing
final game_ids when --allow-targeted-api is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from wnba_props_model.data.normalize import shooting_identity_violations
from wnba_props_model.data.participation_labels import (
    CONFIRMED_ACTIVE,
    CONFIRMED_INACTIVE,
    INFERRED_ELIGIBLE_DNP,
    UNKNOWN_ROSTER_ELIGIBILITY,
    build_conditional_minutes_training_table,
    build_participation_labels,
    participation_counts_by_season,
)
from wnba_props_model.data.injury_workbook import (
    assert_no_onset_leakage,
    eligibility_evidence_from_injury_events,
    identity_summary,
    load_injury_workbook,
)

REPO = Path(__file__).resolve().parents[1]
MISSING_GAME_IDS = [
    {"season": 2023, "game_id": 3368, "game_date": "2023-06-07"},
    {"season": 2026, "game_id": 24935, "game_date": "2026-07-17"},
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _merge_shooting(stats: pd.DataFrame, shooting: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    if "fgm" not in df.columns:
        df["fgm"] = pd.NA
    if "ftm" not in df.columns:
        df["ftm"] = pd.NA
    if shooting is not None and not shooting.empty:
        sl = shooting[["game_id", "player_id", "fgm", "ftm"]].drop_duplicates(
            ["game_id", "player_id"]
        ).rename(columns={"fgm": "_sl_fgm", "ftm": "_sl_ftm"})
        df = df.merge(sl, on=["game_id", "player_id"], how="left")
        df["fgm"] = df["_sl_fgm"].combine_first(df["fgm"])
        df["ftm"] = df["_sl_ftm"].combine_first(df["ftm"])
        df = df.drop(columns=["_sl_fgm", "_sl_ftm"])
    for c in ["fga", "fg3a", "fg3m", "fta", "fgm", "ftm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"fga", "fg3a"}.issubset(df.columns):
        df["fg2a"] = df["fga"] - df["fg3a"]
    if {"fgm", "fg3m"}.issubset(df.columns):
        df["fg2m"] = df["fgm"] - df["fg3m"]
    return df


def _targeted_fetch(game_ids: list[int]) -> tuple[pd.DataFrame, list[dict]]:
    key = os.environ.get("BDL_API_KEY")
    reqs: list[dict] = []
    if not key or not game_ids:
        return pd.DataFrame(), reqs
    rows = []
    for gid in game_ids:
        url = "https://api.balldontlie.io/wnba/v1/player_stats"
        r = requests.get(
            url,
            headers={"Authorization": key},
            params={"game_ids[]": gid, "per_page": 100},
            timeout=30,
        )
        data = r.json().get("data", []) if r.status_code == 200 else []
        reqs.append({
            "endpoint": "/wnba/v1/player_stats",
            "game_id": gid,
            "status": r.status_code,
            "n_rows": len(data) if isinstance(data, list) else 0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        if r.status_code != 200:
            continue
        for s in data:
            rows.append({
                "game_id": s["game"]["id"],
                "player_id": s["player"]["id"],
                "game_date": (s.get("game") or {}).get("date", "")[:10],
                "season": (s.get("game") or {}).get("season"),
                "team_id": (s.get("team") or {}).get("id"),
                "minutes": s.get("min"),
                "pts": s.get("pts"),
                "reb": s.get("reb"),
                "ast": s.get("ast"),
                "stl": s.get("stl"),
                "blk": s.get("blk"),
                "turnover": s.get("turnover"),
                "fgm": s.get("fgm"),
                "fga": s.get("fga"),
                "fg3m": s.get("fg3m"),
                "fg3a": s.get("fg3a"),
                "ftm": s.get("ftm"),
                "fta": s.get("fta"),
                "oreb": s.get("oreb"),
                "dreb": s.get("dreb"),
                "pf": s.get("pf"),
                "plus_minus": s.get("plus_minus"),
                "source": "bdl_targeted",
            })
    return pd.DataFrame(rows), reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/bdl_full")
    ap.add_argument("--shooting-labels", default="data/recovered_v2/wnba_player_shooting_labels.parquet")
    ap.add_argument("--features", default="data/processed/wnba_player_game_features_wide.recovered_v2_20260725.parquet")
    ap.add_argument("--out-dir", default="artifacts/phase2_repair")
    ap.add_argument("--canonical-out", default="data/processed/phase2")
    ap.add_argument("--injury-workbook", default="")
    ap.add_argument("--allow-targeted-api", action="store_true")
    ap.add_argument("--feature-contract-hash", default="adce920e467a64e5")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    canon = Path(args.canonical_out)
    canon.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    raw_stats_path = Path(args.raw_dir) / "wnba_player_game_stats.parquet"
    games_path = Path(args.raw_dir) / "wnba_games.parquet"
    stats = pd.read_parquet(raw_stats_path)
    games = pd.read_parquet(games_path) if games_path.exists() else pd.DataFrame()

    api_requests: list[dict] = []
    present_final = set(stats["game_id"].dropna().astype(int).unique())
    missing = [
        g for g in MISSING_GAME_IDS
        if int(g["game_id"]) not in present_final
    ]
    if args.allow_targeted_api and missing:
        fetched, api_requests = _targeted_fetch([int(g["game_id"]) for g in missing])
        if not fetched.empty:
            # Normalize minutes via existing parser for consistency
            from wnba_props_model.data.normalize import parse_minutes, parse_minutes_flag
            fetched["minutes_raw"] = fetched["minutes"].astype(str)
            fetched["minutes_flag"] = fetched["minutes"].map(parse_minutes_flag)
            fetched["minutes"] = fetched["minutes"].map(parse_minutes)
            stats = pd.concat([stats, fetched], ignore_index=True, sort=False)

    shooting = (
        pd.read_parquet(args.shooting_labels)
        if Path(args.shooting_labels).exists()
        else pd.DataFrame()
    )
    stats = _merge_shooting(stats, shooting)

    # Rebound reconciliation on rebuilt table
    if {"oreb", "dreb", "reb"}.issubset(stats.columns):
        o = pd.to_numeric(stats["oreb"], errors="coerce").fillna(0).astype(int)
        d = pd.to_numeric(stats["dreb"], errors="coerce").fillna(0).astype(int)
        reb = pd.to_numeric(stats["reb"], errors="coerce").fillna(0).astype(int)
        stats["reb"] = reb
        stats["reb_oreb_dreb_sum"] = o + d
        stats["reb_reconcile_flag"] = [
            "match" if a == b else "provider_or_team_reb_discrepancy"
            for a, b in zip(stats["reb_oreb_dreb_sum"], reb, strict=False)
        ]

    if {"fga", "fg3a"}.issubset(stats.columns):
        stats["fg2a"] = pd.to_numeric(stats["fga"], errors="coerce") - pd.to_numeric(
            stats["fg3a"], errors="coerce"
        )
    if {"fgm", "fg3m"}.issubset(stats.columns):
        stats["fg2m"] = pd.to_numeric(stats["fgm"], errors="coerce") - pd.to_numeric(
            stats["fg3m"], errors="coerce"
        )

    stats_path = canon / "wnba_player_game_stats.parquet"
    stats.to_parquet(stats_path, index=False)

    # Shooting-label table from rows with fgm/ftm
    shoot_cols = [
        "game_id", "player_id", "game_date", "season",
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "fg2m", "fg2a",
        "oreb", "dreb", "reb", "pts",
    ]
    have = [c for c in shoot_cols if c in stats.columns]
    shooting_tbl = stats.dropna(subset=[c for c in ["fgm", "ftm"] if c in stats.columns]).copy()
    shooting_tbl = shooting_tbl[have] if have else pd.DataFrame()
    shoot_path = canon / "wnba_player_shooting_labels.parquet"
    if not shooting_tbl.empty:
        shooting_tbl.to_parquet(shoot_path, index=False)

    viol = shooting_identity_violations(shooting_tbl if not shooting_tbl.empty else stats)
    coverage = []
    if not shooting_tbl.empty and "season" in shooting_tbl.columns:
        for season, g in shooting_tbl.groupby("season"):
            coverage.append({
                "season": int(season),
                "rows_with_fgm_ftm": int(len(g)),
                "players": int(g["player_id"].nunique()),
                "games": int(g["game_id"].nunique()),
            })
    pd.DataFrame(coverage).to_csv(out / "SHOOTING_LABEL_RECONCILIATION.csv", index=False)

    reb_rows = []
    if "reb_reconcile_flag" in stats.columns:
        for season, g in stats.groupby("season"):
            vc = g["reb_reconcile_flag"].value_counts().to_dict()
            reb_rows.append({
                "season": int(season),
                "rows": int(len(g)),
                "match": int(vc.get("match", 0)),
                "discrepancy": int(vc.get("provider_or_team_reb_discrepancy", 0)),
            })
    pd.DataFrame(reb_rows).to_csv(out / "REBOUND_RECONCILIATION.csv", index=False)

    # Injury evidence (optional private workbook — never committed)
    evid = pd.DataFrame()
    injury_identity = {"exact_roster_name": 0, "unresolved": 0, "ambiguous_name": 0}
    injury_path = Path(args.injury_workbook) if args.injury_workbook else None
    if injury_path and injury_path.exists():
        roster: dict[str, list[int]] = {}
        # Prefer names observed on box scores (broader than current players table).
        if "player_name" in stats.columns and "player_id" in stats.columns:
            for pid, name in (
                stats[["player_id", "player_name"]].dropna().drop_duplicates().itertuples(index=False)
            ):
                key = " ".join(str(name).strip().lower().split())
                if key:
                    ids = roster.setdefault(key, [])
                    if int(pid) not in ids:
                        ids.append(int(pid))
        players_path = Path(args.raw_dir) / "wnba_players.parquet"
        if players_path.exists():
            pl = pd.read_parquet(players_path)
            name_col = "player_name" if "player_name" in pl.columns else None
            if name_col:
                for _, r in pl.iterrows():
                    key = " ".join(str(r[name_col]).strip().lower().split())
                    if not key:
                        continue
                    ids = roster.setdefault(key, [])
                    if int(r["player_id"]) not in ids:
                        ids.append(int(r["player_id"]))
        # Optional exact aliases from config (never fuzzy).
        alias_path = REPO / "config" / "player_alias_table_v2.json"
        if alias_path.exists():
            try:
                alias_obj = json.loads(alias_path.read_text())
                for name, pid in (alias_obj.get("exact") or alias_obj.get("aliases") or {}).items():
                    if isinstance(pid, dict):
                        continue
                    key = " ".join(str(name).strip().lower().split())
                    ids = roster.setdefault(key, [])
                    if int(pid) not in ids:
                        ids.append(int(pid))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        try:
            events = load_injury_workbook(injury_path, roster_name_to_ids=roster)
            injury_identity = identity_summary(events)
            evid = eligibility_evidence_from_injury_events(events, stats)
            _write_json(out / "INJURY_WORKBOOK_IDENTITY_AGGREGATE.json", {
                "identity_counts": injury_identity,
                "evidence_rows": int(len(evid)),
                "private_rows_committed": False,
            })
        except ImportError as exc:
            _write_json(out / "INJURY_WORKBOOK_IDENTITY_AGGREGATE.json", {
                "identity_counts": injury_identity,
                "evidence_rows": 0,
                "private_rows_committed": False,
                "error": str(exc),
            })

    labels = build_participation_labels(stats, eligibility_evidence=evid if not evid.empty else None)
    labels_path = canon / "wnba_participation_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    counts = participation_counts_by_season(labels)
    # Wide season x class
    wide_rows = []
    for season, g in labels.groupby("season"):
        vc = g["participation_label_class"].value_counts().to_dict()
        wide_rows.append({
            "season": int(season),
            "CONFIRMED_ACTIVE": int(vc.get(CONFIRMED_ACTIVE, 0)),
            "CONFIRMED_INACTIVE": int(vc.get(CONFIRMED_INACTIVE, 0)),
            "INFERRED_ELIGIBLE_DNP": int(vc.get(INFERRED_ELIGIBLE_DNP, 0)),
            "UNKNOWN_ROSTER_ELIGIBILITY": int(vc.get(UNKNOWN_ROSTER_ELIGIBILITY, 0)),
            "training_eligible": int(g["training_eligible"].sum()),
            "inferred_excluded_from_training": int(
                ((g["participation_label_class"] == INFERRED_ELIGIBLE_DNP)
                 & (~g["training_eligible"])).sum()
            ),
            "unknown_excluded_from_training": int(
                ((g["participation_label_class"] == UNKNOWN_ROSTER_ELIGIBILITY)
                 & (~g["training_eligible"])).sum()
            ),
        })
    pd.DataFrame(wide_rows).to_csv(out / "PARTICIPATION_LABEL_COUNTS.csv", index=False)

    # Conditional minutes training table (active only)
    feature_path = Path(args.features)
    cond_rows = 0
    feat_hash = ""
    if feature_path.exists():
        feats = pd.read_parquet(feature_path)
        feat_hash = _sha256_file(feature_path)
        # Use a small safe lag feature set present in the wide matrix
        candidates = [
            c for c in feats.columns
            if c.startswith("player_minutes_") or c.startswith("player_pts_mean_")
            or c in {"player_rest_days", "is_home", "team_rest_days"}
        ]
        # Exclude leakage
        assert_no_onset_leakage(candidates)
        feature_cols = [c for c in candidates if c not in {"actual_minutes", "minutes", "did_play"}][:40]
        if feature_cols:
            cond = build_conditional_minutes_training_table(
                labels,
                feats,
                feature_cols=feature_cols,
                feature_cutoff="prior_game_date_shift1",
                data_hash=_sha256_file(stats_path),
                feature_contract_hash=args.feature_contract_hash,
            )
            cond_path = canon / "conditional_minutes_training.parquet"
            cond.to_parquet(cond_path, index=False)
            cond_rows = int(len(cond))
            _write_json(out / "CONDITIONAL_MINUTES_TRAINING_AUDIT.json", {
                "rows": cond_rows,
                "label_class": CONFIRMED_ACTIVE,
                "feature_cols": feature_cols,
                "feature_cutoff": "prior_game_date_shift1",
                "data_hash": _sha256_file(stats_path),
                "feature_contract_hash": args.feature_contract_hash,
                "features_file_hash": feat_hash,
                "includes_inferred_dnp": False,
                "includes_unknown": False,
                "model_fitted": False,
            })
    else:
        _write_json(out / "CONDITIONAL_MINUTES_TRAINING_AUDIT.json", {
            "rows": 0,
            "note": f"features file missing: {feature_path}",
            "model_fitted": False,
        })

    confirmed_inactive_sources = (
        labels.loc[
            labels["participation_label_class"] == CONFIRMED_INACTIVE, "label_source"
        ]
        .value_counts()
        .to_dict()
    )

    _write_json(out / "PARTICIPATION_LABEL_AUDIT.json", {
        "generated_at_utc": ts,
        "rows": int(len(labels)),
        "counts_by_season": wide_rows,
        "long_counts": counts.to_dict(orient="records"),
        "confirmed_inactive_evidence_sources": confirmed_inactive_sources,
        "inferred_dnp_excluded_from_training": int(
            ((labels["participation_label_class"] == INFERRED_ELIGIBLE_DNP)
             & (~labels["training_eligible"])).sum()
        ),
        "unknown_excluded_from_training": int(
            ((labels["participation_label_class"] == UNKNOWN_ROSTER_ELIGIBILITY)
             & (~labels["training_eligible"])).sum()
        ),
        "injury_identity_counts": injury_identity,
        "training_policy": {
            "CONFIRMED_ACTIVE": {"training_eligible": True, "weight": 1.0},
            "CONFIRMED_INACTIVE": {"training_eligible": True, "weight": 1.0},
            "INFERRED_ELIGIBLE_DNP": {"training_eligible": False, "weight": 0.0},
            "UNKNOWN_ROSTER_ELIGIBILITY": {
                "training_eligible": False,
                "binary": None,
                "weight": 0.0,
            },
        },
        "model_fitted": False,
    })

    _write_json(out / "CANONICAL_REBUILD_AUDIT.json", {
        "generated_at_utc": ts,
        "raw_dir": args.raw_dir,
        "player_game_rows": int(len(stats)),
        "seasons": sorted(int(s) for s in stats["season"].dropna().unique()),
        "shooting_label_rows": int(len(shooting_tbl)),
        "shooting_coverage_by_season": coverage,
        "shooting_identity_violations": viol,
        "rebound_reconciliation": reb_rows,
        "fgm_ftm_source": "recovered_shooting_labels_merged_into_bdl_full_cache",
        "missing_final_games_checked": MISSING_GAME_IDS,
        "missing_final_games_still_absent": missing,
        "targeted_bdl_requests": api_requests,
        "canonical_stats_path": str(stats_path),
        "production_pointer_unchanged": True,
        "model_fitted": False,
        "games_rows": int(len(games)) if not games.empty else 0,
    })

    # Aggregate injury leakage check artifact (no private rows)
    _write_json(out / "INJURY_LEAKAGE_CHECKS.json", {
        "date_returned_prohibited_as_onset_feature": True,
        "total_games_missed_prohibited_as_onset_feature": True,
        "unresolved_identities_cannot_confirm_inactive": True,
        "ambiguous_identities_cannot_confirm_inactive": True,
        "unresolved_2026_returns_remain_open": True,
        "private_injury_rows_committed": False,
    })

    print(json.dumps({
        "player_game_rows": int(len(stats)),
        "shooting_rows": int(len(shooting_tbl)),
        "participation_rows": int(len(labels)),
        "conditional_minutes_rows": cond_rows,
        "api_requests": len(api_requests),
    }, indent=2))


if __name__ == "__main__":
    main()
