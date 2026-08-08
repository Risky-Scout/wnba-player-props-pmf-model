#!/usr/bin/env python3
"""Settle a frozen pregame-verification stage after games are final.

Never regenerates predictions. Never mutates FROZEN_EVIDENCE_LOCK or frozen blobs.
Writes only under artifacts/pregame_verification/<date>/settlement/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]


def _parse_pmf(raw):
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        return sorted(((float(k), float(v)) for k, v in raw.items()), key=lambda x: x[0])
    out = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((float(item[0]), float(item[1])))
    return sorted(out, key=lambda x: x[0])


def _atom_nll(atoms, actual):
    dens = {int(k): p for k, p in atoms}
    p = dens.get(int(round(actual)), 0.0)
    return float(-math.log(max(p, 1e-15)))


def _crps_integer(atoms, actual):
    # Empirical CRPS for discrete non-negative support via cumulative F.
    if not atoms:
        return float("nan")
    ks = [int(k) for k, _ in atoms]
    ps = [p for _, p in atoms]
    kmax = max(max(ks), int(actual))
    pmf = np.zeros(kmax + 1)
    for k, p in zip(ks, ps):
        if 0 <= int(k) <= kmax:
            pmf[int(k)] += p
    cdf = np.cumsum(pmf)
    obs = np.zeros(kmax + 1)
    a = int(actual)
    if a <= kmax:
        obs[a:] = 1.0
    else:
        obs[:] = 1.0
    return float(np.sum((cdf - obs) ** 2))


def _pit(atoms, actual):
    return float(sum(p for k, p in atoms if k < actual) + 0.5 * sum(p for k, p in atoms if k == actual))


def _fetch_final_stats(api_key: str, game_ids: list[int]) -> pd.DataFrame:
    headers = {"Authorization": api_key}
    rows = []
    for gid in game_ids:
        gr = requests.get(
            f"https://api.balldontlie.io/wnba/v1/games/{gid}",
            headers=headers,
            timeout=60,
        )
        gr.raise_for_status()
        game = gr.json().get("data") or gr.json()
        status = str(game.get("status", "")).lower()
        if status not in {"final", "post"}:
            raise SystemExit(f"game {gid} not final (status={status})")
        page = 1
        while True:
            rr = requests.get(
                "https://api.balldontlie.io/wnba/v1/stats",
                headers=headers,
                params={"game_ids[]": gid, "per_page": 100, "page": page},
                timeout=60,
            )
            rr.raise_for_status()
            payload = rr.json()
            data = payload.get("data") or []
            for s in data:
                player = s.get("player") or {}
                mins = s.get("min") or s.get("minutes") or "0"
                try:
                    if isinstance(mins, str) and ":" in mins:
                        mm, ss = mins.split(":")
                        minutes = float(mm) + float(ss) / 60.0
                    else:
                        minutes = float(mins or 0)
                except Exception:
                    minutes = 0.0
                pid = int(player.get("id"))
                did_play = minutes > 0
                rows.append(
                    {
                        "game_id": gid,
                        "player_id": pid,
                        "did_play": did_play,
                        "actual_minutes": minutes,
                        "actual_pts": s.get("pts"),
                        "actual_reb": s.get("reb"),
                        "actual_ast": s.get("ast"),
                        "actual_fg3m": s.get("fg3m"),
                        "actual_stl": s.get("stl"),
                        "actual_blk": s.get("blk"),
                        "actual_turnover": s.get("turnover"),
                        "dnp_status": "DNP" if not did_play else "PLAYED",
                        "void_status": "VOID_DNP" if not did_play else "ACTIVE",
                        "game_status": status,
                    }
                )
            meta = payload.get("meta") or {}
            if page >= int(meta.get("total_pages") or 1):
                break
            page += 1
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-date", default="2026-08-01")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = REPO / "artifacts" / "pregame_verification" / args.game_date
    lock = json.loads((base / "FROZEN_EVIDENCE_LOCK.json").read_text())
    schema = json.loads((base / "POSTGAME_SETTLEMENT_SCHEMA.json").read_text())
    out = base / "settlement"
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        (out / "SETTLEMENT_SUMMARY.json").write_text(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "frozen_stage": lock["frozen_stage"],
                    "workflow_run_id": lock["workflow_run_id"],
                    "message": "Gates only; no BDL pull / metrics.",
                },
                indent=2,
            )
            + "\n"
        )
        return

    game_ids = list(schema["execution_gate"]["target_game_ids"])
    outcomes = _fetch_final_stats(args.api_key, game_ids)
    outcomes.to_parquet(out / "BDL_FINAL_PLAYER_STATS.parquet", index=False)

    pmfs = pd.read_parquet(base / "frozen_extracted" / "stage_core" / "full_pmfs_wide.parquet")
    identity = pd.read_csv(
        base / "frozen_extracted" / "audits" / "PLAYER_IDENTITY_AUDIT.csv"
    )
    exclude = set(int(x) for x in schema["identity_policy"]["primary_exclude_player_ids"])
    rejected = set(int(x) for x in schema["identity_policy"]["rejected_player_ids"])
    accepted = set(
        int(x)
        for x in identity.loc[identity["audit_status"] == "ACCEPTED", "canonical_player_id"]
    )
    primary = accepted - exclude

    # Map stat -> outcome column
    stat_col = {
        "pts": "actual_pts",
        "reb": "actual_reb",
        "ast": "actual_ast",
        "fg3m": "actual_fg3m",
        "stl": "actual_stl",
        "blk": "actual_blk",
        "turnover": "actual_turnover",
    }
    # composites from components when present
    out_idx = outcomes.set_index(["game_id", "player_id"])

    dist_rows = []
    pmf_col = "active_pmf_json" if "active_pmf_json" in pmfs.columns else "pmf_json"
    for _, r in pmfs.iterrows():
        gid, pid, stat = int(r["game_id"]), int(r["player_id"]), str(r["stat"])
        if pid not in primary:
            continue
        if (gid, pid) not in out_idx.index:
            continue
        o = out_idx.loc[(gid, pid)]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[0]
        atoms = _parse_pmf(r.get(pmf_col) or r.get("pmf_json"))
        mean = sum(k * p for k, p in atoms) if atoms else float("nan")
        # actual for atomic stats only; composites computed if components exist
        if stat in stat_col:
            actual = o.get(stat_col[stat])
        elif stat == "pts_reb":
            actual = float(o["actual_pts"]) + float(o["actual_reb"])
        elif stat == "pts_ast":
            actual = float(o["actual_pts"]) + float(o["actual_ast"])
        elif stat == "reb_ast":
            actual = float(o["actual_reb"]) + float(o["actual_ast"])
        elif stat == "pts_reb_ast":
            actual = float(o["actual_pts"]) + float(o["actual_reb"]) + float(o["actual_ast"])
        elif stat == "stocks":
            actual = float(o["actual_stl"]) + float(o["actual_blk"])
        else:
            continue
        if actual is None or (isinstance(actual, float) and math.isnan(actual)):
            continue
        actual_f = float(actual)
        err = actual_f - mean
        dist_rows.append(
            {
                "game_id": gid,
                "player_id": pid,
                "stat": stat,
                "actual": actual_f,
                "predictive_mean": mean,
                "predictive_mean_error": err,
                "mae": abs(err),
                "squared_error": err * err,
                "atom_nll": _atom_nll(atoms, actual_f),
                "crps": _crps_integer(atoms, actual_f),
                "pit_value": _pit(atoms, actual_f),
                "dnp_status": o["dnp_status"],
                "void_status": o["void_status"],
                "identity_bucket": "primary_certified",
            }
        )
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out / "FULL_DISTRIBUTION_METRICS.csv", index=False)

    trace = pd.read_csv(base / "FROZEN_MODEL_MARKET_TRACE.csv")
    qrows = []
    for _, t in trace.iterrows():
        gid, pid = int(t["game_id"]), int(t["player_id"])
        if pid not in primary or pid in rejected:
            continue
        if (gid, pid) not in out_idx.index:
            continue
        o = out_idx.loc[(gid, pid)]
        if isinstance(o, pd.DataFrame):
            o = o.iloc[0]
        if o["void_status"] == "VOID_DNP":
            qrows.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "stat": t["stat"],
                    "book": t["book"],
                    "line": t["line"],
                    "settlement": "VOID_DNP",
                    "binary_log_loss": float("nan"),
                    "brier_score": float("nan"),
                    "model_probability": t["frozen_model_probability_over"],
                    "no_vig_market_probability": t["no_vig_market_prob_over"],
                    "model_minus_market_log_loss_difference": float("nan"),
                    "calibration_label": "INSUFFICIENT_ONE_SLATE_SAMPLE",
                }
            )
            continue
        stat = str(t["stat"])
        if stat in stat_col:
            actual = float(o[stat_col[stat]])
        elif stat == "pts_reb":
            actual = float(o["actual_pts"]) + float(o["actual_reb"])
        elif stat == "pts_ast":
            actual = float(o["actual_pts"]) + float(o["actual_ast"])
        elif stat == "reb_ast":
            actual = float(o["actual_reb"]) + float(o["actual_ast"])
        elif stat == "pts_reb_ast":
            actual = float(o["actual_pts"]) + float(o["actual_reb"]) + float(o["actual_ast"])
        else:
            continue
        line = float(t["line"])
        if abs(actual - line) < 1e-12:
            # push
            qrows.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "stat": stat,
                    "book": t["book"],
                    "line": line,
                    "settlement": "PUSH",
                    "binary_log_loss": float("nan"),
                    "brier_score": float("nan"),
                    "model_probability": t["frozen_model_probability_over"],
                    "no_vig_market_probability": t["no_vig_market_prob_over"],
                    "model_minus_market_log_loss_difference": float("nan"),
                    "calibration_label": "INSUFFICIENT_ONE_SLATE_SAMPLE",
                }
            )
            continue
        y = 1.0 if actual > line else 0.0
        p_m = float(t["frozen_model_probability_over"])
        p_mk = float(t["no_vig_market_prob_over"])
        p_m = min(max(p_m, 1e-15), 1 - 1e-15)
        p_mk = min(max(p_mk, 1e-15), 1 - 1e-15)
        ll_m = -(y * math.log(p_m) + (1 - y) * math.log(1 - p_m))
        ll_mk = -(y * math.log(p_mk) + (1 - y) * math.log(1 - p_mk))
        qrows.append(
            {
                "game_id": gid,
                "player_id": pid,
                "stat": stat,
                "book": t["book"],
                "line": line,
                "settlement": "OVER" if y == 1 else "UNDER",
                "binary_log_loss": ll_m,
                "brier_score": (p_m - y) ** 2,
                "model_probability": p_m,
                "no_vig_market_probability": p_mk,
                "model_minus_market_log_loss_difference": ll_m - ll_mk,
                "calibration_label": "INSUFFICIENT_ONE_SLATE_SAMPLE",
            }
        )
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(out / "QUOTED_LINE_METRICS.csv", index=False)

    summary = {
        "status": "SETTLED",
        "frozen_stage": lock["frozen_stage"],
        "workflow_run_id": lock["workflow_run_id"],
        "prediction_timestamp": lock["prediction_timestamp"],
        "n_outcome_rows": int(len(outcomes)),
        "n_full_distribution_metric_rows": int(len(dist)),
        "n_quoted_line_metric_rows": int(len(qdf)),
        "excluded_player_ids": sorted(exclude),
        "rejected_player_ids_excluded_from_primary": sorted(rejected),
        "calibration_label": "INSUFFICIENT_ONE_SLATE_SAMPLE",
        "prediction_regenerated": False,
        "frozen_evidence_mutated": False,
    }
    (out / "SETTLEMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
