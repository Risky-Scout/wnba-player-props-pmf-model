#!/usr/bin/env python3
"""Thin CLI wrapper around the authoritative V6 ``predict_slate`` inference function.

Daily workflow loads a frozen bundle and must not retrain.
Production mode fails closed on identity, bundle, and PMF integrity errors.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.bundle import BundleIntegrityError, load_bundle
from wnba_props_model.sharp_v6.identity import IdentityResolutionError
from wnba_props_model.sharp_v6.inference import InferenceError, predict_slate
from wnba_props_model.sharp_v6.release import build_deployment_receipt

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]
BDL = "https://api.balldontlie.io/wnba/v1"
ODDS = "https://api.the-odds-api.com/v4/sports/basketball_wnba"


@app.command()
def main(
    date: str = typer.Option(None, "--date", help="Slate date YYYY-MM-DD"),
    bundle_dir: str = typer.Option(
        "artifacts/releases/wnba-pmf-production-v1.1", "--bundle-dir",
        help="Explicit immutable production bundle directory (required path).",
    ),
    out_dir: str = typer.Option(None, "--out-dir"),
    features: str = typer.Option(
        "data/recovered_v2/modeling/wnba_pregame_features_t12.parquet", "--features",
    ),
    stats: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--stats"),
    games: str = typer.Option("data/recovered_v2/wnba_games.parquet", "--games"),
    mode: str = typer.Option(
        "production",
        "--mode",
        help="production | research | validation | offline_fixture",
    ),
    rosters_json: str = typer.Option(
        None, "--rosters-json", help="Optional {team_id: [player_id, ...]} snapshot",
    ),
) -> None:
    if mode not in {"production", "research", "validation", "offline_fixture"}:
        raise SystemExit(f"invalid mode: {mode}")

    ts = datetime.now(timezone.utc).isoformat()
    try:
        bundle = load_bundle(bundle_dir)
    except (BundleIntegrityError, FileNotFoundError) as e:
        raise SystemExit(f"FAIL_CLOSED bundle: {e}") from e

    if bundle.meta.get("retrain_in_daily") or (
        isinstance(bundle.meta.get("manifest"), dict)
        and bundle.meta["manifest"].get("retrain_in_daily")
    ):
        raise SystemExit("bundle misconfigured: daily path must not retrain")

    for path, label in ((features, "features"), (stats, "stats"), (games, "games")):
        if not Path(path).exists():
            raise SystemExit(f"FAIL_CLOSED: missing {label} path {path}")

    feat = pd.read_parquet(features)
    st = pd.read_parquet(stats)
    st["game_date"] = pd.to_datetime(st["game_date"])
    feat["game_date"] = pd.to_datetime(feat["game_date"])
    games_df = pd.read_parquet(games)

    if mode == "offline_fixture":
        # Fixture mode: derive scheduled games from local games table for --date
        start = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        gdf = games_df.copy()
        gdf["game_date"] = pd.to_datetime(gdf["game_date"], errors="coerce")
        day = gdf[gdf["game_date"].dt.strftime("%Y-%m-%d") == start]
        if day.empty:
            raise SystemExit(f"FAIL_CLOSED: no local games for {start}")
        games_live = []
        for r in day.itertuples():
            games_live.append({
                "id": int(r.game_id),
                "game_id": int(r.game_id),
                "date": start,
                "scheduled_tip_utc": f"{start}T19:00:00Z",
                "status": "scheduled",
                "home_team": {"id": int(r.home_team_id)},
                "visitor_team": {"id": int(getattr(r, "visitor_team_id", getattr(r, "away_team_id", -1)))},
            })
        avail = None
        slate_date = start
    else:
        if "BDL_API_KEY" not in os.environ:
            raise SystemExit("FAIL_CLOSED: BDL_API_KEY required in production/research modes")
        hdr = {"Authorization": os.environ["BDL_API_KEY"]}
        start = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            g = requests.get(
                f"{BDL}/games", headers=hdr,
                params={"seasons[]": 2026, "start_date": start, "end_date": "2026-08-15", "per_page": 100},
                timeout=30,
            ).json().get("data", [])
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"FAIL_CLOSED: BDL schedule fetch failed: {e}") from e

        upcoming = [x for x in g if str(x.get("status", "")).lower() not in ("post", "final")]
        if not upcoming:
            raise SystemExit("FAIL_CLOSED: no upcoming games returned for slate window")
        slate_date = min({x["date"][:10] for x in upcoming})
        games_live = [x for x in upcoming if x["date"][:10] == slate_date]
        for gm in games_live:
            gm["scheduled_tip_utc"] = gm.get("datetime") or (gm["date"][:10] + "T19:00:00Z")
            gm["game_id"] = gm["id"]

        try:
            inj = requests.get(f"{BDL}/player_injuries", headers=hdr, timeout=20).json().get("data", [])
            avail = pd.DataFrame([
                {"player_id": r["player"]["id"], "status": r.get("status", "")}
                for r in inj if r.get("player")
            ])
            if mode == "production" and avail.empty:
                # Explicit policy: empty injury feed is allowed but recorded
                avail = pd.DataFrame(columns=["player_id", "status"])
        except Exception as e:  # noqa: BLE001
            if mode == "production":
                raise SystemExit(
                    f"FAIL_CLOSED: availability snapshot required in production: {e}"
                ) from e
            avail = None

    current_rosters = None
    if rosters_json:
        current_rosters = {
            int(k): [int(x) for x in v]
            for k, v in json.loads(Path(rosters_json).read_text()).items()
        }

    try:
        delivery = predict_slate(
            prediction_timestamp=ts,
            scheduled_games=games_live,
            current_rosters=current_rosters,
            availability_snapshot=avail,
            historical_data={"features": feat, "stats": st, "games": games_df},
            model_bundle=bundle,
            mode=mode,
        )
    except (InferenceError, IdentityResolutionError, BundleIntegrityError) as e:
        raise SystemExit(f"FAIL_CLOSED: {e}") from e
    except Exception as e:  # noqa: BLE001
        # Convert unexpected validation exceptions into explicit failed status
        err = {
            "status": "FAILED_GATE",
            "gate": "inference_exception",
            "error_type": type(e).__name__,
            "error": str(e),
            "mode": mode,
        }
        out_err = Path(out_dir or f"deliveries/sharp_v6/{slate_date}/T-live")
        out_err.mkdir(parents=True, exist_ok=True)
        (out_err / "FAILED_GATE.json").write_text(json.dumps(err, indent=2))
        raise SystemExit(f"FAIL_CLOSED: unhandled inference exception: {e}") from e

    if delivery.manifest.get("n_pmfs", 0) <= 0:
        raise SystemExit("FAIL_CLOSED: games exist but PMFs are missing")

    out = Path(out_dir or f"deliveries/sharp_v6/{slate_date}/T-live")
    out.mkdir(parents=True, exist_ok=True)
    delivery.atoms_frame.to_parquet(out / "active_atom_pmfs.parquet", index=False)
    delivery.prices_frame.to_parquet(out / "fair_prices.parquet", index=False)
    delivery.prices_frame.to_csv(out / "pricing_inventory.csv", index=False)
    delivery.participation_frame.to_parquet(out / "participation.parquet", index=False)
    # Legacy-compatible wide PMF for pregame consumers (same calibrated V6 atoms).
    wide_rows = []
    if not delivery.atoms_frame.empty:
        for (gid, pid, stat), g in delivery.atoms_frame.groupby(
            ["game_id", "player_id", "stat"], sort=False
        ):
            g = g.sort_values("atom_value")
            pmf = {
                str(int(r.atom_value)): float(r.atom_probability)
                for r in g.itertuples()
                if float(r.atom_probability) > 0.0
            }
            ovf = float(g["overflow_probability"].iloc[0])
            if ovf > 1e-12:
                k_max = int(g["atom_value"].max()) + 1
                pmf[str(k_max)] = pmf.get(str(k_max), 0.0) + ovf
            pmf_json = json.dumps(pmf)
            wide_rows.append({
                "game_id": int(gid),
                "player_id": int(pid),
                "stat": str(stat),
                "player_name": str(g["player_name"].iloc[0]),
                "active_pmf_json": pmf_json,
                "pmf_json": pmf_json,
                "p_active": float(g["p_active"].iloc[0]),
                "pmf_mean": float(g["predictive_mean"].iloc[0]),
                "source_track": "CALIBRATED_V6_PMF",
            })
    pd.DataFrame(wide_rows).to_parquet(out / "full_pmfs_wide.parquet", index=False)
    if delivery.combo_frame is not None:
        delivery.combo_frame.to_parquet(out / "combo_prices.parquet", index=False)
    if delivery.q1_frame is not None:
        delivery.q1_frame.to_parquet(out / "q1_prices.parquet", index=False)
    if delivery.first_basket_frame is not None:
        delivery.first_basket_frame.to_parquet(out / "first_basket.parquet", index=False)
    (out / "pricing_manifest.json").write_text(
        json.dumps(delivery.manifest, indent=2, default=str)
    )

    # Odds are diagnostic only — never overwrite model_probability
    quotes = _pull_quotes(mode=mode)
    (out / "market_quotes_diagnostic.json").write_text(
        json.dumps(quotes, indent=2, default=str)
    )
    (out / "MARKET_EXTERNAL_NOTE.json").write_text(json.dumps({
        "market_data_role": "external_evaluation_only",
        "overwrites_model_probability": False,
        "positive_ev_rows_are_not_profitability_claims": True,
        "market_superiority": "NOT_PROVEN",
    }, indent=2))

    sample_hash = hashlib.sha256(
        delivery.atoms_frame.sort_values(
            ["game_id", "player_id", "target", "atom_value"]
        ).to_csv(index=False).encode()
    ).hexdigest() if not delivery.atoms_frame.empty else None

    receipt = build_deployment_receipt(
        expected_origin_main_sha=os.environ.get(
            "EXPECTED_ORIGIN_MAIN_SHA",
            bundle.meta.get("manifest", {}).get("code_sha", ""),
        ) or "",
        expected_bundle_hash=bundle.meta.get("model_sha256")
        or bundle.meta.get("manifest", {}).get("model_sha256", ""),
        bundle_dir=bundle_dir,
        deployment_environment=os.environ.get("DEPLOYMENT_ENV", "local"),
        smoke_run_id=os.environ.get("SMOKE_RUN_ID"),
        smoke_result=os.environ.get("SMOKE_RESULT", "local_run"),
        sample_output_hash=sample_hash,
        staged_status="staged",
    )
    # Local run is not remote deployment success
    if receipt.get("smoke_result") != "success":
        receipt["deployment_verified"] = False
        receipt["local_inference_ok"] = True
    (out / "DEPLOYMENT_RECEIPT.json").write_text(json.dumps(receipt, indent=2))

    _append_prospective(delivery, slate_date, bundle)

    typer.echo(
        f"V6 SLATE {slate_date}: mode={mode} games={len(games_live)} "
        f"players={delivery.manifest.get('n_players')} pmfs={delivery.manifest.get('n_pmfs')} "
        f"out={out}"
    )


def _pull_quotes(*, mode: str) -> dict:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {"status": "NO_ODDS_KEY", "role": "diagnostic_only"}
    try:
        events = requests.get(f"{ODDS}/events", params={"apiKey": key}, timeout=20).json()
        out = []
        for ev in events[:8]:
            r = requests.get(
                f"{ODDS}/events/{ev['id']}/odds",
                params={
                    "apiKey": key, "regions": "us",
                    "markets": "player_points,player_rebounds,player_assists,player_threes",
                    "oddsFormat": "american",
                },
                timeout=25,
            )
            out.append({
                "event": ev["id"], "status": r.status_code,
                "remaining": r.headers.get("x-requests-remaining"),
            })
            time.sleep(0.1)
        return {"status": "OK", "events": out, "role": "diagnostic_only", "mode": mode}
    except Exception as e:  # noqa: BLE001
        # Odds failure does not replace model output; diagnostic only
        return {"status": "ERROR", "error": str(e), "role": "diagnostic_only"}


def _append_prospective(delivery, slate_date: str, bundle) -> None:
    import numpy as np
    reg_dir = REPO / "deliveries" / "sharp_v6" / "prospective"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg = reg_dir / "registry.parquet"
    rows = []
    atoms = delivery.atoms_frame
    if atoms.empty:
        return
    bundle_hash = (
        bundle.meta.get("model_sha256")
        or (bundle.meta.get("manifest") or {}).get("model_sha256")
        or ""
    )
    for (gid, pid, tgt), g in atoms.groupby(["game_id", "player_id", "target"]):
        h = hashlib.sha256(
            np.round(g.sort_values("atom_value")["atom_probability"].to_numpy(), 8).tobytes()
        ).hexdigest()[:16]
        tip = str(
            delivery.manifest.get("prediction_timestamp") or slate_date
        )
        pid_key = hashlib.sha256(
            f"v6:{gid}:{pid}:{tgt}:{tip}:{bundle_hash[:16]}".encode()
        ).hexdigest()[:24]
        rows.append({
            "prediction_id": pid_key,
            "forecast_timestamp": delivery.prediction_timestamp,
            "scheduled_tip": tip,
            "slate_date": slate_date,
            "game_id": int(gid),
            "canonical_player_id": int(pid),
            "target": tgt,
            "atom_pmf_hash": h,
            "model_version": "wnba-sharp-pmf-v6",
            "bundle_sha256": bundle_hash,
            "settled": False,
        })
    new = pd.DataFrame(rows)
    if reg.exists():
        old = pd.read_parquet(reg)
        add = new[~new["prediction_id"].isin(set(old["prediction_id"]))]
        combined = pd.concat([old, add], ignore_index=True)
    else:
        combined = new
    combined.to_parquet(reg, index=False)


if __name__ == "__main__":
    app()
