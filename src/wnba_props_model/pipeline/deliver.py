from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.constants import BDL_PROP_TO_STAT
from wnba_props_model.models.market import fair_american, no_vig_two_way, prob_over_from_pmf
from wnba_props_model.models.simulation import json_to_pmf


def add_pge_ladder(pmfs: pd.DataFrame, kmax: int = 20) -> pd.DataFrame:
    out = pmfs.copy()
    for k in range(1, kmax + 1):
        out[f"p_ge_{k}"] = [
            float(json_to_pmf(p)[np.arange(len(json_to_pmf(p))) >= k].sum())
            for p in out["pmf_json"]
        ]
    return out


def build_fair_odds_board(pmfs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pmfs.iterrows():
        pmf = json_to_pmf(r["pmf_json"])
        for line in range(0, min(len(pmf) - 1, 60)):
            p_over = prob_over_from_pmf(pmf, line + 0.5)
            rows.append({
                "game_id": r.get("game_id"),
                "player_id": r["player_id"],
                "player_name": r.get("player_name"),
                "team_id": r.get("team_id"),
                "stat": r["stat"],
                "line": line + 0.5,
                "p_over": p_over,
                "p_under": 1 - p_over,
                "fair_over_american": fair_american(p_over),
                "fair_under_american": fair_american(1 - p_over),
                "role_bucket": r.get("role_bucket"),
                "model_version": r.get("model_version"),
            })
    return pd.DataFrame(rows)


def normalize_player_props_snapshot(raw_props: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in raw_props.iterrows():
        market = r.get("market") or {}
        if isinstance(market, str):
            try:
                market = json.loads(market)
            except Exception:
                market = {}
        stat = BDL_PROP_TO_STAT.get(r.get("prop_type"))
        if stat is None:
            continue
        from wnba_props_model.models.market import shin_no_vig_two_way_with_z  # noqa: PLC0415
        po, pu, z = shin_no_vig_two_way_with_z(market.get("over_odds"), market.get("under_odds"))
        rows.append({
            "game_id": r.get("game_id"),
            "player_id": r.get("player_id"),
            "vendor": r.get("vendor"),
            "prop_type": r.get("prop_type"),
            "stat": stat,
            "line": float(r.get("line_value")),
            "over_odds": market.get("over_odds"),
            "under_odds": market.get("under_odds"),
            "market_prob_over_no_vig": po,
            "shin_z": z,
            "updated_at": r.get("updated_at"),
        })
    return pd.DataFrame(rows)


def build_market_comparison(pmfs: pd.DataFrame, raw_props: pd.DataFrame) -> pd.DataFrame:
    props = normalize_player_props_snapshot(raw_props)
    joined = props.merge(pmfs[["game_id", "player_id", "stat", "pmf_json", "mean", "role_bucket", "model_version"]], on=["game_id", "player_id", "stat"], how="inner")
    model_probs = []
    for _, r in joined.iterrows():
        model_probs.append(prob_over_from_pmf(json_to_pmf(r["pmf_json"]), r["line"]))
    joined["model_prob_over"] = model_probs
    joined["edge_over"] = joined["model_prob_over"] - joined["market_prob_over_no_vig"]
    joined["fair_over_american"] = joined["model_prob_over"].map(fair_american)
    joined["fair_under_american"] = (1 - joined["model_prob_over"]).map(fair_american)

    # Market-implied Poisson mean (Phase 4b)
    from wnba_props_model.models.market import market_implied_mean as _mim  # noqa: PLC0415
    joined["market_implied_mean"] = [
        _mim(float(r["line"]), float(r["market_prob_over_no_vig"]), stat=str(r.get("stat", "")))
        if pd.notna(r.get("market_prob_over_no_vig")) else None
        for _, r in joined.iterrows()
    ]

    # Mean-disagrement flag: |model_mean - market_implied_mean| > 2 (worth investigating)
    if "mean" in joined.columns:
        joined["mean_disagreement"] = (
            (joined["mean"] - joined["market_implied_mean"]).abs() > 2.0
        ).where(joined["market_implied_mean"].notna(), other=False)

    return joined


def build_projection_output(pmfs: pd.DataFrame, game_date: str | None = None) -> pd.DataFrame:
    """Build the standardised projection table per docs/OUTPUT_CONTRACT.md.

    Adds: opponent_abbreviation, home_away, game_date_et, injury_flag, dnp_risk,
    override_applied, override_source, model_version, generated_at,
    and the P(>=k) ladder at common thresholds.
    """
    import datetime as dt
    out = pmfs.copy()

    # Add P(>=k) ladder columns
    for k in [1, 3, 5, 10, 15, 20]:
        col = f"p_ge_{k}"
        if col not in out.columns:
            vals = []
            for pmf_json in out["pmf_json"]:
                pmf = json_to_pmf(pmf_json)
                vals.append(float(pmf[k:].sum()) if len(pmf) > k else 0.0)
            out[col] = vals

    # Ensure required metadata columns exist (may not if overrides weren't applied)
    for col, default in [
        ("injury_flag", False),
        ("dnp_risk", "low"),
        ("override_applied", False),
        ("override_source", None),
    ]:
        if col not in out.columns:
            out[col] = default

    out["game_date_et"] = game_date or (
        pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
        if "game_date" in out.columns else ""
    )
    out["generated_at"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if "model_version" not in out.columns:
        out["model_version"] = "wnba_pmf_v1.0_hgb_calibrated"

    return out


def write_delivery(
    pmfs: pd.DataFrame,
    out_dir: str | Path,
    raw_props: pd.DataFrame | None = None,
    game_date: str | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = add_pge_ladder(pmfs)
    full_path = out / "full_pmfs_wide.parquet"
    full.to_parquet(full_path, index=False)

    # Projection output (per docs/OUTPUT_CONTRACT.md schema)
    proj = build_projection_output(full, game_date=game_date)
    date_tag = game_date or "latest"
    proj_parquet = out / f"player_projections_{date_tag}.parquet"
    proj.to_parquet(proj_parquet, index=False)

    # JSON delivery (drop pmf_json for size)
    proj_json = proj.drop(columns=["pmf_json"], errors="ignore")
    proj_json_path = out / f"player_projections_{date_tag}.json"
    proj_json.to_json(proj_json_path, orient="records", indent=2)

    board = build_fair_odds_board(pmfs)
    board_path = out / "fair_odds_board.parquet"
    board.to_parquet(board_path, index=False)

    paths: dict[str, Path] = {
        "full_pmfs_wide": full_path,
        "player_projections_parquet": proj_parquet,
        "player_projections_json": proj_json_path,
        "fair_odds_board": board_path,
    }

    if raw_props is not None and not raw_props.empty:
        comp = build_market_comparison(pmfs, raw_props)
        comp_path = out / "market_comparison.parquet"
        comp.to_parquet(comp_path, index=False)
        paths["market_comparison"] = comp_path
        edges = comp[comp["edge_over"].abs() >= 0.04].copy()
        edges_path = out / "publishable_edges.parquet"
        edges.to_parquet(edges_path, index=False)
        paths["publishable_edges"] = edges_path

    return paths
