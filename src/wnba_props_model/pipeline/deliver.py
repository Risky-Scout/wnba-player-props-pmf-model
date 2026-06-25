from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.constants import BDL_PROP_TO_STAT, DOMAIN_MAX
from wnba_props_model.models.market import fair_american, no_vig_two_way, prob_over_from_pmf
from wnba_props_model.models.pmf_grid import WNBAPMFGrid, pmfs_df_to_grids
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
    """Build fair odds board using WNBAPMFGrid for push-correct probabilities.

    Produces half-line markets (0.5, 1.5, …) where push probability is always
    zero — market convention for player props. Use WNBAPMFGrid directly for
    integer-line or quarter-line markets (Kalshi/Polymarket).
    """
    rows = []
    for _, r in pmfs.iterrows():
        pmf_arr = json_to_pmf(r["pmf_json"])
        stat = str(r["stat"])
        domain = min(DOMAIN_MAX.get(stat, len(pmf_arr) - 1), len(pmf_arr) - 1)
        grid = WNBAPMFGrid(
            player_id=r.get("player_id", ""),
            player_name=str(r.get("player_name", "")),
            stat_pmfs={stat: pmf_arr},
        )
        for k in range(0, domain):
            line = float(k) + 0.5
            p_over = grid.prob_over(stat, line)
            p_under = grid.prob_under(stat, line)
            rows.append({
                "game_id": r.get("game_id"),
                "player_id": r["player_id"],
                "player_name": r.get("player_name"),
                "team_id": r.get("team_id"),
                "stat": stat,
                "line": line,
                "p_over": p_over,
                "p_under": p_under,
                "p_push": 0.0,  # half-lines never push
                "fair_over_american": fair_american(p_over),
                "fair_under_american": fair_american(p_under),
                "role_bucket": r.get("role_bucket"),
                "model_version": r.get("model_version"),
            })
    return pd.DataFrame(rows)


def build_grids_narratives(pmfs: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one-line narrative per player × stat.

    Useful for rapid human review of the daily slate.
    """
    grids = pmfs_df_to_grids(pmfs, game_context_cols=["game_id", "game_date"])
    rows = []
    for grid in grids:
        for stat in grid.stats:
            rows.append({
                "player_id": grid.player_id,
                "player_name": grid.player_name,
                "stat": stat,
                "narrative": grid.narrative(stat),
                "projected_mean": round(grid.pmf_mean(stat), 3),
                "projected_std": round(grid.pmf_std(stat), 3),
                "projected_minutes": grid.projected_minutes,
                "role_bucket": grid.role_bucket,
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
        from wnba_props_model.models.market import (  # noqa: PLC0415
            shin_no_vig_two_way_with_z, get_no_vig_prob,
        )
        po, pu, z = shin_no_vig_two_way_with_z(market.get("over_odds"), market.get("under_odds"))
        po_power, _ = get_no_vig_prob(market.get("over_odds"), market.get("under_odds"), method="power")
        line_val = float(r.get("line_value") or r.get("line") or 0.0)
        # P4.1: opening line and line movement features
        prop_line_open = r.get("prop_line_open")
        try:
            prop_line_open = float(prop_line_open) if prop_line_open is not None else None
        except (TypeError, ValueError):
            prop_line_open = None
        line_delta = (line_val - prop_line_open) if prop_line_open is not None else None
        # Enhancement 7: market microstructure features
        posted_at = r.get("posted_at") or r.get("opened_at")
        try:
            import pandas as _pd  # noqa: PLC0415
            posted_dt = _pd.Timestamp(posted_at) if posted_at else None
            hours_since_open = float((_pd.Timestamp.now(tz="UTC") - posted_dt.tz_localize("UTC")).total_seconds() / 3600) if posted_dt is not None and posted_dt.tzinfo is None else (float((_pd.Timestamp.now(tz="UTC") - posted_dt).total_seconds() / 3600) if posted_dt is not None else None)
        except Exception:
            hours_since_open = None

        line_move_dir = (
            1 if (line_delta or 0) > 0.05 else (-1 if (line_delta or 0) < -0.05 else 0)
        ) if line_delta is not None else 0

        book_count = r.get("book_count") or r.get("number_of_books_offering")
        try:
            book_count = int(book_count) if book_count is not None else None
        except (TypeError, ValueError):
            book_count = None

        rows.append({
            "game_id": r.get("game_id"),
            "player_id": r.get("player_id"),
            "player_name": r.get("player_name"),
            "vendor": r.get("vendor"),
            "prop_type": r.get("prop_type"),
            "stat": stat,
            "line": line_val,
            "over_odds": market.get("over_odds"),
            "under_odds": market.get("under_odds"),
            "market_prob_over_no_vig": po,
            "market_prob_over_power": po_power,
            "shin_z": z,
            "updated_at": r.get("updated_at"),
            "prop_line_open": prop_line_open,
            "line_delta": line_delta,
            "line_moved_toward_over":  (line_delta > 0.25) if line_delta is not None else None,
            "line_moved_toward_under": (line_delta < -0.25) if line_delta is not None else None,
            # Enhancement 7: market microstructure
            "line_movement_direction":   line_move_dir,
            "line_movement_magnitude":   abs(line_delta) if line_delta is not None else None,
            "hours_since_line_opened":   hours_since_open,
            "number_of_books_offering":  book_count,
        })
    return pd.DataFrame(rows)


def build_market_comparison(pmfs: pd.DataFrame, raw_props: pd.DataFrame) -> pd.DataFrame:
    props = normalize_player_props_snapshot(raw_props)

    # Guard: nothing to compare if props normalisation produced no rows
    if props.empty or not all(c in props.columns for c in ["game_id", "player_id", "stat"]):
        return pd.DataFrame()

    # Resolve the mean column — pmf_engine outputs "pmf_mean"; predict.py combo rows
    # add both "pmf_mean" and "mean".  Normalise to "pmf_mean" before selecting.
    pmfs_sel = pmfs.copy()
    if "pmf_mean" not in pmfs_sel.columns and "mean" in pmfs_sel.columns:
        pmfs_sel["pmf_mean"] = pmfs_sel["mean"]

    # Only select columns that actually exist to avoid KeyError on optional cols.
    _must_have = ["game_id", "player_id", "stat", "pmf_json", "pmf_mean"]
    _optional  = ["role_bucket", "model_version", "player_name", "game_date"]
    sel_cols   = _must_have + [c for c in _optional if c in pmfs_sel.columns]

    # Primary join: game_id + player_id + stat (BDL-sourced props)
    props_with_id = props[props["player_id"].notna() & (props["player_id"] != "")]
    joined = props_with_id.merge(pmfs_sel[sel_cols], on=["game_id", "player_id", "stat"], how="inner")

    # Fallback join for Odds API props where player_id=None: join on player_name + stat.
    # Resolve game_id from PMFs using player_name lookup so downstream audit uses BDL IDs.
    props_no_id = props[props["player_id"].isna() | (props["player_id"] == "")]
    if not props_no_id.empty and "player_name" in props.columns and "player_name" in pmfs_sel.columns:
        import re as _re  # noqa: PLC0415

        def _norm_name(s: str) -> str:
            s = str(s or "").lower().strip()
            s = _re.sub(r"[^a-z ]", "", s)
            return _re.sub(r"\s+", " ", s)

        pmfs_name = pmfs_sel.copy()
        pmfs_name["_norm"] = pmfs_name["player_name"].map(_norm_name)
        props_no_id = props_no_id.copy()
        props_no_id["_norm"] = props_no_id["player_name"].map(_norm_name)

        # Join on normalized name + stat; inherit player_id and game_id from PMFs
        name_sel = [c for c in sel_cols if c not in ("player_id", "game_id")]
        name_sel += ["_norm"]
        fallback = props_no_id.merge(
            pmfs_name[sel_cols + ["_norm"]],
            on=["_norm", "stat"],
            how="inner",
            suffixes=("_prop", "_pmf"),
        )
        # Prefer PMF's game_id and player_id (BDL IDs)
        if not fallback.empty:
            if "game_id_pmf" in fallback.columns:
                fallback["game_id"] = fallback["game_id_pmf"]
                fallback = fallback.drop(columns=["game_id_pmf", "game_id_prop"], errors="ignore")
            if "player_id_pmf" in fallback.columns:
                fallback["player_id"] = fallback["player_id_pmf"]
                fallback = fallback.drop(columns=["player_id_pmf", "player_id_prop"], errors="ignore")
            fallback = fallback.drop(columns=["_norm"], errors="ignore")
            joined = pd.concat([joined, fallback], ignore_index=True)

    import logging as _logging  # noqa: PLC0415
    _logger = _logging.getLogger(__name__)
    _logger.info("build_market_comparison: %d props with BDL id, %d via name fallback -> %d joined rows",
                 len(props_with_id), len(props_no_id), len(joined))

    model_probs = []
    for _, r in joined.iterrows():
        model_probs.append(prob_over_from_pmf(json_to_pmf(r["pmf_json"]), r["line"]))
    joined["model_prob_over"] = model_probs
    joined["edge_over"] = joined["model_prob_over"] - joined["market_prob_over_no_vig"]
    # edge_under: how much the model's under probability exceeds the market's under probability
    joined["edge_under"] = joined["market_prob_over_no_vig"] - joined["model_prob_over"]
    joined["fair_over_american"] = joined["model_prob_over"].map(fair_american)
    joined["fair_under_american"] = (1 - joined["model_prob_over"]).map(fair_american)

    # Market-implied Poisson mean (Phase 4b)
    from wnba_props_model.models.market import market_implied_mean as _mim  # noqa: PLC0415
    joined["market_implied_mean"] = [
        _mim(float(r["line"]), float(r["market_prob_over_no_vig"]), stat=str(r.get("stat", "")))
        if pd.notna(r.get("market_prob_over_no_vig")) else None
        for _, r in joined.iterrows()
    ]

    # Mean-disagreement flag: |model_mean - market_implied_mean| > 2 (worth investigating)
    if "pmf_mean" in joined.columns:
        joined["mean_disagreement"] = (
            (joined["pmf_mean"] - joined["market_implied_mean"]).abs() > 2.0
        ).where(joined["market_implied_mean"].notna(), other=False)

    # Enhancement 7: model vs opening line edge and under-bias indicator
    if "prop_line_open" in joined.columns and "model_prob_over" in joined.columns:
        open_line = joined["prop_line_open"].fillna(joined["line"])
        import numpy as _np  # noqa: PLC0415
        joined["model_vs_opening_edge"] = (
            (joined["model_prob_over"] - joined["market_prob_over_no_vig"]).abs()
        ).where(open_line.notna())

    # under_bias_indicator: role-player props (< 25 min) have historical over-bias from books
    if "role_bucket" in joined.columns:
        joined["under_bias_indicator"] = (
            joined["role_bucket"].isin(["bench", "rotation", "spot"])
        ).astype(int)
    elif "pmf_mean" in joined.columns:
        joined["under_bias_indicator"] = (joined["pmf_mean"] < 15).astype(int)

    # Enhancement 20: game-total coherence scale factor
    try:
        from wnba_props_model.models.game_total_conditioned import mc_condition_player_props  # noqa: PLC0415
        if "market_game_total_line" in joined.columns:
            # Build projection list for conditioning
            proj_rows = []
            for _, r in joined.iterrows():
                proj_rows.append({
                    "player_id": r["player_id"],
                    "team":      r.get("team_id", "unknown"),
                    "pts_projection": float(r.get("pmf_mean", 10.0)) if r.get("stat") == "pts" else 0.0,
                    "stat": r.get("stat"),
                    "line": float(r.get("line", 0.0)),
                })
            game_total = float(joined["market_game_total_line"].median())
            if game_total > 100:
                conditioned = mc_condition_player_props(proj_rows, game_total)
                scale_map = {
                    r["player_id"]: r.get("coherence_scale_factor", 1.0)
                    for r in conditioned
                }
                joined["coherence_scale_factor"] = joined["player_id"].map(scale_map).fillna(1.0)
    except Exception:
        pass

    # Enhancement 16: expected CLV via line predictor (if model artifact exists)
    try:
        from wnba_props_model.models.line_predictor import LinePredictor, build_line_predictor_features  # noqa: PLC0415
        import os as _os  # noqa: PLC0415
        lp_path = _os.environ.get("LINE_PREDICTOR_PATH", "")
        if lp_path and _os.path.exists(lp_path + ".pkl"):
            lp = LinePredictor.load(lp_path)
            expected_clvs = []
            for _, r in joined.iterrows():
                feats = build_line_predictor_features(
                    r.to_dict(),
                    model_projection=float(r.get("pmf_mean", 0.0)),
                    hours_until_game=float(r.get("hours_since_line_opened", 24.0)),
                )
                clv_info = lp.compute_expected_clv(float(r.get("line", 0.0)), feats)
                expected_clvs.append(clv_info.get("expected_clv", 0.0))
            joined["expected_clv"] = expected_clvs
    except Exception:
        pass

    # reverse_line_movement_flag: line moved opposite to model's suggested direction
    if "line_delta" in joined.columns and "edge_over" in joined.columns:
        # Model says over (edge_over > 0), but line moved down (delta < 0) = reverse steam to under
        # Model says under (edge_over < 0), but line moved up (delta > 0) = reverse steam to over
        with_delta = joined["line_delta"].notna()
        joined["reverse_line_movement_flag"] = _np.where(
            with_delta,
            _np.where(
                (joined["edge_over"] > 0.03) & (joined["line_delta"] < -0.1), 1,
                _np.where(
                    (joined["edge_over"] < -0.03) & (joined["line_delta"] > 0.1), 1, 0
                )
            ),
            0,
        ).astype(int)

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

    # WNBAPMFGrid narratives sidecar
    try:
        narratives = build_grids_narratives(pmfs)
        narratives_path = out / "narratives.parquet"
        narratives.to_parquet(narratives_path, index=False)
    except Exception:
        narratives_path = None

    paths: dict[str, Path] = {
        "full_pmfs_wide": full_path,
        "player_projections_parquet": proj_parquet,
        "player_projections_json": proj_json_path,
        "fair_odds_board": board_path,
    }
    if narratives_path is not None:
        paths["narratives"] = narratives_path

    if raw_props is not None and not raw_props.empty:
        comp = build_market_comparison(pmfs, raw_props)
        comp_path = out / "market_comparison.parquet"
        comp.to_parquet(comp_path, index=False)
        paths["market_comparison"] = comp_path
        edges = comp[
            (comp["edge_over"].abs() >= 0.04) | (comp["edge_under"].abs() >= 0.04)
        ].copy()
        edges_path = out / "publishable_edges.parquet"
        edges.to_parquet(edges_path, index=False)
        paths["publishable_edges"] = edges_path

    return paths
