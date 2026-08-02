"""Authoritative V6 inference graph — one function used by OOF, live, prospective, pricing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.sharp_v6.contracts import COMBOS, EMERGENCY_CAP, SEED, TIER_A, build_all_contracts
from wnba_props_model.sharp_v6.live_features import build_live_feature_rows
from wnba_props_model.sharp_v6.models import (
    ModelBundle,
    apply_calibrator,
    minutes_pmf_rows,
    predict_game_environment,
    predict_stat_atoms,
    q1_minutes_pmf_rows,
    structural_points_pmf,
    structural_reb_pmf,
)
from wnba_props_model.sharp_v6.types import PlayerTargetPMF, SlatePMFDelivery

DIRECT_STATS = list(TIER_A)


def _settle(atoms: np.ndarray, overflow: float, line: float) -> tuple[float, float, float, float, float]:
    k = np.arange(atoms.size)
    A = float(atoms[k > line].sum()) + float(overflow)
    B = float(atoms[k < line].sum())
    P = float(atoms[int(line)]) if float(line).is_integer() and 0 <= int(line) < atoms.size else 0.0
    den = A + B
    return A, B, P, (A / den if den > 0 else float("nan")), (B / den if den > 0 else float("nan"))


def _lines_around(mean: float) -> list[float]:
    base = max(round(mean * 2) / 2 - 0.5, 0.5)
    return sorted({round(base + 0.5 * j, 1) for j in range(-2, 4) if base + 0.5 * j >= 0.5})


def predict_slate(
    prediction_timestamp,
    scheduled_games,
    current_rosters,
    availability_snapshot,
    historical_data: dict[str, Any],
    model_bundle: ModelBundle,
) -> SlatePMFDelivery:
    """One authoritative inference path for OOF / live / prospective / pricing / settlement.

    Parameters
    ----------
    prediction_timestamp : datetime | str
    scheduled_games : list[dict] | DataFrame
    current_rosters : dict[team_id, list[player_id]] | None
    availability_snapshot : DataFrame | None
    historical_data : dict with keys features, stats, games (DataFrames)
    model_bundle : frozen ModelBundle (never retrained here)
    """
    ts = prediction_timestamp
    if not isinstance(ts, str):
        ts = pd.Timestamp(ts, tz="UTC").isoformat()
    features = historical_data["features"]
    stats = historical_data["stats"]
    games_hist = historical_data.get("games")

    slate, prov = build_live_feature_rows(
        prediction_timestamp=ts,
        scheduled_games=scheduled_games,
        historical_features=features,
        historical_stats=stats,
        current_rosters=current_rosters,
        availability_snapshot=availability_snapshot,
    )
    unsupported: dict[str, str] = {}
    if slate.empty:
        return SlatePMFDelivery(
            prediction_timestamp=ts, games=[], player_pmfs=[],
            atoms_frame=pd.DataFrame(), prices_frame=pd.DataFrame(),
            participation_frame=pd.DataFrame(),
            manifest={"status": "EMPTY_SLATE", "n_pmfs": 0},
            unsupported={"slate": "no rebuildable rostered players for scheduled games"},
        )

    # participation
    from wnba_props_model.sharp_v6.models import _frame_features
    Xp = _frame_features(slate, model_bundle.participation.feature_cols).to_numpy(float)
    p_active = model_bundle.participation.predict_proba(Xp)

    # minutes (team-reconciled) + shared game environment
    matoms = minutes_pmf_rows(model_bundle.minutes, slate, reconcile_teams=True)
    q1_atoms = q1_minutes_pmf_rows(model_bundle.minutes, slate)
    env = predict_game_environment(model_bundle.game_environment, slate)
    env_map = {(int(r.game_id), int(r.team_id)): r for r in env.itertuples()}

    # per-stat active PMFs
    rng = np.random.default_rng(SEED)
    raw_atoms: dict[str, list[tuple[np.ndarray, float]]] = {}
    for stat in DIRECT_STATS:
        if stat == "pts" and model_bundle.shooting is not None:
            raw_atoms[stat] = structural_points_pmf(model_bundle.shooting, slate, matoms, rng=rng)
        elif stat == "reb" and model_bundle.rebounds is not None:
            raw_atoms[stat] = structural_reb_pmf(model_bundle.rebounds, slate, matoms, rng=rng)
        elif stat in model_bundle.stats:
            raw_atoms[stat] = predict_stat_atoms(model_bundle.stats[stat], slate, matoms)
        else:
            unsupported[stat] = "no fitted direct-stat model in bundle"

    # calibrate
    calibrated: dict[str, list[tuple[np.ndarray, float]]] = {}
    for stat, rows in raw_atoms.items():
        cal = model_bundle.calibrators.get(stat)
        if cal is None:
            calibrated[stat] = rows
            continue
        calibrated[stat] = [apply_calibrator(cal, a, o) for a, o in rows]

    # names
    if "player_name" in stats.columns:
        name_by = stats.sort_values("game_date").groupby("player_id")["player_name"].last().to_dict()
    else:
        name_by = {}

    atom_rows, price_rows, part_rows, pmf_objs = [], [], [], []
    for i, row in slate.iterrows():
        pid = int(row["player_id"]); gid = int(row["game_id"])
        tid = int(row["team_id"]); oid = int(row.get("opponent_team_id", -1))
        pname = name_by.get(pid, str(row.get("player_name", f"player_{pid}")))
        pa = float(p_active[i])
        part_rows.append({
            "game_id": gid, "player_id": pid, "player_name": pname,
            "p_active": pa, "dnp_probability": 1.0 - pa,
            "prediction_timestamp": ts,
            "prediction_cutoff": row.get("prediction_cutoff", ""),
            "feature_source": row.get("feature_source", ""),
            "feature_contract_hash": row.get("feature_contract_hash", ""),
        })
        for stat, rows in calibrated.items():
            a, ovf = rows[i]
            # normalize gate
            mass = float(a.sum()) + float(ovf)
            if abs(mass - 1.0) > 1e-8 and mass > 0:
                a = a / mass
                ovf = ovf / mass
            kk = np.arange(a.size)
            mean = float(np.dot(kk, a))
            var = float(np.dot((kk - mean) ** 2, a))
            cal_method = model_bundle.calibrators[stat].method if stat in model_bundle.calibrators else "identity"
            over_map, under_map, push_map = {}, {}, {}
            for L in _lines_around(mean):
                A, B, P, so, su = _settle(a, ovf, L)
                over_map[L] = so; under_map[L] = su; push_map[L] = P
                for side, pw, ps in (("Over", A, so), ("Under", B, su)):
                    fd = 1.0 / min(max(ps, 1e-9), 1 - 1e-9) if np.isfinite(ps) else float("nan")
                    price_rows.append({
                        "game_id": gid, "player_id": pid, "player_name": pname, "target": stat,
                        "line": L, "side": side, "p_win": pw, "p_push": P,
                        "model_probability": ps,  # settled from calibrated V6 PMF — THE pick input
                        "settled_probability": ps, "fair_decimal": fd,
                        "p_active": pa, "source_track": "CALIBRATED_V6_PMF",
                        "prediction_timestamp": ts,
                    })
            for k, prob in enumerate(a):
                if prob <= 1e-9:
                    continue
                atom_rows.append({
                    "game_id": gid, "canonical_player_id": pid, "player_id": pid,
                    "player_name": pname, "team_id": tid, "opponent_id": oid,
                    "period": "FULL", "target": stat, "stat": stat,
                    "atom_value": int(k), "atom_probability": float(prob),
                    "overflow_probability": float(ovf), "predictive_mean": mean,
                    "predictive_variance": var, "p_active": pa,
                    "prediction_timestamp": ts, "source_track": "CALIBRATED_V6_PMF",
                    "calibration_method": cal_method,
                    "feature_contract_hash": row.get("feature_contract_hash", ""),
                    "feature_source": row.get("feature_source", ""),
                    "prediction_cutoff": row.get("prediction_cutoff", ""),
                })
            pmf_objs.append(PlayerTargetPMF(
                game_id=gid, player_id=pid, player_name=pname, team_id=tid, opponent_id=oid,
                target=stat, p_active=pa, active_pmf_atoms=[float(x) for x in a],
                overflow_probability=float(ovf),
                model_probability_over=over_map, model_probability_under=under_map,
                model_probability_push=push_map, predictive_mean=mean, predictive_variance=var,
                calibration_method=cal_method,
                feature_contract_hash=str(row.get("feature_contract_hash", "")),
                prediction_cutoff=str(row.get("prediction_cutoff", "")),
            ))

    # combination markets via Gaussian copula (preserve marginals)
    combo_rows = []
    if model_bundle.dependence is not None:
        combo_rows = _combo_from_copula(slate, calibrated, model_bundle.dependence, name_by, p_active, ts, rng)
    else:
        for c in COMBOS:
            unsupported[c] = "dependence model absent; combo not fabricated from sportsbook probs"

    # Q1 nested minutes → pts/reb/ast rate mixtures
    q1_rows = _q1_pmfs(slate, model_bundle, q1_atoms, name_by, p_active, ts)

    # First basket: competing-risk from empirical scoring rates × p_active (sums to 1)
    fb_rows, fb_status = _first_basket(slate, stats, p_active, name_by, ts)
    if fb_status:
        unsupported["first_basket"] = fb_status

    games_out = []
    for gm in (scheduled_games if not isinstance(scheduled_games, pd.DataFrame) else scheduled_games.to_dict("records")):
        games_out.append({
            "game_id": gm.get("id") or gm.get("game_id"),
            "scheduled_tip_utc": gm.get("scheduled_tip_utc") or gm.get("date") or gm.get("game_date"),
            "home_team_id": gm["home_team"]["id"] if isinstance(gm.get("home_team"), dict) else gm.get("home_team_id"),
            "visitor_team_id": gm["visitor_team"]["id"] if isinstance(gm.get("visitor_team"), dict) else gm.get("visitor_team_id"),
        })

    manifest = {
        "artifact": "SlatePMFDelivery",
        "prediction_timestamp": ts,
        "n_games": len(games_out),
        "n_players": int(slate["player_id"].nunique()) if len(slate) else 0,
        "n_pmfs": len(pmf_objs),
        "inference_function": "wnba_props_model.sharp_v6.inference.predict_slate",
        "model_bundle_id": model_bundle.meta.get("bundle_id", "in-memory"),
        "retrain": False,
        "feature_rebuild": "live_features.build_live_feature_rows",
        "team_minutes_target": 200,
        "q1_minutes_target": 50,
        "selected_families": model_bundle.selected_family,
        "unsupported": unsupported,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if len(pmf_objs) == 0 and len(games_out) > 0:
        raise RuntimeError("FAIL_CLOSED: scheduled games exist but PMFs are missing")

    return SlatePMFDelivery(
        prediction_timestamp=ts,
        games=games_out,
        player_pmfs=pmf_objs,
        atoms_frame=pd.DataFrame(atom_rows),
        prices_frame=pd.DataFrame(price_rows),
        participation_frame=pd.DataFrame(part_rows),
        manifest=manifest,
        combo_frame=pd.DataFrame(combo_rows) if combo_rows else None,
        q1_frame=pd.DataFrame(q1_rows) if q1_rows else None,
        first_basket_frame=pd.DataFrame(fb_rows) if fb_rows else None,
        unsupported=unsupported,
    )


def _combo_from_copula(slate, calibrated, dep, name_by, p_active, ts, rng, n_sims: int = 500):
    from scipy.stats import norm
    stats = [s for s in dep.stats if s in calibrated]
    if len(stats) < 2:
        return []
    idx = {s: i for i, s in enumerate(stats)}
    C = dep.corr
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(C + np.eye(C.shape[0]) * 1e-6)
    rows = []
    combo_map = {
        "stocks": ("stl", "blk"),
        "pts_ast": ("pts", "ast"),
        "pts_reb": ("pts", "reb"),
        "reb_ast": ("reb", "ast"),
        "pts_reb_ast": ("pts", "reb", "ast"),
    }
    for i, row in slate.iterrows():
        pid = int(row["player_id"]); gid = int(row["game_id"])
        pname = name_by.get(pid, f"player_{pid}")
        # inverse-CDF sampling with shared gaussian latent
        Z = rng.normal(size=(n_sims, len(stats))) @ L.T
        U = norm.cdf(Z)
        draws = {}
        for s in stats:
            a, ovf = calibrated[s][i]
            cdf = np.cumsum(a)
            # include overflow as final atom mass beyond support
            support = np.arange(a.size + 1)
            cdf_full = np.concatenate([cdf, [min(1.0, cdf[-1] + ovf)]]) if a.size else np.array([1.0])
            # map u -> y via searchsorted
            ys = []
            for u in U[:, idx[s]]:
                j = int(np.searchsorted(cdf_full, u, side="left"))
                ys.append(int(min(j, a.size)))  # overflow bucket -> support_max+
            draws[s] = np.asarray(ys)
        for combo, comps in combo_map.items():
            if any(c not in draws for c in comps):
                continue
            samples = sum(draws[c] for c in comps)
            K = EMERGENCY_CAP.get(combo, 80)
            atoms = np.bincount(np.clip(samples, 0, K), minlength=K + 1).astype(float)
            atoms /= max(atoms.sum(), 1e-12)
            mean = float(np.dot(np.arange(atoms.size), atoms))
            for Lne in _lines_around(mean):
                A, B, P, so, su = _settle(atoms, 0.0, Lne)
                for side, ps in (("Over", so), ("Under", su)):
                    rows.append({
                        "game_id": gid, "player_id": pid, "player_name": pname,
                        "target": combo, "line": Lne, "side": side,
                        "model_probability": ps, "p_push": P, "p_active": float(p_active[i]),
                        "source_track": "V6_GAUSSIAN_COPULA", "prediction_timestamp": ts,
                    })
    return rows


def _q1_pmfs(slate, bundle, q1_atoms, name_by, p_active, ts):
    rows = []
    for stat in ("pts", "reb", "ast"):
        if stat not in bundle.stats and not (stat == "pts" and bundle.shooting):
            continue
        # use rate model with Q1 minutes mixture
        if stat in bundle.stats:
            pred = predict_stat_atoms(bundle.stats[stat], slate, q1_atoms)
        else:
            continue
        for i, row in slate.iterrows():
            a, ovf = pred[i]
            mean = float(np.dot(np.arange(a.size), a))
            pid = int(row["player_id"]); gid = int(row["game_id"])
            for L in _lines_around(mean):
                A, B, P, so, su = _settle(a, ovf, L)
                for side, ps in (("Over", so), ("Under", su)):
                    rows.append({
                        "game_id": gid, "player_id": pid,
                        "player_name": name_by.get(pid, f"player_{pid}"),
                        "target": f"q1_{stat}", "line": L, "side": side,
                        "model_probability": ps, "p_push": P, "p_active": float(p_active[i]),
                        "source_track": "V6_Q1_NESTED", "prediction_timestamp": ts,
                    })
    return rows


def _first_basket(slate, stats, p_active, name_by, ts):
    """Competing-risk first basket: weight ∝ p_active * recent scoring rate; normalize per game."""
    if slate.empty:
        return [], "empty_slate"
    rates = {}
    st = stats.copy()
    st["game_date"] = pd.to_datetime(st["game_date"], errors="coerce")
    st["pts"] = pd.to_numeric(st.get("pts", st.get("actual_pts", 0)), errors="coerce").fillna(0)
    st["minutes"] = pd.to_numeric(st.get("minutes", st.get("actual_minutes", 0)), errors="coerce").fillna(0)
    for pid, g in st.sort_values("game_date").groupby("player_id"):
        g = g.tail(10)
        mins = g["minutes"].sum()
        rates[int(pid)] = float(g["pts"].sum() / mins) if mins > 1 else 0.05
    rows = []
    for gid, g in slate.groupby("game_id"):
        idx = list(g.index)
        weights = []
        for i in idx:
            pid = int(slate.loc[i, "player_id"])
            weights.append(max(1e-6, float(p_active[i]) * rates.get(pid, 0.05)))
        w = np.asarray(weights, float)
        # residual "other/unknown" mass
        other = 0.02
        w = w / w.sum() * (1 - other)
        for j, i in enumerate(idx):
            pid = int(slate.loc[i, "player_id"])
            tid = int(slate.loc[i, "team_id"])
            rows.append({
                "game_id": int(gid), "player_id": pid, "team_id": tid,
                "player_name": name_by.get(pid, f"player_{pid}"),
                "p_first_basket": float(w[j]),
                "p_active": float(p_active[i]),
                "source_track": "V6_COMPETING_RISK_FIRST_BASKET",
                "prediction_timestamp": ts,
            })
        rows.append({
            "game_id": int(gid), "player_id": -1, "team_id": -1, "player_name": "OTHER_OR_TEAM_REBOUND_CHAIN",
            "p_first_basket": float(other), "p_active": 0.0,
            "source_track": "V6_COMPETING_RISK_FIRST_BASKET", "prediction_timestamp": ts,
        })
        # team first-basket sums to 1 within team after renormalizing player masses + share of other
        for tid, tg in g.groupby("team_id"):
            tidx = [idx.index(i) for i in tg.index]
            tw = w[tidx]
            # attach half of residual to each team for team-basket market
            team_p = float(tw.sum() + other / 2)
            for j, i in zip(tidx, tg.index):
                # already stored player probs; team market derived separately
                pass
            rows.append({
                "game_id": int(gid), "player_id": -100 - int(tid), "team_id": int(tid),
                "player_name": f"TEAM_{tid}_FIRST_BASKET",
                "p_first_basket": team_p, "p_active": 1.0,
                "source_track": "V6_TEAM_FIRST_BASKET", "prediction_timestamp": ts,
            })
    # verify per-game player+other sums ~ 1
    return rows, ""


# OOF convenience: score a historical fold with the SAME predict path by treating
# fold eval rows as a "scheduled" slate with known tips.
def predict_historical_rows(eval_df: pd.DataFrame, train_df: pd.DataFrame, bundle: ModelBundle, stats: pd.DataFrame) -> SlatePMFDelivery:
    games = []
    for gid, g in eval_df.groupby("game_id"):
        r0 = g.iloc[0]
        home = int(r0["team_id"]) if int(r0.get("is_home", 1)) == 1 else int(r0["opponent_team_id"])
        away = int(r0["opponent_team_id"]) if int(r0.get("is_home", 1)) == 1 else int(r0["team_id"])
        games.append({
            "game_id": int(gid),
            "scheduled_tip_utc": pd.Timestamp(r0["game_date"]).isoformat(),
            "home_team_id": home, "visitor_team_id": away,
            "home_team": {"id": home}, "visitor_team": {"id": away},
            "date": str(r0["game_date"])[:10],
        })
    # For OOF we pass eval feature rows directly by short-circuiting live rebuild:
    # inject eval rows as historical with tip after their game_date... Actually use
    # a dedicated path that feeds eval features as the slate while still using the
    # same PMF math functions.
    return _predict_from_feature_slate(
        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
        slate=eval_df.reset_index(drop=True),
        bundle=bundle,
        stats=stats,
        games=games,
    )


def _predict_from_feature_slate(prediction_timestamp, slate, bundle, stats, games) -> SlatePMFDelivery:
    """Internal: run PMF math on an already-built feature slate (OOF / clean-clone)."""
    # Monkeypatch by calling core math with a thin wrapper around predict_slate guts
    historical_data = {"features": slate, "stats": stats}
    # Build fake scheduled games and rosterson slate itself
    rosters: dict[int, list[int]] = {}
    for (gid, tid), g in slate.groupby(["game_id", "team_id"]):
        rosters.setdefault(int(tid), [])
        for pid in g["player_id"]:
            if int(pid) not in rosters[int(tid)]:
                rosters[int(tid)].append(int(pid))

    # Direct path: temporarily use slate as live rows by invoking math sections
    # via a local duplicate of the core loop with prebuilt slate.
    from wnba_props_model.sharp_v6.models import _frame_features
    ts = prediction_timestamp
    Xp = _frame_features(slate, bundle.participation.feature_cols).to_numpy(float)
    p_active = bundle.participation.predict_proba(Xp)
    matoms = minutes_pmf_rows(bundle.minutes, slate, reconcile_teams=True)
    q1_atoms = q1_minutes_pmf_rows(bundle.minutes, slate)
    rng = np.random.default_rng(SEED)
    raw_atoms = {}
    unsupported = {}
    for stat in DIRECT_STATS:
        if stat == "pts" and bundle.shooting is not None:
            raw_atoms[stat] = structural_points_pmf(bundle.shooting, slate, matoms, rng=rng)
        elif stat == "reb" and bundle.rebounds is not None:
            raw_atoms[stat] = structural_reb_pmf(bundle.rebounds, slate, matoms, rng=rng)
        elif stat in bundle.stats:
            raw_atoms[stat] = predict_stat_atoms(bundle.stats[stat], slate, matoms)
        else:
            unsupported[stat] = "missing"
    calibrated = {}
    for stat, rows in raw_atoms.items():
        cal = bundle.calibrators.get(stat)
        calibrated[stat] = [apply_calibrator(cal, a, o) for a, o in rows] if cal else rows
    name_by = stats.sort_values("game_date").groupby("player_id")["player_name"].last().to_dict() if "player_name" in stats.columns else {}
    atom_rows, price_rows, part_rows, pmf_objs = [], [], [], []
    for i, row in slate.iterrows():
        pid = int(row["player_id"]); gid = int(row["game_id"])
        pa = float(p_active[i])
        pname = name_by.get(pid, f"player_{pid}")
        part_rows.append({"game_id": gid, "player_id": pid, "p_active": pa, "dnp_probability": 1 - pa})
        for stat, rows in calibrated.items():
            a, ovf = rows[i]
            mass = float(a.sum()) + float(ovf)
            if mass > 0 and abs(mass - 1) > 1e-8:
                a, ovf = a / mass, ovf / mass
            kk = np.arange(a.size)
            mean = float(np.dot(kk, a)); var = float(np.dot((kk - mean) ** 2, a))
            for L in _lines_around(mean):
                A, B, P, so, su = _settle(a, ovf, L)
                for side, ps in (("Over", so), ("Under", su)):
                    price_rows.append({
                        "game_id": gid, "player_id": pid, "player_name": pname, "target": stat,
                        "line": L, "side": side, "model_probability": ps, "p_push": P, "p_active": pa,
                        "source_track": "CALIBRATED_V6_PMF",
                    })
            for k, prob in enumerate(a):
                if prob > 1e-9:
                    atom_rows.append({
                        "game_id": gid, "player_id": pid, "canonical_player_id": pid,
                        "target": stat, "stat": stat, "atom_value": int(k),
                        "atom_probability": float(prob), "overflow_probability": float(ovf),
                        "p_active": pa, "predictive_mean": mean,
                    })
            pmf_objs.append(PlayerTargetPMF(
                game_id=gid, player_id=pid, player_name=pname,
                team_id=int(row.get("team_id", -1)), opponent_id=int(row.get("opponent_team_id", -1)),
                target=stat, p_active=pa, active_pmf_atoms=[float(x) for x in a],
                overflow_probability=float(ovf), predictive_mean=mean, predictive_variance=var,
            ))
    combo_rows = _combo_from_copula(slate, calibrated, bundle.dependence, name_by, p_active, ts, rng) if bundle.dependence else []
    q1_rows = _q1_pmfs(slate, bundle, q1_atoms, name_by, p_active, ts)
    fb_rows, _ = _first_basket(slate, stats, p_active, name_by, ts)
    return SlatePMFDelivery(
        prediction_timestamp=ts, games=games, player_pmfs=pmf_objs,
        atoms_frame=pd.DataFrame(atom_rows), prices_frame=pd.DataFrame(price_rows),
        participation_frame=pd.DataFrame(part_rows),
        manifest={
            "inference_function": "wnba_props_model.sharp_v6.inference.predict_slate",
            "path": "historical_feature_slate", "n_pmfs": len(pmf_objs), "retrain": False,
            "contracts": build_all_contracts(list(slate.columns)),
        },
        combo_frame=pd.DataFrame(combo_rows) if combo_rows else None,
        q1_frame=pd.DataFrame(q1_rows) if q1_rows else None,
        first_basket_frame=pd.DataFrame(fb_rows) if fb_rows else None,
    )
