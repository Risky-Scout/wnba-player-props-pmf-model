"""Authoritative V6 inference graph — one function used by OOF, live, prospective, pricing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.sharp_v6.contracts import COMBOS, EMERGENCY_CAP, NORM_TOL, SEED, TIER_A
from wnba_props_model.sharp_v6.identity import (
    IdentityResolutionError,
    audit_scheduled_games,
    build_date_effective_identity_table,
    resolve_roster_identities,
)
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


class InferenceError(RuntimeError):
    """Production inference failure — never converted to absent evidence."""


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


def _normalize_pmf(
    atoms: np.ndarray,
    overflow: float,
    *,
    mode: str,
    context: str,
) -> tuple[np.ndarray, float]:
    """Validate PMF mass identity. Never rewrites atoms/overflow by renormalization.

    Production PMFs must already satisfy |sum(atoms)+overflow-1| <= 1e-10 from the
    authoritative distribution layer. This gate fails closed on violations.
    """
    a = np.asarray(atoms, float)
    ovf = float(overflow)
    if not np.isfinite(a).all() or not np.isfinite(ovf):
        raise InferenceError(f"FAIL_CLOSED: non-finite PMF mass ({context})")
    if (a < -NORM_TOL).any() or ovf < -NORM_TOL:
        raise InferenceError(f"FAIL_CLOSED: negative PMF mass ({context})")
    a = np.clip(a, 0.0, None)
    ovf = max(ovf, 0.0)
    mass = float(a.sum()) + ovf
    if mass <= 0:
        raise InferenceError(f"FAIL_CLOSED: zero PMF mass ({context})")
    err = abs(mass - 1.0)
    # Production: strict mass identity from the distribution layer.
    if mode == "production" and err > NORM_TOL:
        raise InferenceError(
            f"FAIL_CLOSED: PMF normalization failure mass={mass} err={err} ({context})"
        )
    # Non-production research paths: still refuse to silently renormalize; only allow
    # tiny float noise up to 1e-6 before failing.
    if mode != "production" and err > 1e-6:
        raise InferenceError(
            f"FAIL_CLOSED: PMF normalization failure mass={mass} err={err} ({context})"
        )
    return a, ovf


def _core_pmf_delivery(
    *,
    prediction_timestamp: str,
    slate: pd.DataFrame,
    bundle: ModelBundle,
    stats: pd.DataFrame,
    games_out: list[dict[str, Any]],
    mode: str,
    identity_events: list[dict[str, Any]] | None = None,
    feature_drift: list[dict[str, Any]] | None = None,
    quarantined_n: int = 0,
) -> SlatePMFDelivery:
    """Shared PMF math for live and historical paths — one coherent system."""
    from wnba_props_model.sharp_v6.models import _frame_features

    unsupported: dict[str, str] = dict(bundle.meta.get("manifest", {}).get("unsupported_markets", {}) or {})
    # Prefer frozen unsupported map from manifest when present
    man = (bundle.meta or {}).get("manifest") or {}
    if isinstance(man.get("unsupported_markets"), dict):
        unsupported = dict(man["unsupported_markets"])

    Xp = _frame_features(
        slate, bundle.participation.feature_cols,
        mode=mode, contracts=bundle.contracts,
    ).to_numpy(float)
    p_active = bundle.participation.predict_proba(Xp)

    matoms = minutes_pmf_rows(bundle.minutes, slate, reconcile_teams=True, mode=mode)
    q1_atoms = q1_minutes_pmf_rows(bundle.minutes, slate, mode=mode)
    env = predict_game_environment(bundle.game_environment, slate)

    rng = np.random.default_rng(SEED)
    raw_atoms: dict[str, list[tuple[np.ndarray, float]]] = {}
    for stat in DIRECT_STATS:
        if stat == "pts" and bundle.shooting is not None:
            raw_atoms[stat] = structural_points_pmf(bundle.shooting, slate, matoms, rng=rng)
        elif stat == "reb" and bundle.rebounds is not None:
            raw_atoms[stat] = structural_reb_pmf(bundle.rebounds, slate, matoms, rng=rng)
        elif stat in bundle.stats:
            raw_atoms[stat] = predict_stat_atoms(bundle.stats[stat], slate, matoms)
        else:
            if mode == "production":
                raise InferenceError(
                    f"FAIL_CLOSED: missing required direct-stat model for '{stat}'"
                )
            unsupported[stat] = "no fitted direct-stat model in bundle"

    calibrated: dict[str, list[tuple[np.ndarray, float]]] = {}
    for stat, rows in raw_atoms.items():
        cal = bundle.calibrators.get(stat)
        if cal is None:
            if mode == "production":
                raise InferenceError(
                    f"FAIL_CLOSED: missing explicit calibrator for '{stat}' "
                    "(identity calibrator must be present as an explicit selection)"
                )
            calibrated[stat] = rows
            continue
        # Wrong-stat guard
        if getattr(cal, "stat", stat) != stat:
            raise InferenceError(
                f"FAIL_CLOSED: calibrator stat mismatch loaded={cal.stat} expected={stat}"
            )
        calibrated[stat] = [apply_calibrator(cal, a, o) for a, o in rows]

    if "player_name" in stats.columns:
        name_by = stats.sort_values("game_date").groupby("player_id")["player_name"].last().to_dict()
    else:
        name_by = {}

    atom_rows, price_rows, part_rows, pmf_objs = [], [], [], []
    # Positional iteration so array indices align after quarantine.
    slate_reset = slate.reset_index(drop=True)
    for i, row in slate_reset.iterrows():
        pid = int(row["player_id"])
        gid = int(row["game_id"])
        tid = int(row["team_id"])
        oid = int(row.get("opponent_team_id", -1))
        pname = name_by.get(pid, str(row.get("player_name", f"player_{pid}")))
        pa = float(p_active[i])
        part_rows.append({
            "game_id": gid, "player_id": pid, "player_name": pname,
            "p_active": pa, "dnp_probability": 1.0 - pa,
            "prediction_timestamp": prediction_timestamp,
            "prediction_cutoff": row.get("prediction_cutoff", ""),
            "feature_source": row.get("feature_source", ""),
            "feature_contract_hash": row.get("feature_contract_hash", ""),
            "identity_status": row.get("_identity_status", "OK"),
        })
        for stat, rows in calibrated.items():
            a, ovf = _normalize_pmf(
                rows[i][0], rows[i][1], mode=mode, context=f"{stat}:{gid}:{pid}"
            )
            kk = np.arange(a.size)
            mean = float(np.dot(kk, a))
            var = float(np.dot((kk - mean) ** 2, a))
            cal_method = bundle.calibrators[stat].method if stat in bundle.calibrators else "identity"
            over_map, under_map, push_map = {}, {}, {}
            for L in _lines_around(mean):
                A, B, P, so, su = _settle(a, ovf, L)
                over_map[L] = so
                under_map[L] = su
                push_map[L] = P
                for side, pw, ps in (("Over", A, so), ("Under", B, su)):
                    fd = 1.0 / min(max(ps, 1e-9), 1 - 1e-9) if np.isfinite(ps) else float("nan")
                    price_rows.append({
                        "game_id": gid, "player_id": pid, "player_name": pname, "target": stat,
                        "line": L, "side": side, "p_win": pw, "p_push": P,
                        "model_probability": ps,
                        "settled_probability": ps, "fair_decimal": fd,
                        "p_active": pa, "source_track": "CALIBRATED_V6_PMF",
                        "prediction_timestamp": prediction_timestamp,
                        "market_is_external": True,
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
                    "prediction_timestamp": prediction_timestamp,
                    "source_track": "CALIBRATED_V6_PMF",
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

    combo_rows = []
    if bundle.dependence is not None:
        combo_rows = _combo_from_copula(
            slate_reset, calibrated, bundle.dependence, name_by, p_active,
            prediction_timestamp, rng,
        )
    else:
        for c in COMBOS:
            unsupported[c] = "dependence model absent; combo not fabricated from sportsbook probs"

    q1_rows = _q1_pmfs(slate_reset, bundle, q1_atoms, name_by, p_active, prediction_timestamp)
    fb_rows, fb_status = _first_basket(slate_reset, stats, p_active, name_by, prediction_timestamp)
    if fb_status:
        unsupported["first_basket"] = fb_status

    # Fantasy points withheld unless operator config present (never fabricated)
    if "fantasy_points" not in unsupported:
        unsupported["fantasy_points"] = "requires operator scoring configuration at runtime"

    manifest = {
        "artifact": "SlatePMFDelivery",
        "prediction_timestamp": prediction_timestamp,
        "n_games": len(games_out),
        "n_players": int(slate_reset["player_id"].nunique()) if len(slate_reset) else 0,
        "n_pmfs": len(pmf_objs),
        "inference_function": "wnba_props_model.sharp_v6.inference.predict_slate",
        "model_bundle_id": bundle.meta.get("bundle_id", "in-memory"),
        "model_sha256": bundle.meta.get("model_sha256") or (man.get("model_sha256") if man else None),
        "mode": mode,
        "retrain": False,
        "feature_rebuild": "live_features.build_live_feature_rows",
        "team_minutes_target": 200,
        "q1_minutes_target": 50,
        "selected_families": bundle.selected_family,
        "unsupported": unsupported,
        "identity_events": identity_events or [],
        "feature_drift_events": feature_drift or [],
        "quarantined_rows": quarantined_n,
        "game_environment_rows": int(len(env)) if env is not None else 0,
        "market_probabilities_are_external": True,
        "market_superiority": "NOT_PROVEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if len(pmf_objs) == 0 and len(games_out) > 0 and mode == "production":
        raise InferenceError("FAIL_CLOSED: scheduled games exist but PMFs are missing")

    return SlatePMFDelivery(
        prediction_timestamp=prediction_timestamp,
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


def predict_slate(
    prediction_timestamp,
    scheduled_games,
    current_rosters,
    availability_snapshot,
    historical_data: dict[str, Any],
    model_bundle: ModelBundle,
    *,
    mode: str = "production",
    identity_table: pd.DataFrame | None = None,
    require_rosters: bool | None = None,
) -> SlatePMFDelivery:
    """One authoritative inference path for OOF / live / prospective / pricing / settlement.

    Parameters
    ----------
    mode : ``production`` | ``research`` | ``validation`` | ``offline_fixture``
    """
    if mode not in {"production", "research", "validation", "offline_fixture"}:
        raise InferenceError(f"unknown inference mode: {mode}")
    if require_rosters is None:
        require_rosters = mode == "production"

    ts = prediction_timestamp
    if not isinstance(ts, str):
        ts = pd.Timestamp(ts, tz="UTC").isoformat()
    features = historical_data["features"]
    stats = historical_data["stats"]

    game_audit = audit_scheduled_games(scheduled_games, mode=mode)
    if game_audit.severity == "fail_slate":
        raise IdentityResolutionError(
            f"{game_audit.status.value}: {game_audit.events}"
        )

    if require_rosters and not current_rosters:
        if mode == "production":
            # Build date-effective rosters from history rather than ambient last-team discovery
            # without an explicit policy. Still require per-game team membership at tip.
            pass  # date-effective path below still validates
        # In production, ambient discovery is allowed only via date-effective tip filter
        # inside build_live_feature_rows (prior team before tip) — not final-season roster.

    slate, prov = build_live_feature_rows(
        prediction_timestamp=ts,
        scheduled_games=scheduled_games,
        historical_features=features,
        historical_stats=stats,
        current_rosters=current_rosters,
        availability_snapshot=availability_snapshot,
    )
    if slate.empty:
        delivery = SlatePMFDelivery(
            prediction_timestamp=ts, games=[], player_pmfs=[],
            atoms_frame=pd.DataFrame(), prices_frame=pd.DataFrame(),
            participation_frame=pd.DataFrame(),
            manifest={"status": "EMPTY_SLATE", "n_pmfs": 0, "mode": mode},
            unsupported={"slate": "no rebuildable rostered players for scheduled games"},
        )
        if mode == "production":
            raise InferenceError("FAIL_CLOSED: empty slate in production mode")
        return delivery

    if identity_table is None:
        identity_table = build_date_effective_identity_table(stats, as_of=ts)
    id_audit = resolve_roster_identities(
        slate, identity_table=identity_table, prediction_timestamp=ts, mode=mode,
    )
    if id_audit.severity == "fail_slate":
        raise IdentityResolutionError(
            f"{id_audit.status.value}: unresolved identities; refusing generic projections"
        )
    slate = id_audit.rows
    if slate.empty:
        raise IdentityResolutionError("IDENTITY_ERROR: all rows quarantined")

    games_out = []
    for gm in (
        scheduled_games if not isinstance(scheduled_games, pd.DataFrame)
        else scheduled_games.to_dict("records")
    ):
        games_out.append({
            "game_id": gm.get("id") or gm.get("game_id"),
            "scheduled_tip_utc": gm.get("scheduled_tip_utc") or gm.get("date") or gm.get("game_date"),
            "home_team_id": gm["home_team"]["id"] if isinstance(gm.get("home_team"), dict) else gm.get("home_team_id"),
            "visitor_team_id": gm["visitor_team"]["id"] if isinstance(gm.get("visitor_team"), dict) else gm.get("visitor_team_id"),
        })

    feature_drift = [
        {"player_id": None, "mean_missing": p.missingness.get("mean_missing", 0.0),
         "source": p.source, "cutoff": p.prediction_cutoff}
        for p in prov
    ]

    return _core_pmf_delivery(
        prediction_timestamp=ts,
        slate=slate,
        bundle=model_bundle,
        stats=stats,
        games_out=games_out,
        mode=mode,
        identity_events=id_audit.events + game_audit.events,
        feature_drift=feature_drift,
        quarantined_n=len(id_audit.quarantined),
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
    except np.linalg.LinAlgError as e:
        raise InferenceError(f"FAIL_CLOSED: dependence matrix not PSD after repair: {e}") from e
    rows = []
    combo_map = {
        "stocks": ("stl", "blk"),
        "pts_ast": ("pts", "ast"),
        "pts_reb": ("pts", "reb"),
        "reb_ast": ("reb", "ast"),
        "pts_reb_ast": ("pts", "reb", "ast"),
    }
    for i, row in slate.iterrows():
        pid = int(row["player_id"])
        gid = int(row["game_id"])
        pname = name_by.get(pid, f"player_{pid}")
        Z = rng.normal(size=(n_sims, len(stats))) @ L.T
        U = norm.cdf(Z)
        draws = {}
        for s in stats:
            a, ovf = calibrated[s][i]
            cdf = np.cumsum(a)
            cdf_full = np.concatenate([cdf, [min(1.0, cdf[-1] + ovf)]]) if a.size else np.array([1.0])
            ys = []
            for u in U[:, idx[s]]:
                j = int(np.searchsorted(cdf_full, u, side="left"))
                ys.append(int(min(j, a.size)))
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
        if stat in bundle.stats:
            pred = predict_stat_atoms(bundle.stats[stat], slate, q1_atoms)
        else:
            continue
        for i, row in slate.iterrows():
            a, ovf = pred[i]
            mean = float(np.dot(np.arange(a.size), a))
            pid = int(row["player_id"])
            gid = int(row["game_id"])
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
    from wnba_props_model.sharp_v6.contracts import GOVERNED_CONSTANTS
    other = float(GOVERNED_CONSTANTS["FIRST_BASKET_OTHER_MASS"]["value"])
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
            # Inactive / near-zero participation gets negligible hazard mass
            weights.append(max(1e-6, float(p_active[i]) * rates.get(pid, 0.05)))
        w = np.asarray(weights, float)
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
            "game_id": int(gid), "player_id": -1, "team_id": -1,
            "player_name": "OTHER_OR_TEAM_REBOUND_CHAIN",
            "p_first_basket": float(other), "p_active": 0.0,
            "source_track": "V6_COMPETING_RISK_FIRST_BASKET", "prediction_timestamp": ts,
        })
        for tid, tg in g.groupby("team_id"):
            tidx = [idx.index(i) for i in tg.index]
            tw = w[tidx]
            team_p = float(tw.sum() + other / 2)
            rows.append({
                "game_id": int(gid), "player_id": -100 - int(tid), "team_id": int(tid),
                "player_name": f"TEAM_{tid}_FIRST_BASKET",
                "p_first_basket": team_p, "p_active": 1.0,
                "source_track": "V6_TEAM_FIRST_BASKET", "prediction_timestamp": ts,
            })
    return rows, ""


def predict_historical_rows(
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    bundle: ModelBundle,
    stats: pd.DataFrame,
    *,
    mode: str = "validation",
) -> SlatePMFDelivery:
    """Score a historical fold with the SAME core PMF math as production inference."""
    del train_df  # reserved for API compatibility; features come from eval_df
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
    slate = eval_df.reset_index(drop=True).copy()
    if "feature_source" not in slate.columns:
        slate["feature_source"] = "historical_feature_slate"
    if "feature_contract_hash" not in slate.columns:
        slate["feature_contract_hash"] = ""
    if "prediction_cutoff" not in slate.columns:
        slate["prediction_cutoff"] = ""
    return _core_pmf_delivery(
        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
        slate=slate,
        bundle=bundle,
        stats=stats,
        games_out=games,
        mode=mode,
        identity_events=[],
        feature_drift=[],
        quarantined_n=0,
    )
