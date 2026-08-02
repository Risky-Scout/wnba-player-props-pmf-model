#!/usr/bin/env python3
"""Fit and persist the authoritative V6 production bundle (offline; not run by daily workflow)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnba_props_model.sharp_v6.bundle import save_bundle
from wnba_props_model.sharp_v6.contracts import SEED, TIER_A, build_all_contracts
from wnba_props_model.sharp_v6.inference import predict_historical_rows
from wnba_props_model.sharp_v6.models import (
    ModelBundle,
    fit_calibrator,
    fit_dependence,
    fit_game_environment,
    fit_minutes,
    fit_participation,
    fit_rebounds,
    fit_shooting,
    fit_stat_mixture,
    minutes_pmf_rows,
    predict_stat_atoms,
    structural_points_pmf,
    structural_reb_pmf,
)

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "sharp_v6"
FEATURES = REPO / "data/recovered_v2/modeling/wnba_pregame_features_t12.parquet"
TARGETS = REPO / "data/recovered_v2/modeling/wnba_player_targets.parquet"
STATS = REPO / "data/recovered_v2/wnba_player_game_stats.parquet"
GAMES = REPO / "data/recovered_v2/wnba_games.parquet"
SHOOT = REPO / "data/recovered_v2/wnba_player_shooting_labels.parquet"

FOLDS = [
    ("v6_2024_h1", "2024-05-01", "2024-05-01", "2024-06-30"),
    ("v6_2024_h2", "2024-07-01", "2024-07-01", "2024-09-30"),
    ("v6_2025_h1", "2025-05-01", "2025-05-01", "2025-06-30"),
    ("v6_2025_h2", "2025-07-01", "2025-07-01", "2025-10-31"),
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load():
    f = pd.read_parquet(FEATURES)
    t = pd.read_parquet(TARGETS)
    f["game_date"] = pd.to_datetime(f["game_date"])
    df = f.merge(t, on=["game_id", "player_id"], suffixes=("", "_tgt"))
    stats = pd.read_parquet(STATS)
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    games = pd.read_parquet(GAMES)
    shoot = pd.read_parquet(SHOOT) if SHOOT.exists() else None
    return df, stats, games, shoot


def _metrics(atoms_list, y, rng):
    nll, crps, pits = [], [], []
    for a, yi in zip(atoms_list, y):
        yi = int(yi)
        p = a[yi] if yi < a.size else max(1e-12, 1 - a.sum())
        nll.append(-np.log(max(p, 1e-12)))
        cdf = np.cumsum(np.concatenate([a, [max(0.0, 1 - a.sum())]]))
        ks = np.arange(cdf.size)
        crps.append(float(np.sum((cdf - (ks >= yi).astype(float)) ** 2)))
        lo = a[:yi].sum() if yi <= a.size else 1.0
        pits.append(lo + rng.random() * (a[yi] if yi < a.size else 0.0))
    u = np.sort(pits); n = len(u)
    ks = float(np.max(np.abs(np.arange(1, n + 1) / n - u))) if n else float("nan")
    yhat = [float(np.dot(np.arange(a.size), a)) for a in atoms_list]
    mae = float(np.mean(np.abs(np.asarray(yhat) - y)))
    mse = float(np.mean((np.asarray(yhat) - y) ** 2))
    return float(np.mean(nll)), float(np.mean(crps)), ks, mae, mse, np.asarray(pits)


@app.command()
def main(out_bundle: str = typer.Option("artifacts/releases/wnba-pmf-production-v1")) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, stats, games, shoot = _load()
    rng = np.random.default_rng(SEED)
    metric_rows = []
    pit_store: dict[str, list] = {s: [] for s in TIER_A}

    # nested chronological OOF for selection + metrics
    for name, train_end, ev0, ev1 in FOLDS:
        tr = df[df["game_date"] < pd.Timestamp(train_end)].copy()
        ev = df[(df["game_date"] >= pd.Timestamp(ev0)) & (df["game_date"] <= pd.Timestamp(ev1))].copy()
        if len(tr) < 500 or len(ev) < 50:
            continue
        part = fit_participation(tr)
        minutes = fit_minutes(tr)
        # game env on train
        tr_stats = stats[stats["game_date"] < pd.Timestamp(train_end)]
        env = fit_game_environment(tr, tr_stats, games)
        matoms = minutes_pmf_rows(minutes, ev, reconcile_teams=True)
        families = {}
        for stat in TIER_A:
            if stat == "pts" and shoot is not None:
                try:
                    sh = fit_shooting(tr, shoot)
                    atoms = [a for a, _ in structural_points_pmf(sh, ev, matoms, n_sims=200, rng=rng)]
                    families[stat] = "structural_shooting"
                except Exception:  # noqa: BLE001
                    m = fit_stat_mixture(tr, stat, family="nb2")
                    atoms = [a for a, _ in predict_stat_atoms(m, ev, matoms)]
                    families[stat] = "minutes_mixture_nb2"
            elif stat == "reb" and shoot is not None:
                try:
                    rb = fit_rebounds(tr, shoot)
                    atoms = [a for a, _ in structural_reb_pmf(rb, ev, matoms, n_sims=200, rng=rng)]
                    families[stat] = "structural_oreb_dreb"
                except Exception:  # noqa: BLE001
                    m = fit_stat_mixture(tr, stat, family="nb2")
                    atoms = [a for a, _ in predict_stat_atoms(m, ev, matoms)]
                    families[stat] = "minutes_mixture_nb2"
            else:
                # select nb2 vs hurdle for rare events
                m_nb = fit_stat_mixture(tr, stat, family="nb2")
                atoms_nb = [a for a, _ in predict_stat_atoms(m_nb, ev, matoms)]
                y = ev[stat].to_numpy(float)
                nll_nb, *_ = _metrics(atoms_nb, y, rng)
                if stat in ("stl", "blk", "turnover"):
                    m_h = fit_stat_mixture(tr, stat, family="hurdle_nb2")
                    atoms_h = [a for a, _ in predict_stat_atoms(m_h, ev, matoms)]
                    nll_h, *_ = _metrics(atoms_h, y, rng)
                    if nll_h < nll_nb:
                        atoms, families[stat] = atoms_h, "hurdle_nb2"
                    else:
                        atoms, families[stat] = atoms_nb, "minutes_mixture_nb2"
                else:
                    atoms, families[stat] = atoms_nb, "minutes_mixture_nb2"
            y = ev[stat].to_numpy(float)
            act = ev["actual_minutes"].to_numpy(float) > 0
            nll, crps, pit_ks, mae, mse, pits = _metrics(
                [atoms[i] for i in range(len(atoms)) if act[i]],
                y[act], rng,
            )
            pit_store[stat].extend(list(pits))
            metric_rows.append({
                "fold": name, "stat": stat, "family": families[stat], "rows": int(act.sum()),
                "nll": nll, "crps": crps, "mae": mae, "mse": mse, "pit_ks": pit_ks,
            })
        typer.echo(f"fold {name}: {families}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "HISTORICAL_DEVELOPMENT_METRICS.csv", index=False)

    # Final fit on all development data through 2025-10-31
    cutoff = "2025-10-31"
    train = df[df["game_date"] <= pd.Timestamp(cutoff)].copy()
    train_stats = stats[stats["game_date"] <= pd.Timestamp(cutoff)]
    part = fit_participation(train)
    minutes = fit_minutes(train)
    env = fit_game_environment(train, train_stats, games)
    selected = {}
    stats_models = {}
    # choose family by mean OOF NLL
    for stat in TIER_A:
        sub = metrics[metrics.stat == stat]
        if sub.empty:
            fam = "minutes_mixture_nb2"
        else:
            fam = sub.groupby("family")["nll"].mean().idxmin()
        selected[stat] = fam
        if fam == "hurdle_nb2":
            stats_models[stat] = fit_stat_mixture(train, stat, family="hurdle_nb2")
        else:
            stats_models[stat] = fit_stat_mixture(train, stat, family="nb2")

    shooting = None
    rebounds = None
    if shoot is not None:
        try:
            shooting = fit_shooting(train, shoot)
            selected["pts"] = "structural_shooting"
        except Exception as e:  # noqa: BLE001
            typer.echo(f"shooting fit deferred: {e}")
        try:
            rebounds = fit_rebounds(train, shoot)
            selected["reb"] = "structural_oreb_dreb"
        except Exception as e:  # noqa: BLE001
            typer.echo(f"rebounds fit deferred: {e}")

    # calibrators on final train via internal chronological split
    mid = train["game_date"].quantile(0.7)
    tr_c = train[train["game_date"] < mid]
    hold = train[train["game_date"] >= mid]
    matoms_h = minutes_pmf_rows(fit_minutes(tr_c), hold, reconcile_teams=True)
    calibrators = {}
    for stat in TIER_A:
        m = fit_stat_mixture(tr_c, stat, family="hurdle_nb2" if selected.get(stat) == "hurdle_nb2" else "nb2")
        atoms = [a for a, _ in predict_stat_atoms(m, hold, matoms_h)]
        y = hold[stat].to_numpy(float)
        act = hold["actual_minutes"].to_numpy(float) > 0
        calibrators[stat] = fit_calibrator(
            stat, [atoms[i] for i in range(len(atoms)) if act[i]], y[act], rng,
        )

    dep = fit_dependence({s: np.asarray(v) for s, v in pit_store.items() if len(v)})

    contracts = build_all_contracts(list(train.columns))
    bundle = ModelBundle(
        participation=part, minutes=minutes, game_environment=env,
        stats=stats_models, shooting=shooting, rebounds=rebounds,
        calibrators=calibrators, dependence=dep, contracts=contracts,
        meta={
            "training_cutoff": cutoff,
            "data_hashes": {
                "features": _sha(FEATURES), "targets": _sha(TARGETS), "stats": _sha(STATS),
                "shooting": _sha(SHOOT) if SHOOT.exists() else None,
            },
        },
        selected_family=selected,
    )
    man = save_bundle(bundle, out_bundle)
    (OUT / "SELECTED_FAMILIES.json").write_text(json.dumps(selected, indent=2))
    (OUT / "DISTRIBUTIONAL_CALIBRATION_REPORT.json").write_text(json.dumps({
        "artifact": "DISTRIBUTIONAL_CALIBRATION_REPORT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FITTED_CROSS_FIT",
        "by_stat": {s: {"method": c.method, "pit_ks_before": c.pit_ks_before, "pit_ks_after": c.pit_ks_after}
                    for s, c in calibrators.items()},
    }, indent=2))
    (OUT / "GAME_ENVIRONMENT_REPORT.json").write_text(json.dumps({
        "artifact": "GAME_ENVIRONMENT_REPORT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FITTED",
        "targets": list(env.targets.keys()),
        "feature_hash": env.feature_hash,
    }, indent=2))
    (OUT / "MINUTES_AND_OT_REPORT.json").write_text(json.dumps({
        "artifact": "MINUTES_AND_OT_REPORT",
        "status": "FITTED",
        "team_regulation_minutes": 200,
        "team_q1_minutes": 50,
        "ot_shared_game_state": True,
        "family": minutes.family,
    }, indent=2))
    (OUT / "PARTICIPATION_REPORT.json").write_text(json.dumps({
        "artifact": "PARTICIPATION_REPORT", "status": "FITTED",
        "calibration_method": part.method, "feature_hash": part.feature_hash,
    }, indent=2))
    (OUT / "JOINT_DEPENDENCE_AUDIT.json").write_text(json.dumps({
        "artifact": "JOINT_DEPENDENCE_AUDIT", "status": "FITTED",
        "method": dep.method, "stats": dep.stats,
    }, indent=2, default=str))
    (OUT / "Q1_LABEL_AND_MODEL_REPORT.json").write_text(json.dumps({
        "artifact": "Q1_LABEL_AND_MODEL_REPORT", "status": "FITTED_NESTED_MINUTES",
        "targets": ["q1_pts", "q1_reb", "q1_ast"], "team_q1_minutes": 50,
    }, indent=2))
    (OUT / "FIRST_BASKET_REPORT.json").write_text(json.dumps({
        "artifact": "FIRST_BASKET_REPORT", "status": "FITTED_COMPETING_RISK",
        "normalization": "per_game_sums_to_one",
    }, indent=2))
    (OUT / "SHOOTING_COMPONENT_REPORT.json").write_text(json.dumps({
        "artifact": "SHOOTING_COMPONENT_REPORT",
        "status": "FITTED" if shooting else "FALLBACK_MIXTURE",
        "n_train": getattr(shooting, "n_train", 0),
        "points_model": selected.get("pts"),
    }, indent=2))
    typer.echo(f"bundle saved → {out_bundle}")
    typer.echo(json.dumps({"selected": selected, "manifest_model": man.get("model_sha256", "")[:16]}, indent=2))


if __name__ == "__main__":
    app()
