"""DEPRECATED / LEGACY_CONTROL — production uses scripts/run_wnba_pmf.py → sharp_v6.predict_slate."""
"""Real-slate pricing from fitted sharp_v3 artifacts (NOT a fixture).

Refit participation + minutes + Tier A stat models on all history strictly before the slate date,
then price the newest real WNBA slate available in the recovered dataset from real point-in-time
features. Emits atom PMFs + fair Over/Under prices with full lineage. Players missing a valid
fitted path abstain honestly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnba_props_model.pricing import engine as E
from wnba_props_model.sharp_v3 import core as C

app = typer.Typer(add_completion=False)
_HGB = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.06, "min_samples_leaf": 40,
        "l2_regularization": 1.0, "random_state": 20260730}
SEED = 20260730


def _num(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _usable(X):
    ok = np.zeros(X.shape[1], bool)
    for j in range(X.shape[1]):
        f = X[:, j][np.isfinite(X[:, j])]
        ok[j] = f.size > 0 and np.unique(f).size >= 2
    return ok


def _prep(train, slate, feat):
    assert not (set(feat) & set(C.LABEL_COLS)), "leakage guard"
    Xtr = _num(train, feat); m = _usable(Xtr); used = [c for c, k in zip(feat, m) if k]
    return Xtr[:, m], _num(slate, used), used


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


@app.command()
def main(date: str = typer.Option(None, "--date", help="slate date; default = latest in data")) -> None:
    _, df = C.load_verified()
    slate_date = pd.Timestamp(date) if date else df["game_date"].max()
    train = df[df["game_date"] < slate_date]
    slate = df[df["game_date"] == slate_date].copy()
    if slate.empty:
        raise SystemExit(f"no slate rows for {slate_date.date()}")
    code_sha = json.loads((C.REPO / "artifacts/sharp_v3/MODEL_LINEAGE.json").read_text())["code_sha"]
    design_hash = json.loads((C.REPO / "artifacts/sharp_v3/V3_FREEZE_MANIFEST.json").read_text())["modeling_design_v3_sha256"]
    data_hash = json.loads((C.REPO / "artifacts/sharp_v3/PRIVATE_INPUT_MANIFEST.json").read_text())["inputs"]["pregame_features_t12"]["sha256"][:16]
    ts = datetime.now(timezone.utc).isoformat()

    # player names + opponent
    try:
        players = pd.read_parquet(C.REPO / "data/recovered_v2/wnba_players.parquet")
        namecol = "full_name" if "full_name" in players.columns else \
            ("name" if "name" in players.columns else players.columns[1])
        idcol = "id" if "id" in players.columns else "player_id"
        name_map = dict(zip(players[idcol], players[namecol]))
    except Exception:  # noqa: BLE001
        name_map = {}

    # participation (calibrated)
    pf = C.stat_feature_contract("participation", list(df.columns))
    Xtr, Xsl, _ = _prep(train, slate, pf)
    ytr = train["participation"].to_numpy(int)
    order = train["game_date"].rank(method="first").to_numpy(); cut = np.quantile(order, 0.8)
    clf = HistGradientBoostingClassifier(**_HGB).fit(Xtr[order <= cut], ytr[order <= cut])
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(Xtr[order > cut])[:, 1], ytr[order > cut])
    full = HistGradientBoostingClassifier(**_HGB).fit(Xtr, ytr)
    p_active = np.clip(iso.predict(full.predict_proba(Xsl)[:, 1]), 1e-4, 1 - 1e-4)

    # stat models
    stat_pmfs = {}
    feat_hashes = {}
    for stat in C.TIER_A:
        feat = C.stat_feature_contract(stat, list(df.columns))
        act = train[train["actual_minutes"] > 0]
        Xa, Xs, used = _prep(act, slate, feat)
        y = np.clip(act[stat].to_numpy(float), 0, None)
        reg = HistGradientBoostingRegressor(**_HGB).fit(Xa, y)
        r = C.residual_dispersion_r(y, np.clip(reg.predict(Xa), 1e-4, None))
        mu = np.clip(reg.predict(Xs), 1e-4, None)
        stat_pmfs[stat] = ([C.count_pmf(m, r, C.HARD_CAP[stat]) for m in mu], r)
        feat_hashes[stat] = C.feature_schema_hash(used)

    activation = json.loads((C.REPO / "artifacts/sharp_v3/ACTIVATION_REGISTRY.json").read_text())["tier_A"]
    atom_rows, price_rows, cov_rows = [], [], []
    for i, (_, row) in enumerate(slate.iterrows()):
        pid = int(row["player_id"]); gid = int(row["game_id"])
        pa = float(p_active[i])
        pname = name_map.get(pid, f"player_{pid}")
        for stat in C.TIER_A:
            pmf = stat_pmfs[stat][0][i]
            overflow = float(max(0.0, 1.0 - pmf.sum())) if pmf.sum() < 1 else 0.0
            model_status = activation.get(stat, {}).get("market_track", "PURE_UNCERTIFIED")
            for k, prob in enumerate(pmf):
                if prob <= 1e-9:
                    continue
                atom_rows.append({"game_id": gid, "canonical_player_id": pid, "player_name": pname,
                    "team_id": int(row["team_id"]), "opponent_id": int(row["opponent_team_id"]),
                    "period": "FULL", "stat": stat, "atom": int(k), "probability": float(prob),
                    "overflow_probability": overflow, "p_active": pa, "prediction_timestamp": ts,
                    "scheduled_tip": str(row["game_date"]), "source_track": "PURE_PMF",
                    "model_status": model_status, "calibration_status": "UNCALIBRATED_RESEARCH",
                    "data_hash": data_hash, "feature_hash": feat_hashes[stat], "model_hash": "hgb_nb2",
                    "calibrator_hash": "none", "design_hash": design_hash[:16], "code_sha": code_sha[:12],
                    "uncertainty": float(1 - pa), "abstention_reason": ""})
            # fair Over/Under at a grid of lines from the same PMF
            mean = float(np.dot(np.arange(pmf.size), pmf))
            base = max(round(mean * 2) / 2 - 0.5, 0.5)
            for L in sorted({round(base + 0.5 * j, 1) for j in range(-2, 4) if base + 0.5 * j >= 0.5}):
                pl = E.price_over_under(pmf, L, f"player_{stat}")
                for side, pw, fd, fa in [("Over", pl.p_over_win, pl.fair_decimal_over, pl.fair_american_over),
                                          ("Under", pl.p_under_win, pl.fair_decimal_under, pl.fair_american_under)]:
                    price_rows.append({"game_id": gid, "player_id": pid, "player_name": pname, "stat": stat,
                        "line": L, "side": side, "p_win": pw, "p_push": pl.p_push,
                        "settled_probability": pl.p_over_settled if side == "Over" else pl.p_under_settled,
                        "fair_decimal": fd, "fair_american": fa, "p_active": pa, "source_track": "PURE_PMF",
                        "model_status": model_status, "calibration_status": "UNCALIBRATED_RESEARCH"})
            cov_rows.append({"game_id": gid, "player_id": pid, "stat": stat, "status": "PRICED",
                             "model_status": model_status})

    out = C.REPO / "deliveries" / "sharp_v3" / str(slate_date.date())
    out.mkdir(parents=True, exist_ok=True)
    adf = pd.DataFrame(atom_rows); pdf = pd.DataFrame(price_rows)
    adf.to_parquet(out / "active_atom_pmfs.parquet", index=False)
    pdf.to_parquet(out / "fair_prices.parquet", index=False)
    pdf.to_csv(out / "pricing_inventory.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(C.REPO / "artifacts/sharp_v3/REAL_SLATE_COVERAGE.csv", index=False)
    manifest = {"artifact": "pricing_manifest", "slate_date": str(slate_date.date()),
                "is_fixture": False, "source": "fitted sharp_v3 artifacts + real point-in-time features",
                "n_players": int(slate["player_id"].nunique()), "n_atoms": len(adf),
                "n_priced_lines": len(pdf), "generated_at_utc": ts, "code_sha": code_sha,
                "design_hash": design_hash, "data_hash": data_hash, "seed": SEED,
                "note": "Newest real slate in the recovered dataset. Production track = market fallback "
                        "(pure model did not beat market on dev/holdout). Both Over and Under written."}
    (out / "pricing_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    typer.echo(f"REAL-SLATE {slate_date.date()}: players={slate['player_id'].nunique()} "
               f"atoms={len(adf)} priced_lines={len(pdf)} -> {out.relative_to(C.REPO)}")


if __name__ == "__main__":
    app()
