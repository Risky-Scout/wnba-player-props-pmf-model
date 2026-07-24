"""A4 - fit binary P(over) calibrators with a corrected, fail-closed contract.

Rolling-origin, complete-date CV (train dates STRICTLY BEFORE validation dates; whole
game-dates kept together). A calibrator is advanced ONLY when mean CV log loss improves,
mean CV Brier improves, the worst-fold deterioration stays within a frozen tolerance, and
row/date minimums are met. Platt/Beta use a PREREGISTERED finite regularization grid whose
value is chosen only inside rolling-origin folds (never the old near-unregularized C=1e6).

Input contract (A4):
  --scored-input     exact decision-time scored rows (parquet/csv)
  --split-manifest   frozen development/selection dates + proof_date_min
  --mode             development | certified | development-diagnostic

certified mode REQUIRES, per row: quote_pair_status == EXACT_PAIR, binary_score_eligible,
model_prob_over_final present, a decision-time line, a settled non-push outcome, a
model_oof row, no prior_only/failed fit, and NO proof rows. Certified fitting can NOT run
against a closing-consensus table.

development-diagnostic mode may consume a closing-consensus table (--closing); every output
is labeled NOT_EXACT_QUOTE_PROOF and NOT_DEPLOYABLE_POLICY.

Hierarchy: prop x role -> prop -> explicit identity. A role calibrator is only used when the
frozen role row/date minimums are met.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.models.binary_calibrators import (  # noqa: E402
    BETA_C_GRID,
    CALIBRATOR_FAMILIES,
    PLATT_C_GRID,
)
from wnba_props_model.models.market import (  # noqa: E402
    UndefinedSettledProbabilityError,
    settled_probabilities_from_pmf,
)
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

app = typer.Typer(add_completion=False)
KEYS = ["game_id", "player_id", "stat"]
ALL_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
_EPS = 1e-6
_GRID = {"platt": PLATT_C_GRID, "beta": BETA_C_GRID}


def _logloss(y, p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def _brier(y, p):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def _settled_p_over(pmf_json, line):
    try:
        return settled_probabilities_from_pmf(json_to_pmf(pmf_json), float(line)).p_over_settled
    except UndefinedSettledProbabilityError:
        return None


def _rolling_origin_folds(dates_sorted, n_folds):
    """List of (train_dates, val_dates) with train STRICTLY before val (expanding origin)."""
    uniq = list(dates_sorted)
    if len(uniq) < n_folds + 1:
        return []
    blocks = np.array_split(np.arange(len(uniq)), n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_idx = np.concatenate(blocks[:k])
        val_idx = blocks[k]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append(([uniq[i] for i in train_idx], [uniq[i] for i in val_idx]))
    return folds


def _per_fold(g, family_cls, folds, **kw):
    """Per-fold (logloss, brier) arrays (NaN where a fold is unusable)."""
    lls, brs = [], []
    for train_dates, val_dates in folds:
        tr = g[g["_d"].isin(train_dates)]
        va = g[g["_d"].isin(val_dates)]
        if len(tr) < 30 or len(va) == 0 or tr["over_outcome"].nunique() < 2:
            lls.append(np.nan); brs.append(np.nan); continue
        cal = family_cls(**kw).fit(tr["p_over_raw"].to_numpy(), tr["over_outcome"].to_numpy())
        pv = cal.predict(va["p_over_raw"].to_numpy().reshape(-1, 1))
        yv = va["over_outcome"].to_numpy()
        lls.append(_logloss(yv, pv)); brs.append(_brier(yv, pv))
    return np.array(lls, dtype=float), np.array(brs, dtype=float)


def _evaluate(g, fam, folds, worst_tol, min_improve):
    """Return dict: advances, chosen_C, mean_ll, mean_br, worst_fold_delta_ll vs identity."""
    id_ll, id_br = _per_fold(g, CALIBRATOR_FAMILIES["identity"], folds)
    best = None
    grid = _GRID.get(fam, [None])
    for C in grid:
        kw = {"C": C} if C is not None else {}
        c_ll, c_br = _per_fold(g, CALIBRATOR_FAMILIES[fam], folds, **kw)
        ok = np.isfinite(id_ll) & np.isfinite(c_ll) & np.isfinite(id_br) & np.isfinite(c_br)
        if ok.sum() == 0:
            continue
        d_ll = (c_ll - id_ll)[ok]
        d_br = (c_br - id_br)[ok]
        mean_ll, mean_br = float(np.mean(c_ll[ok])), float(np.mean(c_br[ok]))
        rec = {"chosen_C": C, "mean_ll": mean_ll, "mean_br": mean_br,
               "mean_delta_ll": float(np.mean(d_ll)), "mean_delta_br": float(np.mean(d_br)),
               "worst_fold_delta_ll": float(np.max(d_ll))}
        # Select the grid value with the best (lowest) mean CV log loss.
        if best is None or rec["mean_ll"] < best["mean_ll"]:
            best = rec
    if best is None:
        return {"advances": False, "reason": "no_usable_folds"}
    best["advances"] = bool(
        best["mean_delta_ll"] < -min_improve
        and best["mean_delta_br"] < -min_improve
        and best["worst_fold_delta_ll"] <= worst_tol)
    return best


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_best(g, folds, worst_tol, min_improve):
    """Best advancing family for a group, or None. Returns (family, eval_rec)."""
    best_fam, best_rec = None, None
    cand = {}
    for fam in ("platt", "beta", "isotonic"):
        rec = _evaluate(g, fam, folds, worst_tol, min_improve)
        cand[fam] = rec
        if rec.get("advances") and (best_rec is None or rec["mean_ll"] < best_rec["mean_ll"]):
            best_fam, best_rec = fam, rec
    return best_fam, best_rec, cand


def _fit_final(g, fam, rec, out: Path, tag: str):
    import joblib
    kw = {"C": rec["chosen_C"]} if rec.get("chosen_C") is not None else {}
    model = CALIBRATOR_FAMILIES[fam](**kw).fit(g["p_over_raw"].to_numpy(), g["over_outcome"].to_numpy())
    art = out / f"binary_{fam}_{tag}.pkl"
    joblib.dump(model, art)
    return {"method": fam, "path": str(art).replace("\\", "/"), "sha256": _sha256(art),
            "chosen_C": rec.get("chosen_C")}


_CERTIFIED_REQUIRED = ["quote_pair_status", "binary_score_eligible", "model_prob_over_final",
                       "line", "actual_outcome", "oof_prediction_type", "fit_status"]


def _apply_certified_gate(df: pd.DataFrame, proof_date_min: "str | None") -> pd.DataFrame:
    missing = [c for c in _CERTIFIED_REQUIRED if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"certified mode requires columns {missing}; a closing-consensus table can NOT be "
            "used for certified fitting (no EXACT_PAIR/eligibility/settlement provenance).")
    d = df.copy()
    d = d[d["quote_pair_status"].astype(str) == "EXACT_PAIR"]
    d = d[d["binary_score_eligible"] == True]  # noqa: E712
    d = d[d["model_prob_over_final"].notna()]
    d = d[d["line"].notna() & d["actual_outcome"].notna()]
    d = d[d["oof_prediction_type"].astype(str) == "model_oof"]
    d = d[~d["fit_status"].astype(str).isin(["prior_only", "failed_model_fit"])]
    d = d[d["actual_outcome"].astype(float) != d["line"].astype(float)]  # settled non-push
    if proof_date_min and "game_date" in d.columns:
        d = d[pd.to_datetime(d["game_date"], errors="coerce") < pd.Timestamp(proof_date_min)]
    return d


@app.command()
def fit(
    scored_input: str = typer.Option(None, "--scored-input",
                                      help="Exact decision-time scored rows (certified/development)."),
    split_manifest: str = typer.Option(None, "--split-manifest",
                                        help="Frozen development/selection dates + proof_date_min."),
    mode: str = typer.Option("development", "--mode",
                             help="development | certified | development-diagnostic"),
    closing: str = typer.Option(None, "--closing",
                                help="Closing-consensus table (development-diagnostic ONLY)."),
    oof: str = typer.Option("data/oof/oof_player_stat_pmfs.parquet", "--oof",
                            help="OOF PMFs (development-diagnostic ONLY)."),
    split_date: str = typer.Option(None, "--split-date",
                                   help="Legacy selection cutoff (development-diagnostic)."),
    out_dir: str = typer.Option("artifacts/models/calibration", "--out-dir"),
    policy_out: str = typer.Option("config/binary_calibration_policy_v1.json", "--policy-out"),
    selection_out: str = typer.Option("artifacts/models/calibration/binary_calibrator_selection.json",
                                       "--selection-out"),
    cv_folds: int = typer.Option(4, "--cv-folds"),
    min_rows: int = typer.Option(120, "--min-rows"),
    role_min_rows: int = typer.Option(200, "--role-min-rows"),
    role_min_dates: int = typer.Option(15, "--role-min-dates"),
    min_improvement: float = typer.Option(1e-4, "--min-improvement"),
    worst_fold_tolerance: float = typer.Option(0.02, "--worst-fold-tolerance"),
) -> None:
    if mode not in ("development", "certified", "development-diagnostic"):
        raise typer.BadParameter("mode must be development|certified|development-diagnostic")

    proof_date_min = None
    selection_cutoff = None
    if split_manifest:
        sm = json.loads(Path(split_manifest).read_text())
        proof_date_min = sm.get("proof_date_min")
        selection_cutoff = sm.get("selection_date_max") or sm.get("development_date_max") or proof_date_min

    labels: list[str] = []
    if mode == "development-diagnostic":
        # Legacy consensus path - explicitly NON-deployable, NOT exact-quote proof.
        labels = ["NOT_EXACT_QUOTE_PROOF", "NOT_DEPLOYABLE_POLICY"]
        oof_p, cl_p = Path(oof), Path(closing or "")
        if not oof_p.exists() or not cl_p.exists():
            typer.echo(f"[FATAL] development-diagnostic needs --oof and --closing "
                       f"(oof={oof_p.exists()} closing={cl_p.exists()})", err=True)
            raise typer.Exit(1)
        oofd = pd.read_parquet(oof_p).dropna(subset=["pmf_json", "actual_outcome"]).copy()
        if "oof_prediction_type" in oofd.columns:
            oofd = oofd[oofd["oof_prediction_type"] == "model_oof"]
        if "fit_status" in oofd.columns:
            oofd = oofd[~oofd["fit_status"].isin(["prior_only", "failed_model_fit"])]
        cc = pd.read_parquet(cl_p)
        for k in KEYS:
            oofd[k] = oofd[k].astype("string"); cc[k] = cc[k].astype("string")
        gd = "game_date" if "game_date" in oofd.columns else None
        role_cols = [c for c in ("role", "role_bucket") if c in oofd.columns]
        df = cc.merge(oofd[KEYS + ["pmf_json", "actual_outcome"] + role_cols + ([gd] if gd else [])],
                      on=KEYS, how="inner")
        if df.empty:
            typer.echo("[FATAL] no OOF<->closing overlap.", err=True); raise typer.Exit(1)
        if gd is None:
            df["game_date"] = pd.to_datetime(df["commence_time"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["p_over_raw"] = [_settled_p_over(pj, ln) for pj, ln in zip(df["pmf_json"], df["line"])]
        selection_cutoff = selection_cutoff or split_date
    else:
        if not scored_input:
            raise typer.BadParameter(f"mode={mode} requires --scored-input (exact decision-time rows).")
        sp = Path(scored_input)
        if not sp.exists():
            typer.echo(f"[FATAL] missing --scored-input {sp}", err=True); raise typer.Exit(1)
        df = pd.read_parquet(sp) if sp.suffix == ".parquet" else pd.read_csv(sp)
        if mode == "certified":
            if not split_manifest:
                raise typer.BadParameter("certified mode requires --split-manifest.")
            df = _apply_certified_gate(df, proof_date_min)
            labels = ["EXACT_QUOTE_CERTIFIED", "DEPLOYABLE_CANDIDATE"]
        else:
            labels = ["NOT_CERTIFIED", "DEVELOPMENT_ONLY"]
        if "model_prob_over_final" not in df.columns:
            raise typer.BadParameter("scored-input must carry model_prob_over_final (delivery parity).")
        df["p_over_raw"] = df["model_prob_over_final"].astype(float)

    df = df[df["p_over_raw"].notna()].copy()
    df = df[df["actual_outcome"].astype(float) != df["line"].astype(float)].copy()  # push exclusion
    df["over_outcome"] = (df["actual_outcome"].astype(float) > df["line"].astype(float)).astype(int)
    df["_d"] = pd.to_datetime(df["game_date"], errors="coerce")
    if selection_cutoff:
        df = df[df["_d"] < pd.Timestamp(selection_cutoff)]
    role_col = next((c for c in ("role", "role_bucket") if c in df.columns), None)
    selection = df.sort_values("_d")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    policy_props, report = {}, {}
    shipped = []
    for stat in ALL_PROPS:
        g = selection[selection["stat"] == stat]
        rec = {"n_selection": int(len(g)), "n_dates": int(g["_d"].nunique()),
               "positives": int(g["over_outcome"].sum()) if len(g) else 0}
        entry = {"method": "identity"}
        if len(g) < min_rows or g["over_outcome"].nunique() < 2:
            rec["decision"] = "identity_insufficient_data"; policy_props[stat] = entry
            report[stat] = rec; continue
        folds = _rolling_origin_folds(np.sort(g["_d"].dropna().unique()), cv_folds)
        fam, best, cand = _select_best(g, folds, worst_fold_tolerance, min_improvement)
        rec["cv_candidates"] = cand
        if fam is None:
            rec["decision"] = "identity_no_improvement"; policy_props[stat] = entry
            report[stat] = rec; continue
        entry = _fit_final(g, fam, best, out, stat)
        shipped.append(f"{stat}:{fam}")
        rec["decision"] = f"ship_{fam}_C{best.get('chosen_C')}"

        # A4 hierarchy: prop x role overrides, only where role row/date minimums are met.
        roles = {}
        if role_col:
            for role_name, gr in g.groupby(role_col):
                n, nd = len(gr), gr["_d"].nunique()
                if n < role_min_rows or nd < role_min_dates or gr["over_outcome"].nunique() < 2:
                    roles[str(role_name)] = {"method": "identity",
                                             "reason": f"role_min_not_met(rows={n},dates={nd})"}
                    continue
                rfolds = _rolling_origin_folds(np.sort(gr["_d"].dropna().unique()), cv_folds)
                rfam, rbest, _ = _select_best(gr, rfolds, worst_fold_tolerance, min_improvement)
                if rfam is None:
                    roles[str(role_name)] = {"method": "identity", "reason": "role_no_improvement"}
                else:
                    roles[str(role_name)] = _fit_final(gr, rfam, rbest, out, f"{stat}_{role_name}")
                    shipped.append(f"{stat}/{role_name}:{rfam}")
        if roles:
            entry["roles"] = roles
        policy_props[stat] = entry
        report[stat] = rec

    policy = {
        "version": "binary-cal-v2",
        "mode": mode,
        "labels": labels,
        "deployable": bool(mode == "certified"),
        "enabled": any(v.get("method") != "identity" or v.get("roles") for v in policy_props.values()),
        "cv": "rolling_origin_complete_date",
        "regularization": {"platt_C_grid": list(PLATT_C_GRID), "beta_C_grid": list(BETA_C_GRID)},
        "advance_rule": "mean_delta_logloss<0 AND mean_delta_brier<0 AND worst_fold_delta_logloss<=tol",
        "worst_fold_tolerance": worst_fold_tolerance,
        "hierarchy": "prop_x_role -> prop -> identity",
        "role_min_rows": role_min_rows,
        "role_min_dates": role_min_dates,
        "split_manifest": split_manifest,
        "selection_cutoff": str(selection_cutoff) if selection_cutoff else None,
        "proof_date_min": proof_date_min,
        "props": policy_props,
    }
    Path(policy_out).parent.mkdir(parents=True, exist_ok=True)
    Path(policy_out).write_text(json.dumps(policy, indent=2, default=str) + "\n")
    Path(selection_out).parent.mkdir(parents=True, exist_ok=True)
    Path(selection_out).write_text(json.dumps(report, indent=2, default=str) + "\n")
    typer.echo(f"[binary-cal] mode={mode} labels={labels} rows={len(selection):,} folds={cv_folds}")
    for stat in ALL_PROPS:
        typer.echo(f"  {stat:9s}: {report.get(stat, {}).get('decision', 'n/a')}")
    typer.echo(f"[binary-cal] deployable={policy['deployable']} shipped={shipped} -> {policy_out}")


if __name__ == "__main__":
    app()
