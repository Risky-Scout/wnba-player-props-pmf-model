"""Build the G0-v2 exact-quote scored table (owner critical-path step 3+4).

Corrects the G0 baseline's disqualifying limitation (it averaged market_prob_over_no_vig
across books, destroying exact quote identity). G0-v2 keeps EXACT identity:

  * one row per (sportsbook, event, player, prop) at the DECISION-time snapshot,
  * exact line, exact Over/American and Under/American odds,
  * market no-vig via Shin's method computed from that exact Over/Under pair (never
    averaged, never consensus, never closing-as-decision-time),
  * model_prob_over_final produced ONLY by build_probability_lineage (the same single
    source the live delivery path uses -> evaluated==deployed parity) at the exact line,
  * settlement from the OOF actual_outcome (actual > line = over; actual == line = push,
    dropped; binary-ineligible all-push PMFs dropped, never fabricated).

Rolling-origin split: selection = earliest ``--selection-frac`` of distinct game dates,
test = the remainder (an untouched forward window used only for proof).

Outputs (under artifacts/market_feature_proof/G0_v2/):
  * scored_candidates_g0v2.parquet     — all books, every exact comparable row
  * G0_V2_METRICS.csv / .json          — per-prop model/market log loss, Brier, AUC, ECE,
                                          deltas, row/date counts (primary book + pooled)
  * G0_V2_QUOTE_COVERAGE.json          — exact comparable rows + unique dates by prop/book
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from wnba_props_model.models.market import shin_no_vig_two_way  # noqa: E402
from wnba_props_model.models.probability_contract import (  # noqa: E402
    FINAL_PROBABILITY_COLUMN,
)
from wnba_props_model.models.probability_lineage import build_probability_lineage  # noqa: E402
from wnba_props_model.models.simulation import json_to_pmf  # noqa: E402

app = typer.Typer(add_completion=False)

DIRECT_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
EPS = 1e-6


def _clip(x):
    return np.clip(np.asarray(x, float), EPS, 1 - EPS)


def _ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins, weighted by bin count)."""
    y = np.asarray(y, int)
    p = _clip(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _metrics(y, p_model, p_market) -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    pm = _clip(p_model)
    pk = _clip(p_market)
    out = {
        "n_settled": int(len(y)),
        "model_logloss": float(np.mean(-(y * np.log(pm) + (1 - y) * np.log(1 - pm)))),
        "market_logloss": float(np.mean(-(y * np.log(pk) + (1 - y) * np.log(1 - pk)))),
        "model_brier": float(np.mean((pm - y) ** 2)),
        "market_brier": float(np.mean((pk - y) ** 2)),
        "model_ece": _ece(y, pm),
        "market_ece": _ece(y, pk),
    }
    try:
        out["model_auc"] = float(roc_auc_score(y, pm))
        out["market_auc"] = float(roc_auc_score(y, pk))
    except ValueError:
        out["model_auc"] = float("nan"); out["market_auc"] = float("nan")
    out["logloss_delta"] = out["model_logloss"] - out["market_logloss"]
    out["brier_delta"] = out["model_brier"] - out["market_brier"]
    out["auc_delta"] = out["model_auc"] - out["market_auc"]
    return out


def _pair_decision_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """One exact Over/Under pair per (book, event, game, player, prop, line) at decision time."""
    q = quotes[(quotes["snapshot_label"] == "decision")
               & quotes["stat"].isin(DIRECT_PROPS)].copy()
    for c in ("game_id", "player_id"):
        q[c] = q[c].astype(str)
    q["stat"] = q["stat"].astype(str)
    # latest snapshot per identity+side (defensive; decision snapshot is already unique)
    q = q.sort_values("snapshot_time").drop_duplicates(
        ["book", "odds_event_id", "game_id", "player_id", "stat", "line", "side"], keep="last")
    keys = ["book", "odds_event_id", "game_id", "player_id", "stat", "line",
            "game_date", "snapshot_time"]
    over = q[q["side"] == "over"][keys + ["american_odds"]].rename(
        columns={"american_odds": "over_odds"})
    under = q[q["side"] == "under"][["book", "odds_event_id", "game_id", "player_id",
                                     "stat", "line", "american_odds"]].rename(
        columns={"american_odds": "under_odds"})
    pair = over.merge(under, on=["book", "odds_event_id", "game_id", "player_id", "stat", "line"],
                      how="inner")
    return pair


@app.command()
def main(
    oof: str = typer.Option("artifacts/models/calibration/oof_predictions.parquet", "--oof"),
    quotes: str = typer.Option("artifacts/p1/p1_quotes.parquet", "--quotes"),
    out_dir: str = typer.Option("artifacts/market_feature_proof/G0_v2", "--out-dir"),
    primary_book: str = typer.Option("draftkings", "--primary-book",
                                     help="Legacy single-book headline (retained for sensitivity)."),
    quote_policy: str = typer.Option("config/book_quote_priority_v1.json", "--quote-policy",
                                     help="Frozen sportsbook-priority policy for the PRIMARY "
                                          "deterministic one-quote-per-observation selection."),
    selection_frac: float = typer.Option(0.6, "--selection-frac"),
    test_dates: int = typer.Option(0, "--test-dates",
                                   help="If >0, reserve the LAST N distinct game dates as the "
                                        "untouched test/proof window (guarantees N date clusters); "
                                        "overrides --selection-frac."),
) -> None:
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    oof_df = pd.read_parquet(oof)
    oof_df = oof_df[oof_df["actual_outcome"].notna() & oof_df["pmf_json"].notna()].copy()
    for c in ("game_id", "player_id"):
        oof_df[c] = oof_df[c].astype(str)
    oof_df["stat"] = oof_df["stat"].astype(str)
    om_cols = ["game_id", "player_id", "stat", "pmf_json", "actual_outcome"]
    if "role_bucket" in oof_df.columns:
        om_cols.append("role_bucket")
    om = oof_df[om_cols].drop_duplicates(["game_id", "player_id", "stat"])

    pair = _pair_decision_quotes(pd.read_parquet(quotes))
    df = pair.merge(om, on=["game_id", "player_id", "stat"], how="inner")

    # Exact per-book no-vig market P(over) via Shin (no averaging, no consensus).
    mkt = [shin_no_vig_two_way(o, u)[0] for o, u in zip(df["over_odds"], df["under_odds"])]
    df["market_prob_over_no_vig"] = mkt

    # SINGLE SOURCE model probability at the exact quoted line (evaluated==deployed).
    finals, pushes = [], []
    for _, r in df.iterrows():
        lin = build_probability_lineage(
            final_pmf=json_to_pmf(r["pmf_json"]), line=float(r["line"]),
            prop=str(r["stat"]), role=str(r.get("role_bucket", "all")),
            probability_track="pure_forecast")
        finals.append(np.nan if lin.model_prob_over_final is None else float(lin.model_prob_over_final))
        pushes.append(float(lin.model_prob_push))
    df[FINAL_PROBABILITY_COLUMN] = finals
    df["model_prob_push"] = pushes
    df["actual"] = df["actual_outcome"].astype(float)
    df["prop"] = df["stat"]
    df["candidate"] = "G0_v2_exact"

    # Settlement: drop pushes and binary-ineligible/degenerate rows (never fabricate).
    df["_push"] = np.isclose(df["actual"].to_numpy(float), df["line"].to_numpy(float))
    valid = (df[FINAL_PROBABILITY_COLUMN].notna()
             & (df[FINAL_PROBABILITY_COLUMN] > EPS) & (df[FINAL_PROBABILITY_COLUMN] < 1 - EPS)
             & df["market_prob_over_no_vig"].notna() & ~df["_push"])
    df = df[valid].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df["outcome_over"] = (df["actual"] > df["line"]).astype(int)

    # Rolling-origin split by distinct dates. --test-dates reserves the last N dates as the
    # untouched proof window (guarantees N clusters); otherwise use the selection fraction.
    dates = np.sort(df["game_date"].unique())
    if test_dates and len(dates) > test_dates:
        cut = dates[len(dates) - test_dates]
    else:
        cut = dates[int(len(dates) * selection_frac)] if len(dates) > 2 else dates[-1]
    df["split"] = np.where(df["game_date"] < cut, "selection", "test")

    if "role_bucket" not in df.columns:
        df["role_bucket"] = "all"
    df["role_bucket"] = df["role_bucket"].fillna("all").astype(str)

    # Frozen deterministic one-quote-per-observation policy (PRIMARY comparison).
    policy = json.loads(Path(quote_policy).read_text())
    prio = {b: i for i, b in enumerate(policy["priority"])}
    df["_book_rank"] = df["book"].map(lambda b: prio.get(b, 10_000)).astype(int)
    df = df.sort_values(["game_id", "player_id", "prop", "_book_rank", "book", "snapshot_time"])
    first_idx = df.groupby(["game_id", "player_id", "prop"], as_index=False).head(1).index
    df["is_primary"] = df.index.isin(first_idx)

    cols = ["game_date", "game_id", "player_id", "prop", "book", "candidate", "split",
            "actual", "line", FINAL_PROBABILITY_COLUMN, "model_prob_push",
            "market_prob_over_no_vig", "over_odds", "under_odds", "snapshot_time",
            "outcome_over", "role_bucket", "is_primary"]
    scored = df[cols].reset_index(drop=True)
    scored.to_parquet(outp / "scored_candidates_g0v2.parquet", index=False)

    prim = scored[scored["is_primary"]]
    # Coverage report: PRIMARY deterministic one-quote + per-book + all-books pooled (sensitivity).
    cov = {"quote_policy": policy["version"], "primary_scope": "deterministic_one_quote_per_obs",
           "legacy_primary_book": primary_book, "split_cut_date": str(cut),
           "by_prop_primary_deterministic": {}, "by_prop_book": {}, "by_prop_all_books_pooled": {}}
    for prop in DIRECT_PROPS:
        gp = scored[scored["prop"] == prop]
        pr = prim[prim["prop"] == prop]
        cov["by_prop_primary_deterministic"][prop] = {
            "rows": int(len(pr)), "dates": int(pr["game_date"].nunique()),
            "book_mix": {k: int(v) for k, v in pr["book"].value_counts().items()},
            "status": ("OK" if len(pr) else "NO_EXACT_QUOTES")}
        cov["by_prop_book"][prop] = {
            bk: {"rows": int(len(g)), "dates": int(g["game_date"].nunique())}
            for bk, g in gp.groupby("book")}
        cov["by_prop_all_books_pooled"][prop] = {"rows": int(len(gp)), "dates": int(gp["game_date"].nunique())}
    (outp / "G0_V2_QUOTE_COVERAGE.json").write_text(json.dumps(cov, indent=2) + "\n")

    # G0-v2 metrics per prop: PRIMARY deterministic one-quote (headline) + all-books pooled (sensitivity).
    rows = []
    for prop in DIRECT_PROPS:
        for scope, sub in (("primary_deterministic", prim[prim["prop"] == prop]),
                           ("all_books_pooled_SENSITIVITY", scored[scored["prop"] == prop])):
            if len(sub) == 0:
                rows.append({"prop": prop, "scope": scope, "n_settled": 0,
                             "status": "NO_EXACT_QUOTES", "n_dates": 0})
                continue
            m = _metrics(sub["outcome_over"].to_numpy(), sub[FINAL_PROBABILITY_COLUMN].to_numpy(),
                         sub["market_prob_over_no_vig"].to_numpy())
            m.update({"prop": prop, "scope": scope, "n_dates": int(sub["game_date"].nunique()),
                      "status": "EVALUATED"})
            rows.append(m)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outp / "G0_V2_METRICS.csv", index=False)
    (outp / "G0_V2_METRICS.json").write_text(
        json.dumps({"primary_scope": "primary_deterministic (one quote per game_id+player_id+prop)",
                    "quote_policy": policy["version"],
                    "note": ("model log loss/Brier ABOVE market = model does NOT yet beat the "
                             "exact market on that prop (positive delta is worse). all_books_pooled "
                             "is SENSITIVITY ONLY (duplicates outcomes across books). stl/blk/turnover "
                             "have NO exact quotes."),
                    "records": metrics.replace({np.nan: None}).to_dict("records")}, indent=2) + "\n")

    print(f"[g0v2] scored rows={len(scored)} primary(one-quote) rows={len(prim)} "
          f"books={sorted(scored['book'].unique())} split_cut={cut}")
    show = metrics[metrics["scope"] == "primary_deterministic"][
        ["prop", "n_settled", "n_dates", "model_logloss", "market_logloss", "logloss_delta",
         "model_brier", "market_brier", "brier_delta", "model_ece", "market_ece", "status"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    app()
