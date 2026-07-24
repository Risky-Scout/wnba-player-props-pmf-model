#!/usr/bin/env python3
"""Evaluate WNBA prop feature candidates against closing no-vig market probabilities.

The script intentionally separates:
  1) selection: choose one frozen candidate per prop on a selection period;
  2) proof: evaluate only those frozen candidates on an untouched forward test period.

Binary outcome convention:
  over = 1 when actual > line
  over = 0 when actual < line
  pushes are excluded from binary log-loss/Brier/AUC calculations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6
DEFAULT_PROPS = [
    "pts", "reb", "ast", "fg3m", "stl", "blk", "turnover",
    "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast",
]


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV or Parquet.")


def _clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        return float(np.mean(x))
    total = float(np.sum(w))
    if total <= 0:
        return float("nan")
    return float(np.sum(x * w) / total)


def _metrics(
    y: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p_model = _clip_prob(p_model)
    p_market = _clip_prob(p_market)
    ll_model_rows = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    ll_market_rows = -(y * np.log(p_market) + (1 - y) * np.log(1 - p_market))
    br_model_rows = (p_model - y) ** 2
    br_market_rows = (p_market - y) ** 2
    result = {
        "model_logloss": _weighted_mean(ll_model_rows, sample_weight),
        "market_logloss": _weighted_mean(ll_market_rows, sample_weight),
        "model_brier": _weighted_mean(br_model_rows, sample_weight),
        "market_brier": _weighted_mean(br_market_rows, sample_weight),
    }
    result["logloss_delta"] = result["model_logloss"] - result["market_logloss"]
    result["brier_delta"] = result["model_brier"] - result["market_brier"]

    try:
        result["model_auc"] = float(
            roc_auc_score(y, p_model, sample_weight=sample_weight)
        )
        result["market_auc"] = float(
            roc_auc_score(y, p_market, sample_weight=sample_weight)
        )
        result["auc_delta"] = result["model_auc"] - result["market_auc"]
    except ValueError:
        result["model_auc"] = float("nan")
        result["market_auc"] = float("nan")
        result["auc_delta"] = float("nan")
    return result


def _prepare(
    df: pd.DataFrame,
    *,
    prop_col: str,
    candidate_col: str,
    split_col: str,
    date_col: str,
    actual_col: str,
    line_col: str,
    model_prob_col: str,
    market_prob_col: str,
) -> pd.DataFrame:
    required = [
        prop_col, date_col, actual_col, line_col, model_prob_col, market_prob_col
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    if candidate_col not in out.columns:
        out[candidate_col] = "candidate"
    if split_col not in out.columns:
        out[split_col] = "test"

    for c in [actual_col, line_col, model_prob_col, market_prob_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["_push"] = np.isclose(
        out[actual_col].to_numpy(float),
        out[line_col].to_numpy(float),
        equal_nan=False,
    )
    out["_outcome_over"] = (out[actual_col] > out[line_col]).astype(int)
    valid = (
        out[prop_col].notna()
        & out[candidate_col].notna()
        & out[date_col].notna()
        & out[actual_col].notna()
        & out[line_col].notna()
        & out[model_prob_col].between(0, 1, inclusive="both")
        & out[market_prob_col].between(0, 1, inclusive="both")
        & ~out["_push"]
    )
    return out.loc[valid].reset_index(drop=True)


def _point_table(
    df: pd.DataFrame,
    *,
    prop_col: str,
    candidate_col: str,
    date_col: str,
    model_prob_col: str,
    market_prob_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (prop, candidate), g in df.groupby([prop_col, candidate_col], sort=True):
        m = _metrics(
            g["_outcome_over"].to_numpy(),
            g[model_prob_col].to_numpy(),
            g[market_prob_col].to_numpy(),
        )
        rows.append(
            {
                "prop": prop,
                "candidate": candidate,
                "n_settled": int(len(g)),
                "n_clusters": int(g[date_col].nunique()),
                "date_min": str(g[date_col].min()),
                "date_max": str(g[date_col].max()),
                **m,
            }
        )
    return pd.DataFrame(rows)


def _select_candidates(point: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if point.empty:
        raise ValueError("No rows available for candidate selection.")
    ranked = point.copy()
    ranked["rank_logloss"] = ranked.groupby("prop")["model_logloss"].rank(
        method="min", ascending=True
    )
    ranked["rank_brier"] = ranked.groupby("prop")["model_brier"].rank(
        method="min", ascending=True
    )
    ranked["rank_auc"] = ranked.groupby("prop")["model_auc"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    ranked["mean_metric_rank"] = ranked[
        ["rank_logloss", "rank_brier", "rank_auc"]
    ].mean(axis=1)
    ranked = ranked.sort_values(
        ["prop", "mean_metric_rank", "model_logloss", "model_brier", "model_auc"],
        ascending=[True, True, True, True, False],
    )
    winners = ranked.groupby("prop", as_index=False).first()
    selected = dict(zip(winners["prop"].astype(str), winners["candidate"].astype(str)))
    ranked["selected"] = [
        selected.get(str(prop)) == str(candidate)
        for prop, candidate in zip(ranked["prop"], ranked["candidate"])
    ]
    return ranked, selected


def _bootstrap_deltas(
    g: pd.DataFrame,
    *,
    date_col: str,
    model_prob_col: str,
    market_prob_col: str,
    n_boot: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    y = g["_outcome_over"].to_numpy(int)
    p_model = g[model_prob_col].to_numpy(float)
    p_market = g[market_prob_col].to_numpy(float)
    cluster_codes, cluster_labels = pd.factorize(g[date_col], sort=True)
    n_clusters = len(cluster_labels)
    if n_clusters < 2:
        raise ValueError("At least two date clusters are required for bootstrap inference.")

    rng = np.random.default_rng(seed)
    deltas = {
        "logloss_delta": np.full(n_boot, np.nan),
        "brier_delta": np.full(n_boot, np.nan),
        "auc_delta": np.full(n_boot, np.nan),
    }
    probs = np.full(n_clusters, 1.0 / n_clusters)
    for b in range(n_boot):
        sampled_counts = rng.multinomial(n_clusters, probs)
        weights = sampled_counts[cluster_codes].astype(float)
        m = _metrics(y, p_model, p_market, sample_weight=weights)
        for key in deltas:
            deltas[key][b] = m[key]
    return deltas


def _quantile(a: np.ndarray, q: float) -> float:
    clean = np.asarray(a, dtype=float)
    clean = clean[np.isfinite(clean)]
    return float(np.quantile(clean, q)) if len(clean) else float("nan")


def _one_sided_p(a: np.ndarray, better: str) -> float:
    clean = np.asarray(a, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    if better == "negative":
        bad = int(np.sum(clean >= 0))
    elif better == "positive":
        bad = int(np.sum(clean <= 0))
    else:
        raise ValueError(better)
    return float((bad + 1) / (len(clean) + 1))


def _holm_adjust(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    m = len(valid)
    running = 0.0
    for i, (idx, p) in enumerate(valid.items()):
        adjusted = min(1.0, (m - i) * float(p))
        running = max(running, adjusted)
        out.loc[idx] = running
    return out


def _prove(
    df: pd.DataFrame,
    selected: Mapping[str, str],
    *,
    prop_col: str,
    candidate_col: str,
    date_col: str,
    model_prob_col: str,
    market_prob_col: str,
    n_boot: int,
    seed: int,
    min_rows: int,
    min_clusters: int,
    alpha: float,
    min_logloss_delta: float,
    min_brier_delta: float,
    min_auc_delta: float,
) -> pd.DataFrame:
    filtered_parts = []
    for prop, candidate in selected.items():
        part = df[
            (df[prop_col].astype(str) == str(prop))
            & (df[candidate_col].astype(str) == str(candidate))
        ]
        if len(part):
            filtered_parts.append(part)
    if not filtered_parts:
        raise ValueError("No proof rows matched the selected candidate map.")
    proof_df = pd.concat(filtered_parts, ignore_index=True)

    rows: List[Dict[str, object]] = []
    for i, (prop, g) in enumerate(proof_df.groupby(prop_col, sort=True)):
        candidate = selected[str(prop)]
        point = _metrics(
            g["_outcome_over"].to_numpy(),
            g[model_prob_col].to_numpy(),
            g[market_prob_col].to_numpy(),
        )
        row: Dict[str, object] = {
            "prop": str(prop),
            "candidate": str(candidate),
            "n_settled": int(len(g)),
            "n_clusters": int(g[date_col].nunique()),
            "date_min": str(g[date_col].min()),
            "date_max": str(g[date_col].max()),
            **point,
        }
        if len(g) >= min_rows and g[date_col].nunique() >= 2:
            boot = _bootstrap_deltas(
                g,
                date_col=date_col,
                model_prob_col=model_prob_col,
                market_prob_col=market_prob_col,
                n_boot=n_boot,
                seed=seed + 1009 * i,
            )
            for key, direction in [
                ("logloss_delta", "negative"),
                ("brier_delta", "negative"),
                ("auc_delta", "positive"),
            ]:
                row[f"{key}_ci_low"] = _quantile(boot[key], 0.025)
                row[f"{key}_ci_high"] = _quantile(boot[key], 0.975)
                row[f"{key}_p_one_sided"] = _one_sided_p(boot[key], direction)
        else:
            for key in ["logloss_delta", "brier_delta", "auc_delta"]:
                row[f"{key}_ci_low"] = float("nan")
                row[f"{key}_ci_high"] = float("nan")
                row[f"{key}_p_one_sided"] = float("nan")
        rows.append(row)

    result = pd.DataFrame(rows)
    # Holm across the 14 co-primary PROPER-SCORE claims (n_props x {log loss, Brier}) as ONE
    # family, per W0.1. AUC is adjusted in its own family (strict-gate only, not co-primary).
    proper_stack = pd.concat([
        pd.DataFrame({"prop": result["prop"], "_metric": "logloss",
                      "_p": result["logloss_delta_p_one_sided"]}),
        pd.DataFrame({"prop": result["prop"], "_metric": "brier",
                      "_p": result["brier_delta_p_one_sided"]}),
    ], ignore_index=True)
    proper_stack["_p_holm"] = _holm_adjust(proper_stack["_p"])
    ll_holm = proper_stack[proper_stack["_metric"] == "logloss"].set_index("prop")["_p_holm"]
    br_holm = proper_stack[proper_stack["_metric"] == "brier"].set_index("prop")["_p_holm"]
    result["logloss_delta_p_holm"] = result["prop"].map(ll_holm).to_numpy()
    result["brier_delta_p_holm"] = result["prop"].map(br_holm).to_numpy()
    result["auc_delta_p_holm"] = _holm_adjust(result["auc_delta_p_one_sided"])

    # Enough evidence requires BOTH a row floor AND a date-cluster floor (W0.1).
    enough = (result["n_settled"] >= min_rows) & (result["n_clusters"] >= min_clusters)
    ll_pass = (
        result["logloss_delta_ci_high"] < -abs(min_logloss_delta)
    ) & (result["logloss_delta_p_holm"] <= alpha)
    br_pass = (
        result["brier_delta_ci_high"] < -abs(min_brier_delta)
    ) & (result["brier_delta_p_holm"] <= alpha)
    auc_pass = (
        result["auc_delta_ci_low"] > abs(min_auc_delta)
    ) & (result["auc_delta_p_holm"] <= alpha)
    proper = ll_pass & br_pass
    result["gate_logloss"] = np.where(enough, ll_pass, False)
    result["gate_brier"] = np.where(enough, br_pass, False)
    result["gate_auc"] = np.where(enough, auc_pass, False)
    # Proper-score gate: log loss AND Brier superiority. Strict gate: proper-score AND AUC.
    result["proper_score_market_superiority_gate"] = np.where(
        ~enough, "INSUFFICIENT", np.where(proper, "PASS", "FAIL"))
    result["strict_market_superiority_gate"] = np.where(
        ~enough, "INSUFFICIENT", np.where(proper & auc_pass, "PASS", "FAIL"))
    # Back-compat alias: the historical single gate == the strict gate.
    result["market_superiority_gate"] = result["strict_market_superiority_gate"]
    return result.sort_values("prop").reset_index(drop=True)


def _write_markdown_report(
    result: pd.DataFrame,
    path: Path,
    *,
    alpha: float,
    min_rows: int,
    n_boot: int,
) -> None:
    passed = int((result["market_superiority_gate"] == "PASS").sum())
    total = int(len(result))
    lines = [
        "# Market-superiority proof report",
        "",
        f"- Props passing all three gates: **{passed}/{total}**",
        f"- Settled-row minimum: **{min_rows}**",
        f"- Cluster bootstrap replicates: **{n_boot}**",
        f"- Holm-adjusted one-sided alpha: **{alpha:.3f}**",
        "- Delta signs: log loss/Brier negative is better; AUC positive is better.",
        "- Pushes are excluded from binary metrics.",
        "",
        "| Prop | Candidate | N | Δ Log loss (95% CI) | Δ Brier (95% CI) | Δ AUC (95% CI) | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(v: object) -> str:
        try:
            x = float(v)
            return "NA" if not np.isfinite(x) else f"{x:.5f}"
        except Exception:
            return str(v)
    for _, r in result.iterrows():
        ll = f"{fmt(r['logloss_delta'])} [{fmt(r['logloss_delta_ci_low'])}, {fmt(r['logloss_delta_ci_high'])}]"
        br = f"{fmt(r['brier_delta'])} [{fmt(r['brier_delta_ci_low'])}, {fmt(r['brier_delta_ci_high'])}]"
        auc = f"{fmt(r['auc_delta'])} [{fmt(r['auc_delta_ci_low'])}, {fmt(r['auc_delta_ci_high'])}]"
        lines.append(
            f"| {r['prop']} | {r['candidate']} | {int(r['n_settled'])} | "
            f"{ll} | {br} | {auc} | **{r['market_superiority_gate']}** |"
        )
    lines += [
        "",
        "A PASS is evidence only for the frozen candidate, books, line timestamps, date range, and population represented by the input.",
        "It is not a guarantee of future profitability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_self_test(seed: int = 20260720) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    props = DEFAULT_PROPS
    for prop_i, prop in enumerate(props):
        for split_i, split in enumerate(["selection", "test"]):
            start = pd.Timestamp("2025-05-01") + pd.Timedelta(days=90 * split_i)
            for day in range(40):
                date = start + pd.Timedelta(days=day)
                n = 55
                z = rng.normal(0, 1.25, size=n)
                p_true = 1 / (1 + np.exp(-z))
                y = rng.binomial(1, p_true)
                p_market = 1 / (1 + np.exp(-(0.48 * z + rng.normal(0, 0.60, size=n))))
                p_good = 1 / (1 + np.exp(-(0.98 * z + rng.normal(0, 0.10, size=n))))
                p_bad = 1 / (1 + np.exp(-(0.36 * z + rng.normal(0, 0.80, size=n))))
                for candidate, pred in [
                    ("candidate_v1", p_good),
                    ("candidate_bad", p_bad),
                ]:
                    for j in range(n):
                        rows.append(
                            {
                                "game_date": date.date().isoformat(),
                                "prop": prop,
                                "candidate": candidate,
                                "split": split,
                                "actual": int(y[j]),
                                "line": 0.5,
                                "model_prob_over_final": float(pred[j]),
                                "market_prob_over_no_vig": float(p_market[j]),
                            }
                        )
    return pd.DataFrame(rows)


def _load_selected(path: str) -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selected_candidates" in raw:
        raw = raw["selected_candidates"]
    if not isinstance(raw, dict):
        raise ValueError("Selected-candidate file must be a JSON object.")
    return {str(k): str(v) for k, v in raw.items()}


def _proof_mask(df: pd.DataFrame, date_col: str, manifest_path: str) -> "pd.Series":
    """Boolean mask of rows inside the FROZEN proof window (W0.1). Supports either an
    explicit date list ('proof_dates') or an inclusive range ('proof_date_min/max').
    No automatic test-fraction splitting is permitted in prove mode."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    d = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    if man.get("proof_dates"):
        allowed = {str(x)[:10] for x in man["proof_dates"]}
        return d.isin(allowed)
    lo, hi = man.get("proof_date_min"), man.get("proof_date_max")
    if not (lo and hi):
        raise ValueError("split-manifest must define 'proof_dates' or both "
                         "'proof_date_min' and 'proof_date_max'.")
    return (d >= str(lo)[:10]) & (d <= str(hi)[:10])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="CSV or Parquet with settled prop probabilities.")
    ap.add_argument("--output-dir", default="artifacts/market_feature_proof")
    ap.add_argument("--mode", choices=["select", "audit", "prove"], default="prove")
    ap.add_argument("--selected-candidates", help="JSON map used by prove mode.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prop-col", default="prop")
    ap.add_argument("--candidate-col", default="candidate")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--date-col", default="game_date")
    ap.add_argument("--actual-col", default="actual")
    ap.add_argument("--line-col", default="line")
    ap.add_argument("--model-prob-col", default="model_prob_over_final")
    ap.add_argument("--market-prob-col", default="market_prob_over_no_vig")
    ap.add_argument("--selection-split", default="selection")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--min-clusters", type=int, default=30,
                    help="Minimum distinct game-date clusters (hard floor 30 in prove mode).")
    ap.add_argument("--split-manifest",
                    help="Frozen proof-window manifest (JSON) REQUIRED in prove mode: "
                         "{'proof_date_min','proof_date_max'} or {'proof_dates':[...]}. "
                         "Removes automatic test-fraction splitting.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-logloss-delta", type=float, default=0.0)
    ap.add_argument("--min-brier-delta", type=float, default=0.0)
    ap.add_argument("--min-auc-delta", type=float, default=0.0)
    args = ap.parse_args()

    # PR 1A source-of-truth: proof/selection/audit score the delivered final probability.
    # The legacy column model_prob_over is rejected. W0.1: prove mode REQUIRES
    # model_prob_over_final explicitly (no other column, no reconstruction).
    if not args.self_test and args.model_prob_col == "model_prob_over":
        ap.error("legacy column 'model_prob_over' is forbidden; the evaluator scores "
                 "'model_prob_over_final' (the delivered probability).")
    if args.mode == "prove" and not args.self_test and args.model_prob_col != "model_prob_over_final":
        ap.error("prove mode requires --model-prob-col model_prob_over_final "
                 f"(got {args.model_prob_col!r}).")
    # Hard floor on the date-cluster minimum in prove mode.
    if args.mode == "prove":
        args.min_clusters = max(30, int(args.min_clusters))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        raw = _make_self_test(args.seed)
        raw.to_csv(outdir / "synthetic_scored_props.csv", index=False)
    elif args.input:
        raw = _read_table(args.input)
    else:
        ap.error("--input is required unless --self-test is used.")

    df = _prepare(
        raw,
        prop_col=args.prop_col,
        candidate_col=args.candidate_col,
        split_col=args.split_col,
        date_col=args.date_col,
        actual_col=args.actual_col,
        line_col=args.line_col,
        model_prob_col=args.model_prob_col,
        market_prob_col=args.market_prob_col,
    )

    if args.mode == "audit":
        point = _point_table(
            df,
            prop_col=args.prop_col,
            candidate_col=args.candidate_col,
            date_col=args.date_col,
            model_prob_col=args.model_prob_col,
            market_prob_col=args.market_prob_col,
        )
        point.to_csv(outdir / "exploratory_candidate_metrics.csv", index=False)
        print(point.to_string(index=False))
        return 0

    if args.mode == "select" or args.self_test:
        selection = df[df[args.split_col].astype(str) == str(args.selection_split)]
        point = _point_table(
            selection,
            prop_col=args.prop_col,
            candidate_col=args.candidate_col,
            date_col=args.date_col,
            model_prob_col=args.model_prob_col,
            market_prob_col=args.market_prob_col,
        )
        ranked, selected = _select_candidates(point)
        ranked.to_csv(outdir / "selection_metrics.csv", index=False)
        (outdir / "selected_candidates.json").write_text(
            json.dumps(
                {
                    "selection_split": args.selection_split,
                    "selected_candidates": selected,
                    "warning": "Freeze this map before evaluating the untouched test split.",
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        if args.mode == "select" and not args.self_test:
            print(json.dumps(selected, indent=2))
            return 0
    if args.self_test:
        # Mechanics validation only: use the synthetic selection/test split column.
        test = df[df[args.split_col].astype(str) == str(args.test_split)]
    else:
        # W0.1 prove mode: FROZEN candidate manifest + FROZEN split manifest; no auto-derive,
        # no automatic test-fraction splitting (that lives only in development/audit mode).
        if not args.selected_candidates:
            ap.error("prove mode requires --selected-candidates (a candidate manifest frozen "
                     "on a separate selection period).")
        if not args.split_manifest:
            ap.error("prove mode requires --split-manifest (the frozen untouched proof window). "
                     "Automatic test-fraction splitting is disabled in prove mode.")
        selected = _load_selected(args.selected_candidates)
        mask = _proof_mask(df, args.date_col, args.split_manifest)
        test = df[mask]
    if test.empty:
        raise ValueError("No rows found in the frozen proof window / test split.")
    result = _prove(
        test,
        selected,
        prop_col=args.prop_col,
        candidate_col=args.candidate_col,
        date_col=args.date_col,
        model_prob_col=args.model_prob_col,
        market_prob_col=args.market_prob_col,
        n_boot=args.bootstrap,
        seed=args.seed,
        min_rows=args.min_rows,
        min_clusters=args.min_clusters,
        alpha=args.alpha,
        min_logloss_delta=args.min_logloss_delta,
        min_brier_delta=args.min_brier_delta,
        min_auc_delta=args.min_auc_delta,
    )
    result.to_csv(outdir / "market_superiority_proof.csv", index=False)
    (outdir / "market_superiority_proof.json").write_text(
        json.dumps(
            {
                "mode": "prove",
                "self_test": bool(args.self_test),
                "split_manifest": args.split_manifest,
                "candidate_manifest": args.selected_candidates,
                "bootstrap_replicates": args.bootstrap,
                "alpha": args.alpha,
                "minimum_rows": args.min_rows,
                "minimum_clusters": args.min_clusters,
                "all_props_proper_score_pass": bool(
                    len(result) > 0
                    and (result["proper_score_market_superiority_gate"] == "PASS").all()
                ),
                "all_props_strict_pass": bool(
                    len(result) > 0
                    and (result["strict_market_superiority_gate"] == "PASS").all()
                ),
                "results": result.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(
        result,
        outdir / "MARKET_SUPERIORITY_REPORT.md",
        alpha=args.alpha,
        min_rows=args.min_rows,
        n_boot=args.bootstrap,
    )
    print(result[[
        "prop", "candidate", "n_settled", "n_clusters", "logloss_delta",
        "brier_delta", "auc_delta", "proper_score_market_superiority_gate",
        "strict_market_superiority_gate"
    ]].to_string(index=False))

    if args.self_test:
        all_selected_good = all(v == "candidate_v1" for v in selected.values())
        all_pass = bool((result["market_superiority_gate"] == "PASS").all())
        status = {
            "selection_chose_expected_candidate": all_selected_good,
            "all_synthetic_props_pass": all_pass,
            "note": "This validates evaluator mechanics only; it is not WNBA market evidence.",
        }
        (outdir / "SELF_TEST_STATUS.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        if not all_selected_good or not all_pass:
            raise SystemExit("Self-test failed. Inspect output artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
