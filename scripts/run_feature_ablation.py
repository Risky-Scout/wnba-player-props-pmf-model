#!/usr/bin/env python3
"""Driver for the leakage-safe, nested-CV feature-selection / ablation study.

Runs the study per prop and writes ``artifacts/feature_ablation/FEATURE_ABLATION_<PROP>.json``.
Deterministic (fixed seed) and resumable (skips a prop whose artifact already
exists unless ``--force``). A separate ``--summarize`` pass reads the per-prop
artifacts, applies Holm correction across the market-evaluable props to the
selected-vs-market p-values, injects the adjusted p-values back into each
artifact, and writes ``FEATURE_ABLATION_SUMMARY.json``.

Examples::

  # one prop (parallelizable: launch several of these)
  PYTHONPATH=$(pwd)/src python3 scripts/run_feature_ablation.py --props pts
  # all props then summary
  PYTHONPATH=$(pwd)/src python3 scripts/run_feature_ablation.py --props all
  PYTHONPATH=$(pwd)/src python3 scripts/run_feature_ablation.py --summarize
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # benign sklearn/numpy notices; results unaffected

from wnba_props_model.ablation import (
    ALL_PROPS,
    MARKET_PROPS,
    AblationConfig,
    load_inputs,
    run_prop,
)
from wnba_props_model.ablation import metrics as M

OUT_DIR = Path("artifacts/feature_ablation")
HOLM_ALPHA = 0.05


def _artifact_path(out_dir: Path, prop: str) -> Path:
    return out_dir / f"FEATURE_ABLATION_{prop.upper()}.json"


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run(props, out_dir: Path, force: bool, cfg: AblationConfig):
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(cfg)
    for prop in props:
        path = _artifact_path(out_dir, prop)
        if path.exists() and not force:
            print(f"[skip] {prop}: artifact exists ({path})")
            continue
        t0 = time.time()
        print(f"[run ] {prop} ...", flush=True)
        res = run_prop(prop, cfg, inputs)
        json.dump(res, open(path, "w"), indent=2, default=_default)
        dt = time.time() - t0
        if res["kind"] == "binary" and "vs_market" in res:
            v = res["vs_market"]["selected"]
            print(f"[done] {prop} n={res.get('oof_n')} dates={res.get('oof_dates')} "
                  f"sel_dLL={v['delta_log_loss']:.4f} sel_dAUC={v['delta_auc']:.4f} "
                  f"p_ll={v['p_ll_raw']:.4f} closes_gap={res.get('closes_gap_fraction')} "
                  f"({dt:.0f}s) -> {path}")
        else:
            m = res["metrics"]
            print(f"[done] {prop} n={res.get('oof_n')} dates={res.get('oof_dates')} "
                  f"P0_dev={m['P0_naive']['poisson_deviance']:.4f} "
                  f"sel_dev={m['selected']['poisson_deviance']:.4f} ({dt:.0f}s) -> {path}")


def summarize(out_dir: Path):
    arts = {}
    for prop in ALL_PROPS:
        p = _artifact_path(out_dir, prop)
        if p.exists():
            arts[prop] = json.load(open(p))
    if not arts:
        raise SystemExit("no per-prop artifacts found to summarize")

    # Holm across market props on selected-vs-market p-values
    p_ll = {pr: a["vs_market"]["selected"]["p_ll_raw"]
            for pr, a in arts.items() if pr in MARKET_PROPS and "vs_market" in a}
    p_brier = {pr: a["vs_market"]["selected"]["p_brier_raw"]
               for pr, a in arts.items() if pr in MARKET_PROPS and "vs_market" in a}
    holm_ll = M.holm(p_ll) if p_ll else {}
    holm_brier = M.holm(p_brier) if p_brier else {}

    summary = {"holm_family": "selected_vs_market across market props (pts,reb,ast,fg3m)",
               "holm_alpha": HOLM_ALPHA, "per_prop": {}}
    for prop, a in arts.items():
        row = {"kind": a["kind"], "n_rows": a["n_rows"], "n_dates": a["n_dates"],
               "oof_n": a.get("oof_n"), "oof_dates": a.get("oof_dates"),
               "sufficient_data": a["sufficient_data"],
               "most_valuable_group": a["most_valuable_group_only_one"],
               "selected_feature_set": a["selected_feature_set"],
               "top5_features_permutation": [x["feature"] for x in a["importance_permutation_top"][:5]]}
        if prop in MARKET_PROPS and "vs_market" in a:
            hll = holm_ll.get(prop)
            hbr = holm_brier.get(prop)
            a["vs_market"]["selected"]["holm_adjusted_p_ll"] = hll
            a["vs_market"]["selected"]["holm_adjusted_p_brier"] = hbr
            sel = a["metrics"]["selected"]
            mkt = a["metrics"]["market"]
            beats = bool(a["beats_market_pointwise"] and hll is not None and hll <= HOLM_ALPHA)
            info_gap = bool((not np.isfinite(sel["auc"])) or (sel["auc"] <= 0.52 and not beats))
            if beats:
                verdict = "BEATS_MARKET"
            elif info_gap:
                verdict = "INFORMATION_GAP"
            else:
                verdict = "CLOSES_GAP" if (a.get("closes_gap_fraction") or 0) > 0 else "NO_EDGE"
            row.update({
                "selected_log_loss": sel["log_loss"], "market_log_loss": mkt["log_loss"],
                "selected_auc": sel["auc"], "market_auc": mkt["auc"],
                "selected_ece": sel["ece"],
                "delta_log_loss_vs_market": a["vs_market"]["selected"]["delta_log_loss"],
                "delta_auc_vs_market": a["vs_market"]["selected"]["delta_auc"],
                "holm_adjusted_p_ll": hll, "holm_adjusted_p_brier": hbr,
                "closes_gap_fraction": a.get("closes_gap_fraction"),
                "beats_market": beats, "verdict": verdict,
            })
            # persist injected holm back into per-prop artifact
            a["verdict"] = verdict
            json.dump(a, open(_artifact_path(out_dir, prop), "w"), indent=2, default=_default)
        else:
            m = a["metrics"]
            improved = a.get("selected_vs_p0_deviance_delta", 0.0) < 0
            row.update({
                "market_comparison_possible": False,
                "P0_poisson_deviance": m["P0_naive"]["poisson_deviance"],
                "selected_poisson_deviance": m["selected"]["poisson_deviance"],
                "selected_pmf_log_score": m["selected"]["pmf_log_score"],
                "selected_crps": m["selected"]["crps"],
                "selected_improves_over_P0": bool(improved),
                "verdict": "OUTCOME_ONLY_" + ("FEATURES_HELP" if improved else "NO_GAIN"),
            })
            a["verdict"] = row["verdict"]
            json.dump(a, open(_artifact_path(out_dir, prop), "w"), indent=2, default=_default)
        summary["per_prop"][prop] = row

    out = out_dir / "FEATURE_ABLATION_SUMMARY.json"
    json.dump(summary, open(out, "w"), indent=2, default=_default)
    print(f"wrote {out}")
    for pr, r in summary["per_prop"].items():
        print(json.dumps({"prop": pr, "verdict": r.get("verdict"),
                          "sel_auc": round(r.get("selected_auc", float('nan')), 4) if r.get("selected_auc") is not None else None,
                          "dLL": round(r.get("delta_log_loss_vs_market"), 4) if r.get("delta_log_loss_vs_market") is not None else None,
                          "holm_p_ll": r.get("holm_adjusted_p_ll"),
                          "most_valuable_group": r.get("most_valuable_group")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", nargs="+", default=["all"],
                    help="props to run, or 'all' / 'market' / 'count'")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="build summary + Holm from existing per-prop artifacts")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wide-path", default=None,
                    help="override the wide/box-form feature parquet path "
                         "(e.g. a BDL-regenerated standard-pipeline build)")
    ap.add_argument("--player-box-form-status", default=None,
                    help="verbatim provenance label for the player_box_form group, "
                         "recorded in every artifact")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.summarize:
        summarize(out_dir)
        return

    props = []
    for p in args.props:
        if p == "all":
            props = list(ALL_PROPS)
        elif p == "market":
            props += list(MARKET_PROPS)
        elif p == "count":
            props += [x for x in ALL_PROPS if x not in MARKET_PROPS]
        else:
            props.append(p)
    seen = set()
    props = [p for p in props if not (p in seen or seen.add(p))]

    cfg = AblationConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.wide_path is not None:
        cfg.wide_path = args.wide_path
    if args.player_box_form_status is not None:
        cfg.player_box_form_status = args.player_box_form_status
    run(props, out_dir, args.force, cfg)


if __name__ == "__main__":
    main()
