#!/usr/bin/env python3
"""Assemble the Opportunity V2 ablation/metrics report from measured exact-quote metrics.

Only A0 (P0 baseline) and A7 (OPP_V2_RAW) are computable on the current repository data; A1-A6
require forward availability/lineup snapshots and a tracking source (see DATA_AVAILABILITY_AUDIT.json)
and are declared-but-not-runnable historically.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

SRC = "artifacts/opportunity_v2/OPP_V2_EXACT_QUOTE_METRICS.json"


def main() -> None:
    m = json.load(open(SRC))["results"]
    rows = []
    for prop, r in m.items():
        rows.append({
            "prop": prop, "n": r["n"], "dates": r["game_dates"],
            "A0_P0_log_loss": round(r["p0_baseline"]["log_loss"], 5),
            "A7_OPP_V2_log_loss": round(r["opp_v2"]["log_loss"], 5),
            "market_log_loss": round(r["market"]["log_loss"], 5),
            "delta_LL_vs_P0": round(r["delta_vs_p0"]["log_loss"], 5),
            "delta_LL_vs_market": round(r["delta_vs_market"]["log_loss"], 5),
            "delta_Brier_vs_market": round(r["delta_vs_market"]["brier"], 5),
            "A7_AUC": round(r["opp_v2"]["auc"], 4),
            "market_AUC": round(r["market"]["auc"], 4),
            "P0_AUC": round(r["p0_baseline"]["auc"], 4),
            "ci95_dLL_vs_market": r["delta_vs_market"]["ci95_delta_log_loss"],
            "holm_p_vs_market": (round(r["holm_adjusted_p_vs_market"], 4)
                                 if r.get("holm_adjusted_p_vs_market") is not None else None),
            "beats_P0": r["beats_p0"],
            "market_superiority_pass": r["market_superiority_pass"],
        })
    Path("artifacts/opportunity_v2").mkdir(parents=True, exist_ok=True)
    with open("artifacts/opportunity_v2/OPP_V2_ABLATION_REPORT.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for x in rows:
            w.writerow(x)
    report = {
        "candidates_computable": ["A0=P0", "A7=OPP_V2_RAW"],
        "candidates_declared_not_runnable_historically": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "note": "A1-A6 require forward availability/lineup snapshots and a tracking source "
                "(see DATA_AVAILABILITY_AUDIT.json). They are defined but not historically runnable.",
        "rows": rows,
    }
    json.dump(report, open("artifacts/opportunity_v2/OPP_V2_ABLATION_REPORT.json", "w"),
              indent=2, default=str)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
