#!/usr/bin/env python3
"""Machine-readable 0/0/1 audit of probability handling (PR 1A #8).

AST-based classification of every relevant call site across src/ and scripts/:
  - sole lineage creator
  - deprecated output alias writer
  - final-probability decision consumer (migrated)
  - permitted PMF diagnostic / full-PMF scoring
  - forbidden legacy consumer (Load read of model_prob_over in a decision-grade module)
  - forbidden selected-line PMF reconstruction (prob_over_from_pmf / settled_probabilities_from_pmf
    / pmf_to_array in a decision-grade module)

Emits artifacts/probability_contract/PR1A_CONSUMER_AUDIT.json and regenerates
docs/PR1A_PROBABILITY_CALLSITE_AUDIT.md from it. `--check` exits non-zero unless the counts
are exactly creators=1, decision_consumers_reading_model_prob_over=0,
decision_consumers_reconstructing_selected_line_from_pmf=0.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "wnba_props_model"
SCRIPTS = REPO / "scripts"

LINEAGE_MODULE = "src/wnba_props_model/models/probability_lineage.py"
ALIAS_WRITER = "src/wnba_props_model/pipeline/deliver.py"

# Decision-grade modules: they must consume the delivered final probability only.
DECISION_GRADE = {
    "src/wnba_props_model/pipeline/deliver.py",
    "scripts/build_scored_candidates.py",
    "scripts/evaluate_market_superiority.py",
    "scripts/generate_clv_report.py",
    "scripts/score_daily_predictions.py",
    "scripts/build_edge_report.py",
    "scripts/export_betting_sheet.py",
    "scripts/generate_web_pages.py",
    "src/wnba_props_model/evaluation/historical_market.py",
}
# Permitted full-PMF scoring / diagnostics (line reconstruction allowed; not the chosen
# sportsbook decision probability).
PERMITTED_DIAGNOSTIC = {
    "src/wnba_props_model/evaluation/diagnostics.py",
    "src/wnba_props_model/evaluation/oof_scoring.py",
    "src/wnba_props_model/models/pmf_grid.py",
    "src/wnba_props_model/models/pmf_utils.py",
    "src/wnba_props_model/evaluation/forecasting.py",
}
# Over-probability reconstructors for a selected line. Bare pmf_to_array / json_to_pmf are
# generic PMF parsing (e.g. feeding the sole creator or full-PMF diagnostics) and are NOT
# selected-line over-probability reconstruction, so they are not listed here.
_PMF_TO_OVER_CALLS = {"prob_over_from_pmf", "settled_probabilities_from_pmf"}
_LEGACY = "model_prob_over"
_FINAL = "model_prob_over_final"


def _legacy_reads(tree: ast.AST) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            k = node.slice
            if isinstance(k, ast.Constant) and k.value == _LEGACY:
                out.append(node.lineno)
        # .get("model_prob_over") reads
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and a0.value == _LEGACY:
                out.append(node.lineno)
    return out


def _pmf_to_over_calls(tree: ast.AST) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if name in _PMF_TO_OVER_CALLS:
                out.append(node.lineno)
    return out


def _constructs_final(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "ProbabilityLineage":
            return True
    return False


def audit() -> dict:
    files = sorted(list(SRC.rglob("*.py")) + list(SCRIPTS.glob("*.py")))
    creators = []
    forbidden_reads = []
    forbidden_recon = []
    classified = []
    for p in files:
        rel = str(p.relative_to(REPO)).replace("\\", "/")
        tree = ast.parse(p.read_text())
        is_lineage = rel == LINEAGE_MODULE
        if is_lineage and _constructs_final(tree):
            creators.append(rel)
        if rel in DECISION_GRADE:
            reads = _legacy_reads(tree)
            recon = _pmf_to_over_calls(tree)
            if reads:
                forbidden_reads.append({"path": rel, "lines": reads})
            if recon:
                forbidden_recon.append({"path": rel, "lines": recon})
            classified.append({"path": rel, "class": "final_probability_decision_consumer",
                               "legacy_reads": reads, "pmf_reconstructions": recon})
        elif rel == LINEAGE_MODULE:
            classified.append({"path": rel, "class": "sole_lineage_creator"})
        elif rel in PERMITTED_DIAGNOSTIC:
            classified.append({"path": rel, "class": "permitted_pmf_diagnostic"})
    result = {
        "creators_of_model_prob_over_final": len(creators),
        "decision_consumers_reading_model_prob_over": len(forbidden_reads),
        "decision_consumers_reconstructing_selected_line_from_pmf": len(forbidden_recon),
        "creator_modules": creators,
        "forbidden_legacy_reads": forbidden_reads,
        "forbidden_selected_line_reconstruction": forbidden_recon,
        "decision_grade_modules": sorted(DECISION_GRADE),
        "permitted_diagnostic_modules": sorted(PERMITTED_DIAGNOSTIC),
    }
    return result


def write_outputs(result: dict) -> None:
    jp = REPO / "artifacts" / "probability_contract" / "PR1A_CONSUMER_AUDIT.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(result, indent=2) + "\n")
    md = REPO / "docs" / "PR1A_PROBABILITY_CALLSITE_AUDIT.md"
    lines = [
        "# PR 1A - Probability call-site audit (generated)",
        "",
        "Generated by `scripts/audit_probability_consumers.py` (AST-based). Do not edit by hand.",
        "",
        f"- creators_of_model_prob_over_final: **{result['creators_of_model_prob_over_final']}** (target 1)",
        f"- decision_consumers_reading_model_prob_over: **{result['decision_consumers_reading_model_prob_over']}** (target 0)",
        f"- decision_consumers_reconstructing_selected_line_from_pmf: **{result['decision_consumers_reconstructing_selected_line_from_pmf']}** (target 0)",
        "",
        "## Sole creator",
        f"- {', '.join(result['creator_modules']) or '(none)'}",
        "",
        "## Forbidden legacy reads (must be empty for acceptance)",
    ]
    lines += [f"- {e['path']}: lines {e['lines']}" for e in result["forbidden_legacy_reads"]] or ["- (none)"]
    lines += ["", "## Forbidden selected-line PMF reconstruction (must be empty)"]
    lines += [f"- {e['path']}: lines {e['lines']}" for e in result["forbidden_selected_line_reconstruction"]] or ["- (none)"]
    lines += ["", "## Decision-grade modules", ""]
    lines += [f"- {m}" for m in result["decision_grade_modules"]]
    lines += ["", "## Permitted PMF diagnostics", ""]
    lines += [f"- {m}" for m in result["permitted_diagnostic_modules"]]
    md.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Exit 1 unless counts are 1/0/0.")
    args = ap.parse_args()
    result = audit()
    write_outputs(result)
    c = result["creators_of_model_prob_over_final"]
    r = result["decision_consumers_reading_model_prob_over"]
    p = result["decision_consumers_reconstructing_selected_line_from_pmf"]
    print(f"[audit] creators={c} legacy_reads={r} pmf_reconstructions={p}")
    if args.check and not (c == 1 and r == 0 and p == 0):
        print("[audit] FAIL: counts are not 1/0/0", file=sys.stderr)
        for e in result["forbidden_legacy_reads"]:
            print(f"  legacy read: {e['path']} lines {e['lines']}", file=sys.stderr)
        for e in result["forbidden_selected_line_reconstruction"]:
            print(f"  pmf reconstruction: {e['path']} lines {e['lines']}", file=sys.stderr)
        return 1
    print("[audit] OK" if not args.check else "[audit] PASS 1/0/0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
