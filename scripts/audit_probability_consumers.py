#!/usr/bin/env python3
"""Machine-readable source-of-truth audit for the delivered binary probability (PR 1A #8).

AST-based. Enforces FIVE invariants (``--check`` fails unless all hold):

    creators_of_model_prob_over_final = 1
    decision_consumers_reading_model_prob_over = 0
    decision_consumers_reconstructing_selected_line_from_pmf = 0
    post_lineage_writes_to_model_prob_over_final = 0
    unclassified_probability_modules = 0

Classification completeness: every module under src/ or scripts/ containing a probability
trigger symbol MUST appear in the classification registry below (one of the allowed classes),
so a new consumer cannot escape the audit by living outside a hardcoded allowlist.

Final-write detection: only the sole lineage creator (ProbabilityLineage construction) and
the declared lineage serializers (deliver.py, build_scored_candidates.py) may write
model_prob_over_final, and only when the module actually calls build_probability_lineage /
constructs ProbabilityLineage. Every other write (e.g. a post-lineage Venn-Abers assignment)
is a forbidden post-lineage mutation.

Regenerates artifacts/probability_contract/PR1A_CONSUMER_AUDIT.json and
docs/PR1A_PROBABILITY_CALLSITE_AUDIT.md from the JSON.
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

_TRIGGERS = ("model_prob_over", "model_prob_over_final", "model_prob_over_binary_calibrated",
             "prob_over_from_pmf", "settled_probabilities_from_pmf",
             "ProbabilityLineage", "build_probability_lineage")

LINEAGE_MODULE = "src/wnba_props_model/models/probability_lineage.py"
ALLOWED_FINAL_WRITERS = {
    LINEAGE_MODULE,
    "src/wnba_props_model/pipeline/deliver.py",
    "scripts/build_scored_candidates.py",
}

# Every trigger-bearing module must be classified here (completeness gate).
CLASSIFICATION: dict[str, str] = {
    LINEAGE_MODULE: "sole_lineage_creator",
    "src/wnba_props_model/pipeline/deliver.py": "lineage_output_serializer",
    "scripts/build_scored_candidates.py": "lineage_output_serializer",
    # decision consumers (scanned for legacy reads / selected-line reconstruction)
    "scripts/evaluate_market_superiority.py": "final_probability_decision_consumer",
    "scripts/generate_clv_report.py": "final_probability_decision_consumer",
    "scripts/score_daily_predictions.py": "final_probability_decision_consumer",
    "scripts/build_edge_report.py": "final_probability_decision_consumer",
    "scripts/export_betting_sheet.py": "final_probability_decision_consumer",
    "scripts/generate_web_pages.py": "final_probability_decision_consumer",
    "src/wnba_props_model/evaluation/historical_market.py": "final_probability_decision_consumer",
    "scripts/backtest_strategy.py": "final_probability_decision_consumer",
    "scripts/score_outcomes.py": "final_probability_decision_consumer",
    "scripts/track_clv.py": "final_probability_decision_consumer",
    "scripts/verify_gates.py": "final_probability_decision_consumer",
    # permitted PMF diagnostics / probability infrastructure / display renderers
    "src/wnba_props_model/evaluation/diagnostics.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/evaluation/scoring.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/models/pmf_grid.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/models/pmf_utils.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/visualization/pmf_plots.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/models/market.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/models/probability_contract.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/models/binary_probability_calibration.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/pipeline/calibrate.py": "permitted_pmf_diagnostic",
    "src/wnba_props_model/pipeline/recommendation.py": "permitted_pmf_diagnostic",
    "scripts/verify_combo_roundtrip.py": "permitted_pmf_diagnostic",
    "scripts/build_market_superiority_input.py": "permitted_pmf_diagnostic",
    "scripts/fit_binary_prob_calibrators.py": "permitted_pmf_diagnostic",
    "scripts/predict_today.py": "permitted_pmf_diagnostic",
    "scripts/export_html_report.py": "permitted_pmf_diagnostic",
    "scripts/build_historical_review.py": "permitted_pmf_diagnostic",
    # the audit itself references the symbols as detection strings
    "scripts/audit_probability_consumers.py": "audit_tool",
}
_SCANNED_CLASSES = {"final_probability_decision_consumer", "lineage_output_serializer"}
_RECON_CALLS = {"prob_over_from_pmf", "settled_probabilities_from_pmf"}
_LEGACY = "model_prob_over"
_FINAL = "model_prob_over_final"
_FINAL_CONST = "FINAL_PROBABILITY_COLUMN"


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def _trigger_files() -> list[str]:
    out = []
    for p in sorted(list(SRC.rglob("*.py")) + list(SCRIPTS.glob("*.py"))):
        txt = p.read_text()
        if any(t in txt for t in _TRIGGERS):
            out.append(_rel(p))
    return out


def _legacy_reads(tree: ast.AST) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            k = node.slice
            if isinstance(k, ast.Constant) and k.value == _LEGACY:
                out.append(node.lineno)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == _LEGACY):
            out.append(node.lineno)
    return out


def _recon_calls(tree: ast.AST) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if name in _RECON_CALLS:
                out.append(node.lineno)
    return out


def _key_is_final(node) -> bool:
    return (isinstance(node, ast.Constant) and node.value == _FINAL) or \
           (isinstance(node, ast.Name) and node.id == _FINAL_CONST)


def _final_writes(tree: ast.AST) -> list[int]:
    """Detect ASSIGNMENTS / MUTATIONS to an existing DataFrame's model_prob_over_final
    column. Bare dict literals (new-DataFrame construction for display output or synthetic
    test data) are NOT mutations of the lineage-created decision value and are not flagged;
    the dangerous pattern is overwriting an existing final column (e.g. a post-lineage
    Venn-Abers assignment)."""
    out = []
    for node in ast.walk(tree):
        # df["..."] = ... / df.loc[..., "..."] = ... / df.at[..., "..."] = ...
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    sl = t.slice
                    if _key_is_final(sl):
                        out.append(node.lineno)
                    elif isinstance(sl, ast.Tuple) and any(_key_is_final(e) for e in sl.elts):
                        out.append(node.lineno)  # .loc[mask, "model_prob_over_final"] = ...
        # .assign(model_prob_over_final=...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "assign" \
                and any(kw.arg == _FINAL for kw in node.keywords):
            out.append(node.lineno)
        # .rename(columns={... : "model_prob_over_final"}) creating the final column
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "rename":
            for kw in node.keywords:
                if kw.arg == "columns" and isinstance(kw.value, ast.Dict):
                    if any(isinstance(v, ast.Constant) and v.value == _FINAL
                           for v in kw.value.values):
                        out.append(node.lineno)
    return sorted(set(out))


def _has_lineage_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if name in ("build_probability_lineage", "ProbabilityLineage"):
                return True
    return False


def _constructs_lineage(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "ProbabilityLineage":
            return True
    return False


def audit() -> dict:
    triggers = _trigger_files()
    unclassified = [f for f in triggers if f not in CLASSIFICATION]

    creators, forbidden_reads, forbidden_recon, post_writes = [], [], [], []
    per_module = []
    for rel in triggers:
        cls = CLASSIFICATION.get(rel, "UNCLASSIFIED")
        tree = ast.parse((REPO / rel).read_text())
        if rel == LINEAGE_MODULE and _constructs_lineage(tree):
            creators.append(rel)
        reads = _legacy_reads(tree) if cls in _SCANNED_CLASSES else []
        recon = _recon_calls(tree) if cls in _SCANNED_CLASSES else []
        writes = _final_writes(tree)
        # Writes are allowed only in the creator + declared serializers that actually
        # derive the value from a lineage call/construction.
        if writes and not (rel in ALLOWED_FINAL_WRITERS and (_has_lineage_call(tree)
                                                              or rel == LINEAGE_MODULE)):
            post_writes.append({"path": rel, "lines": writes, "class": cls})
        if reads:
            forbidden_reads.append({"path": rel, "lines": reads})
        if recon:
            forbidden_recon.append({"path": rel, "lines": recon})
        per_module.append({"path": rel, "class": cls, "legacy_reads": reads,
                           "pmf_reconstructions": recon, "final_writes": writes})

    return {
        "creators_of_model_prob_over_final": len(creators),
        "decision_consumers_reading_model_prob_over": len(forbidden_reads),
        "decision_consumers_reconstructing_selected_line_from_pmf": len(forbidden_recon),
        "post_lineage_writes_to_model_prob_over_final": len(post_writes),
        "unclassified_probability_modules": len(unclassified),
        "creator_modules": creators,
        "forbidden_legacy_reads": forbidden_reads,
        "forbidden_selected_line_reconstruction": forbidden_recon,
        "forbidden_post_lineage_writes": post_writes,
        "unclassified_modules": unclassified,
        "allowed_final_writers": sorted(ALLOWED_FINAL_WRITERS),
        "classification": {m["path"]: m["class"] for m in per_module},
    }


def write_outputs(result: dict) -> None:
    jp = REPO / "artifacts" / "probability_contract" / "PR1A_CONSUMER_AUDIT.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(result, indent=2) + "\n")
    md = REPO / "docs" / "PR1A_PROBABILITY_CALLSITE_AUDIT.md"
    L = [
        "# PR 1A - Probability call-site audit (generated)",
        "",
        "Generated by `scripts/audit_probability_consumers.py` (AST-based). Do not edit by hand.",
        "",
        f"- creators_of_model_prob_over_final: **{result['creators_of_model_prob_over_final']}** (target 1)",
        f"- decision_consumers_reading_model_prob_over: **{result['decision_consumers_reading_model_prob_over']}** (target 0)",
        f"- decision_consumers_reconstructing_selected_line_from_pmf: **{result['decision_consumers_reconstructing_selected_line_from_pmf']}** (target 0)",
        f"- post_lineage_writes_to_model_prob_over_final: **{result['post_lineage_writes_to_model_prob_over_final']}** (target 0)",
        f"- unclassified_probability_modules: **{result['unclassified_probability_modules']}** (target 0)",
        "",
        "## Allowed final-column writers",
    ]
    L += [f"- {w}" for w in result["allowed_final_writers"]]
    L += ["", "## Module classification", "", "| Module | Class |", "|---|---|"]
    for path, cls in sorted(result["classification"].items()):
        L.append(f"| `{path}` | {cls} |")
    for title, key in [("Forbidden legacy reads", "forbidden_legacy_reads"),
                       ("Forbidden selected-line reconstruction", "forbidden_selected_line_reconstruction"),
                       ("Forbidden post-lineage final writes", "forbidden_post_lineage_writes"),
                       ("Unclassified modules", "unclassified_modules")]:
        L += ["", f"## {title}"]
        vals = result[key]
        L += ([f"- {v}" for v in vals] if vals else ["- (none)"])
    md.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Exit 1 unless counts are 1/0/0/0/0.")
    args = ap.parse_args()
    r = audit()
    write_outputs(r)
    counts = (r["creators_of_model_prob_over_final"],
              r["decision_consumers_reading_model_prob_over"],
              r["decision_consumers_reconstructing_selected_line_from_pmf"],
              r["post_lineage_writes_to_model_prob_over_final"],
              r["unclassified_probability_modules"])
    print(f"[audit] creators={counts[0]} legacy_reads={counts[1]} recon={counts[2]} "
          f"post_lineage_writes={counts[3]} unclassified={counts[4]}")
    if args.check and counts != (1, 0, 0, 0, 0):
        print("[audit] FAIL: counts are not 1/0/0/0/0", file=sys.stderr)
        for e in r["forbidden_legacy_reads"]:
            print(f"  legacy read: {e['path']} {e['lines']}", file=sys.stderr)
        for e in r["forbidden_selected_line_reconstruction"]:
            print(f"  reconstruction: {e['path']} {e['lines']}", file=sys.stderr)
        for e in r["forbidden_post_lineage_writes"]:
            print(f"  post-lineage write: {e['path']} {e['lines']} ({e['class']})", file=sys.stderr)
        for m in r["unclassified_modules"]:
            print(f"  unclassified: {m}", file=sys.stderr)
        return 1
    print("[audit] PASS 1/0/0/0/0" if args.check else "[audit] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
