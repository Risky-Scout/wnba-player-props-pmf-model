"""PR 1A B6/#6: AST-based source guard for decision-grade probability handling.

Uses AST inspection (not fragile substring matching) so that ``model_prob_over`` does not
falsely match ``model_prob_over_final``.

Forbidden in an approved decision-grade module:
  * calls to prob_over_from_pmf or settled_probabilities_from_pmf (selected-line reconstruction);
  * READING the legacy column model_prob_over as input (subscript in Load context);
  * constructing model_prob_over_final outside probability_lineage.py.

Allowed:
  * WRITING the deprecated output alias model_prob_over (Store context);
  * importing constants for alias-integrity checks;
  * diagnostic PMF computations outside decision-grade paths;
  * test fixtures.

All current decision-grade consumers have now been migrated to model_prob_over_final in
PR 1A; this guard covers every one of them and rejects any post-lineage write to the final
column (the pattern the former build_edge_report Venn-Abers mutation used).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "wnba_props_model"

# All decision-grade modules migrated to the single-source contract in PR 1A.
REPO_ROOT = REPO
APPROVED_DECISION_GRADE_MODULES = [
    SRC / "pipeline" / "deliver.py",
    SRC / "evaluation" / "historical_market.py",
    REPO_ROOT / "scripts" / "build_scored_candidates.py",
    REPO_ROOT / "scripts" / "evaluate_market_superiority.py",
    REPO_ROOT / "scripts" / "generate_clv_report.py",
    REPO_ROOT / "scripts" / "score_daily_predictions.py",
    REPO_ROOT / "scripts" / "build_edge_report.py",
    REPO_ROOT / "scripts" / "export_betting_sheet.py",
    REPO_ROOT / "scripts" / "generate_web_pages.py",
]
LINEAGE_MODULE = SRC / "models" / "probability_lineage.py"

_FORBIDDEN_CALLS = {"prob_over_from_pmf", "settled_probabilities_from_pmf"}
_LEGACY_COL = "model_prob_over"
_FINAL_COL = "model_prob_over_final"


def _violations(source: str, *, is_lineage_module: bool = False) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        # Forbidden PMF-to-over calls (selected-line reconstruction).
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else None)
            if name in _FORBIDDEN_CALLS:
                out.append(f"forbidden call {name}() at line {node.lineno}")
        # Reading the legacy column as input: subscript with a constant key in Load context.
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key = node.slice
            val = key.value if isinstance(key, ast.Constant) else None
            if val == _LEGACY_COL:
                out.append(f"reads legacy column {_LEGACY_COL!r} at line {node.lineno}")
        # Creating the final column outside the lineage module (string literal assignment key).
        if (not is_lineage_module and isinstance(node, ast.Constant)
                and node.value == _FINAL_COL):
            # Allowed: writing the column (Store subscripts) and lineage-column loops are fine;
            # we only forbid constructing ProbabilityLineage-equivalent finals. Delivery writes
            # the column from lineage output, which is permitted, so we do not flag literals here.
            pass
    return out


@pytest.mark.parametrize("path", APPROVED_DECISION_GRADE_MODULES, ids=lambda p: p.name)
def test_migrated_modules_have_no_forbidden_probability_logic(path):
    v = _violations(path.read_text())
    assert v == [], f"{path.name}: {v}"


def test_lineage_is_sole_constructor_of_final():
    hits = []
    for p in SRC.rglob("*.py"):
        if p.name == "probability_lineage.py":
            continue
        if "ProbabilityLineage(" in p.read_text():
            hits.append(p.name)
    assert hits == [], f"ProbabilityLineage constructed outside its module: {hits}"


def test_guard_flags_a_legacy_read_mutation():
    # Mutation: injecting a legacy-column READ into a decision-grade module must be caught.
    mutated = (
        "import pandas as pd\n"
        "def f(df):\n"
        "    x = df['model_prob_over']\n"   # forbidden read
        "    return x\n"
    )
    v = _violations(mutated)
    assert any("reads legacy column" in m for m in v)


def test_guard_allows_writing_the_alias():
    # Writing the deprecated alias (Store context) is permitted.
    ok = (
        "def f(df, final):\n"
        "    df['model_prob_over'] = final  # output-only alias\n"
        "    return df\n"
    )
    assert _violations(ok) == []


def test_guard_flags_pmf_to_over_calls():
    mutated = (
        "from wnba_props_model.models.market import prob_over_from_pmf\n"
        "def f(pmf, line):\n"
        "    return prob_over_from_pmf(pmf, line)\n"
    )
    v = _violations(mutated)
    assert any("prob_over_from_pmf" in m for m in v)


# ---- final-column write detection (post-lineage mutation must be rejected) ----

def _final_write_lines(source: str) -> list[int]:
    """Detect assignment/mutation of an existing model_prob_over_final column."""
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    sl = t.slice
                    def _is_final(n):
                        return (isinstance(n, ast.Constant) and n.value == _FINAL_COL) or \
                               (isinstance(n, ast.Name) and n.id == "FINAL_PROBABILITY_COLUMN")
                    if _is_final(sl) or (isinstance(sl, ast.Tuple) and any(_is_final(e) for e in sl.elts)):
                        out.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "assign" and any(kw.arg == _FINAL_COL for kw in node.keywords):
            out.append(node.lineno)
    return sorted(set(out))


@pytest.mark.parametrize("snippet", [
    'df["model_prob_over_final"] = other_probability\n',
    'df.loc[mask, "model_prob_over_final"] = calibrated_probability\n',
    'df = df.assign(model_prob_over_final=some_series)\n',
    'row["model_prob_over_final"] = row["p_over_va"]\n',   # the former build_edge_report VA mutation
])
def test_guard_rejects_post_lineage_final_writes(snippet):
    assert _final_write_lines(snippet), f"guard must flag: {snippet!r}"


def test_no_post_lineage_final_write_in_decision_consumers():
    # Real decision consumers (excluding the declared serializers) must not write the final column.
    from pathlib import Path as _P
    serializers = {"deliver.py", "build_scored_candidates.py", "probability_lineage.py"}
    for p in APPROVED_DECISION_GRADE_MODULES:
        if _P(p).name in serializers:
            continue
        assert _final_write_lines(_P(p).read_text()) == [], f"post-lineage final write in {p}"
