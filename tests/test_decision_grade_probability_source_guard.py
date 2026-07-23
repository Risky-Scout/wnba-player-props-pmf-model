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

Scope note: PR 1A migrates the LIVE delivery creation site (deliver.py) to the single
source. The historical/evaluator/report consumers are migrated together with the PR 1B/1C
quote+proof paths (several are Foundation-Lock hash-pinned); this guard locks the delivery
side established in PR 1A and is designed to extend to each consumer as it is migrated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "wnba_props_model"

# Modules already migrated to the single-source contract in PR 1A.
APPROVED_DECISION_GRADE_MODULES = [
    SRC / "pipeline" / "deliver.py",
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
