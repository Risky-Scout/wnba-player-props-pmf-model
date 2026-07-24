"""A7 - explicit settlement of player-prop quote pairs.

Settlement is a SEPARATE job from quote capture. Every settled row gets exactly one explicit
status and the fields needed to score it. Core rules:

  * A player APPEARANCE settles the market regardless of minutes played.
  * DNP is NOT an Under unless the frozen sportsbook rule says so (default: DNP -> VOID_DNP).
  * Unknown/absent settlement rules FAIL CLOSED (UNRESOLVED); we never guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OVER_WIN = "OVER_WIN"
UNDER_WIN = "UNDER_WIN"
PUSH = "PUSH"
VOID_DNP = "VOID_DNP"
VOID_OTHER = "VOID_OTHER"
CANCELED = "CANCELED"
PENDING = "PENDING"
UNRESOLVED = "UNRESOLVED"

SETTLEMENT_FIELDS = [
    "settlement_status", "sportsbook_rule_id", "did_book_void", "binary_score_eligible",
    "binary_target_over", "pmf_score_eligible", "actual_outcome",
]


@dataclass(frozen=True)
class SportsbookSettlementRule:
    rule_id: str
    dnp_is_under: bool = False   # if True, a DNP settles as UNDER_WIN
    void_on_dnp: bool = True     # if True, a DNP voids the market (default player-prop rule)


# Frozen registry of KNOWN sportsbook rules. A book absent here is UNRESOLVED (fail closed).
SPORTSBOOK_SETTLEMENT_RULES: dict[str, SportsbookSettlementRule] = {
    "wnba_player_prop_standard_v1": SportsbookSettlementRule(
        "wnba_player_prop_standard_v1", dnp_is_under=False, void_on_dnp=True),
}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def settle_one(*, rule: "SportsbookSettlementRule | None", line, appeared, actual_outcome,
               canceled: bool = False, book_void: bool = False) -> dict:
    """Return the explicit settlement dict for one quote pair (fail-closed on unknown rules)."""
    out = {"settlement_status": UNRESOLVED, "sportsbook_rule_id": None, "did_book_void": False,
           "binary_score_eligible": False, "binary_target_over": None,
           "pmf_score_eligible": False, "actual_outcome": _num(actual_outcome)}
    if rule is None:
        return out                                            # unknown rule -> fail closed
    out["sportsbook_rule_id"] = rule.rule_id
    ln = _num(line)

    if canceled:
        out["settlement_status"] = CANCELED; out["did_book_void"] = True; return out
    if book_void:
        out["settlement_status"] = VOID_OTHER; out["did_book_void"] = True; return out

    if appeared is False:
        # DNP: only an Under if the frozen rule explicitly says so; else VOID_DNP.
        if rule.dnp_is_under and ln is not None:
            out.update(settlement_status=UNDER_WIN, binary_score_eligible=True,
                       binary_target_over=0, pmf_score_eligible=False)
        else:
            out.update(settlement_status=VOID_DNP, did_book_void=True)
        return out

    # Appearance path (settles regardless of minutes).
    av = out["actual_outcome"]
    if appeared is None and av is None:
        out["settlement_status"] = PENDING; return out
    if av is None:
        out["settlement_status"] = PENDING; return out
    if ln is None:
        out["settlement_status"] = UNRESOLVED; return out
    out["pmf_score_eligible"] = True
    if av == ln:
        out.update(settlement_status=PUSH, binary_score_eligible=False)   # push: not binary-eligible
    elif av > ln:
        out.update(settlement_status=OVER_WIN, binary_score_eligible=True, binary_target_over=1)
    else:
        out.update(settlement_status=UNDER_WIN, binary_score_eligible=True, binary_target_over=0)
    return out


def settle_frame(df: pd.DataFrame, *, rules: "dict | None" = None,
                 book_col: str = "sportsbook", rule_id_col: str = "sportsbook_rule_id") -> pd.DataFrame:
    """Settle a frame of pairs. A per-row `sportsbook_rule_id` (or a book->rule map) selects the
    frozen rule; rows whose rule is unknown are UNRESOLVED (fail closed)."""
    rules = rules or SPORTSBOOK_SETTLEMENT_RULES
    recs = []
    for _, r in df.iterrows():
        rid = r.get(rule_id_col) if rule_id_col in df.columns else None
        rule = rules.get(str(rid)) if rid else None
        recs.append(settle_one(
            rule=rule, line=r.get("line"),
            appeared=(None if "appeared" not in df.columns else r.get("appeared")),
            actual_outcome=r.get("actual_outcome"),
            canceled=bool(r.get("canceled", False)),
            book_void=bool(r.get("book_void", False))))
    settled = pd.DataFrame(recs, columns=SETTLEMENT_FIELDS)
    keep = [c for c in df.columns if c not in SETTLEMENT_FIELDS]
    return pd.concat([df[keep].reset_index(drop=True), settled], axis=1)
