"""Path B MANDATORY pre-merge acceptance gate (MARKET_DISLOCATION).

This is the fail-closed validator that gates merges. It consumes a Path B live/fixture
scan audit (``LIVE_SCAN_AUDIT.json`` schema, see ``path_b_audit.build_audit``) and asserts
that EVERY displayed row satisfies the 10 acceptance requirements. Any violation makes
``validate_audit`` return a non-passing report; the CLI (``scripts/path_b_acceptance_gate``)
exits non-zero.

Core mandate enforced here:
  * Path B is a MARKET DISLOCATION detector — every row must carry
    ``source_type == "MARKET_DISLOCATION"`` (never MODEL_EDGE).
  * No row is ``actionable`` unless identity + execution (price-survival) + forward-CLV all
    pass — during the validation period that means ``actionable`` must be False everywhere.
  * NO stake sizes / NO Kelly may be emitted — any non-null stake/Kelly field is a violation.
  * Every displayed row must carry full provenance (identity, atomic line, timestamps,
    consensus construction + dispersion, theoretical EV, validation status, reason, source).
  * No-vig must fail closed (config flag) and the audit must disclose rejections by reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

SOURCE_TYPE_MARKET_DISLOCATION = "MARKET_DISLOCATION"

# Requirement 9: the full per-row provenance contract. Every displayed row MUST carry
# every one of these keys (value may be null only where explicitly allowed below).
PROVENANCE_FIELDS: tuple[str, ...] = (
    "event_id",
    "player_name",
    "player_id",
    "player_id_resolved",
    "market_key",
    "is_alternate_market",
    "line",
    "side",
    "bookmaker",
    "displayed_odds",
    "reference_p",
    "consensus_p_over",
    "consensus_n_books",
    "consensus_books",
    "consensus_dispersion_stdev",
    "consensus_dispersion_iqr",
    "consensus_includes_sharp",
    "self_excluded",
    "theoretical_ev_pct",
    "executable_ev_pct",
    "price_survived_30s",
    "price_survived_60s",
    "provider_timestamp",
    "ingestion_timestamp",
    "scan_timestamp",
    "scheduled_tip",
    "quote_age_seconds",
    "validation_status",
    "actionable",
    "rejection_reason",
    "warning_reason",
    "source_type",
)

# Requirement 8: no stake sizes / no Kelly may be emitted during the validation period.
# The presence of any of these keys with a non-null / non-zero value is a hard violation.
FORBIDDEN_STAKE_FIELDS: tuple[str, ...] = (
    "kelly_fraction",
    "kelly",
    "kelly_portfolio",
    "stake",
    "stake_units",
    "bet_size",
    "units",
    "wager",
    "recommended_stake",
)

VALID_VALIDATION_STATUS = frozenset({
    "PENDING_VALIDATION",
    "REJECTED",
    "VALIDATED_EXECUTABLE",
})


@dataclass
class GateViolation:
    code: str
    location: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}] {self.location}: {self.detail}"


@dataclass
class GateReport:
    passed: bool
    violations: list[GateViolation] = field(default_factory=list)
    n_rows_checked: int = 0
    summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_rows_checked": self.n_rows_checked,
            "n_violations": len(self.violations),
            "violations": [
                {"code": v.code, "location": v.location, "detail": v.detail}
                for v in self.violations
            ],
            "summary": self.summary,
        }


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _clv_evidence_qualifies(row: Mapping) -> bool:
    """Fail-closed check of the backtest-CLV evidence embedded on an ``actionable`` row.

    Requires a positive segment sample, a positive mean CLV, and a bootstrap 95% CI whose
    lower bound is strictly > 0 (i.e. the CI excludes zero). Any missing/malformed field or a
    non-positive CI lower bound => not qualified.
    """
    n = row.get("clv_segment_n")
    lo = row.get("clv_ci_low")
    mean = row.get("clv_mean")
    if n is None or lo is None or mean is None:
        return False
    try:
        return int(n) > 0 and float(lo) > 0.0 and float(mean) > 0.0
    except (TypeError, ValueError):
        return False


def _nonzero_number(value) -> bool:
    if value is None:
        return False
    try:
        return abs(float(value)) > 0.0
    except (TypeError, ValueError):
        # A non-numeric non-null stake token still counts as "emitted".
        return True


def validate_row(row: Mapping, location: str) -> list[GateViolation]:
    """Validate one displayed board row against the per-row provenance contract."""
    viol: list[GateViolation] = []

    # Requirement 9: full provenance present.
    for field_name in PROVENANCE_FIELDS:
        if field_name not in row:
            viol.append(GateViolation(
                "MISSING_PROVENANCE_FIELD", location,
                f"required provenance field '{field_name}' absent",
            ))

    # Core mandate: source_type must be MARKET_DISLOCATION.
    if row.get("source_type") != SOURCE_TYPE_MARKET_DISLOCATION:
        viol.append(GateViolation(
            "WRONG_SOURCE_TYPE", location,
            f"source_type={row.get('source_type')!r} != {SOURCE_TYPE_MARKET_DISLOCATION!r}",
        ))

    # Requirement 8: no stake / Kelly emitted.
    for stake_field in FORBIDDEN_STAKE_FIELDS:
        if stake_field in row and _nonzero_number(row.get(stake_field)):
            viol.append(GateViolation(
                "STAKE_EMITTED", location,
                f"forbidden stake/Kelly field '{stake_field}'={row.get(stake_field)!r} "
                "emitted during validation period",
            ))

    # Requirement 1: exact identity — resolved canonical player_id required.
    if not bool(row.get("player_id_resolved")) or _is_missing(row.get("player_id")):
        viol.append(GateViolation(
            "IDENTITY_UNRESOLVED", location,
            f"displayed row lacks resolved canonical player_id "
            f"(player_id={row.get('player_id')!r}, resolved={row.get('player_id_resolved')!r})",
        ))

    # Requirement 2: self-exclusion recorded.
    if row.get("self_excluded") is not True:
        viol.append(GateViolation(
            "SELF_EXCLUSION_NOT_RECORDED", location,
            f"self_excluded={row.get('self_excluded')!r} (candidate book must be excluded "
            "from its own consensus)",
        ))

    # Requirement 8 / core: actionable must be False unless a backtest-CLV segment validates
    # it. Fail closed — actionable=True requires the row to carry qualifying CLV evidence
    # (mean CLV > 0 with a bootstrap 95% CI whose lower bound is strictly > 0), a resolved
    # identity, forward_clv_validated=True, and a VALIDATED_EXECUTABLE status.
    actionable = row.get("actionable")
    if actionable is True:
        fully_validated = (
            row.get("validation_status") == "VALIDATED_EXECUTABLE"
            and bool(row.get("player_id_resolved"))
            and bool(row.get("forward_clv_validated"))
            and _clv_evidence_qualifies(row)
        )
        if not fully_validated:
            viol.append(GateViolation(
                "PREMATURE_ACTIONABLE", location,
                "actionable=True without a qualifying backtest-CLV segment "
                "(need validation_status=VALIDATED_EXECUTABLE, resolved identity, "
                "forward_clv_validated=True, and embedded CLV evidence with a 95% CI "
                "excluding 0)",
            ))
    elif actionable is not False:
        viol.append(GateViolation(
            "ACTIONABLE_NOT_BOOL", location,
            f"actionable={actionable!r} must be a boolean",
        ))

    # Requirement 7: theoretical vs executable EV must be distinct fields.
    if "theoretical_ev_pct" in row and _is_missing(row.get("theoretical_ev_pct")):
        viol.append(GateViolation(
            "MISSING_THEORETICAL_EV", location, "theoretical_ev_pct is null on a displayed row",
        ))

    # Requirement 5: consensus construction must be recorded.
    n_books = row.get("consensus_n_books")
    if not isinstance(n_books, int) or isinstance(n_books, bool):
        viol.append(GateViolation(
            "CONSENSUS_COUNT_INVALID", location,
            f"consensus_n_books={n_books!r} must be an integer",
        ))
    if not isinstance(row.get("consensus_books"), (list, tuple)):
        viol.append(GateViolation(
            "CONSENSUS_BOOKS_INVALID", location,
            "consensus_books must be a list of contributing reference books",
        ))

    # Requirement 3: atomic line — market_key present and side/line well-formed.
    if _is_missing(row.get("market_key")):
        viol.append(GateViolation(
            "ATOMIC_MARKET_MISSING", location, "market_key is required for atomic matching",
        ))
    if row.get("side") not in ("over", "under"):
        viol.append(GateViolation(
            "ATOMIC_SIDE_INVALID", location, f"side={row.get('side')!r} not in (over, under)",
        ))

    # Requirement 4: timestamp integrity. quote_age may be null ONLY when the row carries an
    # explicit malformed-timestamp warning; otherwise a displayed row must have an age.
    if _is_missing(row.get("quote_age_seconds")):
        if row.get("warning_reason") != "malformed_timestamp":
            viol.append(GateViolation(
                "QUOTE_AGE_MISSING", location,
                "quote_age_seconds is null without a malformed_timestamp warning",
            ))

    # Validation status must be a known token.
    if row.get("validation_status") not in VALID_VALIDATION_STATUS:
        viol.append(GateViolation(
            "VALIDATION_STATUS_INVALID", location,
            f"validation_status={row.get('validation_status')!r} not in {sorted(VALID_VALIDATION_STATUS)}",
        ))

    return viol


def _rows_from_audit(audit: Mapping) -> list[tuple[str, Mapping]]:
    """Collect all displayed rows (board_rows + diagnostic_edges) with a location tag."""
    out: list[tuple[str, Mapping]] = []
    for i, r in enumerate(audit.get("board_rows", []) or []):
        out.append((f"board_rows[{i}]", r))
    for i, r in enumerate(audit.get("diagnostic_edges", []) or []):
        out.append((f"diagnostic_edges[{i}]", r))
    return out


def validate_audit(audit: Mapping) -> GateReport:
    """Validate a full Path B scan audit. Returns a GateReport (passed False on any violation)."""
    violations: list[GateViolation] = []

    if not isinstance(audit, Mapping):
        return GateReport(False, [GateViolation("AUDIT_NOT_OBJECT", "$", "audit is not a JSON object")])

    # Top-level source type.
    if audit.get("source_type") != SOURCE_TYPE_MARKET_DISLOCATION:
        violations.append(GateViolation(
            "TOPLEVEL_SOURCE_TYPE", "$.source_type",
            f"top-level source_type={audit.get('source_type')!r} != "
            f"{SOURCE_TYPE_MARKET_DISLOCATION!r}",
        ))

    # Requirement 6: no-vig fail-closed must be declared.
    config = audit.get("config") or {}
    if config.get("no_vig_fail_closed") is not True:
        violations.append(GateViolation(
            "NO_VIG_NOT_FAIL_CLOSED", "$.config.no_vig_fail_closed",
            "config.no_vig_fail_closed must be True (missing opposite side fails closed)",
        ))

    # Requirement 9 / disclosure: rejections by reason must be present.
    rejections = audit.get("rejections") or {}
    if not isinstance(rejections.get("by_reason"), Mapping):
        violations.append(GateViolation(
            "REJECTIONS_NOT_DISCLOSED", "$.rejections.by_reason",
            "rejections.by_reason (counts by reason) must be present",
        ))

    # Requirement 4: latency disclosure.
    latency = audit.get("latency") or {}
    for key in ("max_quote_age_seconds", "median_quote_age_seconds"):
        if key not in latency:
            violations.append(GateViolation(
                "LATENCY_NOT_DISCLOSED", f"$.latency.{key}",
                f"latency.{key} must be reported",
            ))

    # Requirement 7: price-survival block must exist (values may be 0 when credits absent).
    if "price_survival" not in audit:
        violations.append(GateViolation(
            "PRICE_SURVIVAL_MISSING", "$.price_survival",
            "price_survival block (30s/60s recheck results) must be present",
        ))

    # Guard against overstated claims (no profitability/executability assertions).
    for claim_key in ("profitable", "market_superior", "executable_confirmed"):
        if audit.get(claim_key) is True:
            violations.append(GateViolation(
                "OVERSTATED_CLAIM", f"$.{claim_key}",
                f"audit asserts {claim_key}=True — not permitted during validation period",
            ))

    # Per-row provenance.
    rows = _rows_from_audit(audit)
    for location, row in rows:
        violations.extend(validate_row(row, location))

    passed = len(violations) == 0
    summary = {
        "source_type": audit.get("source_type"),
        "n_board_rows": len(audit.get("board_rows", []) or []),
        "n_diagnostic_edges": len(audit.get("diagnostic_edges", []) or []),
        "no_vig_fail_closed": config.get("no_vig_fail_closed"),
        "rejection_reasons": list((rejections.get("by_reason") or {}).keys()),
    }
    return GateReport(passed, violations, n_rows_checked=len(rows), summary=summary)


def load_and_validate(path: str | Path) -> GateReport:
    """Load an audit JSON from disk and validate it."""
    p = Path(path)
    if not p.exists():
        return GateReport(False, [GateViolation("AUDIT_NOT_FOUND", str(p), "audit file does not exist")])
    try:
        audit = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return GateReport(False, [GateViolation("AUDIT_UNREADABLE", str(p), f"cannot parse audit: {exc}")])
    return validate_audit(audit)
