#!/usr/bin/env python3
"""MANDATORY Path B pre-merge acceptance gate (MARKET_DISLOCATION) — fail closed.

Validates one or more Path B scan audits (``LIVE_SCAN_AUDIT.json`` schema) against the 10
acceptance requirements. EXITS NON-ZERO on ANY violation so it can gate merges in CI.

Usage::

  PYTHONPATH=$(pwd)/src python3 scripts/path_b_acceptance_gate.py \
      artifacts/path_b/LIVE_SCAN_AUDIT.json
  # multiple audits (all must pass):
  PYTHONPATH=$(pwd)/src python3 scripts/path_b_acceptance_gate.py audit1.json audit2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wnba_props_model.edge.path_b_gate import load_and_validate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audits", nargs="+", help="Path(s) to Path B scan audit JSON file(s).")
    ap.add_argument("--report-out", default=None,
                    help="Optional path to write the machine-readable gate report JSON.")
    args = ap.parse_args()

    all_passed = True
    reports = []
    for audit_path in args.audits:
        report = load_and_validate(audit_path)
        reports.append({"audit": audit_path, **report.as_dict()})
        status = "PASS" if report.passed else "FAIL"
        print(f"[gate] {status}: {audit_path} "
              f"({report.n_rows_checked} rows checked, {len(report.violations)} violation(s))")
        if not report.passed:
            all_passed = False
            for v in report.violations[:50]:
                print(f"    - {v}")
            if len(report.violations) > 50:
                print(f"    ... and {len(report.violations) - 50} more violation(s)")

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"passed": all_passed, "reports": reports}, indent=2, default=str))
        print(f"[gate] wrote report -> {out}")

    if all_passed:
        print("[gate] ACCEPTANCE GATE PASSED — all audits satisfy the 10 requirements.")
        return 0
    print("[gate] ACCEPTANCE GATE FAILED — fail closed (exit 1).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
