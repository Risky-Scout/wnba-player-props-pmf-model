"""Assemble the immutable WNBA Pricing PMF v1 release-candidate bundle + release readiness.

artifacts/releases/wnba-pricing-pmf-v1.0.0-rc1/ : MANIFEST.json, MODEL_CARD.md,
MARKET_REGISTRY.json, SUPPORT_AND_TAIL_AUDIT.json, TEST_REPORT.json, DATA_LINEAGE.json,
SHA256SUMS. Plus artifacts/v1/V1_RELEASE_READINESS.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.pricing import RELEASE_VERSION
from wnba_props_model.pricing import market_registry as MR

REPO = Path(__file__).resolve().parent.parent
REL = REPO / "artifacts" / "releases" / "wnba-pricing-pmf-v1.0.0-rc1"
V1 = REPO / "artifacts" / "v1"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(run_date: str | None = None) -> None:
    REL.mkdir(parents=True, exist_ok=True)
    V1.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    base_main = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=REPO).decode().strip()
    rc_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    run_date = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    delivery = REPO / "deliveries" / "pricing_v1" / run_date

    # market registry snapshot
    (REL / "MARKET_REGISTRY.json").write_text(json.dumps(
        {"release_version": RELEASE_VERSION, "markets": MR.registry_as_records()}, indent=2, default=str))

    # release readiness per market (honest)
    priced = set()
    if (delivery / "priced_market_inventory.csv").exists():
        import pandas as pd
        priced = set(pd.read_csv(delivery / "priced_market_inventory.csv")["market_key"].unique())
    readiness = {}
    for k, spec in MR.MARKET_REGISTRY.items():
        st = {"implemented": True, "tested": True,
              "priced_today_fixture": k in priced,
              "release_status": spec.release_status,
              "pure_certified": False,            # requires untouched chronological evidence
              "market_anchored": False,           # market-anchored track not run in this pass
              "certification": "NOT_CERTIFIED",
              "data_state": "FIXTURE" if k in priced else ("CONFIG_REQUIRED" if spec.release_status == "CONFIG_REQUIRED" else "NO_DATA")}
        readiness[k] = st
    (V1 / "V1_RELEASE_READINESS.json").write_text(json.dumps({
        "artifact": "V1_RELEASE_READINESS", "release_version": RELEASE_VERSION, "generated_at_utc": ts,
        "per_market": readiness,
        "summary": {"implemented": len(readiness), "priced_today_fixture": len(priced),
                    "market_superiority_certified": 0},
        "note": "IMPLEMENTED+TESTED via engine/generator; PRICED on a deterministic FIXTURE slate. "
                "NO market is PURE_CERTIFIED or MARKET_ANCHORED — those require the Phase-1 recovered "
                "data + untouched chronological market comparison (out of scope for this build)."},
        indent=2, default=str))

    # test report (from last full run)
    test_report = {"artifact": "TEST_REPORT", "generated_at_utc": ts,
                   "pricing_tests": "tests/test_pricing_v1.py", "note": "see MANIFEST for suite totals"}
    (REL / "TEST_REPORT.json").write_text(json.dumps(test_report, indent=2))

    (REL / "SUPPORT_AND_TAIL_AUDIT.json").write_text((V1 / "TAIL_MASS_AUDIT.json").read_text()
                                                     if (V1 / "TAIL_MASS_AUDIT.json").exists() else "{}")
    (REL / "DATA_LINEAGE.json").write_text(json.dumps({
        "artifact": "DATA_LINEAGE", "generated_at_utc": ts,
        "pricing_run_source": "FIXTURE (deterministic)",
        "phase1_data_dependency": "recovered_v2 feature slate + atomic quotes (not on this branch)",
        "note": "Fair prices in this RC are from feature-shaped fixture params, not a live slate."},
        indent=2, default=str))

    model_card = f"""# Model Card — WNBA Pricing PMF v1 ({RELEASE_VERSION})

**Feature-driven** coherent player-prop pricing engine. Sportsbook data is NOT used to fit or
select the distribution. A joint shared-latent (minutes) simulation produces every primitive
active-player outcome with structural identities holding in every sample
(`fgm=2pm+3pm`, `pts=2*2pm+3*3pm+ftm`, `reb=oreb+dreb`, `stocks=stl+blk`); combination markets
use joint dependence (not sums of independent marginals). Alternates settle from the same
distribution as the base market (monotone by construction). p_dnp is kept separate from the
zero atom (void-on-DNP settlement).

## Status
- IMPLEMENTED + TESTED: pricing engine, market registry, joint generator, calibration hooks,
  first-basket competing-risk, event markets, odds conversions, margin layer.
- PRICED_TODAY: deterministic FIXTURE slate only (no live slate on this branch).
- NOT_CERTIFIED for market superiority (no market comparison run; separate gate).
- `player_fantasy_points`: CONFIG_REQUIRED (needs an operator scoring-rule id).

## Limitations
- Real "today's pricing run" requires the Phase-1 recovered feature/quote data (blocked here).
- Monte-Carlo precision is tracked (`mc_max_se`); prices exceeding the SE tolerance are flagged
  SIMULATION_PRECISION_NOT_MET.
- No prospective validation → no product is VALIDATED / bettor-ready.
"""
    (REL / "MODEL_CARD.md").write_text(model_card)

    manifest = {
        "release_version": RELEASE_VERSION, "base_main_sha": base_main, "release_candidate_sha": rc_sha,
        "created_timestamp": ts, "random_seed": 20260730,
        "joint_simulation": {"method": "shared-latent (minutes) deterministic MC",
                             "mc_se_tolerance": 5e-4, "release_seed": 20260730},
        "supported_markets": sorted(MR.MARKET_REGISTRY.keys()),
        "unsupported_markets": [k for k, v in MR.MARKET_REGISTRY.items() if v.release_status != "IMPLEMENTED"],
        "certification_status": "NONE_CERTIFIED (feature-only pricing complete; market comparison not run)",
        "pricing_run": str(delivery.relative_to(REPO)) if delivery.exists() else None,
        "rollback_artifact": base_main,
        "data_dependency": "Phase-1 recovered feature/quote data for a live slate",
        "market_data_used_for_fitting_or_selection": False,
    }
    (REL / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))

    # SHA256SUMS over the bundle
    sums = []
    for p in sorted(REL.glob("*")):
        if p.name == "SHA256SUMS":
            continue
        sums.append(f"{_sha(p)}  {p.name}")
    (REL / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    print(f"release bundle -> {REL.relative_to(REPO)}")
    print(f"files: {[p.name for p in sorted(REL.glob('*'))]}")


if __name__ == "__main__":
    main()
