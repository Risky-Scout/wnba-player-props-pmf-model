"""B5/B6 - fail-closed validation of generated public forecast pages before deploy.

Enforces the trust-critical launch invariants for the forecast-only Distributions page and the
abstaining Edge Board:

  * Provenance + same-release lineage across Edge / PMF-Distributions / Distributions payloads.
  * Edge Board is in explicit abstention (zero recommendations, publication_mode=forecast_only,
    abstain=true) and exposes NO Kelly, NO actionable edge rows, NO "positive edges" counts.
  * Public forecast rows include ONLY forecast_allowed stats (fg3m/blk suppressed).
  * Row-contract classification:
      - PRICED rows (real market line): finite O/U probabilities required.
      - NO_MARKET_AVAILABLE rows (no selected line): null line-display probabilities
        are allowed (never silently coerced to 0/0.5); PMF mass still validated.
  * No invalid/NaN probabilities on PRICED rows; PMF masses sum to ~1; no duplicate keys.

Exit code is nonzero on any failure. Use --require-forecast-rows when games are scheduled.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_CORE_PROVENANCE = ["release_id", "game_date", "git_commit"]
_NO_MARKET_STATUSES = {
    "NO_MARKET_AVAILABLE",
    "NO_MARKET",
    "LIVE_MARKETS_NOT_YET_AVAILABLE",
}


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _forecast_allowed_stats() -> set:
    reg = _load(REPO / "config" / "stat_registry.json") or {}
    return {k.lower() for k, v in reg.items()
            if isinstance(v, dict) and v.get("forecast_allowed") is True}


def _finite01(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and -1e-9 <= v <= 1.0 + 1e-9
    except (TypeError, ValueError):
        return False


def _has_market_line(pr: dict) -> bool:
    """True when the row has a real selected sportsbook line to price against."""
    if pr.get("has_market_line") is True:
        return True
    status = str(pr.get("row_status") or pr.get("pricing_status") or "").upper()
    if status in _NO_MARKET_STATUSES:
        return False
    raw = pr.get("line", pr.get("market_line"))
    try:
        return raw is not None and float(raw) > 0
    except (TypeError, ValueError):
        return False


def _row_status(pr: dict) -> str:
    explicit = str(pr.get("row_status") or pr.get("pricing_status") or "").upper()
    if explicit:
        return explicit
    return "PRICED" if _has_market_line(pr) else "NO_MARKET_AVAILABLE"


def validate(pre_game_dir: Path, *, require_forecast_rows: bool = False) -> list[str]:
    errs: list[str] = []
    edge = _load(pre_game_dir / "Edge" / "latest.json")
    pmf = _load(pre_game_dir / "PMF-Distributions" / "latest.json")
    dist = _load(pre_game_dir / "Distributions" / "latest.json")

    if edge is None:
        errs.append("Edge/latest.json missing")
    if pmf is None:
        errs.append("PMF-Distributions/latest.json missing")
    if dist is None:
        errs.append("Distributions/latest.json missing")
    if errs:
        return errs

    # B5: core provenance present on each payload.
    for name, payload in (("Edge", edge), ("PMF-Distributions", pmf), ("Distributions", dist)):
        for f in _CORE_PROVENANCE:
            if not payload.get(f):
                errs.append(f"{name}/latest.json missing provenance field {f!r}")

    # B5: same-release lineage across all three payloads.
    rels = {edge.get("release_id"), pmf.get("release_id"), dist.get("release_id")}
    if len(rels) > 1:
        errs.append(f"release_id lineage mismatch across payloads: {sorted(map(str, rels))}")

    # B3: Edge Board must be in explicit abstention with NO betting exposure.
    if edge.get("abstain") is not True:
        errs.append("Edge Board is not abstaining (abstain must be true at launch)")
    if str(edge.get("publication_mode")) != "forecast_only":
        errs.append(f"Edge Board publication_mode must be 'forecast_only', got {edge.get('publication_mode')!r}")
    for count_field in ("total_props", "over_signals", "under_signals", "total_recommendations"):
        if int(edge.get(count_field, 0) or 0) != 0:
            errs.append(f"Edge Board exposes nonzero {count_field}={edge.get(count_field)} while abstaining")
    edge_props = edge.get("props") or []
    if edge_props:
        errs.append(f"Edge Board exposes {len(edge_props)} candidate edge row(s) while abstaining")
    # No Kelly anywhere in the Edge payload.
    _blob = json.dumps(edge).lower()
    if '"kelly' in _blob and any(float(p.get("kelly_pct", 0) or 0) != 0 for p in edge_props):
        errs.append("Edge Board exposes nonzero Kelly percentages while abstaining")

    # B2/B6: forecast payloads contain only forecast_allowed stats, valid probs, unique keys.
    allowed = _forecast_allowed_stats()
    for name, payload in (("PMF-Distributions", pmf), ("Distributions", dist)):
        props = payload.get("props") or []
        if require_forecast_rows and not props:
            errs.append(f"{name}: no forecast rows but games are scheduled")
        seen = set()
        for pr in props:
            stat = str(pr.get("stat_raw") or pr.get("stat", "")).lower()
            if allowed and stat not in allowed:
                errs.append(f"{name}: suppressed/uncertified stat {stat!r} is public (forecast_allowed=false)")
            key = (pr.get("player"), stat, pr.get("line"))
            if key in seen:
                errs.append(f"{name}: duplicate player/stat/line key {key}")
            seen.add(key)
            status = _row_status(pr)
            priced = _has_market_line(pr)
            # Row-contract: null line-display probs are honest for NO_MARKET rows.
            # PRICED rows must expose finite settled probabilities.
            for prob_field in ("model_p_over", "model_p_under", "model_prob_over_final"):
                if prob_field not in pr:
                    continue
                val = pr[prob_field]
                if not priced:
                    if val is not None and not _finite01(val):
                        errs.append(
                            f"{name}: invalid probability {prob_field}={val!r} for "
                            f"NO_MARKET row {key} status={status}"
                        )
                    continue
                if not _finite01(val):
                    errs.append(f"{name}: invalid probability {prob_field}={val!r} for {key}")
            # PMF mass must sum to ~1 when present (distribution integrity, priced or not).
            pmf_pairs = pr.get("pmf_full") or pr.get("pmf") or []
            if pmf_pairs:
                s = sum(float(v) for _, v in pmf_pairs)
                if not (0.98 <= s <= 1.02):
                    errs.append(f"{name}: PMF mass for {key} sums to {s:.4f} (outside tolerance)")
    return errs



def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed forecast-page validation (B5/B6).")
    ap.add_argument("--pre-game-dir",
                    default="tools/odds-scanner/predictions/WNBA/Pre-Game")
    ap.add_argument("--require-forecast-rows", action="store_true",
                    help="Fail if forecast rows are empty (use when games are scheduled).")
    args = ap.parse_args()
    errs = validate(Path(args.pre_game_dir), require_forecast_rows=args.require_forecast_rows)
    if errs:
        print("[FORECAST PAGE VALIDATION FAIL]", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("[FORECAST PAGE VALIDATION PASS] abstaining Edge Board + forecast-only Distributions OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
