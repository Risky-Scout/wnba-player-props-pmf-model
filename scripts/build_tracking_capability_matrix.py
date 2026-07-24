"""LANE 2 (W2) - tracking/hustle capability matrix.

Reads the hash-verified tracking + hustle assets and classifies each feature the Blueprint-C
structural candidates need as one of:

    DIRECTLY_AVAILABLE - present at player+game grain with adequate coverage
    DERIVABLE          - computable from present columns (documented formula)
    PROXY_ONLY         - only a proxy exists (e.g. box-score 3PA for FG3M attempts)
    UNAVAILABLE        - not present, or present only at the wrong grain

Emits artifacts/tracking/capability_matrix.json + a short markdown report with coverage stats.
Hustle is expected to be deferred until a player-grain extract exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRACKING = REPO / "data" / "processed" / "wnba_tracking_2021_2026.parquet"
HUSTLE = REPO / "data" / "processed" / "wnba_hustle_2021_2026.parquet"
OUT_JSON = REPO / "artifacts" / "tracking" / "capability_matrix.json"
OUT_MD = REPO / "artifacts" / "tracking" / "CAPABILITY_MATRIX.md"

DIRECT, DERIVABLE, PROXY, UNAVAILABLE = (
    "DIRECTLY_AVAILABLE", "DERIVABLE", "PROXY_ONLY", "UNAVAILABLE")

# Blueprint-C structural feature needs -> tracking columns backing them.
TRACKING_FEATURES = {
    "reb_chances_total": ["reboundChancesTotal"],
    "reb_chances_offensive": ["reboundChancesOffensive"],
    "reb_chances_defensive": ["reboundChancesDefensive"],
    "touches": ["touches"],
    "passes": ["passes"],
    "assists_tracked": ["assists"],
    "secondary_assists": ["secondaryAssists"],
    "free_throw_assists": ["freeThrowAssists"],
    "contested_fga": ["contestedFieldGoalsAttempted"],
    "contested_fgm": ["contestedFieldGoalsMade"],
    "defended_at_rim_fga": ["defendedAtRimFieldGoalsAttempted"],
    "speed": ["speed"],
    "distance": ["distance"],
}
# Derived features and the columns they need.
DERIVED_FEATURES = {
    "assist_opportunity_proxy": ["passes", "assists", "secondaryAssists", "freeThrowAssists"],
}
# Features that tracking does NOT provide directly (documented proxies).
PROXY_FEATURES = {
    # FG3M attempts are not a tracking column; use box-score 3PA as the attempts proxy.
    "fg3m_attempts": {"proxy": "box_score_3PA", "reason": "no 3PA column in tracking extract"},
}
PLAYER_KEY_CANDIDATES = ["personId", "PLAYER_ID", "player_id"]
GAME_KEY_CANDIDATES = ["gameId", "GAME_ID", "game_id"]
_MIN_COVERAGE = 0.80


def _coverage(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(df[col].notna().mean())


def _player_grain_ok(df: pd.DataFrame) -> tuple[bool, str]:
    pk = next((c for c in PLAYER_KEY_CANDIDATES if c in df.columns), None)
    if pk is None:
        return False, "no player id column"
    n = int(df[pk].nunique())
    # A player-grain table must have many distinct players (a degenerate single id => wrong grain).
    return (n > 10), f"{pk} nunique={n}"


def build() -> dict:
    tr = pd.read_parquet(TRACKING) if TRACKING.exists() else pd.DataFrame()
    hu = pd.read_parquet(HUSTLE) if HUSTLE.exists() else pd.DataFrame()

    tr_grain_ok, tr_grain = _player_grain_ok(tr)
    hu_grain_ok, hu_grain = _player_grain_ok(hu)
    tr_games = int(tr[next((c for c in GAME_KEY_CANDIDATES if c in tr.columns), "gameId")].nunique()) if len(tr) else 0

    features: dict[str, dict] = {}
    for feat, cols in TRACKING_FEATURES.items():
        cov = min((_coverage(tr, c) for c in cols), default=0.0)
        present = all(c in tr.columns for c in cols)
        if present and tr_grain_ok and cov >= _MIN_COVERAGE:
            status = DIRECT
        elif present and tr_grain_ok:
            status = PROXY  # present but sparse coverage -> proxy-grade only
        else:
            status = UNAVAILABLE
        features[feat] = {"status": status, "source": "tracking", "columns": cols,
                          "coverage": round(cov, 4)}
    for feat, cols in DERIVED_FEATURES.items():
        present = all(c in tr.columns for c in cols)
        features[feat] = {"status": DERIVABLE if (present and tr_grain_ok) else UNAVAILABLE,
                          "source": "tracking(derived)", "columns": cols,
                          "formula": "potential_assists ~= f(passes, assists, secondaryAssists, freeThrowAssists)"}
    for feat, meta in PROXY_FEATURES.items():
        features[feat] = {"status": PROXY, "source": meta["proxy"], "reason": meta["reason"]}

    # Hustle: player-degenerate extract -> deferred/unavailable at player grain.
    hustle_status = UNAVAILABLE if not hu_grain_ok else PROXY
    for feat in ("deflections", "contested_shots", "loose_balls_recovered", "box_outs",
                 "screen_assists", "charges_drawn"):
        features[feat] = {"status": hustle_status, "source": "hustle",
                          "note": f"hustle grain: {hu_grain} (deferred until player-grain extract)"}

    matrix = {
        "version": "tracking-capability-v1",
        "tracking": {"rows": int(len(tr)), "games": tr_games, "player_grain_ok": tr_grain_ok,
                     "grain": tr_grain, "player_key": next((c for c in PLAYER_KEY_CANDIDATES if c in tr.columns), None)},
        "hustle": {"rows": int(len(hu)), "player_grain_ok": hu_grain_ok, "grain": hu_grain,
                   "deferred": not hu_grain_ok},
        "min_coverage_threshold": _MIN_COVERAGE,
        "features": features,
        "summary": {s: sorted(f for f, m in features.items() if m["status"] == s)
                    for s in (DIRECT, DERIVABLE, PROXY, UNAVAILABLE)},
    }
    return matrix


def write(matrix: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(matrix, indent=2) + "\n")
    lines = ["# Tracking / Hustle Capability Matrix", "",
             f"- tracking rows: {matrix['tracking']['rows']:,} across {matrix['tracking']['games']:,} games "
             f"(player grain: {matrix['tracking']['player_grain_ok']}, {matrix['tracking']['grain']})",
             f"- hustle rows: {matrix['hustle']['rows']:,} (player grain: {matrix['hustle']['player_grain_ok']}, "
             f"{matrix['hustle']['grain']}) -> deferred: {matrix['hustle']['deferred']}", "",
             "| feature | status | source | coverage |", "|---|---|---|---|"]
    for feat, m in sorted(matrix["features"].items()):
        lines.append(f"| {feat} | `{m['status']}` | {m.get('source','')} | {m.get('coverage','-')} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    m = build()
    write(m)
    print(f"[capability] tracking games={m['tracking']['games']} player_grain_ok={m['tracking']['player_grain_ok']}")
    for status, feats in m["summary"].items():
        print(f"  {status}: {feats}")
    print(f"[capability] -> {OUT_JSON}")
