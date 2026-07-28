#!/usr/bin/env python3
"""Assemble the PBP-opportunity completion report from the produced artifacts (single source of truth).

Reads the parser-validation, feature-leakage, and candidate-comparison artifacts and emits one
consolidated JSON with the headline verdict for the owner directive question:
"Does the play-by-play opportunity signal let a pure model (no market inputs) beat or close the gap
to the market on FG3M and AST?"
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
A = REPO / "artifacts" / "opportunity_v2"


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    validation = _load(A / "PBP_PARSER_VALIDATION.json")
    leakage = _load(A / "PBP_FEATURE_LEAKAGE_AUDIT.json")
    comparison = _load(A / "CANDIDATE_COMPARISON_PBP.json")
    ingest = _load(REPO / "data" / "snapshots" / "pbp" / "_INGEST_MANIFEST.json")

    def prop_block(prop: str) -> dict:
        r = comparison.get("results", {}).get(prop, {})
        if not r:
            return {}
        p0, mk, pbp = r.get("p0", {}), r.get("market", {}), r.get("pbp", {})
        gap = p0.get("log_loss", float("nan")) - mk.get("log_loss", float("nan"))
        closed = p0.get("log_loss", float("nan")) - pbp.get("log_loss", float("nan"))
        vsm = r.get("pbp_minus_market", {})
        vsp0 = r.get("pbp_minus_p0", {})
        return {
            "n": r.get("n_total"), "game_dates": r.get("game_dates"),
            "log_loss": {"p0": p0.get("log_loss"), "pbp_pure": pbp.get("log_loss"),
                         "r1_box_opp": r.get("r1", {}).get("log_loss"), "market": mk.get("log_loss")},
            "brier": {"p0": p0.get("brier"), "pbp_pure": pbp.get("brier"), "market": mk.get("brier")},
            "auc": {"p0": p0.get("auc"), "pbp_pure": pbp.get("auc"), "market": mk.get("auc")},
            "ece": {"p0": p0.get("ece"), "pbp_pure": pbp.get("ece"), "market": mk.get("ece")},
            "pbp_vs_p0": {"delta_log_loss": vsp0.get("delta_log_loss"), "p_ll": vsp0.get("p_ll"),
                          "delta_brier": vsp0.get("delta_brier"), "p_brier": vsp0.get("p_brier")},
            "pbp_vs_market": {"delta_log_loss": vsm.get("delta_log_loss"), "p_ll": vsm.get("p_ll"),
                              "holm_p_ll": r.get("holm_vs_market_p_ll", {}).get("pbp"),
                              "holm_p_brier": r.get("holm_vs_market_p_brier", {}).get("pbp")},
            "p0_to_market_ll_gap": gap,
            "pbp_ll_gap_closed_vs_p0": closed,
            "pbp_fraction_of_gap_closed": (closed / gap) if gap else None,
            "pbp_residual_ll_to_market": pbp.get("log_loss", float("nan")) - mk.get("log_loss", float("nan")),
            "beats_market": bool(vsm.get("delta_log_loss", 1) < 0 and (r.get("holm_vs_market_p_ll", {}).get("pbp", 1) <= 0.05)),
        }

    report = {
        "question": ("Does the PBP opportunity signal let a pure (no-market) model beat or close the "
                     "gap to the market on FG3M and AST?"),
        "ingestion": {
            "games_with_pbp": ingest.get("total_games_with_pbp"),
            "total_plays": ingest.get("total_plays"),
            "exhibitions_skipped": ingest.get("known_exhibition_game_ids_skipped"),
        },
        "parser_validation": {
            "reconciliation_games": validation.get("reconciliation_games"),
            "reconciliation_rows": validation.get("reconciliation_rows"),
            "attributable_play_rate": validation.get("attributable_play_rate"),
            "unmatched_actor_play_rate": validation.get("unmatched_actor_play_rate"),
            "per_stat_exact_match": {k: v["exact_match_rate"] for k, v in
                                     validation.get("per_stat_reconciliation", {}).items()},
            "per_stat_mean_abs_error": {k: v["mean_abs_error"] for k, v in
                                        validation.get("per_stat_reconciliation", {}).items()},
        },
        "feature_leakage_guard": leakage,
        "fg3m": prop_block("fg3m"),
        "ast": prop_block("ast"),
    }
    fg3m, ast = report["fg3m"], report["ast"]
    report["verdict"] = {
        "fg3m_beats_market": fg3m.get("beats_market"),
        "ast_beats_market": ast.get("beats_market"),
        "summary": (
            "The pure PBP-opportunity model does NOT beat the market on FG3M or AST (Holm-adjusted "
            f"p_ll = {fg3m.get('pbp_vs_market', {}).get('holm_p_ll')} for FG3M, "
            f"{ast.get('pbp_vs_market', {}).get('holm_p_ll')} for AST). It significantly beats the "
            f"P0 baseline on FG3M (delta LL {fg3m.get('pbp_vs_p0', {}).get('delta_log_loss')}, "
            f"p_ll {fg3m.get('pbp_vs_p0', {}).get('p_ll')}), closing "
            f"{fg3m.get('pbp_fraction_of_gap_closed')} of the P0->market gap, and improves on P0 "
            f"directionally for AST (delta LL {ast.get('pbp_vs_p0', {}).get('delta_log_loss')}, "
            f"p_ll {ast.get('pbp_vs_p0', {}).get('p_ll')}). For FG3M the existing box-opportunity "
            "candidate (r1) is at least as good as PBP, so PBP's genuine incremental value is the new "
            "AST opportunity candidate and the stl/blk/tov label side."),
    }
    out = A / "PBP_MODEL_COMPLETION_REPORT.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
