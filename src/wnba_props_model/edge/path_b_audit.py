"""Build the Path B MARKET_DISLOCATION audit artifacts from a scan.

Converts the enriched ``scan_soft_book_edges`` board (+ rejection ledger) into the audit
JSON that ``path_b_gate.validate_audit`` checks, plus the companion audit artifacts
(consensus construction, quote latency). Every displayed row carries full provenance and
``actionable=False``; no stake/Kelly is emitted.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from wnba_props_model.edge.path_b_gate import (
    PROVENANCE_FIELDS,
    SOURCE_TYPE_MARKET_DISLOCATION,
)

AUDIT_SCHEMA_VERSION = "path_b_dislocation_audit_v1"


def _clean(value):
    """JSON-safe scalar (NaN/NaT -> None, numpy scalars -> python scalars)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def row_provenance(row: Mapping) -> dict:
    """Extract the full provenance contract from one board row (gate-shaped)."""
    return {f: _clean(row.get(f)) for f in PROVENANCE_FIELDS}


def _latency_summary(board: pd.DataFrame, max_quote_age_seconds) -> dict:
    ages = []
    if len(board) and "quote_age_seconds" in board.columns:
        ages = [float(a) for a in board["quote_age_seconds"].tolist() if a is not None and not pd.isna(a)]
    return {
        "n_quotes_with_age": len(ages),
        "max_quote_age_seconds": (max(ages) if ages else None),
        "median_quote_age_seconds": (float(np.median(ages)) if ages else None),
        "configured_max_quote_age_seconds": max_quote_age_seconds,
        "n_malformed_timestamp": int(
            (board["warning_reason"] == "malformed_timestamp").sum()
        ) if len(board) and "warning_reason" in board.columns else 0,
    }


def _consensus_summary(board: pd.DataFrame, reference_books) -> dict:
    if not len(board):
        return {
            "n_rows": 0,
            "n_rows_with_sharp_reference": 0,
            "reference_books": sorted(str(b) for b in reference_books),
            "median_consensus_n_books": None,
            "median_dispersion_stdev": None,
        }
    n_sharp = int(board["consensus_includes_sharp"].sum()) if "consensus_includes_sharp" in board else 0
    disp = [float(x) for x in board.get("consensus_dispersion_stdev", pd.Series(dtype=float)).tolist()
            if x is not None and not pd.isna(x)]
    return {
        "n_rows": int(len(board)),
        "n_rows_with_sharp_reference": n_sharp,
        "reference_books": sorted(str(b) for b in reference_books),
        "median_consensus_n_books": int(np.median(board["consensus_n_books"]))
        if "consensus_n_books" in board else None,
        "median_dispersion_stdev": (float(np.median(disp)) if disp else None),
        "self_exclusion_enforced": bool(board["self_excluded"].all())
        if "self_excluded" in board else False,
    }


def _rejections_summary(rejections: pd.DataFrame) -> dict:
    by_reason: dict[str, int] = {}
    if rejections is not None and len(rejections):
        by_reason = {str(k): int(v) for k, v in
                     rejections["reason"].value_counts().to_dict().items()}
    return {"total": int(sum(by_reason.values())), "by_reason": by_reason}


def build_audit(
    board: pd.DataFrame,
    rejections: pd.DataFrame,
    *,
    game_date: str,
    config: dict,
    discovery: dict,
    credit_usage: dict,
    price_survival: dict | None = None,
    diagnostic_ev_threshold: float = 0.025,
    max_board_rows: int = 2000,
) -> dict:
    """Assemble the full LIVE_SCAN_AUDIT.json dict (gate-validated schema)."""
    now = datetime.now(timezone.utc).isoformat()
    reference_books = config.get("reference_books", [])

    board = board if board is not None else pd.DataFrame()
    board_rows = [row_provenance(r) for _, r in board.head(max_board_rows).iterrows()]

    # Diagnostic edges: >= threshold theoretical EV, actionable stays False.
    diagnostic_edges: list[dict] = []
    if len(board) and "theoretical_ev_frac" in board.columns:
        diag = board[board["theoretical_ev_frac"] >= diagnostic_ev_threshold]
        diag = diag.sort_values("theoretical_ev_frac", ascending=False)
        diagnostic_edges = [row_provenance(r) for _, r in diag.iterrows()]

    config_block = dict(config)
    config_block.setdefault("no_vig_fail_closed", True)
    config_block["diagnostic_ev_threshold_pct"] = round(diagnostic_ev_threshold * 100.0, 4)

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE_MARKET_DISLOCATION,
        "generated_at": now,
        "scan_timestamp": now,
        "game_date": game_date,
        "disclaimer": (
            "MARKET DISLOCATION diagnostics only. Rows are NOT model edges and NOT claimed "
            "profitable/executable/market-superior. THEORETICAL_EV != EXECUTABLE_EV. No row is "
            "actionable and no stake/Kelly is emitted until identity + latency + forward-CLV "
            "validation passes."
        ),
        "profitable": False,
        "market_superior": False,
        "executable_confirmed": False,
        "config": config_block,
        "discovery": discovery,
        "rejections": _rejections_summary(rejections),
        "consensus": _consensus_summary(board, reference_books),
        "latency": _latency_summary(board, config.get("max_quote_age_seconds")),
        "price_survival": price_survival or {
            "rechecked": 0,
            "survived_30s": 0,
            "survived_60s": 0,
            "note": "no forward price-survival recheck performed (credits unavailable or disabled)",
            "details": [],
        },
        "api_credit_usage": credit_usage,
        "summary": {
            "n_board_rows": int(len(board)),
            "n_diagnostic_edges": len(diagnostic_edges),
            "n_qualifying": int(board["qualified"].sum()) if len(board) and "qualified" in board else 0,
        },
        "diagnostic_edges": diagnostic_edges,
        "board_rows": board_rows,
    }


_CSV_COLS = [
    "event_id", "player_name", "player_id", "player_id_resolved", "market_key",
    "is_alternate_market", "line", "side", "bookmaker", "displayed_odds", "reference_p",
    "consensus_p_over", "consensus_n_books", "consensus_includes_sharp",
    "consensus_dispersion_stdev", "self_excluded", "theoretical_ev_pct",
    "executable_ev_pct", "price_survived_30s", "price_survived_60s", "quote_age_seconds",
    "provider_timestamp", "scan_timestamp", "validation_status", "actionable",
    "rejection_reason", "warning_reason", "source_type",
]


def board_to_sample_csv(board: pd.DataFrame, path, max_rows: int = 500) -> int:
    """Write an EDGE_BOARD_SAMPLE.csv of displayed rows with provenance. Returns row count."""
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if board is None or not len(board):
        pd.DataFrame(columns=_CSV_COLS).to_csv(p, index=False)
        return 0
    cols = [c for c in _CSV_COLS if c in board.columns]
    out = board.head(max_rows)[cols].copy()
    out.to_csv(p, index=False)
    return len(out)
