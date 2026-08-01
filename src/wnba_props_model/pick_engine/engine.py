"""Pick-engine orchestrator: candidates -> gates -> probabilities -> ranked board."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wnba_props_model.pick_engine.constants import (
    DEFAULT_MIN_REFERENCE_BOOKS,
    DEFAULT_TOP_N,
    MARKET_KEY_TO_STAT,
    RETROSPECTIVE_LABEL,
    STAT_TO_MARKET_KEY,
)
from wnba_props_model.pick_engine.gates import evaluate_gates, normalize_stat
from wnba_props_model.pick_engine.odds_math import (
    american_to_decimal,
    break_even_probability,
    expected_value,
    side_settlement_probs,
)
from wnba_props_model.pick_engine.probabilities import (
    assert_tracks_distinct,
    pick_probability,
    production_probability_for_side,
    pure_settled_from_active_pmf,
    side_pure_probability,
)
from wnba_props_model.pick_engine.ranking import provisional_picks, rank_candidates
from wnba_props_model.pick_engine.reference import build_reference_probability
from wnba_props_model.pick_engine.reliability import (
    ReliabilityWeights,
    conservative_probability,
    default_reliability_weights,
    uncertainty_components,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class PickEngineResult:
    ranked: pd.DataFrame
    provisional: pd.DataFrame
    abstentions: pd.DataFrame
    candidates: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


def _pmf_index(pmfs: pd.DataFrame) -> dict[tuple[Any, Any, str], pd.Series]:
    idx: dict[tuple[Any, Any, str], pd.Series] = {}
    for _, r in pmfs.iterrows():
        stat = normalize_stat(str(r.get("stat", "")))
        if stat is None:
            continue
        key = (r.get("game_id"), r.get("player_id"), stat)
        idx[key] = r
    return idx


def _name_index(pmfs: pd.DataFrame) -> dict[tuple[str, str], list[pd.Series]]:
    out: dict[tuple[str, str], list[pd.Series]] = {}
    for _, r in pmfs.iterrows():
        stat = normalize_stat(str(r.get("stat", "")))
        if stat is None:
            continue
        key = (str(r.get("player_name", "")).strip().lower(), stat)
        out.setdefault(key, []).append(r)
    return out


def _lookup_production(
    fair: pd.DataFrame | None,
    *,
    game_id,
    player_id,
    stat: str,
    line: float,
) -> float | None:
    if fair is None or fair.empty:
        return None
    m = (
        (fair["game_id"] == game_id)
        & (fair["player_id"] == player_id)
        & (fair["stat"].astype(str) == stat)
        & np.isclose(fair["line"].astype(float), float(line), atol=1e-9)
    )
    sub = fair.loc[m]
    if sub.empty:
        return None
    return float(sub.iloc[0]["p_over"])


def build_candidates(
    *,
    quotes: pd.DataFrame,
    pmfs: pd.DataFrame,
    fair_odds: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    game_map: dict[str, dict[str, Any]] | None = None,
    reliability: ReliabilityWeights | None = None,
    prediction_timestamp: str | datetime | None = None,
    asof_timestamp: str | datetime | None = None,
    lineage_hashes: dict[str, str] | None = None,
    min_reference_books: int = DEFAULT_MIN_REFERENCE_BOOKS,
    rejected_player_ids: set[Any] | None = None,
    team_mismatch_player_ids: set[Any] | None = None,
    retrospective: bool = False,
    quote_freshness_hours: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build scored candidate sides and abstention rows."""
    rel = reliability or default_reliability_weights()
    hashes = lineage_hashes or {}
    pred_ts = _parse_ts(prediction_timestamp)
    asof = _parse_ts(asof_timestamp) or pred_ts or _utc_now()
    rejected_player_ids = rejected_player_ids or set()
    team_mismatch_player_ids = team_mismatch_player_ids or set()
    # Retrospective boards reconstruct pre-tip executable prices against a frozen
    # PMF that may have been produced earlier the same slate window. Freshness is
    # evaluated against asof (quote board time), and prediction_cutoff relaxes to tip.
    freshness_hours = quote_freshness_hours
    if freshness_hours is None:
        freshness_hours = 18.0 if retrospective else None

    # Identity maps
    identity_by_id: dict[Any, dict] = {}
    if identity is not None and not identity.empty:
        for _, r in identity.iterrows():
            identity_by_id[r.get("canonical_player_id")] = r.to_dict()

    out_status: dict[Any, str] = {}
    if injuries is not None and not injuries.empty:
        # Flexible injury schemas
        for _, r in injuries.iterrows():
            pid = r.get("player_id") or r.get("canonical_player_id")
            status = str(r.get("status") or r.get("injury_status") or "").upper()
            if pid is not None:
                out_status[pid] = status

    pmf_by_key = _pmf_index(pmfs)
    pmf_by_name = _name_index(pmfs)
    game_map = game_map or {}

    q = quotes.copy()
    if "sportsbook" in q.columns and "book" not in q.columns:
        q = q.rename(columns={"sportsbook": "book"})
    if "market_key" not in q.columns and "stat" in q.columns:
        q["market_key"] = q["stat"].map(lambda s: STAT_TO_MARKET_KEY.get(str(s), str(s)))

    # Normalize soft-book stat keys already short-form (pts/reb/...).
    if "stat" in q.columns:
        q["stat_norm"] = q["stat"].map(lambda s: normalize_stat(str(s)))
    else:
        q["stat_norm"] = q["market_key"].map(lambda s: normalize_stat(str(s)))

    candidates: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []

    for _, qr in q.iterrows():
        side = str(qr.get("side", "")).strip().lower()
        if side not in {"over", "under"}:
            continue
        market_key = str(qr.get("market_key") or "")
        stat = qr.get("stat_norm") or normalize_stat(market_key)
        player_name = str(qr.get("player_name") or "").strip()
        book = str(qr.get("book") or "")
        try:
            line = float(qr["line"])
            american = float(qr["american_odds"])
            decimal = american_to_decimal(american)
        except Exception:  # noqa: BLE001
            abstentions.append(
                {
                    "player": player_name,
                    "stat": stat,
                    "line": qr.get("line"),
                    "side": side,
                    "sportsbook": book,
                    "reason": "ABSTAIN_STALE_QUOTE",
                    "detail": "invalid_executable_odds",
                }
            )
            continue

        event_id = str(qr.get("event_id") or "")
        ginfo = game_map.get(event_id, {})
        game_id = qr.get("game_id") or ginfo.get("game_id") or ginfo.get("bdl_game_id")
        tip = qr.get("commence_time") or ginfo.get("scheduled_tip_utc")

        # Resolve player identity via PMF name match within event/game when possible.
        pmf_row = None
        player_id = qr.get("player_id")
        if player_id is not None and game_id is not None and stat is not None:
            pmf_row = pmf_by_key.get((game_id, player_id, stat))
        if pmf_row is None and stat is not None:
            matches = pmf_by_name.get((player_name.lower(), stat), [])
            if game_id is not None:
                matches = [m for m in matches if m.get("game_id") == game_id]
            if len(matches) == 1:
                pmf_row = matches[0]
                player_id = pmf_row.get("player_id")
                game_id = pmf_row.get("game_id")
            elif len(matches) > 1:
                abstentions.append(
                    {
                        "player": player_name,
                        "stat": stat,
                        "line": line,
                        "side": side,
                        "sportsbook": book,
                        "reason": "ABSTAIN_IDENTITY",
                        "detail": "ambiguous_player_match",
                    }
                )
                continue

        identity_rejected = (
            player_id in rejected_player_ids
            or (
                player_id in identity_by_id
                and str(identity_by_id[player_id].get("audit_status", "")).upper() == "REJECTED"
            )
        )
        team_mismatch = player_id in team_mismatch_player_ids or (
            player_id in identity_by_id
            and "mismatch" in str(identity_by_id[player_id].get("reject_reason") or "").lower()
        )

        active_pmf = None
        role = "all"
        team = ""
        opponent = ""
        p_dnp = None
        if pmf_row is not None:
            active_pmf = pmf_row.get("active_pmf_json") or pmf_row.get("pmf_json")
            role = str(pmf_row.get("role_bucket") or "all")
            team = str(pmf_row.get("team_abbreviation") or "")
            opponent = str(
                pmf_row.get("opponent_team_abbreviation")
                or pmf_row.get("opponent_abbreviation")
                or ""
            )
            p_dnp = pmf_row.get("p_dnp")

        settled = (
            pure_settled_from_active_pmf(active_pmf, line)
            if active_pmf is not None
            else {"valid": False, "reason": "ABSTAIN_MISSING_PURE_PROBABILITY"}
        )
        pure_p = side_pure_probability(settled, side) if settled.get("valid") else None

        prod_over = _lookup_production(
            fair_odds, game_id=game_id, player_id=player_id, stat=str(stat), line=line
        )
        # Production may equal pure on current main; keep field separate regardless.
        if prod_over is None and settled.get("valid"):
            prod_over = settled.get("pure_probability_over")
        prod_p = production_probability_for_side(production_p_over=prod_over, side=side)

        ref = build_reference_probability(
            q,
            event_id=event_id,
            player_name=player_name,
            stat=str(qr.get("stat") or stat),
            line=line,
            side=side,
            candidate_book=book,
            asof=asof,
            min_books=min_reference_books,
        )
        # If no external reference, still allow pure-model ranking without reference tier.
        if ref.has_valid_reference and ref.reference_probability is not None:
            ref_p = float(ref.reference_probability)
            ref_tier_ok = True
        else:
            ref_p = None
            ref_tier_ok = False

        w = rel.weight_for(stat=str(stat or ""), role=role)
        if pure_p is not None and ref_p is not None:
            p_pick = pick_probability(pure_p, ref_p, w)
        elif pure_p is not None:
            # No reference: pick track follows pure (w applied vs 0.5 neutral only for ranking).
            p_pick = float(pure_p)
            w = float(w)  # retain reliability for conservative bound
        else:
            p_pick = None

        quote_ts = qr.get("book_last_update") or qr.get("collected_utc")
        quote_age = None
        qt = _parse_ts(quote_ts)
        if qt is not None:
            quote_age = max(0.0, (asof - qt).total_seconds() / 3600.0)

        avail_status = out_status.get(player_id, "UNKNOWN")
        confirmed_out = avail_status in {"OUT", "CONFIRMED_OUT"}

        unc = uncertainty_components(
            calibration_uncertainty=0.02,
            segment_reliability=max(w, 0.0),
            role_uncertainty=0.03 if role in {"bench", "spot"} else 0.01,
            availability_uncertainty=0.05 if p_dnp is not None and float(p_dnp) > 0.2 else 0.01,
            ood_uncertainty=0.0,
            quote_freshness_penalty=min(0.2, (quote_age or 0.0) / 24.0),
            model_disagreement=0.0,
        )

        p_win = p_lose = p_push = None
        raw_edge = shrunk_edge = raw_ev = cons_ev = None
        if (
            pure_p is not None
            and settled.get("valid")
            and settled.get("p_over_unconditional") is not None
        ):
            # EV uses unconditional win/lose/push masses (push returns stake).
            p_win, p_lose, p_push = side_settlement_probs(
                side=side,
                p_over_unc=float(settled["p_over_unconditional"]),
                p_under_unc=float(settled["p_under_unconditional"]),
                p_push=float(settled["p_push"] or 0.0),
                line=line,
            )
            # For pick-edge vs break-even use pick/pure settled-style probabilities.
            p_be = break_even_probability(decimal)
            raw_edge = float(pure_p) - p_be
            if p_pick is not None:
                shrunk_edge = float(p_pick) - p_be
            # Raw EV from pure unconditional decomposition at executable price.
            raw_ev = expected_value(
                p_win=p_win, p_lose=p_lose, p_push=p_push, decimal_odds=decimal
            )
            # Conservative EV: scale win prob toward pick conservative bound.
            if p_pick is not None:
                p_cons = conservative_probability(
                    p_pick, reliability_weight=w, uncertainty=unc["uncertainty_total"]
                )
                # Map settled-style conservative p into win mass while preserving push.
                if p_push is not None and p_push < 1.0:
                    win_cons = p_cons * (1.0 - p_push)
                    lose_cons = (1.0 - p_cons) * (1.0 - p_push)
                else:
                    win_cons, lose_cons = p_cons, 1.0 - p_cons
                cons_ev = expected_value(
                    p_win=win_cons,
                    p_lose=lose_cons,
                    p_push=float(p_push or 0.0),
                    decimal_odds=decimal,
                )
            try:
                if p_pick is not None and prod_p is not None and ref_p is not None:
                    assert_tracks_distinct(
                        pure_probability=float(pure_p),
                        reference_market_probability=ref_p,
                        production_probability=prod_p,
                        pick_prob=float(p_pick),
                    )
            except AssertionError:
                # Fail closed into abstention rather than emitting a corrupted row.
                abstentions.append(
                    {
                        "player": player_name,
                        "stat": stat,
                        "line": line,
                        "side": side,
                        "sportsbook": book,
                        "reason": "ABSTAIN_MISSING_PURE_PROBABILITY",
                        "detail": "track_collision_production_suppressed_alpha",
                    }
                )
                continue

        tip_ts = _parse_ts(tip)
        if retrospective and tip_ts is not None:
            # Require quote before tip; do not reject solely because the frozen
            # PMF timestamp precedes the reconstructed pre-tip quote board.
            pred_cutoff = tip_ts.isoformat()
        else:
            pred_cutoff = pred_ts.isoformat() if pred_ts else None
        gate_row = {
            "market_key": market_key if market_key in MARKET_KEY_TO_STAT else STAT_TO_MARKET_KEY.get(str(stat), market_key),
            "stat": stat,
            "canonical_game_id": game_id,
            "canonical_player_id": player_id,
            "game_id_valid": game_id is not None,
            "player_id_valid": player_id is not None,
            "current_team_valid": bool(team) or pmf_row is not None,
            "identity_rejected": identity_rejected,
            "team_mismatch": team_mismatch,
            "confirmed_out": confirmed_out,
            "availability_status": avail_status,
            "availability_timestamp": None,
            "prediction_timestamp": pred_ts.isoformat() if pred_ts else None,
            "prediction_cutoff": pred_cutoff,
            "provider_quote_timestamp": quote_ts,
            "scheduled_tip_utc": tip,
            "asof_timestamp": asof.isoformat(),
            "active_pmf": active_pmf,
            "pure_probability": pure_p,
            "executable_price_available": True,
            "period": qr.get("period") or "game",
            "ood_flag": False,
        }
        if freshness_hours is not None:
            gate_row["quote_freshness_hours"] = float(freshness_hours)
        gate = evaluate_gates(gate_row)
        row = {
            "game": f"{ginfo.get('away_team', qr.get('away_team', ''))}@{ginfo.get('home_team', qr.get('home_team', ''))}",
            "game_id": game_id,
            "event_id": event_id,
            "scheduled_tip": tip,
            "player": player_name,
            "player_id": player_id,
            "team": team or ginfo.get("team"),
            "opponent": opponent,
            "stat": stat,
            "market_key": gate_row["market_key"],
            "period": "game",
            "line": line,
            "side": side,
            "sportsbook": book,
            "provider": qr.get("provider") or "the-odds-api",
            "american_odds": american,
            "decimal_odds": decimal,
            "pure_probability": pure_p,
            "reference_probability": ref_p,
            "production_probability": prod_p,
            "pick_probability": p_pick,
            "break_even_probability": break_even_probability(decimal),
            "p_win": p_win,
            "p_lose": p_lose,
            "p_push": p_push,
            "raw_probability_edge": raw_edge,
            "shrunken_probability_edge": shrunk_edge,
            "raw_expected_value": raw_ev,
            "conservative_expected_value": cons_ev,
            "reliability_weight": w,
            "uncertainty": unc["uncertainty_total"],
            "uncertainty_components": json.dumps(unc, sort_keys=True),
            "quote_age": quote_age,
            "availability_status": avail_status,
            "reference_tier_ok": ref_tier_ok,
            "reference_books": ",".join(ref.reference_books),
            "consensus_dispersion": ref.consensus_dispersion,
            "provider_quote_timestamp": quote_ts,
            "market_last_update": qr.get("book_last_update"),
            "ingestion_timestamp": qr.get("collected_utc"),
            "prediction_timestamp": pred_ts.isoformat() if pred_ts else None,
            "valid": gate.ok,
            "abstain_reason": gate.reason,
            "ood_warning": False,
            "availability_warning": bool(p_dnp is not None and float(p_dnp) > 0.35),
            "model_hash": hashes.get("model_hash"),
            "calibrator_hash": hashes.get("calibrator_hash"),
            "feature_hash": hashes.get("feature_hash"),
            "data_hash": hashes.get("data_hash"),
            "quote_hash": hashes.get("quote_hash"),
            "availability_hash": hashes.get("availability_hash"),
            "weights_hash": rel.weights_hash,
        }
        if gate.ok:
            candidates.append(row)
        else:
            abstentions.append(
                {
                    "player": player_name,
                    "player_id": player_id,
                    "game_id": game_id,
                    "stat": stat,
                    "line": line,
                    "side": side,
                    "sportsbook": book,
                    "american_odds": american,
                    "reason": gate.reason,
                    "detail": "",
                }
            )

    return pd.DataFrame(candidates), pd.DataFrame(abstentions)


def run_pick_engine(
    *,
    quotes: pd.DataFrame,
    pmfs: pd.DataFrame,
    fair_odds: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    game_map: dict[str, dict[str, Any]] | None = None,
    reliability: ReliabilityWeights | None = None,
    prediction_timestamp: str | datetime | None = None,
    asof_timestamp: str | datetime | None = None,
    lineage_hashes: dict[str, str] | None = None,
    top_n: int = DEFAULT_TOP_N,
    board_label: str = "",
    min_reference_books: int = DEFAULT_MIN_REFERENCE_BOOKS,
    rejected_player_ids: set[Any] | None = None,
    team_mismatch_player_ids: set[Any] | None = None,
    certification_by_stat: dict | None = None,
    retrospective: bool = False,
    quote_freshness_hours: float | None = None,
) -> PickEngineResult:
    retrospective = retrospective or (board_label == RETROSPECTIVE_LABEL)
    candidates, abstentions = build_candidates(
        quotes=quotes,
        pmfs=pmfs,
        fair_odds=fair_odds,
        identity=identity,
        injuries=injuries,
        game_map=game_map,
        reliability=reliability,
        prediction_timestamp=prediction_timestamp,
        asof_timestamp=asof_timestamp,
        lineage_hashes=lineage_hashes,
        min_reference_books=min_reference_books,
        rejected_player_ids=rejected_player_ids,
        team_mismatch_player_ids=team_mismatch_player_ids,
        retrospective=retrospective,
        quote_freshness_hours=quote_freshness_hours,
    )
    ranked = rank_candidates(
        candidates,
        top_n=top_n,
        certification_by_stat=certification_by_stat,
        board_label=board_label,
    )
    provisional = provisional_picks(ranked)
    # Full valid candidate board for diagnostics (ranked top_n is the published board).
    manifest = {
        "generated_at_utc": _utc_now().isoformat(),
        "board_label": board_label,
        "n_quote_rows": int(len(quotes)) if quotes is not None else 0,
        "n_valid_candidates": int(len(candidates)),
        "n_ranked_selections": int(len(ranked)),
        "n_provisional_picks": int(len(provisional)),
        "n_abstentions": int(len(abstentions)),
        "abstentions_by_reason": (
            abstentions["reason"].value_counts().to_dict() if not abstentions.empty else {}
        ),
        "weights_hash": (reliability or default_reliability_weights()).weights_hash,
        "lineage_hashes": lineage_hashes or {},
        "retrospective": board_label == RETROSPECTIVE_LABEL,
    }
    return PickEngineResult(
        ranked=ranked,
        provisional=provisional,
        abstentions=abstentions,
        candidates=candidates,
        manifest=manifest,
    )


def write_pick_engine_delivery(
    result: PickEngineResult,
    out_dir: str | Path,
) -> dict[str, str]:
    """Write ranked_selections/provisional_picks/abstentions/pick_manifest under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "ranked_selections": str(out / "ranked_selections.csv"),
        "provisional_picks": str(out / "provisional_picks.csv"),
        "abstentions": str(out / "abstentions.csv"),
        "pick_manifest": str(out / "pick_manifest.json"),
    }
    result.ranked.to_csv(paths["ranked_selections"], index=False)
    result.provisional.to_csv(paths["provisional_picks"], index=False)
    result.abstentions.to_csv(paths["abstentions"], index=False)
    manifest = dict(result.manifest)
    manifest["output_paths"] = paths
    manifest["manifest_hash"] = _sha(manifest)
    Path(paths["pick_manifest"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths
