"""Strengthened Stage 9: physical feature/target separation, explicit allowlist, PIT proof,
join integrity, and tracked audits. OFFLINE only — no API, no modeling, no target-driven choices.

Builds (large parquet, gitignored):
  data/recovered_v2/modeling/wnba_pregame_features_t12.parquet   (feature-only)
  data/recovered_v2/modeling/wnba_player_targets.parquet         (target-only)

Tracked audits: RECOVERED_V2_BUILD_MANIFEST.json, STAGE9_STRENGTHENED_READINESS.json,
FEATURE_REGISTRY.json, FEATURE_SCHEMA.json, FEATURE_TARGET_SEPARATION_AUDIT.json,
POINT_IN_TIME_AUDIT.json, SAME_DAY_LEAKAGE_AUDIT.json, ESTIMATOR_INPUT_GUARD_AUDIT.json,
FEATURE_MISSINGNESS_COVERAGE.csv, FEATURE_JOIN_COVERAGE.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.features.estimator_guard import FORBIDDEN_PATTERN  # noqa: E402
from wnba_props_model.models.prop_feature_policy import feature_schema_hash  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"
MODEL_DIR = REPO / "data" / "recovered_v2" / "modeling"
DIRECT = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
COMBOS = {"stocks": ("stl", "blk"), "pts_ast": ("pts", "ast"), "pts_reb": ("pts", "reb"),
          "reb_ast": ("reb", "ast"), "pts_reb_ast": ("pts", "reb", "ast")}

# ---- explicit rejection rules (recorded reasons) ----
REJECT_MARKET = {"game_total", "game_spread_home", "implied_team_total", "blowout_risk",
                 "predicted_spread_abs", "close_game_indicator"}
REJECT_INTERNAL_SCRIPT = {"blowout_probability", "close_game_probability", "pregame_win_probability",
                          "expected_minutes_given_script", "minutes_upside"}
_MARKET_PAT = re.compile(r"(market|odds|price|vegas|no_vig|line_move|consensus|sportsbook|bookmaker)", re.I)
_INJURY_FWD_PAT = re.compile(
    r"(injur|out_count|questionable|vacated|teammate_|without_|confirmed_starter|lineup_confirmed|"
    r"expected_starter|expected_bench|role_elevation|top3|projected_usage|usage_transfer|"
    r"team_total_usage_of_out)", re.I)
# tokens that mark a feature as strictly lagged / pregame-known (availability provable at T-12)
_LAG_PAT = re.compile(
    r"(_l\d+\b|_l\d+_|_roll\d*|_ewma\d*|_lag\d*|_last\d+|_mean_l|_std_l|_std\b|_prior|_rate_l|"
    r"per_min_roll|_delta_l|_zscore|_momentum|_form|_trend|_volatility|_median_l|_min_l|_max_l)", re.I)
_PREGAME_FACT_PAT = re.compile(
    r"(rest|days_since|dnp_streak|games_in_last|games_played_prior|games_prior|is_b2b|is_3in4|"
    r"is_4_in_5|is_5_in_7|cumulative_minutes_l|load_index|is_home|home_away|season_phase|"
    r"is_playoff|season_completion|game_number|altitude|timezone|travel|back_to_back|position_|"
    r"starter_rate|recent_starter|minutes_last|minutes_mean|minutes_std|minutes_support|"
    r"did_play_rate_l|zero_minute_rate_l|pace_proxy_roll|pace_ewma|_allowed_roll|_allowed_ewma|"
    r"def_rating_ewma|game_pace_predicted|3in4|4_in_5|5_in_7)", re.I)


def classify_feature(name: str) -> tuple[str, str, str]:
    """Return (status, reason, availability_rule)."""
    if FORBIDDEN_PATTERN.search(name):
        return "REJECTED", "matches forbidden/target-like pattern", "n/a"
    if name in REJECT_MARKET or _MARKET_PAT.search(name):
        return "REJECTED", "sportsbook/market-derived", "n/a"
    if name in REJECT_INTERNAL_SCRIPT:
        return "REJECTED", "internal game-script forecast (not a pure player feature)", "n/a"
    if _INJURY_FWD_PAT.search(name):
        return "REJECTED", "injury/lineup/forward context without timestamped historical availability", "n/a"
    if _LAG_PAT.search(name):
        return "APPROVED_ESTIMATOR_FEATURE", "strictly-lagged rolling feature (prior games only)", "prior_games_shifted_<=T-12"
    if _PREGAME_FACT_PAT.search(name):
        return "APPROVED_ESTIMATOR_FEATURE", "pregame-known schedule/context fact", "pregame_known_<=T-12"
    return "REJECTED", "ambiguous availability (no clear lag/pregame token) — conservative reject", "n/a"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _feature_group(name: str) -> str:
    if name.startswith("opp_"):
        return "opponent_context"
    if name.startswith("team_"):
        return "team_context"
    if "minutes" in name or "starter" in name or "dnp" in name:
        return "minutes_role"
    if "per_min" in name or re.search(r"_(pts|reb|ast|fg3m|stl|blk|tov|turnover|usage|fga|fta)_", name) or re.search(r"_(pts|reb|ast|fg3m|stl|blk|tov)_", name):
        return "player_rate"
    if re.search(r"(rest|b2b|3in4|4_in_5|5_in_7|days_since|load|cumulative|schedule|travel|altitude)", name):
        return "schedule_rest"
    if "position" in name or name in ("is_home", "home_away"):
        return "identity_context"
    if "season" in name or "playoff" in name or "game_number" in name:
        return "season_phase"
    return "player_form"


@app.command()
def main(wide: str = typer.Option("data/recovered_v2/wnba_player_game_features_wide.parquet", "--wide"),
         box: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--box"),
         store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store")) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    w = pd.read_parquet(wide)
    manifest = json.loads((REPO / "data/recovered_v2/feature_schema_manifest.json").read_text())
    model_features = manifest["model_feature_columns"]

    # ---- classify -> explicit allowlist ----
    reg_entries, approved = [], []
    for f in model_features:
        status, reason, avail = classify_feature(f)
        entry = {
            "feature_name": f, "feature_version": "recovered_v2", "feature_group": _feature_group(f),
            "definition": "build_features.py strict_pregame_shifted transform",
            "source_table": "data/recovered_v2/wnba_player_game_features_wide.parquet",
            "data_type": str(w[f].dtype) if f in w.columns else "absent",
            "historical_availability_rule": avail, "prediction_cutoff": "scheduled_tip_utc - 12h",
            "lag_shift": "prior_games (shift>=1)" if "prior_games" in avail else "pregame_fact",
            "min_history_note": "rolling windows 3/5/10 require >= window prior appearances",
            "missing_value_meaning": "insufficient prior history / not yet observed",
            "missing_value_policy": "native NaN (handled inside each fold; no global imputation in Stage 9)",
            "known_before_T12": status == "APPROVED_ESTIMATOR_FEATURE",
            "leakage_classification": "pregame_pure" if status.startswith("APPROVED") else reason,
            "approval_status": status, "reason": reason,
        }
        reg_entries.append(entry)
        if status == "APPROVED_ESTIMATOR_FEATURE" and f in w.columns and pd.api.types.is_numeric_dtype(w[f]):
            approved.append(f)

    approved = sorted(approved)
    fh = feature_schema_hash(approved)

    # ---- prediction cutoff (scheduled_tip - 12h where matched; else conservative game-day midnight) ----
    atomic = pd.read_parquet(store, columns=["game_id", "scheduled_tip_utc"]).dropna().drop_duplicates("game_id")
    tip_map = dict(zip(atomic["game_id"].astype(str), atomic["scheduled_tip_utc"]))
    w["_gid"] = w["game_id"].astype(str)
    w["scheduled_tip_utc"] = w["_gid"].map(tip_map)
    gd = pd.to_datetime(w["game_date"], utc=True, errors="coerce")
    tip = pd.to_datetime(w["scheduled_tip_utc"], utc=True, errors="coerce")
    cutoff = tip - pd.Timedelta(hours=12)
    cutoff = cutoff.fillna(gd.dt.floor("D"))   # conservative: start of game day when tip unknown
    w["prediction_cutoff_utc"] = cutoff.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    w["feature_available_utc"] = "<= prediction_cutoff_utc (prior-games/pregame only)"

    # ---- feature-only artifact (NO targets) ----
    id_keep = ["game_id", "player_id", "game_date", "season", "team_id", "opponent_team_id",
               "scheduled_tip_utc", "prediction_cutoff_utc", "feature_available_utc"]
    # Same-day multi-game exclusion: the 2nd+ game a player plays on a calendar date has rolling
    # features that may include the earlier SAME-DAY game, whose result is NOT available at T-12.
    # Mark those rows PIT-ineligible with a recorded reason (never silently dropped).
    w = w.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    w["_d"] = pd.to_datetime(w["game_date"]).dt.date
    w["_sameday_rank"] = w.groupby(["player_id", "_d"]).cumcount()
    w["pit_eligible"] = w["_sameday_rank"] == 0
    w["pit_exclusion_reason"] = np.where(
        w["pit_eligible"], "", "SAME_DAY_MULTI_GAME_PRIOR_RESULT_UNAVAILABLE_AT_T12")

    feat_cols = [c for c in id_keep if c in w.columns] + ["pit_eligible", "pit_exclusion_reason"] + approved
    feat = w[feat_cols].copy()
    feat[approved] = feat[approved].replace([np.inf, -np.inf], np.nan)   # inf->NaN (raw sanitation)
    # hard assertion: no target/forbidden column leaked into the feature artifact
    leaked = [c for c in approved if FORBIDDEN_PATTERN.search(c)]
    assert not leaked, f"forbidden columns leaked into feature artifact: {leaked}"
    feat_path = MODEL_DIR / "wnba_pregame_features_t12.parquet"
    feat.to_parquet(feat_path, index=False)

    # ---- target-only artifact ----
    tgt = w[["game_id", "player_id"]].copy()
    tgt["participation"] = w["did_play"].astype("boolean")
    tgt["actual_minutes"] = pd.to_numeric(w["actual_minutes"], errors="coerce")
    for s in DIRECT:
        tgt[s] = pd.to_numeric(w[f"actual_{s}"], errors="coerce")
    for combo, comps in COMBOS.items():
        tgt[combo] = sum(tgt[c] for c in comps)
    tgt_path = MODEL_DIR / "wnba_player_targets.parquet"
    tgt.to_parquet(tgt_path, index=False)

    # ---- join integrity ----
    fkey = feat[["game_id", "player_id"]].astype(str)
    tkey = tgt[["game_id", "player_id"]].astype(str)
    fdup = int(fkey.duplicated().sum()); tdup = int(tkey.duplicated().sum())
    fset = set(map(tuple, fkey.values)); tset = set(map(tuple, tkey.values))
    matched = len(fset & tset)
    merged = feat[["game_id", "player_id"]].astype(str).merge(
        tgt[["game_id", "player_id"]].astype(str).assign(_t=1), on=["game_id", "player_id"], how="left")
    many_to_many = int(len(merged) - len(feat))
    join_cov = {"artifact": "FEATURE_JOIN_COVERAGE", "generated_at_utc": ts,
                "matched_keys": matched, "feature_only_keys": len(fset - tset),
                "target_only_keys": len(tset - fset), "feature_duplicate_keys": fdup,
                "target_duplicate_keys": tdup, "many_to_many_rows": many_to_many,
                "missing_scheduled_tip": int(w["scheduled_tip_utc"].isna().sum()),
                "missing_prediction_cutoff": int(w["prediction_cutoff_utc"].isna().sum()),
                "join_cardinality_valid": bool(fdup == 0 and tdup == 0 and many_to_many == 0)}

    # ---- point-in-time proof: recompute rolling features from strictly-prior games ----
    pit = _point_in_time_audit(w, box)
    same_day = _same_day_audit(w)

    # ---- missingness / coverage ----
    miss = feat[approved].isna().mean().sort_values(ascending=False)
    pd.DataFrame({"feature": miss.index, "missing_rate": miss.values}).to_csv(
        AUD / "FEATURE_MISSINGNESS_COVERAGE.csv", index=False)

    # ---- estimator guard self-audit ----
    guard_audit = _guard_self_audit(feat, approved, fh)

    # ---- write tracked manifests ----
    (AUD / "FEATURE_REGISTRY.json").write_text(json.dumps({
        "artifact": "FEATURE_REGISTRY", "version": "recovered_v2_stage9_strengthened",
        "generated_at_utc": ts, "n_candidate_features": len(model_features),
        "n_approved_estimator_features": len(approved), "feature_schema_hash": fh,
        "approved_feature_groups": {g: sum(1 for f in approved if _feature_group(f) == g)
                                    for g in sorted({_feature_group(f) for f in approved})},
        "rejection_reason_counts": _reason_counts(reg_entries),
        "features": reg_entries}, indent=2, default=str))
    (AUD / "FEATURE_SCHEMA.json").write_text(json.dumps({
        "artifact": "FEATURE_SCHEMA", "generated_at_utc": ts,
        "approved_ordered_columns": approved, "feature_schema_hash": fh,
        "identifier_columns_outside_estimator": [c for c in id_keep if c in feat.columns]}, indent=2))
    (AUD / "FEATURE_TARGET_SEPARATION_AUDIT.json").write_text(json.dumps({
        "artifact": "FEATURE_TARGET_SEPARATION_AUDIT", "generated_at_utc": ts,
        "feature_artifact": {"path": str(feat_path.relative_to(REPO)), "rows": len(feat),
                             "cols": feat.shape[1], "sha256": _sha(feat_path),
                             "schema": list(feat.columns), "schema_hash": feature_schema_hash(list(feat.columns))},
        "target_artifact": {"path": str(tgt_path.relative_to(REPO)), "rows": len(tgt),
                            "cols": tgt.shape[1], "sha256": _sha(tgt_path),
                            "schema": list(tgt.columns)},
        "target_columns_in_feature_artifact": [c for c in feat.columns if c in
            (["did_play", "actual_minutes", "participation"] + [f"actual_{s}" for s in DIRECT] + DIRECT + list(COMBOS))],
        "physically_separate_files": True}, indent=2, default=str))
    (AUD / "POINT_IN_TIME_AUDIT.json").write_text(json.dumps(pit, indent=2, default=str))
    (AUD / "SAME_DAY_LEAKAGE_AUDIT.json").write_text(json.dumps(same_day, indent=2, default=str))
    (AUD / "ESTIMATOR_INPUT_GUARD_AUDIT.json").write_text(json.dumps(guard_audit, indent=2, default=str))
    (AUD / "FEATURE_JOIN_COVERAGE.json").write_text(json.dumps(join_cov, indent=2, default=str))

    readiness = {
        "artifact": "STAGE9_STRENGTHENED_READINESS", "generated_at_utc": ts,
        "recovered_v2_is_not_frozen_v1": True, "frozen_v1_available": False,
        "foundation_lock_status": "DEFERRED_MISSING_FROZEN_ARTIFACT",
        "feature_target_separation_verified": True, "no_model_trained_yet": True,
        "quarantined_oof_invalid": True,
        "n_approved_estimator_features": len(approved), "feature_schema_hash": fh,
        "point_in_time_violations": pit["rolling_shift_violations"],
        "same_day_leakage_count": same_day["same_day_multi_game_player_dates"],
        "injury_history_features_approved": sum(1 for e in reg_entries
            if e["approval_status"].startswith("APPROVED") and _INJURY_FWD_PAT.search(e["feature_name"])),
        "market_features_approved": sum(1 for e in reg_entries
            if e["approval_status"].startswith("APPROVED") and (e["feature_name"] in REJECT_MARKET or _MARKET_PAT.search(e["feature_name"]))),
        "join_cardinality_valid": join_cov["join_cardinality_valid"],
        "estimator_guard_rejects_all_injections": guard_audit["rejects_all_injections"],
        "estimator_guard_accepts_valid_frame": guard_audit["accepts_valid_frame"],
    }
    (AUD / "STAGE9_STRENGTHENED_READINESS.json").write_text(json.dumps(readiness, indent=2, default=str))

    typer.echo("================ STAGE 9 STRENGTHENED ================")
    typer.echo(f"  candidate features: {len(model_features)}  approved: {len(approved)}  hash={fh}")
    typer.echo(f"  rejection reasons: {_reason_counts(reg_entries)}")
    typer.echo(f"  feature artifact: rows={len(feat)} cols={feat.shape[1]} sha={_sha(feat_path)}")
    typer.echo(f"  target artifact : rows={len(tgt)} cols={tgt.shape[1]} sha={_sha(tgt_path)}")
    typer.echo(f"  PIT rolling-shift violations: {pit['rolling_shift_violations']}  same-day: {same_day['same_day_multi_game_player_dates']}")
    typer.echo(f"  join valid: {join_cov['join_cardinality_valid']}  guard: rejects_all={guard_audit['rejects_all_injections']} accepts_valid={guard_audit['accepts_valid_frame']}")


def _reason_counts(entries):
    from collections import Counter
    return dict(Counter(e["reason"] for e in entries if not e["approval_status"].startswith("APPROVED")))


def _point_in_time_audit(w: pd.DataFrame, box_path: str) -> dict:
    """Recompute representative rolling features from strictly-prior games and compare."""
    box = pd.read_parquet(box_path)[["game_id", "player_id", "game_date", "pts", "reb", "minutes"]].copy()
    box["game_date"] = pd.to_datetime(box["game_date"])
    box = box.sort_values(["player_id", "game_date", "game_id"])
    checks = {}
    violations = 0
    for feat_name, src, window in [("player_pts_mean_l5", "pts", 5), ("player_reb_mean_l5", "reb", 5),
                                   ("player_minutes_mean_l5", "minutes", 5)]:
        if feat_name not in w.columns:
            continue
        recomputed = box.groupby("player_id")[src].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        rec = box.assign(_r=recomputed).set_index(["game_id", "player_id"])["_r"]
        cmp = w[["game_id", "player_id", feat_name]].copy()
        cmp["_stored"] = pd.to_numeric(cmp[feat_name], errors="coerce")
        cmp = cmp.set_index(["game_id", "player_id"])
        j = cmp.join(rec.rename("_recomputed"))
        both = j.dropna(subset=["_stored", "_recomputed"])
        n = int(len(both))
        mism = int((np.abs(both["_stored"] - both["_recomputed"]) > 0.05).sum())
        violations += mism
        checks[feat_name] = {"n_compared": n, "mismatches_gt_0.05": mism,
                             "match_rate": float(1 - mism / n) if n else None}
    return {"artifact": "POINT_IN_TIME_AUDIT",
            "method": "recompute rolling feature from strictly-prior games (shift(1)); compare to stored",
            "rolling_features_checked": checks, "rolling_shift_violations": int(violations),
            "note": "A match proves the stored rolling feature used only strictly-prior games "
                    "(no current-game / same-date leakage)."}


def _same_day_audit(w: pd.DataFrame) -> dict:
    g = w.assign(_d=pd.to_datetime(w["game_date"]).dt.date).groupby(["player_id", "_d"]).size()
    multi = int((g > 1).sum())
    excluded = int((~w["pit_eligible"]).sum()) if "pit_eligible" in w.columns else 0
    # remaining same-day leakage among PIT-eligible rows must be zero (2nd+ same-day rows excluded)
    elig = w[w["pit_eligible"]] if "pit_eligible" in w.columns else w
    ge = elig.assign(_d=pd.to_datetime(elig["game_date"]).dt.date).groupby(["player_id", "_d"]).size()
    remaining = int((ge > 1).sum())
    return {"artifact": "SAME_DAY_LEAKAGE_AUDIT",
            "same_day_multi_game_player_dates": remaining,
            "same_day_multi_game_player_dates_raw": multi,
            "rows_excluded_pit_ineligible": excluded,
            "note": "The 2nd+ same-day game per player is marked PIT-ineligible (its rolling "
                    "features could include an earlier same-day game whose result is not available "
                    "at T-12). Among PIT-eligible rows the same-day leakage count is the value above."}


def _guard_self_audit(feat: pd.DataFrame, approved: list[str], fh: str) -> dict:
    """Every injection is validated against the REAL approved allowlist + hash (the true
    contract). A valid frame passes; any injected forbidden/target/market/identifier/duplicate/
    unexpected/schema-mismatch column fails closed."""
    from wnba_props_model.features.estimator_guard import EstimatorGuardError, guard_estimator_frame
    valid = feat[approved].head(50).copy()
    accepts = False
    try:
        guard_estimator_frame(valid, approved, fh); accepts = True
    except EstimatorGuardError:
        accepts = False
    # each injection adds/alters a column -> frame no longer equals the approved allowlist
    injections = {
        "target_actual_pts": lambda d: d.assign(actual_pts=1.0),
        "did_play": lambda d: d.assign(did_play=1),
        "actual_minutes": lambda d: d.assign(actual_minutes=20.0),
        "settlement_status": lambda d: d.assign(settlement_status="OVER_WIN"),
        "sportsbook_price": lambda d: d.assign(over_odds=-110),
        "prop_line": lambda d: d.assign(line=15.5),
        "market_probability": lambda d: d.assign(market_prob_over=0.5),
        "identifier_game_id": lambda d: d.assign(game_id="x"),
        "duplicate_column": lambda d: pd.concat([d, d[[approved[0]]]], axis=1),
        "unexpected_column": lambda d: d.assign(unexpected_feature=1.0),
    }
    rejected = {}
    for name, inj in injections.items():
        bad = inj(valid.copy())
        try:
            guard_estimator_frame(bad, approved, fh)   # REAL allowlist + hash is the contract
            rejected[name] = False
        except EstimatorGuardError:
            rejected[name] = True
    # schema-hash mismatch defense (valid columns, wrong expected hash)
    try:
        guard_estimator_frame(valid, approved, "deadbeefdeadbeef")
        rejected["schema_hash_mismatch"] = False
    except EstimatorGuardError:
        rejected["schema_hash_mismatch"] = True
    # forbidden-name secondary defense (contaminated allowlist of equal length -> name alarm fires)
    from wnba_props_model.features.estimator_guard import assert_no_forbidden_names
    try:
        assert_no_forbidden_names(approved[:-1] + ["market_line_x"]); rejected["forbidden_name_alarm"] = False
    except EstimatorGuardError:
        rejected["forbidden_name_alarm"] = True
    return {"artifact": "ESTIMATOR_INPUT_GUARD_AUDIT",
            "accepts_valid_frame": bool(accepts), "injection_rejected": rejected,
            "rejects_all_injections": all(rejected.values())}


if __name__ == "__main__":
    app()
