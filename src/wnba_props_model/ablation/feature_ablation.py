"""Leakage-safe, nested-CV feature-selection / ablation harness for WNBA player props.

This module assembles a BROAD union of strictly-lagged (as-of strictly before
tip) candidate features, organised into named GROUPS, and runs a rigorous
per-prop study:

* nested rolling-origin (expanding-window) CV -- feature SELECTION and any
  tuning happen inside the inner folds of each outer fold and never see the
  outer evaluation block (enforced by :func:`assert_nested_cv_integrity`);
* feature IMPORTANCE via permutation (gradient-boosted model) and an
  L1-regularised path;
* group ABLATION -- leave-one-group-out and only-one-group;
* SELECTION -- greedy forward + L1, consensus across outer folds;
* final OUTER-OOF evaluation reusing the frozen market-superiority scores
  (LogLoss / Brier / AUC / ECE / calibration + paired date-cluster bootstrap +
  Holm) for the market-evaluable props, and proper count scores
  (Poisson deviance / PMF log-score / CRPS) for the outcome-only props.

Feature GROUPS
--------------
``player_pbp_rate``   strictly-lagged per-player EWMA PBP rates (pbp_features).
``player_box_form``   rolling box aggregates / usage / shooting / form.
``opponent_defense``  opponent prior-game allowed rates (+ built oppdef_*).
``pace_env``          team & opponent pace / possession proxies.
``schedule``          rest, back-to-back, density, home/away, season phase.
``role``              minutes level/trend, starter proxy, rotation shape.
``dispersion``        rolling variance / CV of the stat and minutes.

Forward-only features (require tonight's availability / lineup / game script)
are NOT historically constructible and are excluded from modeling; they are
recorded separately as recommended-but-requires-live-availability. Market-
derived columns are also excluded so the study measures NON-market signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.pmf_builders import (
    poisson_or_nbinom_pmf,
    settled_over_probability,
)

from . import metrics as M
from .opponent_defense import OppDefConfig, build_opponent_defense_features

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for the ablation harness") from e

MARKET_PROPS = ("pts", "reb", "ast", "fg3m")
COUNT_PROPS = ("stl", "blk", "tov")
ALL_PROPS = MARKET_PROPS + COUNT_PROPS

# prop -> wide/box column stem
_WIDE_STEM = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m",
              "stl": "stl", "blk": "blk", "tov": "turnover"}

GROUP_NAMES = ("player_pbp_rate", "player_box_form", "opponent_defense",
               "pace_env", "schedule", "role", "dispersion")

# Columns that are current-game outcomes / identifiers / metadata -> never features.
_DROP_EXACT = {
    "player_id", "player_name", "team_id", "team_abbreviation", "game_id", "game_date",
    "season", "position", "actual_minutes", "minutes", "minutes_raw", "minutes_flag",
    "actual_pts", "actual_reb", "actual_ast", "actual_fg3m", "actual_turnover",
    "actual_stl", "actual_blk", "oreb", "dreb", "fga", "fg3a", "fta", "pf", "plus_minus",
    "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast", "stocks", "source", "pull_timestamp_utc",
    "home_away", "opponent_team_id", "opponent_team_abbreviation", "did_play",
    "started_proxy", "player_is_confirmed_starter", "zero_minute_flag", "non_playing_flag",
    "stat_line_all_zero_flag", "missing_team_flag", "missing_opponent_flag",
    "missing_game_date_flag", "feature_build_timestamp_utc", "split",
    "def_team_id", "oppdef_games_prior",
}

# Forward-only: require tonight's availability / lineup / game script (NOT historical).
_FORWARD_ONLY_PATTERNS = [
    r"teammate_\d+_is_out", r"teammate_\d+_usage_rate", r"without_\d+_.*_delta",
    r"projected_usage_given_absences", r"usage_transfer_delta", r"teammate_injury_flag",
    r"team_top3_scorers_available", r"expected_minutes_given_script", r"minutes_upside",
    r"usage_share_delta", r"vacated_minutes_l1", r"vacated_pts_l1", r"player_role_elevation",
    r"player_injured_l1",
]
# Market-derived: excluded so the study isolates NON-market signal.
_MARKET_PATTERNS = [
    r"player_market_.*", r"player_line_movement_prev", r"pregame_win_probability",
    r"blowout_probability", r"close_game_probability",
]

# Ordered (regex -> group); first match wins.  Applied to WIDE numeric feature cols.
_GROUP_RULES = [
    (r"pace_proxy", "pace_env"),
    (r"^opp_", "opponent_defense"),
    (r"_def_pi$", "opponent_defense"),
    (r"_vs_opp_", "opponent_defense"),
    (r"^oppdef_.*_allowed_", "opponent_defense"),
    (r"^oppdef_(poss|team_poss)", "pace_env"),
    (r"(rest|back_to_back|_b2b|3in4|4_in_5|5_in_7|games_in_last_7|days_since_last|"
     r"dnp_streak|load_index|cumulative_minutes|schedule_fatigue|timezone|altitude|"
     r"games_prior|games_played_prior|did_play_rate|zero_minute_rate)", "schedule"),
    (r"(season_game_number|game_number_in_season|season_completion_pct|is_playoff_game|"
     r"season_phase|^is_home$|^rest_days$|^is_back_to_back$)", "schedule"),
    (r"(_std_l|_std_$|_std_|volatility|form_vs_season_ratio|"
     r"rotation_minutes_(std|q\d|bimodal|p_over|p_under))", "dispersion"),
    (r"^team_", "pace_env"),
    (r"(starter|projected_minutes|rotation_minutes|_minutes_last|_minutes_mean|"
     r"_minutes_median|_minutes_min|_minutes_max|_minutes_support|_minutes_ewma|"
     r"_minutes_form|_minutes_momentum|_minutes_zscore|_minutes_season|usage_bucket|"
     r"^role|player_usage_proxy)", "role"),
    (r"^player_minutes", "role"),
]


def _matches_any(name: str, patterns) -> bool:
    return any(re.search(p, name) for p in patterns)


# --------------------------------------------------------------------------- #
# frame assembly
# --------------------------------------------------------------------------- #
@dataclass
class AblationConfig:
    n_outer: int = 5
    min_train_dates: int = 12
    n_inner: int = 3
    min_inner_train_dates: int = 6
    prerank_top_k: int = 25
    forward_max_features: int = 12
    forward_tol: float = 1e-4
    perm_repeats: int = 5
    bootstrap_iters: int = 3000
    seed: int = 20260728
    min_history_games: int = 3
    hgb_max_iter: int = 160
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 20
    consensus_frac: float = 0.5
    # data paths
    quotes_path: str = "artifacts/market_feature_proof/G0_v2/PRIMARY_DETERMINISTIC_SCORED_ROWS.parquet"
    wide_path: str = "data/processed/wnba_player_game_features_wide.recovered_v2_20260725.parquet"
    pbp_feats_path: str = "data/processed/wnba_pbp_opportunity_features.parquet"
    box_path: str = "data/processed/wnba_player_game_stats.parquet"
    stlblktov_path: str = "data/processed/wnba_stlblktov_labels.parquet"
    # provenance of the wide/box-form feature source, recorded verbatim in every
    # per-prop artifact so the study is honest about whether the license-restricted
    # ``player_box_form`` group was available on the machine that ran it.
    player_box_form_status: str = (
        "INCLUDED_recovered_v2_license_restricted_wide_features")


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    out = []
    for c in num:
        if c in _DROP_EXACT:
            continue
        if _matches_any(c, _FORWARD_ONLY_PATTERNS) or _matches_any(c, _MARKET_PATTERNS):
            continue
        out.append(c)
    return out


def assign_groups(columns: list[str], pbp_cols: list[str]) -> dict[str, list[str]]:
    """Assign feature columns to groups. ``pbp_cols`` -> player_pbp_rate; the rest
    via the ordered rule list, default player_box_form."""
    groups: dict[str, list[str]] = {g: [] for g in GROUP_NAMES}
    groups["player_pbp_rate"] = list(pbp_cols)
    for c in columns:
        if c in pbp_cols:
            continue
        assigned = None
        for pat, grp in _GROUP_RULES:
            if re.search(pat, c):
                assigned = grp
                break
        groups[assigned or "player_box_form"].append(c)
    return {g: sorted(set(cols)) for g, cols in groups.items()}


def audit_forward_only_and_market(wide_cols: list[str]) -> dict:
    fwd = sorted([c for c in wide_cols if _matches_any(c, _FORWARD_ONLY_PATTERNS)])
    mkt = sorted([c for c in wide_cols if _matches_any(c, _MARKET_PATTERNS)])
    return {"forward_only_excluded": fwd, "market_derived_excluded": mkt}


def assemble_frame(prop: str, cfg: AblationConfig, *,
                   wide: pd.DataFrame, pbp: pd.DataFrame, box: pd.DataFrame,
                   quotes: pd.DataFrame | None, stlblktov: pd.DataFrame | None,
                   oppdef: pd.DataFrame):
    """Build the modeling frame + group map for one prop.

    Returns (frame, groups, kind, meta). ``kind`` in {"binary","count"}.
    """
    stem = _WIDE_STEM[prop]
    kind = "binary" if prop in MARKET_PROPS else "count"

    for c in ("game_id", "player_id"):
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce")

    # pbp features (prefix to avoid clashing with wide naming); keep only rate/pct cols
    pbp_rate_cols = [c for c in pbp.columns
                     if c.startswith("player_") and ("ewma" in c or c == "player_fg3_pct_prior")]
    pbp_small = pbp[["game_id", "player_id", "player_games_played_prior"] + pbp_rate_cols].copy()
    ren = {c: f"pbp_{c[len('player_'):]}" for c in pbp_rate_cols}
    pbp_small = pbp_small.rename(columns=ren)
    pbp_feat_names = list(ren.values())

    # wide numeric features
    wide_feat_cols = _numeric_feature_columns(wide)
    wide_small = wide[["game_id", "player_id"] + wide_feat_cols].drop_duplicates(["game_id", "player_id"])

    # target rows
    if kind == "binary":
        q = quotes[quotes["prop"] == prop].copy()
        for c in ("game_id", "player_id"):
            q[c] = pd.to_numeric(q[c], errors="coerce")
        q = q[q["binary_score_eligible"].astype(bool) & q["outcome_over"].isin([0, 1])].copy()
        base = q[["game_id", "player_id", "game_date", "prop", "line", "outcome_over",
                  "actual", "market_prob_over_no_vig", "model_prob_over_final", "oof_fold"]].rename(
            columns={"market_prob_over_no_vig": "market", "model_prob_over_final": "p0_delivered"})
        base["y"] = base["outcome_over"].astype(int)
    else:
        lab = stlblktov.copy()
        for c in ("game_id", "player_id"):
            lab[c] = pd.to_numeric(lab[c], errors="coerce")
        lab["game_date"] = pd.to_datetime(lab["game_date"], errors="coerce")
        lab = lab[lab["did_play"].astype("boolean").fillna(False)]
        base = lab[["game_id", "player_id", "game_date", prop]].rename(columns={prop: "y"})
        base["actual"] = base["y"].astype(float)
        base["prop"] = prop

    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")
    frame = (base
             .merge(pbp_small, on=["game_id", "player_id"], how="left")
             .merge(wide_small, on=["game_id", "player_id"], how="left")
             .merge(oppdef.drop(columns=[c for c in ("team_id", "opponent_team_id", "game_date")
                                         if c in oppdef.columns], errors="ignore"),
                    on=["game_id", "player_id"], how="left"))

    # minimum-history filter (needs prior games to have any lagged signal)
    if "player_games_played_prior" in frame.columns:
        frame = frame[frame["player_games_played_prior"].fillna(0) >= cfg.min_history_games].copy()

    frame = frame.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)

    # groups over all assembled feature columns
    oppdef_cols = [c for c in oppdef.columns if c.startswith("oppdef_")]
    all_feat = sorted(set(pbp_feat_names + wide_feat_cols + oppdef_cols))
    all_feat = [c for c in all_feat if c in frame.columns]
    # drop degenerate features (constant / all-NaN) -- HGB binning requires >=2 distinct values
    degenerate = [c for c in all_feat if frame[c].nunique(dropna=True) < 2]
    all_feat = [c for c in all_feat if c not in degenerate]
    pbp_feat_names = [c for c in pbp_feat_names if c in all_feat]
    groups = assign_groups(all_feat, pbp_feat_names)
    groups = {g: [c for c in cols if c in frame.columns] for g, cols in groups.items()}

    meta = {"n_rows": int(len(frame)),
            "n_dates": int(frame["game_date"].nunique()),
            "pbp_feature_names": pbp_feat_names,
            "oppdef_feature_names": sorted([c for c in oppdef_cols if c in all_feat]),
            "degenerate_dropped": sorted(degenerate),
            "n_candidate_features": len(all_feat)}
    return frame, groups, kind, meta


# --------------------------------------------------------------------------- #
# nested rolling-origin folds
# --------------------------------------------------------------------------- #
def make_expanding_folds(dates_sorted: np.ndarray, n_folds: int, min_train_dates: int):
    """Expanding-window folds over the sorted unique dates.

    Returns list of (train_dates, val_dates) tuples; train dates are strictly
    earlier than the fold's val dates. Folds with too little history are skipped.
    """
    uniq = np.array(sorted(set(pd.to_datetime(dates_sorted).tolist())))
    n = len(uniq)
    if n <= min_train_dates + 1:
        return []
    tail = uniq[min_train_dates:]
    blocks = np.array_split(tail, n_folds)
    folds = []
    for blk in blocks:
        if len(blk) == 0:
            continue
        vstart = blk[0]
        train = uniq[uniq < vstart]
        if len(train) < min_train_dates:
            continue
        folds.append((set(train.tolist()), set(blk.tolist())))
    return folds


def assert_nested_cv_integrity(dates: np.ndarray, cfg: AblationConfig) -> dict:
    """Verify that (1) every outer fold's train dates precede its val dates, and
    (2) every inner fold's train+val dates are a subset of the outer train dates
    (selection therefore can never see the outer eval block). Raises on violation.
    """
    outer = make_expanding_folds(dates, cfg.n_outer, cfg.min_train_dates)
    if not outer:
        raise AssertionError("no outer folds constructed (insufficient dates)")
    checks = 0
    for tr, va in outer:
        if max(tr) >= min(va):
            raise AssertionError("outer fold leakage: max(train_date) >= min(val_date)")
        inner = make_expanding_folds(np.array(sorted(tr)), cfg.n_inner, cfg.min_inner_train_dates)
        for itr, iva in inner:
            if not (itr | iva).issubset(tr):
                raise AssertionError("inner fold escaped outer-train dates (selection leakage)")
            if max(itr) >= min(iva):
                raise AssertionError("inner fold leakage: max(train_date) >= min(val_date)")
            if iva & va:
                raise AssertionError("inner val overlaps outer val (selection saw eval fold)")
            checks += 1
    return {"outer_folds": len(outer), "inner_checks": checks, "integrity_ok": True}


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _hgb_clf(cfg: AblationConfig, seed: int):
    return HistGradientBoostingClassifier(
        max_iter=cfg.hgb_max_iter, learning_rate=cfg.hgb_learning_rate,
        max_leaf_nodes=cfg.hgb_max_leaf_nodes, min_samples_leaf=cfg.hgb_min_samples_leaf,
        l2_regularization=1.0, early_stopping=False, random_state=seed)


def _hgb_poisson(cfg: AblationConfig, seed: int):
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=cfg.hgb_max_iter, learning_rate=cfg.hgb_learning_rate,
        max_leaf_nodes=cfg.hgb_max_leaf_nodes, min_samples_leaf=cfg.hgb_min_samples_leaf,
        l2_regularization=1.0, early_stopping=False, random_state=seed)


def _fit_predict(frame, feat_cols, tr_idx, va_idx, kind, cfg, seed):
    X = frame[feat_cols].to_numpy(dtype=float)
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr = frame["y"].to_numpy()[tr_idx]
    if kind == "binary":
        if len(np.unique(ytr)) < 2:
            return np.full(len(va_idx), float(ytr.mean()))
        m = _hgb_clf(cfg, seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    m = _hgb_poisson(cfg, seed)
    m.fit(Xtr, np.clip(ytr, 0, None))
    return np.clip(m.predict(Xva), 1e-6, None)


def _mom_dispersion(actual: np.ndarray) -> float | None:
    a = np.asarray(actual, float)
    m, v = float(np.mean(a)), float(np.var(a))
    if m <= 1e-9 or v <= m:
        return None
    return float(np.clip(m * m / (v - m), 0.5, 500.0))


# --------------------------------------------------------------------------- #
# OOF prediction for a fixed feature spec (used by ablation + baselines)
# --------------------------------------------------------------------------- #
def oof_predict(frame, feat_cols, outer_folds, kind, cfg):
    """Produce outer-OOF predictions for a fixed feature set. Returns a dict with
    row index -> prediction, plus fold id, plus (for counts) predicted mean +
    train dispersion so a PMF can be built downstream."""
    date = frame["game_date"]
    preds = np.full(len(frame), np.nan)
    fold_id = np.full(len(frame), -1)
    disp = np.full(len(frame), np.nan)
    for k, (tr_dates, va_dates) in enumerate(outer_folds):
        tr_idx = np.where(date.isin(tr_dates).to_numpy())[0]
        va_idx = np.where(date.isin(va_dates).to_numpy())[0]
        if len(tr_idx) < 30 or len(va_idx) == 0:
            continue
        p = _fit_predict(frame, feat_cols, tr_idx, va_idx, kind, cfg, cfg.seed + k)
        preds[va_idx] = p
        fold_id[va_idx] = k
        if kind == "count":
            disp[va_idx] = _mom_dispersion(frame["y"].to_numpy()[tr_idx]) or np.nan
    mask = ~np.isnan(preds)
    return {"pred": preds, "fold": fold_id, "disp": disp, "mask": mask}


# --------------------------------------------------------------------------- #
# importance: permutation (GBM) + L1 path
# --------------------------------------------------------------------------- #
def _loss(y, p, kind):
    return M.log_loss(y, p) if kind == "binary" else M.poisson_deviance(y, p)


def permutation_importance(frame, feat_cols, outer_folds, kind, cfg):
    """Permutation importance aggregated across outer folds (permute on the val
    block only). Importance = mean increase in loss when a feature is shuffled."""
    rng = np.random.default_rng(cfg.seed)
    imp = {c: [] for c in feat_cols}
    date = frame["game_date"]
    y_all = frame["y"].to_numpy()
    for k, (tr_dates, va_dates) in enumerate(outer_folds):
        tr_idx = np.where(date.isin(tr_dates).to_numpy())[0]
        va_idx = np.where(date.isin(va_dates).to_numpy())[0]
        if len(tr_idx) < 30 or len(va_idx) < 5:
            continue
        yva = y_all[va_idx]
        if kind == "binary" and len(np.unique(y_all[tr_idx])) < 2:
            continue
        X = frame[feat_cols].to_numpy(dtype=float)
        Xtr = X[tr_idx]
        if kind == "binary":
            mdl = _hgb_clf(cfg, cfg.seed + k)
            mdl.fit(Xtr, y_all[tr_idx])
            base_p = mdl.predict_proba(X[va_idx])[:, 1]
        else:
            mdl = _hgb_poisson(cfg, cfg.seed + k)
            mdl.fit(Xtr, np.clip(y_all[tr_idx], 0, None))
            base_p = np.clip(mdl.predict(X[va_idx]), 1e-6, None)
        base_loss = _loss(yva, base_p, kind)
        Xva = X[va_idx].copy()
        for j, c in enumerate(feat_cols):
            deltas = []
            for _ in range(cfg.perm_repeats):
                saved = Xva[:, j].copy()
                Xva[:, j] = rng.permutation(saved)
                if kind == "binary":
                    pp = mdl.predict_proba(Xva)[:, 1]
                else:
                    pp = np.clip(mdl.predict(Xva), 1e-6, None)
                Xva[:, j] = saved
                deltas.append(_loss(yva, pp, kind) - base_loss)
            imp[c].append(float(np.mean(deltas)))
    return {c: float(np.mean(v)) if v else 0.0 for c, v in imp.items()}


def l1_importance(frame, feat_cols, outer_folds, kind, cfg):
    """L1 path importance: selection frequency + mean |standardized coef| across
    outer-fold TRAIN sets at a moderate penalty."""
    freq = {c: 0 for c in feat_cols}
    coefmag = {c: [] for c in feat_cols}
    date = frame["game_date"]
    y_all = frame["y"].to_numpy()
    nfolds = 0
    for k, (tr_dates, _va) in enumerate(outer_folds):
        tr_idx = np.where(date.isin(tr_dates).to_numpy())[0]
        if len(tr_idx) < 40:
            continue
        X = frame[feat_cols].to_numpy(dtype=float)
        Xtr = np.nan_to_num(X[tr_idx], nan=0.0)
        ytr = y_all[tr_idx]
        sc = StandardScaler().fit(Xtr)
        Xs = sc.transform(Xtr)
        if kind == "binary":
            if len(np.unique(ytr)) < 2:
                continue
            mdl = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.05,
                                     max_iter=2000, random_state=cfg.seed + k)
            mdl.fit(Xs, ytr)
            coef = mdl.coef_[0]
        else:
            mdl = Lasso(alpha=0.02, max_iter=5000, random_state=cfg.seed + k)
            mdl.fit(Xs, np.log1p(np.clip(ytr, 0, None)))
            coef = mdl.coef_
        nfolds += 1
        for c, w in zip(feat_cols, coef):
            coefmag[c].append(abs(float(w)))
            if abs(w) > 1e-8:
                freq[c] += 1
    return {c: {"select_freq": (freq[c] / nfolds if nfolds else 0.0),
                "mean_abs_coef": float(np.mean(coefmag[c])) if coefmag[c] else 0.0}
            for c in feat_cols}


# --------------------------------------------------------------------------- #
# selection: greedy forward (inner CV) + L1, consensus across outer folds
# --------------------------------------------------------------------------- #
def _prerank(frame, feat_cols, tr_idx, kind, cfg, seed):
    """Cheap pre-rank of candidate features by standardized L1 |coef| on train."""
    X = np.nan_to_num(frame[feat_cols].to_numpy(dtype=float)[tr_idx], nan=0.0)
    y = frame["y"].to_numpy()[tr_idx]
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return feat_cols[: cfg.prerank_top_k]
        mdl = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.1,
                                 max_iter=2000, random_state=seed)
        mdl.fit(Xs, y)
        mag = np.abs(mdl.coef_[0])
    else:
        mdl = Lasso(alpha=0.01, max_iter=5000, random_state=seed)
        mdl.fit(Xs, np.log1p(np.clip(y, 0, None)))
        mag = np.abs(mdl.coef_)
    order = np.argsort(-mag)
    return [feat_cols[i] for i in order[: cfg.prerank_top_k]]


def _inner_cv_loss(frame, feat_cols, tr_dates, kind, cfg):
    """Mean inner-val loss for a feature set, using inner expanding folds of the
    outer-train dates only."""
    inner = make_expanding_folds(np.array(sorted(tr_dates)), cfg.n_inner, cfg.min_inner_train_dates)
    if not inner:
        return float("inf")
    date = frame["game_date"]
    losses = []
    for j, (itr, iva) in enumerate(inner):
        tr_idx = np.where(date.isin(itr).to_numpy())[0]
        va_idx = np.where(date.isin(iva).to_numpy())[0]
        if len(tr_idx) < 30 or len(va_idx) < 5:
            continue
        p = _fit_predict(frame, feat_cols, tr_idx, va_idx, kind, cfg, cfg.seed + 100 + j)
        losses.append(_loss(frame["y"].to_numpy()[va_idx], p, kind))
    return float(np.mean(losses)) if losses else float("inf")


def greedy_forward_select(frame, anchors, candidates, tr_dates, kind, cfg):
    """Greedy forward selection inside the inner folds of one outer fold."""
    selected = list(anchors)
    best = _inner_cv_loss(frame, selected, tr_dates, kind, cfg) if selected else float("inf")
    pool = [c for c in candidates if c not in selected]
    improved = True
    while improved and len([s for s in selected if s not in anchors]) < cfg.forward_max_features:
        improved = False
        best_add, best_loss = None, best
        for c in pool:
            loss = _inner_cv_loss(frame, selected + [c], tr_dates, kind, cfg)
            if loss < best_loss - cfg.forward_tol:
                best_loss, best_add = loss, c
        if best_add is not None:
            selected.append(best_add)
            pool.remove(best_add)
            best = best_loss
            improved = True
    return [s for s in selected if s not in anchors], best


def consensus_selection(frame, groups, anchors, kind, cfg, outer_folds):
    """Run forward + L1 selection per outer fold (train only) and take a robust
    consensus. Returns (consensus_features, per_fold_selected, l1_sets)."""
    all_feats = sorted({c for cols in groups.values() for c in cols})
    date = frame["game_date"]
    per_fold, l1_sets, counts = [], [], {}
    per_fold_map: dict[int, list[str]] = {}
    for k, (tr_dates, _va) in enumerate(outer_folds):
        tr_idx = np.where(date.isin(tr_dates).to_numpy())[0]
        if len(tr_idx) < 40:
            continue
        cands = _prerank(frame, all_feats, tr_idx, kind, cfg, cfg.seed + k)
        fwd, _loss_ = greedy_forward_select(frame, anchors, cands, tr_dates, kind, cfg)
        per_fold.append(fwd)
        per_fold_map[k] = fwd
        # L1 nonzero at moderate penalty
        X = np.nan_to_num(frame[all_feats].to_numpy(dtype=float)[tr_idx], nan=0.0)
        y = frame["y"].to_numpy()[tr_idx]
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        if kind == "binary" and len(np.unique(y)) >= 2:
            mdl = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.05,
                                     max_iter=2000, random_state=cfg.seed + k)
            mdl.fit(Xs, y)
            l1 = [all_feats[i] for i in np.where(np.abs(mdl.coef_[0]) > 1e-8)[0]]
        elif kind == "count":
            mdl = Lasso(alpha=0.02, max_iter=5000, random_state=cfg.seed + k)
            mdl.fit(Xs, np.log1p(np.clip(y, 0, None)))
            l1 = [all_feats[i] for i in np.where(np.abs(mdl.coef_) > 1e-8)[0]]
        else:
            l1 = []
        l1_sets.append(l1)
        for c in set(fwd) | set(l1):
            counts[c] = counts.get(c, 0) + 1
    n = max(len(per_fold), 1)
    thresh = int(np.ceil(cfg.consensus_frac * n))
    consensus = sorted([c for c, v in counts.items() if v >= thresh])
    return consensus, per_fold, l1_sets, per_fold_map, {"n_selection_folds": len(per_fold),
                                                        "consensus_threshold_folds": thresh}


def oof_predict_perfold(frame, anchors, per_fold_map, outer_folds, kind, cfg):
    """OOF using a fold-specific selected feature set (honest nested CV): fold k
    is predicted with ANCHOR + the features selected on fold k's train only."""
    date = frame["game_date"]
    preds = np.full(len(frame), np.nan)
    fold_id = np.full(len(frame), -1)
    disp = np.full(len(frame), np.nan)
    for k, (tr_dates, va_dates) in enumerate(outer_folds):
        feats = list(anchors) + list(per_fold_map.get(k, []))
        if not feats:
            feats = list(anchors) if anchors else per_fold_map.get(k, [])
        tr_idx = np.where(date.isin(tr_dates).to_numpy())[0]
        va_idx = np.where(date.isin(va_dates).to_numpy())[0]
        if len(tr_idx) < 30 or len(va_idx) == 0 or not feats:
            continue
        p = _fit_predict(frame, feats, tr_idx, va_idx, kind, cfg, cfg.seed + k)
        preds[va_idx] = p
        fold_id[va_idx] = k
        if kind == "count":
            disp[va_idx] = _mom_dispersion(frame["y"].to_numpy()[tr_idx]) or np.nan
    return {"pred": preds, "fold": fold_id, "disp": disp, "mask": ~np.isnan(preds)}


# --------------------------------------------------------------------------- #
# scoring an OOF result
# --------------------------------------------------------------------------- #
def _binary_metrics(y, p):
    ci_i, sl = M.calibration_intercept_slope(y, p)
    return {"log_loss": M.log_loss(y, p), "brier": M.brier(y, p), "auc": M.auc(y, p),
            "ece": M.expected_calibration_error(y, p),
            "calibration_intercept": ci_i, "calibration_slope": sl}


def _count_pmfs(mean, disp, cap=80):
    out = []
    for mu, r in zip(mean, disp):
        if not np.isfinite(mu):
            out.append(None); continue
        rr = None if (not np.isfinite(r)) else float(r)
        out.append(poisson_or_nbinom_pmf(float(max(mu, 1e-6)), rr, maximum_cap=cap))
    return out


def _count_metrics(y, mean, disp):
    pmfs = _count_pmfs(mean, disp)
    return {"poisson_deviance": M.poisson_deviance(y, mean),
            "pmf_log_score": M.pmf_log_score(pmfs, y),
            "crps": M.crps_discrete(pmfs, y),
            "mean_predicted": float(np.mean(mean)), "mean_actual": float(np.mean(y))}


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #
def load_inputs(cfg: AblationConfig) -> dict:
    wide = pd.read_parquet(cfg.wide_path)
    pbp = pd.read_parquet(cfg.pbp_feats_path)
    box = pd.read_parquet(cfg.box_path)
    quotes = pd.read_parquet(cfg.quotes_path)
    stlblktov = pd.read_parquet(cfg.stlblktov_path)
    if "stat" in quotes.columns and "prop" not in quotes.columns:
        quotes = quotes.rename(columns={"stat": "prop"})
    oppdef = build_opponent_defense_features(box, OppDefConfig())
    return {"wide": wide, "pbp": pbp, "box": box, "quotes": quotes,
            "stlblktov": stlblktov, "oppdef": oppdef}


def _naive_features(frame, prop) -> list[str]:
    stem = _WIDE_STEM[prop]
    cands = [f"player_{stem}_mean_l5", f"player_{stem}_mean_l10", "player_minutes_mean_l5"]
    return [c for c in cands if c in frame.columns]


def _ranked(d: dict, topn: int | None = None, key=lambda kv: -kv[1]):
    items = sorted(d.items(), key=key)
    if topn:
        items = items[:topn]
    return [{"feature": k, "value": (v if not isinstance(v, dict) else v)} for k, v in items]


# --------------------------------------------------------------------------- #
# per-prop orchestrator
# --------------------------------------------------------------------------- #
def run_prop(prop: str, cfg: AblationConfig, inputs: dict) -> dict:
    kind = "binary" if prop in MARKET_PROPS else "count"
    frame, groups, kind, meta = assemble_frame(
        prop, cfg, wide=inputs["wide"], pbp=inputs["pbp"], box=inputs["box"],
        quotes=inputs.get("quotes"), stlblktov=inputs.get("stlblktov"), oppdef=inputs["oppdef"])

    n_rows, n_dates = meta["n_rows"], meta["n_dates"]
    sufficient = (n_rows >= 300 and n_dates >= 30)
    dates = frame["game_date"].to_numpy()
    outer_folds = make_expanding_folds(dates, cfg.n_outer, cfg.min_train_dates)
    integrity = assert_nested_cv_integrity(dates, cfg)

    anchors = ["line"] if kind == "binary" and "line" in frame.columns else []
    all_group_feats = sorted({c for cols in groups.values() for c in cols})
    y = frame["y"].to_numpy()

    # ---- importance (both methods) on the full candidate space ----
    perm = permutation_importance(frame, all_group_feats, outer_folds, kind, cfg)
    l1 = l1_importance(frame, all_group_feats, outer_folds, kind, cfg)
    group_of = {c: g for g, cols in groups.items() for c in cols}
    ranked_perm = [{"feature": k, "delta_loss": v, "group": group_of.get(k)}
                   for k, v in sorted(perm.items(), key=lambda kv: -kv[1])]
    ranked_l1 = [{"feature": k, "select_freq": v["select_freq"],
                  "mean_abs_coef": v["mean_abs_coef"], "group": group_of.get(k)}
                 for k, v in sorted(l1.items(), key=lambda kv: -kv[1]["mean_abs_coef"])]

    # ---- baseline / all-features OOF ----
    p0_spec = anchors + _naive_features(frame, prop)
    all_spec = anchors + all_group_feats
    oof_p0 = oof_predict(frame, p0_spec, outer_folds, kind, cfg)
    oof_all = oof_predict(frame, all_spec, outer_folds, kind, cfg)

    # ---- selection (forward + L1, consensus) ----
    consensus, per_fold, l1_sets, per_fold_map, sel_meta = consensus_selection(
        frame, groups, anchors, kind, cfg, outer_folds)
    oof_sel = oof_predict_perfold(frame, anchors, per_fold_map, outer_folds, kind, cfg)
    selected_empty = not oof_sel["mask"].any()
    if selected_empty:
        oof_sel = oof_p0  # fall back so metrics compute; flagged as info gap

    # ---- group ablation: only-one-group + leave-one-group-out ----
    only_one, logo = {}, {}
    all_mask = oof_all["mask"]
    for g, cols in groups.items():
        if not cols:
            continue
        oo = oof_predict(frame, anchors + cols, outer_folds, kind, cfg)
        lo_cols = [c for c in all_group_feats if c not in cols]
        lg = oof_predict(frame, anchors + lo_cols, outer_folds, kind, cfg)
        if kind == "binary":
            m_oo = _binary_metrics(y[oo["mask"]], oo["pred"][oo["mask"]])
            m_lg = _binary_metrics(y[lg["mask"]], lg["pred"][lg["mask"]])
            m_all = _binary_metrics(y[all_mask], oof_all["pred"][all_mask])
            only_one[g] = {"n": int(oo["mask"].sum()), **m_oo}
            logo[g] = {"n": int(lg["mask"].sum()),
                       "delta_log_loss_vs_all": m_lg["log_loss"] - m_all["log_loss"],
                       "delta_brier_vs_all": m_lg["brier"] - m_all["brier"],
                       "delta_auc_vs_all": m_lg["auc"] - m_all["auc"], **m_lg}
        else:
            m_oo = _count_metrics(y[oo["mask"]], oo["pred"][oo["mask"]], oo["disp"][oo["mask"]])
            m_lg = _count_metrics(y[lg["mask"]], lg["pred"][lg["mask"]], lg["disp"][lg["mask"]])
            m_all = _count_metrics(y[all_mask], oof_all["pred"][all_mask], oof_all["disp"][all_mask])
            only_one[g] = {"n": int(oo["mask"].sum()), **m_oo}
            logo[g] = {"n": int(lg["mask"].sum()),
                       "delta_poisson_deviance_vs_all": m_lg["poisson_deviance"] - m_all["poisson_deviance"],
                       "delta_crps_vs_all": m_lg["crps"] - m_all["crps"], **m_lg}

    # most valuable group by only-one-group ranking
    if kind == "binary":
        best_group = max(only_one, key=lambda g: (only_one[g]["auc"] if np.isfinite(only_one[g]["auc"]) else -1))
    else:
        best_group = min(only_one, key=lambda g: only_one[g]["poisson_deviance"])

    # ---- final metrics on common evaluation mask ----
    result = {
        "prop": prop, "kind": kind, "n_rows": n_rows, "n_dates": n_dates,
        "sufficient_data": bool(sufficient),
        "min_data_note": None if sufficient else f"only {n_rows} rows / {n_dates} dates (<300/30)",
        "player_box_form_group": cfg.player_box_form_status,
        "feature_groups": {g: cols for g, cols in groups.items()},
        "group_sizes": {g: len(cols) for g, cols in groups.items()},
        "pbp_feature_names": meta["pbp_feature_names"],
        "oppdef_feature_names": meta["oppdef_feature_names"],
        "nested_cv_integrity": integrity,
        "outer_folds": len(outer_folds),
        "importance_permutation_top": ranked_perm[:20],
        "importance_l1_top": ranked_l1[:20],
        "only_one_group": only_one,
        "leave_one_group_out": logo,
        "most_valuable_group_only_one": best_group,
        "selected_feature_set": consensus,
        "selected_per_fold": {int(k): v for k, v in per_fold_map.items()},
        "selection_meta": sel_meta,
        "selected_empty_info_gap": bool(selected_empty),
        "config": {"n_outer": cfg.n_outer, "n_inner": cfg.n_inner, "seed": cfg.seed,
                   "prerank_top_k": cfg.prerank_top_k, "forward_max_features": cfg.forward_max_features,
                   "bootstrap_iters": cfg.bootstrap_iters, "hgb_max_iter": cfg.hgb_max_iter},
    }

    if kind == "binary":
        market = frame["market"].to_numpy()
        p0_deliv = frame["p0_delivered"].to_numpy()
        cm = oof_p0["mask"] & oof_all["mask"] & oof_sel["mask"] & np.isfinite(market)
        gd = frame["game_date"].to_numpy()[cm]
        yy = y[cm]
        specs = {"P0_naive": oof_p0["pred"][cm], "all_features": oof_all["pred"][cm],
                 "selected": oof_sel["pred"][cm], "market": market[cm],
                 "production_baseline": p0_deliv[cm]}
        result["oof_n"] = int(cm.sum())
        result["oof_dates"] = int(pd.Series(gd).nunique())
        result["metrics"] = {name: _binary_metrics(yy, p) for name, p in specs.items()}
        # paired bootstrap vs market for P0, all, selected
        vs = {}
        for name in ("P0_naive", "all_features", "selected"):
            ci_ll, ci_bs, p_ll, p_brier = M.paired_bootstrap(
                yy, specs[name], specs["market"], gd, cfg.bootstrap_iters, cfg.seed)
            vs[name] = {"delta_log_loss": M.log_loss(yy, specs[name]) - M.log_loss(yy, specs["market"]),
                        "delta_brier": M.brier(yy, specs[name]) - M.brier(yy, specs["market"]),
                        "delta_auc": M.auc(yy, specs[name]) - M.auc(yy, specs["market"]),
                        "ci95_delta_log_loss": ci_ll, "ci95_delta_brier": ci_bs,
                        "p_ll_raw": p_ll, "p_brier_raw": p_brier}
        result["vs_market"] = vs
        ll_mkt = result["metrics"]["market"]["log_loss"]
        ll_p0 = result["metrics"]["P0_naive"]["log_loss"]
        ll_sel = result["metrics"]["selected"]["log_loss"]
        gap_p0 = ll_p0 - ll_mkt
        gap_sel = ll_sel - ll_mkt
        result["closes_gap_fraction"] = (float((gap_p0 - gap_sel) / gap_p0)
                                         if abs(gap_p0) > 1e-9 else None)
        result["beats_market_pointwise"] = bool(gap_sel < 0)
        result["selected_auc"] = result["metrics"]["selected"]["auc"]
        result["market_auc"] = result["metrics"]["market"]["auc"]
    else:
        cm = oof_p0["mask"] & oof_all["mask"] & oof_sel["mask"]
        yy = y[cm]
        result["oof_n"] = int(cm.sum())
        result["oof_dates"] = int(pd.Series(frame["game_date"].to_numpy()[cm]).nunique())
        result["metrics"] = {
            "P0_naive": _count_metrics(yy, oof_p0["pred"][cm], oof_p0["disp"][cm]),
            "all_features": _count_metrics(yy, oof_all["pred"][cm], oof_all["disp"][cm]),
            "selected": _count_metrics(yy, oof_sel["pred"][cm], oof_sel["disp"][cm]),
        }
        dev_p0 = result["metrics"]["P0_naive"]["poisson_deviance"]
        dev_sel = result["metrics"]["selected"]["poisson_deviance"]
        result["selected_vs_p0_deviance_delta"] = dev_sel - dev_p0
        result["market_comparison_possible"] = False
        result["market_note"] = "no sportsbook offers stl/blk/tov props (0 books) -> outcome-only"
    return result
