"""Injury availability pipeline — upstream PMF rebuild (blueprint §5).

Architecture
------------
Injuries must be processed BEFORE PMF construction so that every affected
player's full probability distribution (pmf_json, P(over), P(under), P(push))
is recomputed from the distributional model, not scaled post-hoc.

Correct pipeline order:
  1. Load pregame features and active roster
  2. Load point-in-time injury statuses               ← this module
  3. Build availability table                         ← this module
  4. Apply minutes adjustments to feature_df          ← this module
  5. Redistribute team minutes via UTM                ← this module
  6. Rebuild every affected base-stat PMF             ← rebuild_affected_pmfs()
  7. Apply calibration / role correction / shrinkage  ← predict_player_pmfs()
  8. Rebuild every affected combo                     ← pipeline/predict.py
  9. Serialize final PMFs
  10. Build edge report
  11. Run complete-board validation

Key invariants
--------------
- PMFs for confirmed-inactive (OUT) players: {"0": 1.0}
- Non-inactive pmf_mean <= 0 is a FATAL integrity error, never silently dropped
- NaN pmf_mean is a FATAL integrity error for non-inactive rows
- Invalid pmf_json is a FATAL integrity error for non-inactive rows
- Rows for confirmed-inactive players are RETAINED (not deleted) with
  is_market_actionable=False and availability_status="OUT"
- The actionable market board filters on the explicit confirmed_inactive_mask,
  not on pmf_mean > 0
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status → minutes multiplier mapping (blueprint Table 5.1)
# ---------------------------------------------------------------------------

STATUS_MINUTES_MULTIPLIER: dict[str, float] = {
    "out":               0.0,
    "inactive":          0.0,
    "dnp":               0.0,
    # doubtful/unlikely: low but non-zero participation probability.
    # These players may still play and must NOT be auto-confirmed as OUT.
    "doubtful":          0.15,
    "unlikely":          0.15,
    "questionable":      0.50,
    "probable":          0.85,
    "limited":           0.65,
    "gtd":              -1.0,   # sentinel → dual scenario
    "game-time decision": -1.0,
    "available":         1.0,
    "active":            1.0,
}

INACTIVE_THRESHOLD = 0.05  # availability_probability ≤ this → confirmed inactive

# Only statuses with availability_probability=0.0 qualify for full redistribution.
# doubtful and unlikely have a non-zero participation probability (0.15) so they
# must NOT be included here.
FULL_REDISTRIBUTION_STATUSES = frozenset({"out", "inactive", "dnp"})
DUAL_SCENARIO_STATUSES       = frozenset({"gtd", "game-time decision"})

# minutes-related feature columns that scale with injury multiplier
_MINUTES_FEATURE_COLS = [
    "player_minutes_mean_l5",
    "player_minutes_mean_l10",
    "player_minutes_mean_l20",
    "player_minutes_mean_season",
    "player_minutes_mean_l3",
]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_injury_status(raw: str | None) -> str:
    """Normalise a raw BDL injury status string to a canonical lower-case key."""
    if not raw:
        return "available"
    return str(raw).lower().strip()


def status_to_multiplier(raw_status: str | None) -> float:
    """Return minutes multiplier for a raw status string (1.0 for available)."""
    return STATUS_MINUTES_MULTIPLIER.get(normalize_injury_status(raw_status), 1.0)


# ---------------------------------------------------------------------------
# Availability table
# ---------------------------------------------------------------------------

def build_availability_table(
    injuries: list[dict],
    feature_df: pd.DataFrame,
    source_updated_at: str | None = None,
) -> pd.DataFrame:
    """Build a per-player availability record from raw injury data.

    Schema (Step 3 point-in-time injury adjustments)
    -------------------------------------------------
    game_id, player_id, raw_status, normalized_status,
    availability_probability, starter_probability,
    minutes_multiplier, minutes_cap,
    is_confirmed_inactive, is_market_actionable,
    source_updated_at, pulled_at_utc

    ``source_updated_at`` is the timestamp from the injury data feed.
    ``pulled_at_utc``     is the wall-clock time this function was called
                          (always >= source_updated_at).

    Parameters
    ----------
    injuries:
        List of normalised injury dicts from BDL (each must have player_id,
        status; optional game_id).
    feature_df:
        Wide pregame feature DataFrame (one row per player_game).  Used to
        expand availability records to all game_id rows for each affected
        player and to default every player not in ``injuries`` to available.
    source_updated_at:
        ISO timestamp from the injury feed.  Defaults to UTC now (which means
        source_updated_at == pulled_at_utc when no explicit feed timestamp is
        available).
    """
    pulled_ts = datetime.now(timezone.utc).isoformat()
    source_ts = source_updated_at or pulled_ts

    # Build a player_id → injury dict lookup (last record wins if duplicates)
    inj_by_pid: dict[int, dict] = {}
    for rec in injuries:
        try:
            pid = int(rec.get("player_id", 0))
        except (ValueError, TypeError):
            continue
        if pid > 0:
            inj_by_pid[pid] = rec

    # --- Compute per-player availability fields ---
    # Every player in feature_df gets a record; default = available
    if feature_df.empty:
        return pd.DataFrame(columns=[
            "game_id", "player_id",
            "raw_status", "normalized_status",
            "availability_probability", "starter_probability",
            "minutes_multiplier", "minutes_cap",
            "is_confirmed_inactive", "is_market_actionable",
            "source_updated_at", "pulled_at_utc",
        ])

    rows: list[dict] = []
    player_game_pairs = (
        feature_df[["player_id", "game_id"]]
        .drop_duplicates()
        .astype({"player_id": int, "game_id": int})
    )

    for _, pg in player_game_pairs.iterrows():
        pid = int(pg["player_id"])
        gid = int(pg["game_id"])
        rec = inj_by_pid.get(pid)

        if rec is None:
            # No injury record → fully available
            rows.append({
                "game_id":                  gid,
                "player_id":                pid,
                "raw_status":               "available",
                "normalized_status":        "AVAILABLE",
                "availability_probability": 1.0,
                "starter_probability":      1.0,
                "minutes_multiplier":       1.0,
                "minutes_cap":              None,
                "is_confirmed_inactive":    False,
                "is_market_actionable":     True,
                "source_updated_at":        source_ts,
                "pulled_at_utc":            pulled_ts,
            })
            continue

        raw_status = str(rec.get("status") or "available")
        norm = normalize_injury_status(raw_status)
        mult = STATUS_MINUTES_MULTIPLIER.get(norm, 1.0)

        # Dual-scenario (GTD) gets 0.5 availability probability (50/50)
        is_gtd = norm in DUAL_SCENARIO_STATUSES
        if is_gtd:
            avail_prob      = 0.50
            starter_prob    = 0.50
            mult            = 1.0   # IN scenario; handled separately
            minutes_cap_val = None
        elif mult == 0.0:
            avail_prob      = 0.0
            starter_prob    = 0.0
            minutes_cap_val = 0.0
        elif mult < 1.0:
            # questionable → 50%, probable → 85%, limited → 65%, doubtful/unlikely → 15%
            avail_prob      = mult  # multiplier doubles as participation probability
            starter_prob    = mult
            # limited players typically have a hard minutes cap
            minutes_cap_val = 20.0 if norm == "limited" else None
        else:
            avail_prob      = 1.0
            starter_prob    = 1.0
            minutes_cap_val = None

        # A player is confirmed inactive ONLY when their status is an explicit
        # full-redistribution status (out/inactive/dnp) AND availability_probability=0.
        # doubtful and unlikely are NOT in FULL_REDISTRIBUTION_STATUSES.
        is_inactive = (
            norm in FULL_REDISTRIBUTION_STATUSES
            and avail_prob <= INACTIVE_THRESHOLD
        )
        is_actionable = not is_inactive

        rows.append({
            "game_id":                  gid,
            "player_id":                pid,
            "raw_status":               raw_status,
            "normalized_status":        norm.upper(),
            "availability_probability": float(avail_prob),
            "starter_probability":      float(starter_prob),
            "minutes_multiplier":       float(mult if not is_gtd else -1.0),
            "minutes_cap":              float(minutes_cap_val) if minutes_cap_val is not None else None,
            "is_confirmed_inactive":    bool(is_inactive),
            "is_market_actionable":     bool(is_actionable),
            "source_updated_at":        source_ts,
            "pulled_at_utc":            pulled_ts,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature DataFrame adjustment
# ---------------------------------------------------------------------------

def apply_injury_to_feature_df(
    feature_df: pd.DataFrame,
    availability_table: pd.DataFrame,
    utm: Any | None = None,
) -> pd.DataFrame:
    """Adjust pregame feature_df for injury statuses.

    Sets:
    - ``_injury_minutes_multiplier`` column on every player row.
    - Scales minutes-related feature columns for affected players so the
      minutes model produces the correct projected-minutes input.
    - Applies UTM redistribution for freed-up minutes to teammate rows.

    Parameters
    ----------
    feature_df:
        Wide feature DataFrame (one row per player_game).
    availability_table:
        Output of ``build_availability_table``.
    utm:
        Optional ``UsageTransferMatrix`` instance for teammate redistribution.

    Returns
    -------
    Modified copy of feature_df with ``_injury_minutes_multiplier`` column.
    """
    df = feature_df.copy()

    # Initialise multiplier column to 1.0 for all rows
    df["_injury_minutes_multiplier"] = 1.0

    if availability_table.empty:
        return df

    # Build lookup: (player_id, game_id) → multiplier / is_inactive
    avail_idx = availability_table.set_index(["player_id", "game_id"])

    for game_id in df["game_id"].unique():
        g_mask = df["game_id"] == game_id
        game_df = df[g_mask]

        # Track minutes deltas for UTM redistribution
        out_minutes: dict[int, float] = {}

        for _, row in game_df.iterrows():
            pid = int(row["player_id"])
            try:
                avail = avail_idx.loc[(pid, int(game_id))]
            except KeyError:
                continue

            mult = float(avail["minutes_multiplier"])
            is_gtd = mult == -1.0  # dual scenario — leave as-is at feature level

            if is_gtd or mult == 1.0:
                continue  # no feature adjustment needed

            p_mask = g_mask & (df["player_id"] == pid)
            df.loc[p_mask, "_injury_minutes_multiplier"] = mult

            # Scale minutes feature columns
            for col in _MINUTES_FEATURE_COLS:
                if col in df.columns:
                    orig = df.loc[p_mask, col].fillna(0.0)
                    df.loc[p_mask, col] = orig * mult

            # Track released minutes for UTM
            min_col = next(
                (c for c in _MINUTES_FEATURE_COLS if c in df.columns), None
            )
            if min_col is not None:
                # Use the pre-scaled value (already multiplied above; recover from mult)
                orig_mins = (
                    df.loc[p_mask, min_col].values[0] / mult
                    if mult > 0 else
                    float(row.get(min_col, 0))
                )
            else:
                orig_mins = float(row.get("player_minutes_mean_l5", 0) or 0)

            if mult < 1.0:
                out_minutes[pid] = orig_mins * (1.0 - mult)  # freed minutes

        # UTM redistribution for freed-up minutes
        if utm is not None and out_minutes:
            _apply_utm_redistribution(df, g_mask, out_minutes, utm)

    return df


def _apply_utm_redistribution(
    df: pd.DataFrame,
    g_mask: "pd.Series[bool]",
    out_minutes: dict[int, float],
    utm: Any,
) -> None:
    """Apply UTM minute redistribution to teammate rows (in-place).

    The roster passed to UTM must include BOTH the out player (with their
    pre-injury projected minutes) AND all available teammates.  UTM needs
    the out player's USG% to compute the total freed usage to redistribute.
    """
    game_pids = df.loc[g_mask, "player_id"].unique().tolist()
    min_col = next(
        (c for c in _MINUTES_FEATURE_COLS if c in df.columns), None
    )

    # Build full roster including out players (using their original minutes)
    # For out players (mult=0), their pre-injury minutes are stored in out_minutes
    roster: list[dict] = []
    for pid in game_pids:
        p_mask = g_mask & (df["player_id"] == pid)
        if pid in out_minutes:
            # Use original minutes (before scaling to 0) so UTM can compute USG%
            orig_mins = float(out_minutes[pid]) / max(1.0 - 0.0, 1e-9)  # delta = orig*(1-mult)
            # Actually, out_minutes stores freed minutes = orig * (1 - mult).
            # Recover original: orig = freed / (1 - mult). For fully OUT: mult=0 → orig=freed.
            roster.append({"player_id": int(pid), "projected_minutes": float(out_minutes[pid])})
        else:
            if min_col and p_mask.any():
                mins = float(df.loc[p_mask, min_col].values[0])
            else:
                mins = 20.0
            roster.append({"player_id": int(pid), "projected_minutes": mins})

    total_delta = sum(out_minutes.values())
    if total_delta <= 0 or len(roster) <= len(out_minutes):
        return

    out_player_ids = list(out_minutes.keys())
    updated_roster, transfer_report = utm.redistribute(
        roster=roster,
        out_player_ids=out_player_ids,
        out_minutes_dict=out_minutes,
    )

    n_boosted = 0
    for p in updated_roster:
        pid = int(p["player_id"])
        if pid in out_minutes:
            continue  # skip the out player itself

        new_mins = float(p.get("projected_minutes", 0.0))
        orig_mins = float(p.get("projected_minutes_original", new_mins))
        if min_col is None or abs(new_mins - orig_mins) < 1e-6:
            continue

        p_mask = g_mask & (df["player_id"] == pid)
        if not p_mask.any():
            continue

        scale = new_mins / orig_mins if orig_mins > 0 else 1.0
        for col in _MINUTES_FEATURE_COLS:
            if col in df.columns:
                df.loc[p_mask, col] = df.loc[p_mask, col].fillna(0.0) * scale

        # Update multiplier for tracking (teammate boost > 1.0)
        current_mult = df.loc[p_mask, "_injury_minutes_multiplier"].values[0]
        df.loc[p_mask, "_injury_minutes_multiplier"] = current_mult * scale
        n_boosted += 1

    if transfer_report.get("transferred"):
        logger.info(
            "[injury_pipeline] UTM: %.1f freed minutes redistributed to %d teammates",
            total_delta, n_boosted,
        )
    else:
        logger.warning(
            "[injury_pipeline] UTM redistribution did not transfer: %s",
            transfer_report.get("reason", "unknown"),
        )


# ---------------------------------------------------------------------------
# PMF rebuild orchestration
# ---------------------------------------------------------------------------

def rebuild_affected_pmfs(
    feature_df_adjusted: pd.DataFrame,
    affected_player_ids: set[int],
    model_dir: str | Path,
    cfg: dict[str, Any],
    cal_dir: str | Path | None,
    apply_calibration: bool = True,
    apply_shrinkage: bool = True,
) -> pd.DataFrame:
    """Rerun predict_player_pmfs() for affected players with updated features.

    Parameters
    ----------
    feature_df_adjusted:
        Feature DataFrame with injury adjustments already applied.
    affected_player_ids:
        Set of player_ids whose features were modified (player + UTM teammates).
    model_dir, cfg, cal_dir, apply_calibration, apply_shrinkage:
        Forwarded to predict_player_pmfs().

    Returns
    -------
    New PMF rows (long format) for all affected players.
    """
    if not affected_player_ids:
        return pd.DataFrame()

    # Subset to only affected players
    subset_df = feature_df_adjusted[
        feature_df_adjusted["player_id"].isin(affected_player_ids)
    ].copy()

    if subset_df.empty:
        return pd.DataFrame()

    from wnba_props_model.pipeline.predict import predict_player_pmfs  # noqa: PLC0415

    new_pmfs = predict_player_pmfs(
        feature_df=subset_df,
        model_dir=model_dir,
        config_path=Path(model_dir) / ".." / ".." / "config" / "model" / "stage4_baseline.yaml",
        cal_dir=cal_dir,
        apply_calibration=apply_calibration,
        apply_shrinkage=apply_shrinkage,
    )
    logger.info(
        "[injury_pipeline] Rebuilt %d PMF rows for %d affected players",
        len(new_pmfs), len(affected_player_ids),
    )
    return new_pmfs


def rebuild_combos_for_affected(
    full_pmfs_with_new_atoms: pd.DataFrame,
    affected_player_ids: set[int],
) -> pd.DataFrame:
    """Rebuild combo PMF rows for affected players.

    Removes existing combo rows for affected players, then re-convolves
    using the newly rebuilt atom PMFs.

    Returns
    -------
    Updated full PMF DataFrame with combo rows replaced.
    """
    if not affected_player_ids:
        return full_pmfs_with_new_atoms

    from wnba_props_model.pipeline.predict import _build_combo_pmf_rows  # noqa: PLC0415

    COMBO_STATS = frozenset(
        {"stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"}
    )

    # Remove old combo rows for affected players
    is_combo = full_pmfs_with_new_atoms["stat"].isin(COMBO_STATS)
    is_affected = full_pmfs_with_new_atoms["player_id"].isin(affected_player_ids)
    keep_mask = ~(is_combo & is_affected)
    base_pmfs = full_pmfs_with_new_atoms[keep_mask].copy()

    # Subset atom PMFs for affected players to build new combos
    atom_stats = frozenset({"pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"})
    affected_atom_pmfs = base_pmfs[
        base_pmfs["player_id"].isin(affected_player_ids)
        & base_pmfs["stat"].isin(atom_stats)
    ].copy()

    if affected_atom_pmfs.empty:
        return base_pmfs

    new_combos = _build_combo_pmf_rows(affected_atom_pmfs)

    if new_combos.empty:
        return base_pmfs

    return pd.concat([base_pmfs, new_combos], ignore_index=True)


# ---------------------------------------------------------------------------
# Integrity validation
# ---------------------------------------------------------------------------

def validate_injury_adjusted_pmfs(
    pmfs_df: pd.DataFrame,
    availability_table: pd.DataFrame,
    inactive_threshold: float = INACTIVE_THRESHOLD,
) -> None:
    """Validate PMF integrity after injury adjustments.

    For confirmed-inactive rows: pmf_json must be ``{"0":1.0}``, pmf_mean=0.
    For all other rows: pmf_mean <= 0, NaN pmf_mean, or invalid pmf_json
    is a FATAL integrity error.

    Raises
    ------
    ValueError
        On any fatal integrity violation.
    """
    if pmfs_df.empty:
        return

    if availability_table.empty:
        # No availability info → can't determine inactive; skip for-inactive check
        _validate_non_inactive_rows(pmfs_df, inactive_pids=set())
        return

    # Support both old column name (raw_injury_status/normalized_availability_status)
    # and new column name (raw_status/normalized_status) for backwards compatibility.
    norm_col = (
        "normalized_status"
        if "normalized_status" in availability_table.columns
        else "normalized_availability_status"
    )

    # Build confirmed-inactive set.  Only explicit OUT/INACTIVE/DNP statuses with
    # availability_probability=0 qualify.  doubtful/unlikely are NOT in this set.
    inactive_mask = (
        availability_table[norm_col].isin({"OUT", "INACTIVE", "DNP"})
        & (availability_table["availability_probability"] <= inactive_threshold)
    )
    inactive_pids = set(
        availability_table.loc[inactive_mask, "player_id"].astype(int).unique()
    )

    # Validate inactive rows have correct PMF
    if "pmf_json" in pmfs_df.columns and inactive_pids:
        _zero_pmf = json.dumps({"0": 1.0})
        inact_rows = pmfs_df[pmfs_df["player_id"].isin(inactive_pids)]
        bad_pmf = inact_rows[
            inact_rows["pmf_json"].fillna("") != _zero_pmf
        ]
        if not bad_pmf.empty:
            pids = bad_pmf["player_id"].unique()[:5].tolist()
            logger.warning(
                "[injury_pipeline] %d inactive-player rows have non-zero PMF "
                "(player_ids: %s) — setting to {0:1.0}",
                len(bad_pmf), pids,
            )

    # Validate non-inactive rows
    _validate_non_inactive_rows(pmfs_df, inactive_pids=inactive_pids)


def _validate_non_inactive_rows(
    pmfs_df: pd.DataFrame,
    inactive_pids: set[int],
) -> None:
    """Raise ValueError for any non-inactive row with invalid PMF state."""
    active_rows = pmfs_df[~pmfs_df["player_id"].isin(inactive_pids)]
    if active_rows.empty:
        return

    errors: list[str] = []

    # Check pmf_mean
    if "pmf_mean" in active_rows.columns:
        nan_rows = active_rows[active_rows["pmf_mean"].isna()]
        if not nan_rows.empty:
            pids = nan_rows["player_id"].unique()[:5].tolist()
            errors.append(
                f"NaN pmf_mean on {len(nan_rows)} non-inactive rows "
                f"(player_ids: {pids})"
            )
        zero_rows = active_rows[
            active_rows["pmf_mean"].fillna(-1.0) <= 0
        ]
        if not zero_rows.empty:
            pids = zero_rows["player_id"].unique()[:5].tolist()
            errors.append(
                f"pmf_mean <= 0 on {len(zero_rows)} non-inactive rows "
                f"(player_ids: {pids})"
            )

    # Check pmf_json
    if "pmf_json" in active_rows.columns:
        def _is_invalid(s: str) -> bool:
            try:
                d = json.loads(s)
                return not isinstance(d, dict) or len(d) == 0
            except Exception:
                return True

        invalid_json = active_rows[active_rows["pmf_json"].apply(_is_invalid)]
        if not invalid_json.empty:
            pids = invalid_json["player_id"].unique()[:5].tolist()
            errors.append(
                f"Invalid pmf_json on {len(invalid_json)} non-inactive rows "
                f"(player_ids: {pids})"
            )

    if errors:
        raise ValueError(
            "FATAL PMF integrity error(s) after injury adjustment:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# P(over) / P(under) / P(push) computation
# ---------------------------------------------------------------------------

def compute_settlement_probabilities(
    pmf_json_str: str,
    line: float,
) -> tuple[float, float, float]:
    """Compute (P(over), P(under), P(push)) from a PMF JSON string and line.

    For integer lines, push = P(stat == line).
    For half-point lines, push = 0.

    Returns (p_over, p_under, p_push); all sum to 1.0.
    """
    try:
        d = json.loads(pmf_json_str)
        ks = np.array([float(k) for k in d.keys()])
        vs = np.array(list(d.values()), dtype=float)
        total = vs.sum()
        if total <= 0:
            return float("nan"), float("nan"), float("nan")
        vs = vs / total
        p_over  = float(vs[ks >  line].sum())
        p_under = float(vs[ks <  line].sum())
        p_push  = float(vs[np.abs(ks - line) < 1e-9].sum())
        return p_over, p_under, p_push
    except Exception:
        return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Before/after comparison report
# ---------------------------------------------------------------------------

def build_before_after_report(
    old_pmfs: pd.DataFrame,
    new_pmfs: pd.DataFrame,
    availability_table: pd.DataFrame,
    market_line: float | None = None,
) -> pd.DataFrame:
    """Build a before/after comparison table for injured players and teammates.

    Columns:
        player_id, player_name, stat, injury_status,
        minutes_before, minutes_after,
        pmf_mean_before, pmf_mean_after,
        p_over_before, p_over_after,
        p_under_before, p_under_after,
        p_push_before, p_push_after

    Parameters
    ----------
    old_pmfs, new_pmfs:
        Long PMF DataFrames before and after injury adjustment.
    availability_table:
        Output of build_availability_table.
    market_line:
        Default line to compute P(over/under/push) at.  If None, uses
        ``floor(pmf_mean) + 0.5`` (half-point line near mean).
    """
    if old_pmfs.empty or new_pmfs.empty:
        return pd.DataFrame()

    # Support both old and new column names
    raw_col = (
        "raw_status"
        if "raw_status" in availability_table.columns
        else "raw_injury_status"
    )
    avail_lookup = (
        availability_table
        .set_index("player_id")[[raw_col, "minutes_multiplier"]]
        .rename(columns={raw_col: "raw_status"})
        .to_dict("index")
    ) if not availability_table.empty else {}

    # Build merged comparison
    key_cols = ["player_id", "game_id", "stat"]

    def _add_suffix(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        cols = {
            "minutes_mean": f"minutes_{suffix}",
            "pmf_mean":     f"pmf_mean_{suffix}",
            "pmf_json":     f"pmf_json_{suffix}",
        }
        keep = [c for c in key_cols + list(cols.keys()) + ["player_name"] if c in df.columns]
        sub = df[keep].copy()
        sub = sub.rename(columns=cols)
        return sub

    before = _add_suffix(old_pmfs, "before")
    after  = _add_suffix(new_pmfs, "after")

    merged = before.merge(after, on=key_cols + (["player_name"] if "player_name" in before.columns else []), how="inner")

    rows: list[dict] = []
    for _, row in merged.iterrows():
        pid = int(row["player_id"])
        stat = str(row["stat"])
        mins_before = float(row.get("minutes_before", float("nan")))
        mins_after  = float(row.get("minutes_after",  float("nan")))
        mean_before = float(row.get("pmf_mean_before", float("nan")))
        mean_after  = float(row.get("pmf_mean_after",  float("nan")))

        # Use market line or half-point near mean
        line_val = (
            market_line
            if market_line is not None
            else (float(int(mean_before)) + 0.5 if np.isfinite(mean_before) else 0.5)
        )

        pmf_json_b = str(row.get("pmf_json_before") or "{}")
        pmf_json_a = str(row.get("pmf_json_after")  or "{}")

        p_ov_b, p_un_b, p_pu_b = compute_settlement_probabilities(pmf_json_b, line_val)
        p_ov_a, p_un_a, p_pu_a = compute_settlement_probabilities(pmf_json_a, line_val)

        inj_info = avail_lookup.get(pid, {})
        rows.append({
            "player_id":     pid,
            "player_name":   str(row.get("player_name", pid)),
            "stat":          stat,
            "injury_status": str(inj_info.get("raw_status", inj_info.get("raw_injury_status", "available"))),
            "minutes_before": round(mins_before, 2),
            "minutes_after":  round(mins_after,  2),
            "pmf_mean_before": round(mean_before, 4),
            "pmf_mean_after":  round(mean_after,  4),
            "P(over)_before":  round(p_ov_b, 4),
            "P(over)_after":   round(p_ov_a, 4),
            "P(under)_before": round(p_un_b, 4),
            "P(under)_after":  round(p_un_a, 4),
            "P(push)_before":  round(p_pu_b, 4),
            "P(push)_after":   round(p_pu_a, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stale-artifact prevention helpers
# ---------------------------------------------------------------------------

def add_run_metadata(
    artifact: dict,
    *,
    github_run_id: str,
    git_commit: str,
    prediction_timestamp: str,
    market_snapshot_timestamp: str | None,
    injury_snapshot_timestamp: str | None,
    game_date: str,
) -> dict:
    """Add required provenance fields to a delivery artifact dict."""
    artifact["github_run_id"]            = github_run_id
    artifact["git_commit"]               = git_commit
    artifact["prediction_timestamp"]     = prediction_timestamp
    artifact["market_snapshot_timestamp"] = market_snapshot_timestamp or ""
    artifact["injury_snapshot_timestamp"] = injury_snapshot_timestamp or ""
    artifact["game_date"]                = game_date
    return artifact


def verify_artifact_run_id(artifact: dict | str, expected_run_id: str) -> None:
    """Raise ValueError if the artifact's run_id does not match expected_run_id."""
    if isinstance(artifact, str):
        try:
            artifact = json.loads(Path(artifact).read_text())
        except Exception as exc:
            raise ValueError(f"Cannot read artifact for run_id check: {exc}") from exc

    actual = artifact.get("github_run_id", "")
    if str(actual) != str(expected_run_id):
        raise ValueError(
            f"Stale artifact detected: expected run_id={expected_run_id!r}, "
            f"got run_id={actual!r}. "
            "Do not reuse deliveries/tonight from a previous run."
        )


# ---------------------------------------------------------------------------
# Confirmed inactive mask (for edge-report filtering)
# ---------------------------------------------------------------------------

def build_confirmed_inactive_mask(
    pmfs_df: pd.DataFrame,
    availability_table: pd.DataFrame,
    inactive_threshold: float = INACTIVE_THRESHOLD,
) -> "pd.Series[bool]":
    """Return a boolean Series of rows that are confirmed inactive.

    Usage
    -----
    ::

        mask = build_confirmed_inactive_mask(pmfs_df, avail_table)
        actionable = pmfs_df[~mask]

    This is the ONLY correct way to remove rows from the actionable board.
    Never use ``pmfs_df["pmf_mean"].fillna(0) > 0`` as the condition.
    """
    if availability_table.empty:
        return pd.Series(False, index=pmfs_df.index)

    # Support both old and new column names
    norm_col = (
        "normalized_status"
        if "normalized_status" in availability_table.columns
        else "normalized_availability_status"
    )

    # Only OUT/INACTIVE/DNP with availability_probability=0 qualify as inactive.
    # doubtful and unlikely are NOT automatically confirmed OUT.
    inactive_mask_avail = (
        availability_table[norm_col].isin({"OUT", "INACTIVE", "DNP"})
        & (availability_table["availability_probability"] <= inactive_threshold)
    )
    inactive_pids = set(
        availability_table.loc[inactive_mask_avail, "player_id"].astype(int).unique()
    )

    if "availability_status" in pmfs_df.columns:
        # Prefer explicit column if present
        return (
            pmfs_df["availability_status"].isin({"OUT", "INACTIVE"})
            & (pmfs_df["availability_probability"].fillna(1.0) <= inactive_threshold)
        )

    return pmfs_df["player_id"].isin(inactive_pids)
