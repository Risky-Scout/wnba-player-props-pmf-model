"""Vacated-opportunity feature builder (Path A).

Given the set of players who are OUT for a team on a date, redistribute their
STRICTLY-PRIOR minutes / usage / 3PA / possessions to the available teammates. This
quantifies the opportunity vacated by absences — the raw material for "next man up"
prop edges (a bench player who inherits 12 extra minutes and 4 extra 3PA is a very
different bet than his season baseline).

Leakage discipline: this function only ever consumes the ``prior`` frame the caller
passes in. That frame MUST be computed as-of BEFORE tip (rolling aggregates over games
strictly prior to the game date). Nothing here reads the game being predicted. And the
absence set itself must come from a PRE-tip availability pull (see
``scripts/collect_availability.py``) — which is exactly why this feature can only be
evaluated on forward-accrued data, not backfilled (there is no historical pregame
availability archive to reconstruct absences from).

Redistribution model: for each metric, the OUT players' combined prior total is
distributed to the available teammates in proportion to each teammate's own prior share
of that metric (bigger contributors absorb more). If the available teammates have zero
prior mass for a metric, the vacated total is split evenly among them. Projected minutes
are capped (default 40) to avoid unphysical totals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Default metric columns to redistribute (prior, as-of before tip).
DEFAULT_METRICS: tuple[str, ...] = ("minutes", "usage", "fg3a", "possessions")
DEFAULT_MINUTES_CAP = 40.0


@dataclass(frozen=True)
class VacatedConfig:
    metrics: tuple[str, ...] = DEFAULT_METRICS
    minutes_cap: float = DEFAULT_MINUTES_CAP
    minutes_col: str = "minutes"
    player_col: str = "player_id"
    team_col: str = "team_id"
    extra_passthrough: tuple[str, ...] = field(default_factory=tuple)


def redistribute_vacated_opportunity(
    prior: pd.DataFrame,
    out_player_ids,
    team_id,
    config: VacatedConfig | None = None,
) -> pd.DataFrame:
    """Redistribute OUT players' prior opportunity to available teammates.

    Parameters
    ----------
    prior : DataFrame
        Per-player STRICTLY-PRIOR (as-of before tip) aggregates. Must contain the player
        and team id columns plus every metric column in ``config.metrics``.
    out_player_ids : iterable
        Player ids that are OUT for ``team_id`` on the date (from a pre-tip pull).
    team_id :
        The team whose absences are being redistributed.
    config : VacatedConfig | None
        Column names / metrics / minutes cap.

    Returns
    -------
    DataFrame
        One row per AVAILABLE teammate, with for each metric ``M``:
        ``prior_M``, ``vacated_M_added``, ``proj_M`` (= prior + added, minutes capped).
        Plus ``player_id``, ``team_id``, ``n_out``, ``is_beneficiary`` (added > 0), and any
        ``extra_passthrough`` columns. Empty (typed) frame if no available teammates.
    """
    cfg = config or VacatedConfig()
    out_ids = set(out_player_ids or [])

    cols = (
        [cfg.player_col, cfg.team_col, "n_out", "is_beneficiary"]
        + [c for m in cfg.metrics for c in (f"prior_{m}", f"vacated_{m}_added", f"proj_{m}")]
        + list(cfg.extra_passthrough)
    )
    if prior is None or len(prior) == 0:
        return pd.DataFrame(columns=cols)

    team_prior = prior[prior[cfg.team_col] == team_id].copy()
    if team_prior.empty:
        return pd.DataFrame(columns=cols)

    is_out = team_prior[cfg.player_col].isin(out_ids)
    out_df = team_prior[is_out]
    avail = team_prior[~is_out].copy()
    if avail.empty:
        return pd.DataFrame(columns=cols)

    n_out = len(out_df)
    result = pd.DataFrame({
        cfg.player_col: avail[cfg.player_col].to_numpy(),
        cfg.team_col: avail[cfg.team_col].to_numpy(),
        "n_out": n_out,
    })

    added_any = np.zeros(len(avail), dtype=float)
    for m in cfg.metrics:
        prior_vals = pd.to_numeric(avail[m], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        vacated_total = float(
            pd.to_numeric(out_df[m], errors="coerce").fillna(0.0).sum()
        ) if n_out else 0.0

        total_avail = prior_vals.sum()
        if vacated_total <= 0.0:
            weights = np.zeros(len(avail), dtype=float)
        elif total_avail > 0.0:
            weights = prior_vals / total_avail
        else:
            # No prior mass among available players -> split the vacated total evenly.
            weights = np.full(len(avail), 1.0 / len(avail), dtype=float)

        added = vacated_total * weights
        proj = prior_vals + added
        if m == cfg.minutes_col:
            proj = np.minimum(proj, cfg.minutes_cap)
        result[f"prior_{m}"] = prior_vals
        result[f"vacated_{m}_added"] = added
        result[f"proj_{m}"] = proj
        added_any = added_any + added

    result["is_beneficiary"] = added_any > 1e-9
    for c in cfg.extra_passthrough:
        if c in avail.columns:
            result[c] = avail[c].to_numpy()

    return result.reset_index(drop=True)[cols]


def build_vacated_features_for_slate(
    prior: pd.DataFrame,
    absences_by_team: dict,
    config: VacatedConfig | None = None,
) -> pd.DataFrame:
    """Apply :func:`redistribute_vacated_opportunity` across a whole slate.

    ``absences_by_team`` maps ``team_id -> iterable of OUT player_ids``. Returns the
    concatenation of per-team beneficiary frames (empty typed frame if none).
    """
    cfg = config or VacatedConfig()
    frames = [
        redistribute_vacated_opportunity(prior, out_ids, team_id, cfg)
        for team_id, out_ids in (absences_by_team or {}).items()
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        return redistribute_vacated_opportunity(prior.head(0), [], None, cfg)
    return pd.concat(frames, ignore_index=True)
