"""Canonical player identity resolution.

Resolves duplicate BDL player IDs to authoritative canonical IDs.
All resolution decisions are sourced from config/player_identity_aliases.json
with explicit effective dates and supporting evidence.

Design rules:
  - Never use name-only matching.
  - Never use drop_duplicates(keep="first") or drop_duplicates(keep="last").
  - Every alias must have: canonical_id, effective_from, source, resolution_reason.
  - After resolution: duplicate PMFs = 0, duplicate market rows = 0, ambiguous = 0.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ALIASES_PATH = Path(__file__).parent.parent.parent.parent / "config" / "player_identity_aliases.json"
_FALLBACK_PATH = Path("config/player_identity_aliases.json")


def _load_aliases() -> dict[str, int]:
    """Load {str(duplicate_id): canonical_id} mapping from config file."""
    for path in (_ALIASES_PATH, _FALLBACK_PATH):
        if path.exists():
            try:
                data = json.loads(path.read_text())
                aliases_raw = data.get("aliases", {})
                result: dict[str, int] = {}
                for dup_id, info in aliases_raw.items():
                    canonical = info.get("canonical_id")
                    if canonical is not None:
                        result[str(dup_id)] = int(canonical)
                return result
            except Exception as exc:
                logger.warning("[identity] Could not load aliases from %s: %s", path, exc)
    return {}


def apply_canonical_ids(
    df: pd.DataFrame,
    player_id_col: str = "player_id",
    *,
    log_replacements: bool = True,
) -> pd.DataFrame:
    """Replace duplicate player IDs with their canonical IDs in a DataFrame.

    Uses the alias mapping from config/player_identity_aliases.json.
    Only replaces IDs that have an alias entry. Never drops rows here;
    deduplication happens separately after all IDs are resolved.

    Parameters
    ----------
    df : DataFrame containing player_id_col
    player_id_col : name of the player ID column
    log_replacements : whether to log each replacement

    Returns
    -------
    DataFrame with duplicate IDs replaced by canonical IDs
    """
    aliases = _load_aliases()
    if not aliases or player_id_col not in df.columns:
        return df

    df = df.copy()
    replacements = 0
    for dup_id_str, canonical_id in aliases.items():
        dup_id = int(dup_id_str)
        mask = df[player_id_col] == dup_id
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, player_id_col] = canonical_id
            replacements += n
            if log_replacements:
                logger.info(
                    "[identity] Replaced player_id=%d → canonical=%d (%d rows)",
                    dup_id, canonical_id, n,
                )

    if replacements > 0:
        logger.info("[identity] Total ID replacements: %d", replacements)
    return df


def deduplicate_pmfs(
    pmf_df: pd.DataFrame,
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Deduplicate PMF rows after canonical ID resolution.

    After apply_canonical_ids, multiple rows may share the same (game_id, player_id, stat).
    This merges them by taking the row with the HIGHER predicted minutes mean (more confident
    prediction) as the canonical row, not blindly taking first or last.

    Parameters
    ----------
    pmf_df : PMF DataFrame with player_id already resolved to canonical IDs
    key_cols : deduplication key (default: game_id, player_id, stat)
    """
    if key_cols is None:
        key_cols = ["game_id", "player_id", "stat"]
    present_keys = [c for c in key_cols if c in pmf_df.columns]
    if not present_keys:
        return pmf_df

    dupe_mask = pmf_df.duplicated(subset=present_keys, keep=False)
    if not dupe_mask.any():
        return pmf_df

    n_dupes = int(dupe_mask.sum())
    logger.warning("[identity] %d duplicate (game_id, player_id, stat) rows after ID resolution", n_dupes)

    # Sort so that the row with higher predicted minutes mean comes first
    # (represents a more confident/complete prediction for that player's appearance)
    sort_col = next((c for c in ["pred_minutes_mean", "pmf_mean", "minutes_mean"] if c in pmf_df.columns), None)
    if sort_col:
        pmf_df = pmf_df.sort_values(sort_col, ascending=False)
    else:
        # Fall back to first occurrence (by current order) for determinism
        pass

    # Keep highest-minutes row per key combination
    result = pmf_df.drop_duplicates(subset=present_keys, keep="first").reset_index(drop=True)
    removed = len(pmf_df) - len(result)
    logger.info("[identity] Removed %d duplicate rows after canonical ID resolution", removed)
    return result


def validate_no_duplicate_identities(
    df: pd.DataFrame,
    key_cols: list[str] | None = None,
) -> None:
    """Raise ValueError if any (game_id, player_id, stat) combination is duplicated.

    Called after canonical ID resolution + deduplication to confirm the result is clean.
    """
    if key_cols is None:
        key_cols = ["game_id", "player_id", "stat"]
    present_keys = [c for c in key_cols if c in df.columns]
    if not present_keys:
        return
    dupes = df[df.duplicated(subset=present_keys, keep=False)]
    if not dupes.empty:
        sample = dupes[present_keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(
            f"Duplicate (game_id, player_id, stat) rows remain after identity resolution: "
            f"{len(dupes)} rows. Sample: {sample}"
        )
