import pandas as pd
import numpy as np
from pathlib import Path


def join_historical_props_to_oof(
    oof_parquet: str,
    props_parquet: str,
    output_parquet: str,
) -> pd.DataFrame:
    """
    Join historical prop lines to OOF predictions after OOF generation
    but before calibration. Safe: OOF predictions were generated without
    lines. Lines are needed only for Venn-Abers and beta calibration fitting.
    """
    oof = pd.read_parquet(oof_parquet)
    props = pd.read_parquet(props_parquet)

    cols_needed = [c for c in ['player_id', 'game_id', 'stat', 'line',
                                'over_odds', 'under_odds'] if c in props.columns]
    merge_keys = [c for c in ['player_id', 'game_id', 'stat'] if c in props.columns]
    merged = oof.merge(props[cols_needed], on=merge_keys, how='left')

    n_total = len(oof)
    n_with_lines = merged['line'].notna().sum() if 'line' in merged.columns else 0
    coverage = n_with_lines / n_total * 100 if n_total > 0 else 0
    print(f"[OOF Line Join] {n_with_lines}/{n_total} rows have lines ({coverage:.1f}% coverage)")

    merged.to_parquet(output_parquet, index=False)
    return merged
