# Baseline Data Inventory

- generated: 2026-07-24T22:42:07.743129+00:00
- W1 ready: **False**

## Required datasets for W1

| dataset | role | status |
|---|---|---|
| wnba_games | canonical | `MISSING` |
| wnba_player_game_stats | canonical | `MISSING` |
| wnba_player_game_features_wide | generated | `MISSING` |
| wnba_player_game_features_long | generated | `MISSING` |
| feature_schema_manifest | generated | `MISSING` |

## Owner blockers (exact)

- **wnba_games** (canonical): expected `data/processed/wnba_games.parquet`; registry_entry=True, api_redownload_possible=True; affected_props=ALL (features cannot be rebuilt)
- **wnba_player_game_stats** (canonical): expected `data/processed/wnba_player_game_stats.parquet`; registry_entry=True, api_redownload_possible=True; affected_props=ALL (features cannot be rebuilt)
- **wnba_player_game_features_wide** (generated): expected `data/processed/wnba_player_game_features_wide.parquet`; registry_entry=False, api_redownload_possible=False; affected_props=['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'turnover']
- **wnba_player_game_features_long** (generated): expected `data/processed/wnba_player_game_features_long.parquet`; registry_entry=False, api_redownload_possible=False; affected_props=['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'turnover']
- **feature_schema_manifest** (generated): expected `artifacts/models/stage4_baseline/feature_manifest.json`; registry_entry=False, api_redownload_possible=False; affected_props=['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'turnover']
