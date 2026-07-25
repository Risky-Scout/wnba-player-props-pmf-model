# Clean-Fetch Verification

- generated: 2026-07-25T01:25:12.618436+00:00

- **local_run_ready**: True
- **durable_data_ready**: False
- **clean_fetch_verified**: False
- **reproducible_run_ready**: False

| dataset | local | remote asset | status |
|---|---|---|---|
| wnba_games | True | False | `LOCAL_ONLY_UNPUBLISHED` |
| wnba_player_game_stats | True | False | `LOCAL_ONLY_UNPUBLISHED` |
| wnba_player_game_features_wide | True | False | `LOCAL_ONLY_UNPUBLISHED` |
| wnba_player_game_features_long | True | False | `LOCAL_ONLY_UNPUBLISHED` |
| feature_schema_manifest | True | False | `LOCAL_ONLY_UNPUBLISHED` |

## Blocker

Remote releases processed-data-v1 / processed-features-v2 do not exist; no write credential available to publish (cursor account push=false; GH_TOKEN not injected). Owner must publish or inject a write-scoped token.
