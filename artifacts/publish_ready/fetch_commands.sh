#!/usr/bin/env bash
set -euo pipefail
python scripts/fetch_data.py wnba_player_game_stats
python scripts/fetch_data.py wnba_games
python scripts/fetch_data.py wnba_player_game_features_wide
python scripts/fetch_data.py oof_predictions
python scripts/fetch_data.py atomic_sides
python scripts/fetch_data.py atomic_pairs
