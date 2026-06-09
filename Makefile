.PHONY: test compile

compile:
	python -m compileall src scripts

test:
	pytest -q

pull-history:
	python scripts/pull_bdl_history.py --start-season 2022 --end-season 2026

train-player:
	python scripts/train_player_pmfs.py --player-stats data/raw/bdl/wnba_player_game_stats.parquet --games data/raw/bdl/wnba_games.parquet

train-totals:
	python scripts/build_game_totals.py --games data/raw/bdl/wnba_games.parquet

oof:
	python scripts/build_oof_pmfs.py --player-stats data/raw/bdl/wnba_player_game_stats.parquet --games data/raw/bdl/wnba_games.parquet --out-path data/processed/oof_pmfs.parquet --draws 5000

calibrate:
	python scripts/fit_calibrators.py --oof-pmfs data/processed/oof_pmfs.parquet
