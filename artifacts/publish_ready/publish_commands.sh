#!/usr/bin/env bash
set -euo pipefail
# RUN ONLY against a PRIVATE store (see PUBLICATION_STATUS.json). GH_TOKEN must have write scope.
gh release create oof-data-v1 'artifacts/models/calibration/oof_predictions.parquet' --repo Risky-Scout/wnba-player-props-pmf-model --title 'oof-data-v1' --notes 'immutable oof_predictions' || gh release upload oof-data-v1 'artifacts/models/calibration/oof_predictions.parquet' --repo Risky-Scout/wnba-player-props-pmf-model --clobber
