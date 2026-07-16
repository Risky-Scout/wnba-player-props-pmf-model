Full model-retrain sentinel for "Daily WNBA PMF Pipeline".

Bumping this file (any content change) on main fires exactly one full model
retrain, which regenerates artifacts/models/stage4_baseline/artifact_manifest_model.json
with config_hash = sha256(config/model/stage4_baseline.yaml)[:16] for the CURRENT
main, then dispatches pregame_initial. Use this when workflow_dispatch is
unavailable (app token lacks actions:write).

retrain-request: 2026-07-16T08:00Z
reason: weekly_calibration run 29466162581 committed an "Auto-update dispersion r
        from OOF" change to config/model/stage4_baseline.yaml (commit 531dcc6).
        The existing daily model artifact's config_hash (88be734e12563a11) no
        longer matches main's stage4_baseline.yaml hash (d2604a17f085b2ad),
        blocking pregame_initial's BLOCKING artifact-manifest validation. A fresh
        retrain on current main re-syncs the model config_hash (and trains the
        production model with the quantile-loss objective + rebuilt dispersion).
