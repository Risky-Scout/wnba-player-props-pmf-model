Full OOF rebuild sentinel for "Weekly OOF Refresh & Calibration".

Bumping this file (any content change) on main fires exactly one full
out-of-fold rebuild + calibrator refit, then auto-triggers pregame_initial.
Use this when workflow_dispatch is unavailable (app token lacks actions:write).

rebuild-request: 2026-07-16T02:00Z
reason: Refit calibrators on quantile-loss (median) OOF to remove systematic
        under-prediction. rate_model.py switched pts/reb/ast HGB objective to
        quantile=0.5 (median) on 2026-07-10; the production calibrators
        (2026-07-13, run 29249816558) were fit on a reused MSE-era OOF artifact.
