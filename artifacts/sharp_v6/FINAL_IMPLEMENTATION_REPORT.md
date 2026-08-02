# WNBA Sharp PMF V6 — Final Implementation Report (Consolidation)

## Authoritative production
- Inference: `wnba_props_model.sharp_v6.inference.predict_slate`
- Bundle: `artifacts/releases/wnba-pmf-production-v1/`
- Daily workflow: `.github/workflows/wnba_pmf_daily.yml`
- CLI: `python scripts/run_wnba_pmf.py --bundle-dir artifacts/releases/wnba-pmf-production-v1`

## Fitted components
- Participation (HGB + selected isotonic/identity)
- Minutes discrete PMF with team regulation=200, Q1=50, shared OT state
- Game environment (possessions/pace/FGA/3PA/FTA/misses/reb opps/assist/tov/OT)
- Direct stats with minutes mixture; pts structural shooting; reb structural OREB+DREB
- STL/BLK/TOV family selection via chronological NLL (hurdle vs NB2)
- Persisted cross-fit calibrators (identity only when selected)
- Gaussian copula dependence for combo markets
- Q1 nested minutes; first-basket competing risk

## Live slate (2026-08-02)
- 3 games, 128 players, 896 PMFs, 188 ranked picks
- No market superiority claimed

## Legacy
- sharp_v3/v4/v5 marked RESEARCH_ONLY / DEPRECATED
- Stage-4 `predict_today.py` LEGACY_CONTROL; daily_pipeline superseded
