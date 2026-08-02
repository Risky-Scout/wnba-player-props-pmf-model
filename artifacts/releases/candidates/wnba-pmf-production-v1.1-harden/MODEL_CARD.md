# wnba-pmf-production-v1.1

Authoritative WNBA player-prop PMF bundle (generated from MANIFEST).

- Inference: `wnba_props_model.sharp_v6.inference.predict_slate`
- Code SHA: `987c07fab0315ac026c446ed0ca7d9c9aedca17c`
- Bundle hash (model_bundle.pkl): `3005c790e705843b7e9ce8633a0855bf99b53fe23aab92112fdd405df51b3b92`
- Feature-contract hash: `c2fd1d592d88e3f755779258ae8f5b9b1e089bbd0f4908d7b9f3f055b522a94e`
- Training cutoff: `2025-10-31`
- Selected families: `{"pts": "structural_shooting", "reb": "structural_oreb_dreb", "ast": "minutes_mixture_nb2", "fg3m": "minutes_mixture_nb2", "stl": "hurdle_nb2", "blk": "minutes_mixture_nb2", "turnover": "hurdle_nb2"}`
- Calibration: `{"pts": "identity", "reb": "identity", "ast": "identity", "fg3m": "identity", "stl": "identity", "blk": "identity", "turnover": "identity"}`
- Supported markets: `["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover", "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast", "q1_pts", "q1_reb", "q1_ast", "first_basket"]`
- Withheld: `{"fantasy_points": "requires operator scoring configuration at runtime", "double_double": "derived from joint sims; enabled when dependence present", "triple_double": "derived from joint sims; enabled when dependence present"}`
- Daily inference loads this bundle and does not retrain.
- Command: `python scripts/run_wnba_pmf.py --bundle-dir artifacts/releases/wnba-pmf-production-v1.1`
