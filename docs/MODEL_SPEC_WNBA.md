# WNBA PMF Player Props + Game Totals Model Spec

## Player prop architecture

The model is a WNBA port of the NBA PMF design:

1. Build ex-ante player-game features.
2. Predict a minutes quantile distribution.
3. Predict per-minute rate quantile distributions for high-frequency counting stats.
4. Convert minutes × rate into a discrete PMF through a Poisson layer.
5. Use sparse hurdle models for steals and blocks.
6. Build combo PMFs by discrete convolution.
7. Fit role-aware CDF/PIT calibrators from walk-forward OOF PMFs.
8. Deliver full atom PMFs, fair prices, no-vig market comparisons, and after-game scoring.

## Direct stats

`pts`, `reb`, `ast`, `fg3m`, `tov`, `stl`, `blk`

## Combo stats

`pa`, `pr`, `ra`, `pra`, `stocks`

## WNBA-specific changes

- Regulation game length is 40 minutes, so player minutes are clipped to [0, 40].
- No BDL lineup endpoint is documented for WNBA; the champion path uses `expected_starter` and recent minutes proxies.
- WNBA prop market list is narrower than NBA. The delivery layer maps only supported BDL player prop types to model stats.

## Game totals model

The game total model trains quantile regressors for `game_total = home_score + away_score` using:

- home/away recent scoring
- home/away points allowed
- rest days
- postseason flag
- optional no-vig market total for diagnostics only, not as a core model feature unless explicitly training a market-residual variant

Output is a full `game_total` PMF over integer totals, so every total line can be priced by direct tail sum.

## Calibration

Use the same philosophy as the NBA model:

- OOF PMFs only for calibrator fitting.
- Randomized PIT for discrete PMFs.
- Isotonic CDF remap with role-aware shrinkage.
- `inactive_risk` global-only.
- Guarded fallback when stat×role NLL is worsened.
- Report NLL, RPS/CRPS, PIT mean/std/KS, ECE, mean error, and variance error.

## Gates

| Gate | Threshold |
|---|---:|
| PIT KS | <= 0.075 |
| ECE | <= 0.025 |
| abs(mean error) | <= 0.15 |
| abs(variance error) | <= 0.20 |
| UCB95 log-loss delta | < -0.0025 |
| UCB95 Brier delta | < -0.0010 |
| Market superiority min rows | 100 per stat×role cell |
| Production simulation draws | 50,000 |
| OOF walk-forward window | 28 days |
| Minimum OOF training history | 365 days |
