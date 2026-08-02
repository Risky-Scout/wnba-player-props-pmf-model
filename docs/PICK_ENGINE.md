# WNBA Pick Engine v1

Daily model-selection board for WNBA player props. The engine ranks valid
candidate sides from the independent active-player PMF against exact executable
sportsbook prices, using a same-time external reference market for shrinkage —
without requiring long-run market-superiority certification before showing
selections.

## Probability tracks (always separate)

| Track | Meaning |
|---|---|
| `pure_probability` | Active-conditional settled O/U from the frozen pure PMF (`active_pmf_json` → `settled_probabilities_from_pmf`). No sportsbook price, consensus, closing line, or zero-residual tilt. |
| `reference_market_probability` | Same-time no-vig consensus from **other** books (candidate book excluded). |
| `production_probability` | Conservative fair/pricing probability (may be market-consistent). **Not** the pick signal. |
| `pick_probability` | Selection probability: \(\mathrm{logit}(p_\text{pick})=\mathrm{logit}(p_\text{ref})+w\cdot[\mathrm{logit}(p_\text{pure})-\mathrm{logit}(p_\text{ref})]\), \(w\in[0,1]\). |

Never set `pick_probability = production_probability` merely because production
uses a market-consistent PMF. Never treat a zero-residual market probability
compared to its own market as a model conclusion.

## Supported markets (v1)

`player_points`, `player_rebounds`, `player_assists`, `player_threes`,
`player_steals`, `player_blocks`, `player_turnovers`.

Combination markets are excluded until a fitted joint model exists.

## Outputs

```
deliveries/pick_engine/<date>/<timestamp>/
  ranked_selections.csv
  provisional_picks.csv
  abstentions.csv
  pick_manifest.json
```

Selection statuses:

- `DAILY_RANKED_SELECTION` — valid ranked model opinion (not necessarily +EV)
- `PROVISIONAL_MODEL_PICK` — validity gates pass, raw & conservative EV > 0, reliability > 0, no unresolved OOD/availability warning (certification **not** required)
- `CERTIFIED_MODEL_PICK` — also passes the separate long-run prospective gate
- `NO_POSITIVE_CONSERVATIVE_EV` — ranked opinion without positive conservative EV

## Commands

```bash
# Fit reliability weights from chronological OOF / scored history
PYTHONPATH=src python scripts/fit_pick_engine_reliability.py

# Live / dry-run slate
PYTHONPATH=src python scripts/run_pick_engine.py \
  --date 2026-08-02 \
  --quotes data/snapshots/soft_book_quotes/snapshot_date_utc=2026-08-02 \
  --pmfs deliveries/tonight/full_pmfs_wide.parquet \
  --fair-odds deliveries/tonight/fair_odds_board.parquet

# August 1 frozen retrospective (does not modify frozen forecasts)
PYTHONPATH=src python scripts/replay_aug1_pick_engine.py
```

## Certification (separate)

Provisional picks are never blocked while certification accumulates. Certified
status requires per-stat / approved segment: ≥300 settled rows, ≥30 game dates,
chronological prospective evidence, date-clustered inference, positive log-loss
or EV evidence, acceptable calibration, no catastrophic period, and
multiple-testing correction.

## PR #99

`cursor/wnba-sharp-pmf-v3` (PR #99) remains an unfinished research branch and is
not merged by this work. August 1 frozen forecasts are immutable prospective
evidence.
