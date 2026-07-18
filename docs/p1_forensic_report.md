# P1 Forensic Report — origin of the +64.6% / +113% ROI (artifact 29626281415)

## Verdict on the prior run: INVALID (not a real evaluation)

### Exact ROI mechanism
Every graded wager used a **synthetic consensus price** — the *median of American
odds across books* produced by `build_consensus()`. American odds do not average:
the median of values straddling the ±100 boundary (e.g. `median(-110, +100) = -5`)
and other small-magnitude results yield **invalid "odds"** such as `-0.5, -1.0, -2.0`.

`profit_at_american(-0.5, win) = 100/|-0.5| = 200 units` on a 1-unit stake. The
permissive `_valid_american()` (accepted any finite non-zero number) let these
through, so a handful of "wins" at fabricated −0.5/−1.0 prices paid 40–200 units
each and inflated both sides' ROI. The three −0.5 Under rows alone contributed
~600 of the 1,469 total Under profit units.

### Counts found
- Invalid `american_odds` in the raw quotes `(-100, 100)`: **0** (raw quotes were valid).
- Invalid consensus prices `|odds| < 100` (median-straddle artifacts): **146 Under + 172 Over = 318**
  in the closing consensus; **67 Under + 92 Over** entered the graded eval table.
- **100% of graded bets** priced off synthetic consensus medians (non-executable).
- Stale-closing groups (>1 closing line per book/player/stat under the buggy
  `game_id+player_id+stat+book+line` grouping): **2,090 of 13,048**.
- Extreme price outliers surviving as 1-book medians: over_odds up to **+4000**,
  under_odds down to **−10000**.

### Root causes (all confirmed in code)
1. `select_open_close()` grouped by `...+book+line` → stale earlier lines labeled closing.
2. `build_consensus()` created synthetic median Over/Under prices, used for ROI.
3. `_valid_american()` accepted any finite non-zero number (e.g. −0.5).
4. American odds were averaged/medianed directly (`avg_american_odds`, consensus medians).
5. Event coverage denominator used all daily-snapshot events, not eligible canonical OOF games.
6. The replay used a simplified selector, not the production recommendation policy.

Repair replaces synthetic prices with **exact executable quote-level rows** (quote_id),
enforces strict American-odds validation, groups closing selection by book only,
computes bet-level P&L from executable prices, fixes the coverage denominator, shares
the production selector, and reconstructs fold-safe calibration.

### Repaired result (executable prices, same OOF, same period)
Grading the identical recommendations against the **exact executable book price** at a
decision snapshot (tip − 12h), with fold-safe walk-forward calibration:

| | prior (synthetic median) | repaired (executable) |
|---|---|---|
| Under ROI | **+64.6%** | **+3.5%** (95% CI −3.3% .. +10.7%, spans 0) |
| Over ROI | **+113%** | −(spans 0) |
| invalid-price rows graded | 67 Under / 92 Over | **0** |
| coverage denominator | all snapshot events | **eligible canonical OOF games (85.1%)** |
| model vs market log-loss | — | model **worse** (+0.10) |

The Under lean is **INCONCLUSIVE** on repaired data (interval spans zero and the model
scores worse than the market on the graded bets) — a statistical conclusion, not a
plausibility cap. The +64.6%/+113% figures were entirely synthetic-price artifacts.
