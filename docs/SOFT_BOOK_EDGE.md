# Soft-Book +EV Edge Board (Definition B)

Pure **market-vs-market** line shopping for WNBA player props. No PMF model, no
information edge — this finds individual books whose posted price is better (for the
bettor) than the no-vig consensus of the wider book set. This is where realistic prop
profit lives for a price-taker.

---

## Two edges — do NOT conflate them

| | Definition A (model vs market) | **Definition B (soft book vs sharp consensus)** |
|---|---|---|
| Source | `deliver.py` (`model_prob_over_final` vs `market_prob_over_no_vig`) | `src/wnba_props_model/edge/soft_book_scan.py` |
| Requires | a model that beats the closing line | nothing — just multiple book prices |
| Our finding | ablation showed the model does **not** beat the sharp line | a soft price is objectively +EV vs the consensus fair line |
| Used by this board | ❌ ignored | ✅ **this is the board** |

Definition B needs no forecasting skill. If Book X pays `+120` on an outcome the rest of
the market fairly prices at 50%, betting Book X is +EV regardless of who wins tonight —
because you are getting paid more than the fair probability warrants, over many such bets.

---

## Method

For each `(event, player, stat, line)` group:

1. **Per-book de-vig.** For every book that posts **both** the over and the under at that
   line, remove the vig from its two-sided price with Shin's method
   (`shin_no_vig_two_way`) to get that book's fair `P(over)` / `P(under)`. A one-sided
   book cannot be de-vigged and is dropped.

2. **Robust consensus.** The consensus fair `P(over)` is the **median** of the per-book
   fair `P(over)` across all books posting a two-sided price. When scoring a *specific*
   book, that same book is **excluded from its own consensus** (no self-reference). We
   record `consensus_n_books` = number of other books used.

   *Why median-of-all rather than sharp-only?* The median is robust to a single mispriced
   outlier and does not require hard-coding which books are "sharp". Known-sharper books
   (`pinnacle`, `betonlineag`, `lowvig`) are **annotated** (`is_sharp_book`,
   `sharp_consensus_p_over`) so a human can prefer edges that also agree with the sharp
   subset — but the qualifying EV is always computed against the median-of-all consensus.

3. **Expected value.** For each book+side, with `fair_p` the consensus fair probability
   for that side and `decimal_profit` the net payout multiple of the book's offered
   American odds:

   ```
   EV = fair_p * decimal_profit - (1 - fair_p)
   ```

   where `decimal_profit = odds/100` for positive American odds and `100/|odds|` for
   negative. `EV > 0` means the book's price pays more than the fair probability warrants.

4. **Qualify.** A row is flagged (`qualified = true`) when `EV >= threshold`
   (default **2.5%**, configurable) **and** `consensus_n_books >= min_consensus_books`
   (default **3**).

### Guards (junk protection)

- `consensus_n_books >= 3` after self-exclusion (thin markets never qualify).
- Valid American odds: `|odds| >= 100`.
- Both sides present for the scored book (required to de-vig it).
- Stale rows dropped: `commence_time` already in the past.

---

## Outputs

- **Authoritative board:** `artifacts/edge_board/SOFT_BOOK_EDGE_<date>.json`
  - `summary` — event / book / scored-row / qualifying counts, books seen, books per stat.
  - `board` — the qualifying +EV plays (sorted by EV desc) with columns: `player_name`,
    `team`, `stat`, `line`, `side`, `book`, `offered_odds`, `fair_p`, `ev_pct`,
    `consensus_n_books`, `consensus_p_over`, `is_sharp_book`, `best_book`, `best_odds`, …
  - `top_by_ev_all` — top 20 by EV including below-threshold near-misses (transparency).

- **Render board (odds-scanner schema):**
  `tools/odds-scanner/predictions/WNBA/Soft-Book-Edge/{latest,<date>}.json` — the same
  data reshaped into the pre-game render schema (`games → players → stat_projections →
  calibrated_p_over`) so the static odds-scanner renders it.

Raw two-sided quote snapshots are written under `data/snapshots/soft_book_quotes/`
(**gitignored — never committed**).

---

## How to read the board

In the odds-scanner card:

- **Player + stat pill** — the prop.
- **Meta line** (`hardrockbet +105 · EV +3.6%`) — the soft book, its offered odds, and the
  EV as a **positive magnitude**.
- **Big number** — the line.
- **No-vig Over%** — `p_over`, the consensus fair `P(over)` (always the over probability,
  regardless of which side is the +EV play).
- **Edge badge** — the SIDE and color: green `▲ OVER` for over plays, red `▼ UNDER` for
  under plays. In the shared render widget `calibrated_p_over.edge_vs_market` carries the
  **signed** EV (positive → over/green, negative → under/red); the `|value|` is the EV.
  The meta line always shows the EV as a positive number to avoid ambiguity.
- **Kelly** — quarter-Kelly stake fraction from the consensus fair prob and the offered odds.

To view locally (relative fetch requires a server, per `AGENTS.md`):

```bash
cd tools/odds-scanner && python -m http.server 8899
# then open:
#   http://localhost:8899/index.html?pregame_src=predictions/WNBA/Soft-Book-Edge/latest.json
```

---

## Running it

```bash
# 1. Collect two-sided quotes for today's slate across all US books (us,us2).
PYTHONPATH=$(pwd)/src python3 scripts/collect_soft_book_quotes.py            # today UTC
# 2. Scan + build the board.
PYTHONPATH=$(pwd)/src python3 scripts/build_soft_book_edge_board.py --date $(date -u +%Y-%m-%d)
```

Automated by `.github/workflows/soft_book_edge.yml` (hourly through the afternoon/evening
ET window, fail-open, commits only the board JSON — never raw data, no-ops out of season).

---

## Practical notes

- `pts / reb / ast / fg3m` post across ~9–10 US books early in the season and are the
  primary source of soft-book edges. `stl / blk / tov` are thin/unoffered right now and
  usually contribute nothing until deeper into the season (the collector still requests
  them so they accrue automatically once books post them).
- A single night frequently shows **few or zero** qualifying edges — efficient markets are
  the norm. The value is cumulative: run the board multiple times pre-tip, every day, and
  the +EV plays accrue over the week. The machinery + cron is the durable deliverable.
