# CLV Methodology

## Summary

This document defines the correct terminology and computation for Closing Line Value (CLV) in the WNBA player-prop model.

---

## Definitions

### model_edge_at_entry (NOT CLV)

The difference between the structural model's probability and the entry-time no-vig market probability:

```
model_edge_at_entry = model_p_over - entry_no_vig_p_over
```

This is **not CLV**. It requires only a current market quote. Positive = model is more bullish than the market at entry time.

### model_edge_vs_open (NOT CLV)

The difference between the structural model's probability and the opening-line no-vig probability:

```
model_edge_vs_open = model_p_over - opening_no_vig_p_over
```

This is **not CLV**.

### True CLV — Same-Line Price CLV

Requires an archived closing quote. Only valid when entry and close use the **same line**:

```
same_line_price_clv = closing_no_vig_p_selected - entry_no_vig_p_selected
```

Sign convention: positive = bet became more valuable (closing market assigned higher probability to the bet's selected side, meaning entry odds were generous).

### True CLV — Side-Adjusted Line CLV

```
# For over ticket:
line_clv = closing_line - entry_line   (positive = better number at entry)

# For under ticket:
line_clv = entry_line - closing_line   (positive = better number at entry)
```

### Ticket EV at Close

Requires a monotonic closing probability curve at the entry line. Only computed when sufficient alternate-line or cross-market data exists. Returns `NOT_AVAILABLE` otherwise.

---

## No-Vig Computation

```python
def american_to_implied(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

p_over_nv = p_over_raw / (p_over_raw + p_under_raw)
p_under_nv = p_under_raw / (p_over_raw + p_under_raw)
```

---

## What Is NOT Available

- True CLV requires an **archived closing quote** pulled before tip-off.
- BALLDONTLIE does not provide unlimited historical prop snapshots.
- Historical CLV can only be computed where snapshots were archived.
- Do not backfill fabricated closing quotes.

---

## Closing Quote Selection

The definitive closing quote for a given (game, player, stat, market_type, vendor) is:

```
pulled_at_utc < scheduled_start_utc
source_updated_at <= scheduled_start_utc
```

The **latest** valid quote satisfying both conditions is the close.

---

## Directional Movement (Descriptive Only)

Line movement direction (whether the line moved up or down from open to close) is a **descriptive** metric only. It is not the primary CLV definition and must not be reported as CLV.

The `backtest_clv.py` script computes `model_edge_vs_open_agreement_rate`, which measures whether the model's significant-edge direction agreed with line movement direction. This is explicitly labeled "NOT CLV" in all outputs.

---

## Opening, Entry, and Closing Concepts

Three distinct quotes must be tracked separately:

| Concept | Definition |
|---------|-----------|
| **Opening quote** | First quote observed for the market |
| **Entry quote** | Hypothetical or actual price at time of bet placement |
| **Closing quote** | Latest valid quote before scheduled tip-off |

Where no wager-execution log exists, results are labeled **hypothetical quote CLV**.

---

## Implementation

See `src/wnba_props_model/evaluation/clv.py` for the full implementation.
