"""Market-vs-market edge detection (Definition B: soft book vs sharp consensus).

This package contains the soft-book +EV line-shopping scan. It is deliberately
independent of the PMF model: it finds individual books whose posted price is
better (for the bettor) than the no-vig consensus of the wider book set. No
model and no information edge is required — it is pure market-vs-market shopping.

See ``docs/SOFT_BOOK_EDGE.md`` for the definitions, devig method, and EV formula.
"""

from wnba_props_model.edge.soft_book_scan import (
    SHARP_BOOKS,
    american_to_decimal_profit,
    ev_fraction,
    scan_soft_book_edges,
)

__all__ = [
    "SHARP_BOOKS",
    "american_to_decimal_profit",
    "ev_fraction",
    "scan_soft_book_edges",
]
