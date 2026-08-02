"""PMF feature-selection scoring — distinct from binary Over-classifier selection.

The legacy ablation harness selected features by training a **binary Over classifier** with
the sportsbook line as an input feature and scoring log-loss / Brier / AUC of ``P(Y>L)``.
That is *binary* feature selection. The production model instead estimates a coherent count
PMF ``P(Y=y|X)`` and must derive every line's probability from that one distribution.

This module provides the **PMF selection** scoring surface required by the directive:

    count log score, CRPS, mean bias, variance calibration, zero-rate calibration,
    tail calibration, and line-level log loss / Brier derived from the PMF via push-safe
    settlement — so a feature set is judged by the distribution it produces, not by a
    surrogate binary classifier.

It also provides :func:`assert_cdf_coherent`, the monotonicity guard an *optional* binary
residual head must pass before it may replace the PMF: ``P(Y>L2) <= P(Y>L1)`` whenever
``L2 > L1``. The sportsbook line may enter the evaluator / a distributional correction layer,
never the primary PMF generator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wnba_props_model.ablation import metrics as M
from wnba_props_model.opportunity.pmf_builders import (
    pmf_mean,
    pmf_variance,
    settled_over_probability,
)


@dataclass(frozen=True)
class PMFSelectionScores:
    n: int
    count_log_score: float          # mean -log P(Y=y) under the PMF (lower is better)
    crps: float                     # mean discrete CRPS (lower is better)
    mean_bias: float                # mean(E[Y]) - mean(y)   (0 is best)
    variance_calibration: float     # mean(Var_pmf) / var(residual)  (~1 is best)
    zero_rate_calibration: float    # mean P(Y=0) - empirical P(y=0)  (0 is best)
    tail_calibration: float         # mean P(Y>=q95_emp) - 0.05  (0 is best)
    line_log_loss: float | None     # log loss of settled P(over) at provided lines
    line_brier: float | None        # Brier of settled P(over) at provided lines

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "count_log_score": self.count_log_score,
            "crps": self.crps,
            "mean_bias": self.mean_bias,
            "variance_calibration": self.variance_calibration,
            "zero_rate_calibration": self.zero_rate_calibration,
            "tail_calibration": self.tail_calibration,
            "line_log_loss": self.line_log_loss,
            "line_brier": self.line_brier,
        }


def score_pmfs(
    pmfs: list[np.ndarray | None],
    y: np.ndarray,
    lines: np.ndarray | None = None,
) -> PMFSelectionScores:
    """Score a list of count PMFs against observed integer outcomes ``y``.

    ``pmfs[i]`` is a 1-D array over ``0..K_i`` (may be ``None`` to skip a row). ``lines`` is an
    optional per-row sportsbook line used only to derive line-level log loss / Brier from the
    PMF via push-safe settlement (the line is NOT used to build the PMF).
    """
    y = np.asarray(y, dtype=float)
    valid = [i for i, p in enumerate(pmfs) if p is not None]
    if not valid:
        raise ValueError("score_pmfs: no valid PMFs")

    means = np.array([pmf_mean(pmfs[i]) for i in valid])
    varis = np.array([pmf_variance(pmfs[i]) for i in valid])
    yv = y[valid]

    residual_var = float(np.var(yv - means)) if len(yv) > 1 else float("nan")
    var_cal = float(np.mean(varis) / residual_var) if residual_var > 1e-9 else float("nan")

    # zero-rate calibration
    p_zero = np.array([pmfs[i][0] if pmfs[i].size > 0 else 0.0 for i in valid])
    zero_cal = float(np.mean(p_zero) - np.mean(yv == 0))

    # tail calibration at the empirical 95th percentile
    q95 = float(np.quantile(yv, 0.95)) if len(yv) else 0.0
    k95 = int(np.ceil(q95))
    p_tail = np.array([float(pmfs[i][k95:].sum()) if pmfs[i].size > k95 else 0.0 for i in valid])
    tail_cal = float(np.mean(p_tail) - np.mean(yv >= k95))

    line_ll = line_bs = None
    if lines is not None:
        lv = np.asarray(lines, dtype=float)[valid]
        p_over, y_over = [], []
        for j, i in enumerate(valid):
            L = lv[j]
            if not np.isfinite(L):
                continue
            po, _pu, _pp = settled_over_probability(pmfs[i], float(L))
            # push rows (integer line, y == L) are not settled
            if float(L).is_integer() and yv[j] == L:
                continue
            p_over.append(po)
            y_over.append(1.0 if yv[j] > L else 0.0)
        if p_over:
            p_over = np.clip(np.array(p_over), 1e-6, 1 - 1e-6)
            y_over = np.array(y_over)
            line_ll = M.log_loss(y_over, p_over)
            line_bs = M.brier(y_over, p_over)

    return PMFSelectionScores(
        n=len(valid),
        count_log_score=float(M.pmf_log_score([pmfs[i] for i in valid], yv)),
        crps=float(M.crps_discrete([pmfs[i] for i in valid], yv)),
        mean_bias=float(np.mean(means) - np.mean(yv)),
        variance_calibration=var_cal,
        zero_rate_calibration=zero_cal,
        tail_calibration=tail_cal,
        line_log_loss=line_ll,
        line_brier=line_bs,
    )


def pmf_over_probabilities(pmf: np.ndarray, lines) -> np.ndarray:
    """Raw survival P(Y>L) for each line, derived from a single PMF (always coherent)."""
    a = np.asarray(pmf, dtype=float)
    a = a / a.sum()
    k = np.arange(a.size)
    return np.array([float(a[k > float(L)].sum()) for L in lines])


def assert_cdf_coherent(lines, p_over, tol: float = 1e-9) -> None:
    """A coherent line model must satisfy ``P(Y>L2) <= P(Y>L1)`` whenever ``L2 > L1``.

    This is the guard an **optional line-aware binary residual head** must pass before it
    may replace the PMF: independently-fit per-line probabilities can otherwise contradict
    each other. A single PMF is coherent by construction (see :func:`pmf_over_probabilities`);
    a binary head is not, so its per-line outputs are what must be checked here.

    Args:
        lines: per-line values (any order).
        p_over: matching P(over) for each line (e.g. from the binary head).
    """
    ls = np.asarray([float(x) for x in lines], dtype=float)
    ps = np.asarray([float(x) for x in p_over], dtype=float)
    order = np.argsort(ls)
    ls, ps = ls[order], ps[order]
    for i in range(1, len(ls)):
        if ls[i] > ls[i - 1] and ps[i] > ps[i - 1] + tol:
            raise ValueError(
                f"incoherent line model: P(Y>{ls[i]}) = {ps[i]:.6f} > "
                f"P(Y>{ls[i-1]}) = {ps[i-1]:.6f} (survival must be non-increasing in the line)")


def better_on_pmf(candidate: PMFSelectionScores, reference: PMFSelectionScores,
                  material: float = 1e-4) -> dict:
    """Advancement decision on PMF proper scores (directive S20): a candidate advances only
    if it does not materially worsen the count log score or CRPS versus the reference."""
    d_cls = candidate.count_log_score - reference.count_log_score
    d_crps = candidate.crps - reference.crps
    return {
        "delta_count_log_score": d_cls,
        "delta_crps": d_crps,
        "count_log_score_not_worse": d_cls <= material,
        "crps_not_worse": d_crps <= material,
        "advances": (d_cls <= material) and (d_crps <= material),
    }
