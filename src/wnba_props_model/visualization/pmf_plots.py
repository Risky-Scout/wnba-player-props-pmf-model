"""PenaltyBlog-standard PMF visualization functions.

All plotting functions return matplotlib Figure objects for maximum flexibility:
they can be shown interactively, saved to PNG, or embedded in HTML as base64.

Key plot types
--------------
plot_player_pmf        — Individual player PMF bar chart (blue/green/gray coloring)
plot_game_total_pmf    — Game total distribution with home/away inset subplots
plot_pmf_grid_heatmap  — 2D heatmap: players × outcomes, sorted by mean desc
plot_calibration_curve — PenaltyBlog reliability diagram (10-bin, diagonal guide)
"""
from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from wnba_props_model.models.pmf_grid import WNBAPMFGrid
    from wnba_props_model.models.team_score import WNBATeamScorePMFGrid

logger = logging.getLogger(__name__)

# Colour palette (colorblind-safe)
_BLUE = "#4C72B0"
_GREEN = "#55A868"
_GRAY = "#CCCCCC"
_RED = "#E84040"
_ORANGE = "#FFA500"
_DARK = "#333333"


def _get_plt():
    """Lazy import of matplotlib to avoid hard dependency at module import time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for PMF visualization. "
            "Install with: pip install matplotlib"
        )


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return b64


def plot_player_pmf(
    grid: "WNBAPMFGrid",
    stat: str,
    market_line: float | None = None,
) -> "Figure":
    """PenaltyBlog-style PMF bar chart for one player × stat.

    Bar coloring:
      - Blue  = under (k < market_line)
      - Green = over  (k > market_line)
      - Gray  = push  (k == market_line at integer lines)

    Vertical dashed line at market_line.
    Title: "A.Wilson Pts — 62% Over 17.5"

    Parameters
    ----------
    grid : WNBAPMFGrid
    stat : str
        Stat name (e.g. 'pts', 'reb').
    market_line : float, optional
        If provided, colors the bars and adds the dashed line.
    """
    plt = _get_plt()
    from wnba_props_model.constants import DOMAIN_MAX

    pmf = grid._pmf(stat)
    ks = np.arange(len(pmf))
    domain = min(DOMAIN_MAX.get(stat, len(pmf) - 1), len(pmf) - 1)
    ks = ks[: domain + 1]
    pmf_trim = pmf[: domain + 1]

    # Trim trailing near-zero mass for readability
    nonzero = np.where(pmf_trim > 5e-4)[0]
    if len(nonzero):
        trim_to = nonzero[-1] + 2
        ks = ks[:trim_to]
        pmf_trim = pmf_trim[:trim_to]

    colors = []
    for k in ks:
        if market_line is None:
            colors.append(_BLUE)
        elif abs(k - market_line) < 1e-9:
            colors.append(_GRAY)
        elif k > market_line:
            colors.append(_GREEN)
        else:
            colors.append(_BLUE)

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(ks, pmf_trim, color=colors, edgecolor="white", linewidth=0.5, width=0.85)

    for bar, p in zip(bars, pmf_trim):
        if p > 0.025:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pmf_trim.max() * 0.015,
                f"{p:.0%}",
                ha="center", va="bottom", fontsize=7, color=_DARK, fontweight="bold",
            )

    if market_line is not None:
        ax.axvline(market_line, color=_RED, linewidth=1.8, linestyle="--", alpha=0.85)
        p_over = grid.prob_over(stat, market_line)
        p_push = grid.push_prob(stat, market_line)
        push_str = f" | {p_push:.0%} push" if p_push > 0.005 else ""
        title = (
            f"{grid.player_name} — {stat.upper()} | "
            f"{p_over:.0%} Over {market_line}{push_str} · "
            f"Mean: {grid.pmf_mean(stat):.1f}"
        )
    else:
        title = (
            f"{grid.player_name} — {stat.upper()} | "
            f"Mean: {grid.pmf_mean(stat):.1f} · σ: {grid.pmf_std(stat):.1f}"
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Outcome (count)", fontsize=9)
    ax.set_ylabel("Probability", fontsize=9)

    if len(ks) <= 20:
        ax.set_xticks(ks)
    else:
        ax.set_xticks(ks[::2])
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Role bucket label
    if grid.role_bucket not in ("unknown", ""):
        ax.text(
            0.99, 0.97, grid.role_bucket,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray", style="italic",
        )

    fig.tight_layout()
    return fig


def plot_game_total_pmf(
    grid: "WNBATeamScorePMFGrid",
    market_line: float | None = None,
) -> "Figure":
    """Game total PMF bar chart with home/away score inset subplots.

    Main panel: total score PMF centered near the expected total.
    Inset (upper right): home and away score distributions overlaid.

    Parameters
    ----------
    grid : WNBATeamScorePMFGrid
    market_line : float, optional
        If provided, shows over/under split and dashed line.
    """
    plt = _get_plt()

    total_pmf = grid.total_score_pmf
    total_mean = grid.expected_total()

    # Trim to ±30 around the mean for readability
    lo = max(0, int(total_mean) - 25)
    hi = min(len(total_pmf) - 1, int(total_mean) + 25)
    ks = np.arange(lo, hi + 1)
    pmf_trim = total_pmf[lo: hi + 1]

    colors = []
    for k in ks:
        if market_line is None:
            colors.append(_BLUE)
        elif k > market_line:
            colors.append(_GREEN)
        else:
            colors.append(_BLUE)

    fig = plt.figure(figsize=(12, 5))
    ax_main = fig.add_axes([0.07, 0.12, 0.60, 0.78])

    ax_main.bar(ks, pmf_trim, color=colors, edgecolor="white", linewidth=0.3, width=0.9)

    if market_line is not None:
        ax_main.axvline(market_line, color=_RED, linewidth=2.0, linestyle="--", alpha=0.9)
        p_over = grid.total_over(market_line)
        title = (
            f"{grid.home_team} vs {grid.away_team} — Game Total | "
            f"{p_over:.0%} Over {market_line} · E[total]={total_mean:.1f}"
        )
    else:
        title = (
            f"{grid.home_team} vs {grid.away_team} — Game Total | "
            f"E[total]={total_mean:.1f}"
        )

    ax_main.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax_main.set_xlabel("Combined Score", fontsize=9)
    ax_main.set_ylabel("Probability", fontsize=9)
    ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax_main.tick_params(axis="both", labelsize=8)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Team score inset subplot (upper right)
    ax_inset = fig.add_axes([0.72, 0.35, 0.25, 0.55])

    home_pmf = grid.home_score_pmf
    away_pmf = grid.away_score_pmf
    h_mean = grid.expected_home_score()
    a_mean = grid.expected_away_score()

    lo_t = max(0, int(min(h_mean, a_mean)) - 12)
    hi_t = min(max(len(home_pmf), len(away_pmf)) - 1, int(max(h_mean, a_mean)) + 12)
    ks_t = np.arange(lo_t, hi_t + 1)

    ax_inset.plot(ks_t, home_pmf[lo_t: hi_t + 1], color=_BLUE, lw=2, label=f"{grid.home_team} ({h_mean:.0f})")
    ax_inset.plot(ks_t, away_pmf[lo_t: hi_t + 1], color=_GREEN, lw=2, label=f"{grid.away_team} ({a_mean:.0f})", linestyle="--")
    ax_inset.set_title("Team Scores", fontsize=8, pad=4)
    ax_inset.legend(fontsize=7, frameon=False)
    ax_inset.tick_params(axis="both", labelsize=7)
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)
    ax_inset.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    return fig


def plot_pmf_grid_heatmap(
    player_grids: list["WNBAPMFGrid"],
    stat: str,
    max_players: int = 15,
    max_outcomes: int = 25,
) -> "Figure":
    """2D heatmap: players × outcomes, sorted by pmf_mean descending.

    Parameters
    ----------
    player_grids : list[WNBAPMFGrid]
    stat : str
    max_players : int
        Truncate to this many players (sorted by mean).
    max_outcomes : int
        Truncate outcome axis to this many columns.
    """
    plt = _get_plt()

    grids = [g for g in player_grids if g.has_stat(stat)]
    if not grids:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No players with stat '{stat}'", ha="center", va="center", transform=ax.transAxes)
        return fig

    grids = sorted(grids, key=lambda g: g.pmf_mean(stat), reverse=True)[:max_players]

    player_names = [g.player_name for g in grids]
    max_k = max(len(g._pmf(stat)) for g in grids)
    max_k = min(max_k, max_outcomes + 1)

    matrix = np.zeros((len(grids), max_k))
    for i, g in enumerate(grids):
        pmf = g._pmf(stat)
        n = min(len(pmf), max_k)
        matrix[i, :n] = pmf[:n]

    fig, ax = plt.subplots(figsize=(max(10, max_k * 0.55), max(4, len(grids) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)

    ax.set_yticks(range(len(player_names)))
    ax.set_yticklabels(player_names, fontsize=8)
    ax.set_xticks(range(0, max_k, max(1, max_k // 10)))
    ax.set_xticklabels(range(0, max_k, max(1, max_k // 10)), fontsize=8)
    ax.set_xlabel("Outcome", fontsize=9)
    ax.set_title(f"{stat.upper()} PMF Grid — {len(grids)} Players (sorted by mean)", fontsize=10, pad=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Probability")

    for i in range(len(grids)):
        mean_k = grids[i].pmf_mean(stat)
        if 0 <= mean_k < max_k:
            ax.axvline(mean_k, ymin=i / len(grids), ymax=(i + 1) / len(grids),
                       color="black", linewidth=1.5, alpha=0.6)

    fig.tight_layout()
    return fig


def plot_calibration_curve(
    oof_df: pd.DataFrame,
    stat: str,
    n_bins: int = 10,
) -> "Figure":
    """PenaltyBlog reliability diagram for model calibration.

    Layout:
    - Main panel: predicted probability vs actual frequency (10-bin reliability)
    - Sub panel: histogram of predicted probabilities (bin count)
    - Diagonal: perfect calibration reference
    - Shaded region: acceptable calibration zone (±5%)

    Parameters
    ----------
    oof_df : pd.DataFrame
        Columns: model_prob_over (float), actual_over (0/1), optionally stat (str).
    stat : str
        Stat name for title.
    n_bins : int
        Number of calibration bins.
    """
    plt = _get_plt()

    df = oof_df.copy()
    if "stat" in df.columns:
        df = df[df["stat"] == stat]

    if df.empty or "model_prob_over" not in df.columns or "actual_over" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, f"No calibration data for {stat}", ha="center", va="center", transform=ax.transAxes)
        return fig

    probs = df["model_prob_over"].values.astype(float)
    actuals = df["actual_over"].values.astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means: list[float] = []
    bin_freqs: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_means.append(float(probs[mask].mean()))
            bin_freqs.append(float(actuals[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_means.append(float((lo + hi) / 2))
            bin_freqs.append(float("nan"))
            bin_counts.append(0)

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_axes([0.12, 0.33, 0.82, 0.60])
    ax2 = fig.add_axes([0.12, 0.10, 0.82, 0.20])

    # Perfect calibration diagonal
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Perfect calibration")

    # ±5% shaded zone
    ax1.fill_between([0, 1], [0.05, 1.05], [-0.05, 0.95], alpha=0.08, color="gray")

    # Reliability dots
    for xm, ym, cnt in zip(bin_means, bin_freqs, bin_counts):
        if not np.isnan(ym):
            size = max(20, min(cnt / 5, 200))
            color = _RED if abs(ym - xm) > 0.05 else _GREEN
            ax1.scatter(xm, ym, s=size, color=color, zorder=5, edgecolors="white", linewidth=0.8)

    valid_x = [x for x, y in zip(bin_means, bin_freqs) if not np.isnan(y)]
    valid_y = [y for y in bin_freqs if not np.isnan(y)]
    if valid_x:
        ax1.plot(valid_x, valid_y, "-o", color=_BLUE, lw=1.5, ms=0)

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("Actual frequency", fontsize=9)
    ax1.set_title(f"{stat.upper()} Calibration Curve (n={len(df):,})", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=8)
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xticklabels([])

    # Bin count histogram
    bar_colors = [_RED if abs(bm - bc) > 0.05 and not np.isnan(bm) else _BLUE
                  for bm, bc in zip(bin_freqs, bin_centers)]
    ax2.bar(bin_centers, bin_counts, width=(bins[1] - bins[0]) * 0.85,
            color=bar_colors, alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability", fontsize=9)
    ax2.set_ylabel("Count", fontsize=8)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    return fig


def fig_to_base64(fig) -> str:
    """Export a matplotlib Figure to a base64 PNG string for HTML embedding."""
    return _fig_to_base64(fig)
