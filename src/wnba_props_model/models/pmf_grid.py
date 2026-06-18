"""WNBAPMFGrid — PenaltyBlog-style PMF output object for WNBA player props.

Wraps the full discrete PMF distribution for every stat in a single, clean,
internally-consistent object with one API for all market calculations.

Modeled after PenaltyBlog's FootballProbabilityGrid:
  - One object, many markets: over/under at any line, push handling at integer
    lines, quarter-line Kalshi markets, edge, Kelly stake, narrative, plots.
  - Internally consistent: every market derived from the same PMF array.
  - Correct push semantics: P(over) + P(under) + P(push) == 1.0 always.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from wnba_props_model.constants import DOMAIN_MAX
from wnba_props_model.models.market import fair_american, kelly_from_edge_and_prob
from wnba_props_model.models.simulation import normalize_pmf


class WNBAPMFGrid:
    """Player-level PMF grid for all target stats.

    Parameters
    ----------
    player_id : int | str
    player_name : str
    stat_pmfs : dict[str, np.ndarray]
        Mapping from stat name to normalized PMF array (index = outcome value).
    projected_minutes : float
    role_bucket : str
    game_context : dict
        Arbitrary metadata (game_id, game_date, team_id, opponent_team_id, …).
    """

    def __init__(
        self,
        player_id: int | str,
        player_name: str,
        stat_pmfs: dict[str, np.ndarray],
        projected_minutes: float = 0.0,
        role_bucket: str = "unknown",
        game_context: dict[str, Any] | None = None,
    ) -> None:
        self.player_id = player_id
        self.player_name = player_name
        # Normalize and store copies
        self._pmfs: dict[str, np.ndarray] = {
            stat: normalize_pmf(arr).copy()
            for stat, arr in stat_pmfs.items()
        }
        self.projected_minutes = float(projected_minutes)
        self.role_bucket = role_bucket
        self.game_context: dict[str, Any] = game_context or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pmf(self, stat: str) -> np.ndarray:
        if stat not in self._pmfs:
            raise KeyError(f"Stat '{stat}' not in WNBAPMFGrid for player {self.player_name}. "
                           f"Available: {sorted(self._pmfs)}")
        return self._pmfs[stat]

    def _check_line(self, line: float) -> tuple[float, bool, int | None]:
        """Return (line, is_half_line, integer_k).

        is_half_line=True  → line like 17.5 → no push possible
        is_half_line=False → line like 17.0 → push at k==17 is possible
        """
        frac = line % 1.0
        is_half = abs(frac - 0.5) < 1e-9
        int_k = None if is_half else int(round(line))
        return line, is_half, int_k

    # ------------------------------------------------------------------
    # Core probability accessors
    # ------------------------------------------------------------------

    def prob_over(self, stat: str, line: float) -> float:
        """P(Y > line) — exact tail sum from the PMF.

        Correct for both half-lines (no push) and integer lines (push possible
        but not counted in over). E.g.:
          line=17.5 → P(Y >= 18) = sum(pmf[18:])
          line=17.0 → P(Y >= 18) = sum(pmf[18:])  [push at k=17 is separate]
        """
        pmf = self._pmf(stat)
        k_min = int(np.floor(line)) + 1  # first integer strictly above line
        if k_min >= len(pmf):
            return 0.0
        return float(pmf[k_min:].sum())

    def prob_under(self, stat: str, line: float) -> float:
        """P(Y < line) — exact head sum from the PMF.

        For half-lines: P(Y <= floor(line)) = P(Y < line).
        For integer lines: P(Y < line) = P(Y <= line - 1).
        """
        pmf = self._pmf(stat)
        k_max = int(np.ceil(line)) - 1  # last integer strictly below line
        if k_max < 0:
            return 0.0
        k_max = min(k_max, len(pmf) - 1)
        return float(pmf[: k_max + 1].sum())

    def push_prob(self, stat: str, line: float) -> float:
        """P(Y == line) — non-zero only at integer lines.

        Returns 0 for half-lines (e.g. 17.5 — no push possible).
        """
        _, is_half, int_k = self._check_line(line)
        if is_half or int_k is None:
            return 0.0
        pmf = self._pmf(stat)
        if int_k < 0 or int_k >= len(pmf):
            return 0.0
        return float(pmf[int_k])

    def prob_exactly(self, stat: str, k: int) -> float:
        """P(Y == k) — single atom probability."""
        pmf = self._pmf(stat)
        if k < 0 or k >= len(pmf):
            return 0.0
        return float(pmf[k])

    def quarter_line_probs(self, stat: str, line: float) -> dict[str, float]:
        """Over/push/under for Kalshi/Polymarket quarter-line markets.

        A quarter line (e.g. 12.25) splits the stake 50/50 across the two
        nearest half-lines or integer lines:
          12.25 → 50% at 12.0 + 50% at 12.5
          12.75 → 50% at 12.5 + 50% at 13.0

        Returns {'win': p, 'push': p, 'lose': p} summing to 1.0.
        Betting "over" at a quarter line:
          win  = average over-probability across the two neighbouring lines
          push = average push-probability (only non-zero at integer sub-line)
          lose = average under-probability
        """
        frac = line % 1.0
        is_quarter = abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9

        if not is_quarter:
            # Fall back: treat as a standard line
            return {
                "win": self.prob_over(stat, line),
                "push": self.push_prob(stat, line),
                "lose": self.prob_under(stat, line),
            }

        lo = np.floor(line * 2) / 2  # nearest 0.5 below
        hi = lo + 0.5

        lo_over = self.prob_over(stat, lo)
        lo_push = self.push_prob(stat, lo)
        lo_under = self.prob_under(stat, lo)

        hi_over = self.prob_over(stat, hi)
        hi_push = self.push_prob(stat, hi)
        hi_under = self.prob_under(stat, hi)

        return {
            "win": 0.5 * lo_over + 0.5 * hi_over,
            "push": 0.5 * lo_push + 0.5 * hi_push,
            "lose": 0.5 * lo_under + 0.5 * hi_under,
        }

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def pmf_mean(self, stat: str) -> float:
        """E[Y] — expected value of the distribution."""
        pmf = self._pmf(stat)
        return float(np.dot(np.arange(len(pmf)), pmf))

    def pmf_std(self, stat: str) -> float:
        """Standard deviation of the distribution."""
        pmf = self._pmf(stat)
        ks = np.arange(len(pmf))
        mu = float(np.dot(ks, pmf))
        return float(np.sqrt(np.dot((ks - mu) ** 2, pmf)))

    def pmf_median(self, stat: str) -> float:
        """Median of the discrete distribution (50th percentile)."""
        return self.percentile(stat, 50.0)

    def percentile(self, stat: str, p: float) -> float:
        """p-th percentile of the discrete PMF (p in [0, 100])."""
        pmf = self._pmf(stat)
        cdf = np.cumsum(pmf)
        target = p / 100.0
        idx = int(np.searchsorted(cdf, target - 1e-9))
        return float(min(idx, len(pmf) - 1))

    # ------------------------------------------------------------------
    # Edge and Kelly
    # ------------------------------------------------------------------

    def edge(self, stat: str, line: float, market_prob_over: float) -> float:
        """Model edge vs. market on the over.

        edge = model_prob_over - market_prob_over_no_vig
        Positive → model thinks over is more likely than market prices.
        """
        return self.prob_over(stat, line) - market_prob_over

    def kelly_stake(
        self,
        stat: str,
        line: float,
        market_prob_over: float,
        over_american_odds: float = -110.0,
        bankroll_fraction: float = 0.25,
    ) -> float:
        """Fractional Kelly stake size (fraction of bankroll).

        Uses quarter-Kelly by default (bankroll_fraction=0.25).
        Returns 0 if edge is negative.
        """
        model_p = self.prob_over(stat, line)
        e = model_p - market_prob_over
        if e <= 0:
            return 0.0
        return float(kelly_from_edge_and_prob(e, model_p, fractional_kelly=bankroll_fraction))

    # ------------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------------

    def narrative(self, stat: str, market_line: float | None = None) -> str:
        """One-line human-readable projection summary.

        Example: "A.Wilson: 18.3 pts projected (±4.1), 62% over 17.5, 38% under 17.5"
        """
        mu = self.pmf_mean(stat)
        sd = self.pmf_std(stat)
        name = self.player_name

        if market_line is not None:
            p_over = self.prob_over(stat, market_line)
            p_under = self.prob_under(stat, market_line)
            p_push = self.push_prob(stat, market_line)
            push_str = f", {p_push * 100:.0f}% push" if p_push > 0.005 else ""
            return (
                f"{name}: {mu:.1f} {stat} projected (±{sd:.1f}), "
                f"{p_over * 100:.0f}% over {market_line}"
                f"{push_str}, {p_under * 100:.0f}% under {market_line}"
            )
        return f"{name}: {mu:.1f} {stat} projected (±{sd:.1f}), role={self.role_bucket}"

    def to_dict(self, half_line_step: float = 0.5) -> dict[str, Any]:
        """Serializable dict with all markets at 0.5-increment lines.

        Suitable for JSON export, parquet storage, and downstream systems.
        """
        base = {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "projected_minutes": self.projected_minutes,
            "role_bucket": self.role_bucket,
            **self.game_context,
        }
        stats_out: dict[str, Any] = {}
        for stat, pmf in self._pmfs.items():
            domain = DOMAIN_MAX.get(stat, len(pmf) - 1)
            mu = float(np.dot(np.arange(len(pmf)), pmf))
            sd_val = float(np.sqrt(np.dot((np.arange(len(pmf)) - mu) ** 2, pmf)))
            lines: list[dict[str, Any]] = []
            line = 0.5
            while line <= domain:
                p_over = self.prob_over(stat, line)
                p_under = self.prob_under(stat, line)
                p_push = self.push_prob(stat, line)
                lines.append({
                    "line": line,
                    "p_over": round(p_over, 6),
                    "p_under": round(p_under, 6),
                    "p_push": round(p_push, 6),
                    "fair_over_american": round(fair_american(p_over), 1) if p_over > 0 else None,
                    "fair_under_american": round(fair_american(p_under), 1) if p_under > 0 else None,
                })
                line += half_line_step
            stats_out[stat] = {
                "mean": round(mu, 4),
                "std": round(sd_val, 4),
                "median": self.pmf_median(stat),
                "markets": lines,
            }
        base["stats"] = stats_out
        return base

    def plot_pmf(self, stat: str, market_line: float | None = None):
        """Clean bar chart of the PMF distribution.

        Blue bars = under, green bars = over, gray bar = push at integer line.
        Returns a matplotlib Figure (or None if matplotlib unavailable).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        pmf = self._pmf(stat)
        ks = np.arange(len(pmf))
        domain = min(DOMAIN_MAX.get(stat, len(pmf) - 1), len(pmf) - 1)
        ks = ks[: domain + 1]
        pmf_trim = pmf[: domain + 1]

        # Trim trailing near-zero mass for readability
        nonzero = np.where(pmf_trim > 1e-4)[0]
        if len(nonzero):
            ks = ks[: nonzero[-1] + 2]
            pmf_trim = pmf_trim[: nonzero[-1] + 2]

        colors = []
        for k in ks:
            if market_line is None:
                colors.append("#4C72B0")
            elif k > market_line:
                colors.append("#55A868")  # green = over
            elif abs(k - market_line) < 1e-9:
                colors.append("#CCCCCC")  # gray = push
            else:
                colors.append("#4C72B0")  # blue = under

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(ks, pmf_trim, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars with probability if > 2%
        for bar, p in zip(bars, pmf_trim):
            if p > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{p:.1%}",
                    ha="center", va="bottom", fontsize=7, color="#333333",
                )

        if market_line is not None:
            ax.axvline(market_line, color="#E84040", linewidth=1.5, linestyle="--", label=f"Line: {market_line}")
            p_over = self.prob_over(stat, market_line)
            title = f"{self.player_name} — {stat.upper()} Distribution | {p_over:.0%} Over {market_line}"
        else:
            mu = self.pmf_mean(stat)
            title = f"{self.player_name} — {stat.upper()} Distribution | Mean: {mu:.1f}"

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Outcome", fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_xticks(ks)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if market_line is not None:
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def plot_pmf_base64(self, stat: str, market_line: float | None = None) -> str | None:
        """Return base64-encoded PNG of the PMF plot (for HTML embedding)."""
        fig = self.plot_pmf(stat, market_line)
        if fig is None:
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return base64.b64encode(buf.read()).decode("utf-8")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        stats = ", ".join(sorted(self._pmfs))
        return f"WNBAPMFGrid(player={self.player_name!r}, stats=[{stats}])"

    @property
    def stats(self) -> list[str]:
        return sorted(self._pmfs)

    def has_stat(self, stat: str) -> bool:
        return stat in self._pmfs


def pmfs_df_to_grids(pmfs_df, game_context_cols: list[str] | None = None) -> list[WNBAPMFGrid]:
    """Convert a long-format PMF DataFrame to a list of WNBAPMFGrid objects.

    Parameters
    ----------
    pmfs_df : pd.DataFrame
        Long-format PMF table (one row per player × stat) with columns:
        player_id, player_name, stat, pmf_json (or pmf), minutes_mean,
        role_bucket, plus optional context columns.
    game_context_cols : list[str], optional
        Column names to include in game_context dict (e.g. ["game_id", "game_date"]).
    """
    import json
    import pandas as pd
    from wnba_props_model.models.simulation import json_to_pmf

    ctx_cols = game_context_cols or ["game_id", "game_date", "team_id",
                                      "opponent_team_id", "is_home"]

    grids: list[WNBAPMFGrid] = []
    key_cols = ["player_id"]
    if "game_id" in pmfs_df.columns:
        key_cols.append("game_id")

    for keys, group in pmfs_df.groupby(key_cols):
        row0 = group.iloc[0]
        player_id = row0.get("player_id", "unknown")
        player_name = str(row0.get("player_name", "Unknown"))
        projected_minutes = float(row0.get("minutes_mean", 0.0))
        role_bucket = str(row0.get("role_bucket", "unknown"))
        game_context = {
            c: row0.get(c) for c in ctx_cols if c in pmfs_df.columns
        }

        stat_pmfs: dict[str, np.ndarray] = {}
        for _, r in group.iterrows():
            stat = str(r["stat"])
            if "pmf" in r and isinstance(r["pmf"], np.ndarray):
                arr = r["pmf"]
            elif "pmf_json" in r and pd.notna(r["pmf_json"]):
                arr = json_to_pmf(r["pmf_json"])
            else:
                continue
            stat_pmfs[stat] = arr

        if stat_pmfs:
            grids.append(WNBAPMFGrid(
                player_id=player_id,
                player_name=player_name,
                stat_pmfs=stat_pmfs,
                projected_minutes=projected_minutes,
                role_bucket=role_bucket,
                game_context=game_context,
            ))

    return grids
