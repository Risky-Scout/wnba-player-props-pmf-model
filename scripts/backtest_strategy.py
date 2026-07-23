"""Strategy backtest using PenaltyBlog Backtest framework.

Loads OOF PMF predictions joined with market data, runs a fractional-Kelly
betting strategy, and produces an equity curve, per-stat ROI bar chart, and
Kelly stake distribution histogram.

Usage:
    python scripts/backtest_strategy.py \
        --oof-pmfs artifacts/models/oof_pmfs.parquet \
        --clv-tracking data/clv_tracking/results.parquet \
        --out-dir artifacts/reports \
        --edge-threshold 0.04 \
        --kelly-fraction 0.25 \
        --bankroll 10000
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

_DEFAULT_EDGE = 0.04
_DEFAULT_KELLY = 0.25
_DEFAULT_BANKROLL = 10_000.0


def _american_to_decimal(odds: float | int | None) -> float | None:
    """Convert American odds to decimal."""
    if odds is None:
        return None
    o = float(odds)
    if o > 0:
        return o / 100.0 + 1.0
    if o < 0:
        return 100.0 / abs(o) + 1.0
    return None


def _fractional_kelly(
    model_prob: float,
    market_prob_no_vig: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly stake fraction.

    Returns 0 if there is no edge.
    """
    b = decimal_odds - 1.0
    edge = model_prob - market_prob_no_vig
    if edge <= 0 or b <= 0:
        return 0.0
    full_kelly = (b * model_prob - (1.0 - model_prob)) / b
    return float(max(0.0, full_kelly * fraction))


def _prepare_backtest_df(
    oof_pmfs_path: str | None,
    clv_path: str | None,
) -> pd.DataFrame:
    """Load and join OOF PMF predictions with CLV tracking data.

    Returns a flat DataFrame with one row per (player, game, stat) with
    at least columns: date, model_prob_over, market_prob_over_no_vig,
    over_odds_american, actual_outcome, stat, line, edge_over, edge_under.
    """
    frames = []

    if clv_path and Path(clv_path).exists():
        clv = pd.read_parquet(clv_path)
        frames.append(clv)

    if oof_pmfs_path and Path(oof_pmfs_path).exists():
        oof = pd.read_parquet(oof_pmfs_path)
        frames.append(oof)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalise date column
    date_col = None
    for c in ["game_date", "date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.date

    # Required bet-decision columns
    required = {"model_prob_over_final", "market_prob_over_no_vig", "actual_outcome"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("backtest_df missing columns: %s", missing)
        return pd.DataFrame()

    return df.dropna(subset=["date", "model_prob_over_final", "market_prob_over_no_vig"]).copy()


def _run_backtest(
    df: pd.DataFrame,
    edge_threshold: float,
    kelly_fraction: float,
    bankroll: float,
) -> tuple[dict, pd.DataFrame]:
    """Run PenaltyBlog Backtest and return (results_dict, bet_history_df)."""
    try:
        from penaltyblog.backtest import Backtest  # noqa: PLC0415
    except ImportError:
        logger.error("penaltyblog.backtest not available — returning empty results")
        return {}, pd.DataFrame()

    start_date = str(df["date"].min())
    end_date   = str(df["date"].max())

    def logic(ctx) -> None:
        row = ctx.fixture

        # --- Over bet ---
        edge_over = float(row.get("edge_over", 0.0) or 0.0)
        if edge_over > edge_threshold:
            mp = float(row.get("model_prob_over_final", 0.5))
            mkt_p = float(row.get("market_prob_over_no_vig", 0.5))
            over_odds = row.get("over_odds")
            dec = _american_to_decimal(over_odds)
            if dec and dec > 1.0:
                stake_frac = _fractional_kelly(mp, mkt_p, dec, kelly_fraction)
                stake = bankroll * stake_frac
                if stake > 0.01:
                    actual = row.get("actual_outcome")
                    line   = row.get("line", 0.5)
                    win = 1 if (actual is not None and not pd.isna(actual) and float(actual) > float(line)) else 0
                    ctx.account.place_bet(odds=dec, stake=stake, outcome=win)

        # --- Under bet ---
        edge_under = float(row.get("edge_under", 0.0) or 0.0)
        if edge_under > edge_threshold:
            model_p_under = 1.0 - float(row.get("model_prob_over_final", 0.5))
            mkt_p_under   = 1.0 - float(row.get("market_prob_over_no_vig", 0.5))
            under_odds = row.get("under_odds")
            dec = _american_to_decimal(under_odds)
            if dec and dec > 1.0:
                stake_frac = _fractional_kelly(model_p_under, mkt_p_under, dec, kelly_fraction)
                stake = bankroll * stake_frac
                if stake > 0.01:
                    actual = row.get("actual_outcome")
                    line   = row.get("line", 0.5)
                    win = 1 if (actual is not None and not pd.isna(actual) and float(actual) <= float(line)) else 0
                    ctx.account.place_bet(odds=dec, stake=stake, outcome=win)

    bt = Backtest(df, start_date=start_date, end_date=end_date)
    bt.start(bankroll=bankroll, logic=logic)
    results = bt.results()

    history_df = pd.DataFrame(bt.account.history) if bt.account.history else pd.DataFrame()
    equity_curve = list(bt.account.tracker)

    return results, history_df, equity_curve


def _generate_charts(
    df: pd.DataFrame,
    history_df: pd.DataFrame,
    equity_curve: list[float],
    bankroll: float,
    out_dir: Path,
) -> None:
    """Generate equity curve, per-stat ROI, and Kelly distribution charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not available — skipping chart generation")
        return

    # --- 1. Equity curve ---
    if equity_curve:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity_curve, color="steelblue", linewidth=1.5)
        ax.axhline(y=bankroll, color="gray", linestyle="--", alpha=0.6, label="Starting bankroll")
        ax.set_title("Strategy Equity Curve", fontsize=14)
        ax.set_xlabel("Bet number")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        equity_path = out_dir / "backtest_equity_curve.png"
        fig.savefig(equity_path, dpi=120)
        plt.close(fig)
        typer.echo(f"Equity curve → {equity_path}")

    # --- 2. Per-stat ROI bar chart ---
    if "stat" in df.columns and not history_df.empty:
        try:
            # Merge history back with df to get stat labels
            df_reset = df.reset_index(drop=True)
            df_reset["bet_idx"] = range(len(df_reset))
            stat_roi: dict[str, list[float]] = {}
            for _, r in history_df.iterrows():
                # Approximate: profit per stat based on available data
                stat = None
                if "stat" in history_df.columns:
                    stat = str(r.get("stat", "unknown"))
                else:
                    stat = "all"
                profit_pct = r.get("profit", 0.0) / max(r.get("stake", 1.0), 1.0) * 100
                stat_roi.setdefault(stat, []).append(float(profit_pct))

            if stat_roi:
                stats = list(stat_roi.keys())
                roi_vals = [np.mean(v) for v in stat_roi.values()]
                colors = ["green" if v >= 0 else "red" for v in roi_vals]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(stats, roi_vals, color=colors)
                ax.axhline(y=0, color="black", linewidth=0.8)
                ax.set_title("Per-Stat Mean ROI per Bet (%)", fontsize=14)
                ax.set_xlabel("Stat")
                ax.set_ylabel("ROI per bet (%)")
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                roi_path = out_dir / "backtest_per_stat_roi.png"
                fig.savefig(roi_path, dpi=120)
                plt.close(fig)
                typer.echo(f"Per-stat ROI → {roi_path}")
        except Exception as exc:
            logger.warning("Per-stat ROI chart failed: %s", exc)

    # --- 3. Kelly stake distribution histogram ---
    if not history_df.empty and "stake" in history_df.columns:
        stakes = history_df["stake"].dropna()
        if len(stakes) > 2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(stakes / bankroll * 100, bins=30, color="steelblue", edgecolor="white")
            ax.set_title("Kelly Stake Distribution (% of Bankroll)", fontsize=13)
            ax.set_xlabel("Stake (% of bankroll)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            kelly_path = out_dir / "backtest_kelly_dist.png"
            fig.savefig(kelly_path, dpi=120)
            plt.close(fig)
            typer.echo(f"Kelly distribution → {kelly_path}")


@app.command()
def main(
    oof_pmfs: str | None = typer.Option(None, "--oof-pmfs",
        help="OOF PMF predictions parquet (artifacts/models/oof_pmfs.parquet)."),
    clv_tracking: str | None = typer.Option(None, "--clv-tracking",
        help="CLV tracking results parquet (data/clv_tracking/results.parquet)."),
    out_dir: str = typer.Option("artifacts/reports", "--out-dir",
        help="Output directory for charts and results."),
    edge_threshold: float = typer.Option(_DEFAULT_EDGE, "--edge-threshold",
        help="Minimum edge (model_prob - market_prob) to place a bet."),
    kelly_fraction: float = typer.Option(_DEFAULT_KELLY, "--kelly-fraction",
        help="Fractional Kelly multiplier (0.25 = quarter Kelly)."),
    bankroll: float = typer.Option(_DEFAULT_BANKROLL, "--bankroll",
        help="Starting bankroll in $."),
) -> None:
    """Run strategy backtest using PenaltyBlog Backtest framework."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading backtest data …")
    df = _prepare_backtest_df(oof_pmfs, clv_tracking)

    if df.empty:
        typer.echo("[WARN] No backtestable rows found. Ensure oof_pmfs or clv_tracking paths exist.")
        raise typer.Exit(0)

    typer.echo(f"Loaded {len(df):,} rows spanning {df['date'].min()} → {df['date'].max()}")
    typer.echo(f"Edge threshold: {edge_threshold:.2f}, Kelly: {kelly_fraction:.2f}×, Bankroll: ${bankroll:,.0f}")

    results, history_df, equity_curve = _run_backtest(df, edge_threshold, kelly_fraction, bankroll)

    if not results:
        typer.echo("[WARN] Backtest returned no results — check penaltyblog installation.")
        raise typer.Exit(0)

    # --- Save results ---
    results_path = out / f"backtest_{date.today().isoformat()}.json"
    with open(results_path, "w") as f:
        json.dump({k: (float(v) if v is not None else None) for k, v in results.items()}, f, indent=2)

    if not history_df.empty:
        history_path = out / f"backtest_{date.today().isoformat()}.parquet"
        history_df.to_parquet(history_path, index=False)
        typer.echo(f"Bet history → {history_path}")

    # --- Print summary ---
    typer.echo(f"\n{'='*55}")
    typer.echo("Backtest Summary")
    typer.echo(f"{'='*55}")
    for k, v in results.items():
        if isinstance(v, float):
            typer.echo(f"  {k:<25} {v:+.2f}")
        else:
            typer.echo(f"  {k:<25} {v}")
    typer.echo(f"{'='*55}")
    typer.echo(f"Results JSON → {results_path}")

    # --- Generate charts ---
    _generate_charts(df, history_df, equity_curve, bankroll, out)


if __name__ == "__main__":
    app()
