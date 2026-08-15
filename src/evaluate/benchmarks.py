"""
Phase 5 — benchmarks.

SPY buy-and-hold and an equal-weight, daily-rebalanced basket of the
research universe, both restricted to the same stitched OOS date range
as the walk-forward strategies for an apples-to-apples comparison.

Both benchmarks are quoted gross (zero transaction cost) — standard
convention for a passive reference curve, not a proposed tradable
strategy. SPY is used ONLY here, never as part of the strategy
universe (hard constraint from the task brief).

Reuses `build_next_day_returns` for the equal-weight basket so it is
subject to the identical no-look-ahead, NYSE-session-aware return
construction as every strategy in this project — not a separate,
possibly-inconsistent return calculation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import YAHOO_DIR, calculate_metrics  # noqa: E402


def spy_buy_and_hold(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:

    spy_path = YAHOO_DIR / "SPY.parquet"

    if not spy_path.exists():
        raise FileNotFoundError(
            f"SPY price file not found: {spy_path}. Run "
            "download_ticker(ticker='SPY', output_dir=YAHOO_DIR) first."
        )

    prices = pd.read_parquet(spy_path)[["date", "adj_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)]

    prices["net_return"] = prices["adj_close"].pct_change()
    prices = prices.dropna(subset=["net_return"])

    daily = pd.DataFrame(
        {
            "date": prices["date"].to_numpy(),
            "net_return": prices["net_return"].to_numpy(),
            "turnover": 0.0,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
        }
    )

    return daily


def equal_weight_basket(
    yahoo_returns: pd.DataFrame,
    universe_yahoo_tickers: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:

    universe_returns = yahoo_returns[
        yahoo_returns["ticker"].isin(universe_yahoo_tickers)
        & yahoo_returns["date"].between(start, end)
        & yahoo_returns["next_return"].notna()
    ]

    grouped = (
        universe_returns.groupby("date")["next_return"]
        .mean()
        .reset_index()
        .rename(columns={"next_return": "net_return"})
        .sort_values("date")
    )

    grouped["turnover"] = 0.0
    grouped["gross_exposure"] = 1.0
    grouped["net_exposure"] = 1.0

    return grouped.reset_index(drop=True)


def benchmark_metrics(daily: pd.DataFrame) -> dict:
    return calculate_metrics(daily)
