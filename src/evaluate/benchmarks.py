"""
Phase 5 — benchmarks.

SPY buy-and-hold and an equal-weight, daily-rebalanced basket of the
research universe, both restricted to the same stitched OOS date range
as the walk-forward strategies for an apples-to-apples comparison.

Both benchmarks are quoted gross (zero transaction cost) — standard
convention for a passive reference curve, not a proposed tradable
strategy. SPY is used ONLY here, never as part of the strategy
universe (hard constraint from the task brief).

Reuses `build_next_day_returns` for BOTH benchmarks — the equal-weight
basket and SPY alike — so both are subject to the identical
no-look-ahead, NYSE-session-aware, t -> next-session return
construction as every strategy in this project (signal on day t,
return realized t -> t+1, row labeled at date t). SPY must not be
built with a naive `pct_change()` (t-1 -> t, labeled at date t): that
is one trading session out of alignment with every strategy and with
the equal-weight basket, which silently corrupts any date-aligned
comparison — correlation against strategy return streams above all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    YAHOO_DIR,
    build_next_day_returns,
    calculate_metrics,
)


def spy_buy_and_hold(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:

    spy_path = YAHOO_DIR / "SPY.parquet"

    if not spy_path.exists():
        raise FileNotFoundError(
            f"SPY price file not found: {spy_path}. Run "
            "download_ticker(ticker='SPY', output_dir=YAHOO_DIR) first."
        )

    prices = pd.read_parquet(spy_path)[["date", "ticker", "adj_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    returns = build_next_day_returns(prices)
    returns = returns[
        returns["date"].between(start, end) & returns["next_return"].notna()
    ].sort_values("date")

    daily = pd.DataFrame(
        {
            "date": returns["date"].to_numpy(),
            "net_return": returns["next_return"].to_numpy(),
            "turnover": 0.0,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
        }
    )

    return daily.reset_index(drop=True)


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
