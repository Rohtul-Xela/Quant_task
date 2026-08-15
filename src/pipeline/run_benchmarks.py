"""
Phase 5 — benchmarks.

SPY buy-and-hold and an equal-weight daily-rebalanced basket of the
research universe, both over the same stitched OOS date range used by
Phase 2/3/4 (2013-01-01 through the 2026-08-11 research cutoff).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
)
from src.evaluate.benchmarks import (  # noqa: E402
    benchmark_metrics,
    equal_weight_basket,
    spy_buy_and_hold,
)
from src.walkforward.windows import generate_windows  # noqa: E402


FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"
RESULTS_CSV = OUTPUT_DIR / "benchmark_results.csv"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("PHASE 5 — BENCHMARKS")

    df = pd.read_parquet(FEATURE_FILE, columns=["date", "yahoo_ticker"])
    df["date"] = pd.to_datetime(df["date"])

    data_start = df["date"].min()
    data_end = df["date"].max()

    windows = generate_windows(
        data_start, data_end, is_years=5, oos_years=1, step_years=1
    )

    oos_start = windows[0].oos_start
    oos_end = windows[-1].oos_end

    print(f"Stitched OOS date range: {oos_start.date()} -> {oos_end.date()}")

    universe_yahoo_tickers = set(
        df["yahoo_ticker"].astype(str).str.strip().str.upper().unique()
    )
    print(f"Research universe: {len(universe_yahoo_tickers)} yahoo tickers")

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)

    print_section("SPY BUY-AND-HOLD")

    spy_daily = spy_buy_and_hold(oos_start, oos_end)
    spy_metrics = benchmark_metrics(spy_daily)
    spy_metrics["strategy_id"] = "benchmark_spy_buy_hold"
    spy_daily.to_parquet(EQUITY_DIR / "benchmark_spy_buy_hold.parquet", index=False)

    print(
        f"  Sharpe={spy_metrics['sharpe']:.4f}  "
        f"NetReturn={spy_metrics['net_return']:.4f}  "
        f"MaxDD={spy_metrics['max_drawdown']:.4f}"
    )

    print_section("EQUAL-WEIGHT UNIVERSE BASKET")

    ew_daily = equal_weight_basket(yahoo_returns, universe_yahoo_tickers, oos_start, oos_end)
    ew_metrics = benchmark_metrics(ew_daily)
    ew_metrics["strategy_id"] = "benchmark_equal_weight_basket"
    ew_daily.to_parquet(EQUITY_DIR / "benchmark_equal_weight_basket.parquet", index=False)

    print(
        f"  Sharpe={ew_metrics['sharpe']:.4f}  "
        f"NetReturn={ew_metrics['net_return']:.4f}  "
        f"MaxDD={ew_metrics['max_drawdown']:.4f}"
    )

    print_section("SAVING RESULTS")

    results_df = pd.DataFrame([spy_metrics, ew_metrics])
    results_df.to_csv(RESULTS_CSV, index=False)

    print(f"Saved: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
