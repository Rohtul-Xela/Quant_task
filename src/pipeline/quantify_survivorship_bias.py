"""
Quantify the direction and magnitude of survivorship bias.

"Today's list" is defined as the PIT-reconstructed constituent set on
the research end date (2026-08-11) — i.e. the tickers a naive,
non-PIT-aware backtest would have used for the entire history, since
that's the only list such a backtest would ever think to pull. This
reuses the PIT dataset already built (`pit_features.parquet`) rather
than re-deriving "current membership" from a separate source, so it's
guaranteed consistent with what the rest of the project calls PIT.

Method: take the best finalist's already-computed stitched-OOS signal
(`combo__sma_rsi__or__signal.parquet`), restrict it to only the
today's-list tickers, and run it through the same `run_backtest` used
everywhere else in the project. Compare against the already-computed
full-PIT-universe result for the same line in `walkforward_results.csv`.
Same signal, same window, same cost model — the only thing that
changes is which tickers were eligible to trade, which isolates the
survivorship-bias effect from every other variable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    DEFAULT_COST_BPS,
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    calculate_metrics,
    load_yahoo_prices,
    run_backtest,
)

FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"

TARGET_LINE_ID = "combo__sma_rsi__or"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("SURVIVORSHIP BIAS — DIRECTION AND MAGNITUDE")

    df = pd.read_parquet(FEATURE_FILE, columns=["date", "source_ticker"])
    df["date"] = pd.to_datetime(df["date"])

    research_end_date = df["date"].max()
    todays_list = set(
        df.loc[df["date"] == research_end_date, "source_ticker"].unique()
    )

    print(f"Research end date: {research_end_date.date()}")
    print(f"Tickers on that date ('today's list'): {len(todays_list)}")

    full_pit_tickers = df["source_ticker"].nunique()
    print(f"Tickers ever in the PIT-reconstructed universe: {full_pit_tickers}")
    print(
        f"'Today's list' is {len(todays_list) / full_pit_tickers:.1%} of the "
        "full PIT universe — the rest are names that were in the S&P 500 at "
        "some point in 2008-2026 but are not constituents today (delisted, "
        "acquired, dropped from the index, etc.)."
    )

    signal_path = EQUITY_DIR / f"{TARGET_LINE_ID}__signal.parquet"

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Stitched signal not found for {TARGET_LINE_ID}: {signal_path}"
        )

    full_signal = pd.read_parquet(signal_path)
    full_signal["adj_close"] = 1.0  # not used numerically by run_backtest

    survivorship_signal = full_signal[
        full_signal["source_ticker"].isin(todays_list)
    ].copy()

    print(
        f"\nFull-PIT stitched signal: {full_signal['source_ticker'].nunique()} tickers, "
        f"{len(full_signal)} rows"
    )
    print(
        f"Today's-list-only signal: {survivorship_signal['source_ticker'].nunique()} "
        f"tickers, {len(survivorship_signal)} rows"
    )

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    print_section("BACKTESTING BOTH UNIVERSES (same signal, same window, same cost)")

    full_result = run_backtest(
        full_signal,
        cost_bps=DEFAULT_COST_BPS,
        strategy_id=f"{TARGET_LINE_ID}__full_pit",
        yahoo_prices=yahoo_prices,
        yahoo_returns=yahoo_returns,
    )

    survivorship_result = run_backtest(
        survivorship_signal,
        cost_bps=DEFAULT_COST_BPS,
        strategy_id=f"{TARGET_LINE_ID}__todays_list_only",
        yahoo_prices=yahoo_prices,
        yahoo_returns=yahoo_returns,
    )

    full_metrics = calculate_metrics(full_result.daily)
    survivorship_metrics = calculate_metrics(survivorship_result.daily)

    rows = [
        {"universe": "full_pit_reconstructed", **full_metrics},
        {"universe": "todays_list_only_survivorship_biased", **survivorship_metrics},
    ]

    result_df = pd.DataFrame(rows)

    sharpe_delta = survivorship_metrics["sharpe"] - full_metrics["sharpe"]
    return_delta = survivorship_metrics["net_return"] - full_metrics["net_return"]

    direction = "OVERSTATES" if sharpe_delta > 0 else "UNDERSTATES"

    print_section("RESULT")

    print(
        result_df[
            ["universe", "sharpe", "net_return", "max_drawdown"]
        ].to_string(index=False)
    )

    print(
        f"\nUsing today's list only instead of PIT-reconstructed membership "
        f"{direction} the Sharpe ratio by {sharpe_delta:+.4f} "
        f"({full_metrics['sharpe']:.4f} -> {survivorship_metrics['sharpe']:.4f}) "
        f"and net return by {return_delta:+.4f} "
        f"({full_metrics['net_return']:.4f} -> {survivorship_metrics['net_return']:.4f})."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_DIR / "survivorship_bias_quantification.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'survivorship_bias_quantification.csv'}")


if __name__ == "__main__":
    main()
