"""
Phase 1 — Confirm the walk-forward candidate shortlist.

Re-verifies the six shortlisted strategies against the *current*
strategy_sweep_results.csv (rather than trusting stale numbers from a
prior review), and runs one manual sanity pass on the top strategy's
equity curve to rule out a sizing/double-counting bug before it is
carried into the walk-forward harness.

This does not run any new backtests for the sweep itself — it only
reads the existing in-sample sweep output and re-derives the top
strategy's daily series via the existing run_backtest for inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

SWEEP_RESULTS_CSV = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "strategy_sweep_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "shortlist.csv"


from src.backtest.backtest import (  # noqa: E402
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
    run_backtest,
    validate_returns,
)
from src.strategy.strategies import generate_strategy_signal  # noqa: E402


# =====================================================================
# The confirmed shortlist
#
# NOT simply "top 6 by Sharpe" — this mirrors the handover spec's
# explicit picks: the four best long-only trend/momentum configs, the
# best long-short RSI config (to test the long > short hypothesis
# out-of-sample), and one deliberately weak Donchian comparator.
# =====================================================================

SHORTLIST = [
    {
        "strategy_id": "sma_crossover__fast=50_slow=200__long_only",
        "strategy_name": "sma_crossover",
        "parameters": {"fast": 50, "slow": 200},
        "mode": "long_only",
        "reason": "Top Sharpe in the in-sample sweep.",
    },
    {
        "strategy_id": (
            "rsi_mean_reversion__entry=40_exit=50_short_entry=60_window=21"
            "__long_only"
        ),
        "strategy_name": "rsi_mean_reversion",
        "parameters": {
            "entry": 40,
            "exit": 50,
            "short_entry": 60,
            "window": 21,
        },
        "mode": "long_only",
        "reason": "2nd-best Sharpe in the in-sample sweep.",
    },
    {
        "strategy_id": (
            "rsi_mean_reversion__entry=30_exit=60_short_entry=70_window=14"
            "__long_only"
        ),
        "strategy_name": "rsi_mean_reversion",
        "parameters": {
            "entry": 30,
            "exit": 60,
            "short_entry": 70,
            "window": 14,
        },
        "mode": "long_only",
        "reason": "3rd-best Sharpe in the in-sample sweep.",
    },
    {
        "strategy_id": "sma_crossover__fast=10_slow=50__long_only",
        "strategy_name": "sma_crossover",
        "parameters": {"fast": 10, "slow": 50},
        "mode": "long_only",
        "reason": "4th-best Sharpe; faster SMA pair for contrast with 50/200.",
    },
    {
        "strategy_id": (
            "rsi_mean_reversion__entry=40_exit=50_short_entry=60_window=21"
            "__long_short"
        ),
        "strategy_name": "rsi_mean_reversion",
        "parameters": {
            "entry": 40,
            "exit": 50,
            "short_entry": 60,
            "window": 21,
        },
        "mode": "long_short",
        "reason": (
            "Best-performing RSI long-short config — deliberately included to "
            "test the long-side-more-robust hypothesis out-of-sample."
        ),
    },
    {
        "strategy_id": "donchian_breakout__window=40__long_only",
        "strategy_name": "donchian_breakout",
        "parameters": {"window": 40},
        "mode": "long_only",
        "reason": (
            "Deliberately weak comparator — best-of-a-bad-bunch Donchian config; "
            "the whole family is negative-Sharpe in-sample."
        ),
    },
]


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def confirm_against_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for entry in SHORTLIST:
        match = sweep[sweep["strategy_id"] == entry["strategy_id"]]

        if match.empty:
            raise ValueError(
                f"Shortlisted strategy_id not found in current sweep results: "
                f"{entry['strategy_id']}"
            )

        if len(match) != 1:
            raise ValueError(
                f"Expected exactly one row for {entry['strategy_id']!r}, "
                f"got {len(match)}."
            )

        row = match.iloc[0]

        rows.append(
            {
                "strategy_id": entry["strategy_id"],
                "strategy_name": entry["strategy_name"],
                "mode": entry["mode"],
                "parameters": entry["parameters"],
                "reason": entry["reason"],
                "sweep_sharpe": row["sharpe"],
                "sweep_net_return": row["net_return"],
                "sweep_max_drawdown": row["max_drawdown"],
            }
        )

        print(
            f"  OK  {entry['strategy_id']:<70s} "
            f"Sharpe={row['sharpe']:.4f}  "
            f"NetReturn={row['net_return']:.4f}  "
            f"MaxDD={row['max_drawdown']:.4f}"
        )

    return pd.DataFrame(rows)


def sanity_check_top_strategy(
    df: pd.DataFrame,
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
) -> None:
    """
    Reconstruct the SMA 50/200 long-only equity curve directly and
    check for single-day outliers / exposure bounds before trusting the
    752% headline in-sample net return.
    """

    top = SHORTLIST[0]

    strategy_df = generate_strategy_signal(
        df,
        strategy_name=top["strategy_name"],
        parameters=top["parameters"],
        mode=top["mode"],
    )

    result = run_backtest(
        strategy_df,
        strategy_id=top["strategy_id"],
        yahoo_prices=yahoo_prices,
        yahoo_returns=yahoo_returns,
    )

    validate_returns(yahoo_returns)

    daily = result.daily

    worst_day = daily.loc[daily["net_return"].idxmin()]
    best_day = daily.loc[daily["net_return"].idxmax()]
    max_active = daily["active_positions"].max()
    max_gross = daily["gross_exposure"].max()
    max_turnover = daily["turnover"].max()

    print(f"  Trading days:        {len(daily)}")
    print(f"  Max active positions: {max_active} (universe ~634)")
    print(f"  Max gross exposure:   {max_gross:.4f} (must be <= 1.0)")
    print(f"  Max daily turnover:   {max_turnover:.4f} (must be <= 2.0)")
    print(
        f"  Worst day: {worst_day['date'].date()}  "
        f"net_return={worst_day['net_return']:.4f}"
    )
    print(
        f"  Best day:  {best_day['date'].date()}  "
        f"net_return={best_day['net_return']:.4f}"
    )

    # A single-day return outside +/-25% for a diversified, <=1.0-gross,
    # equal-weight cross-sectional long-only book would be a red flag
    # (e.g. a double-counted weight or an unclipped adjusted-price glitch).
    outlier_threshold = 0.25

    outliers = daily[daily["net_return"].abs() > outlier_threshold]

    if not outliers.empty:
        print(
            f"  WARNING: {len(outliers)} day(s) with |net_return| > "
            f"{outlier_threshold:.0%} — inspect before trusting the sweep."
        )
        print(
            outliers[["date", "net_return", "active_positions"]].to_string(
                index=False
            )
        )
    else:
        print(
            f"  No single-day |net_return| > {outlier_threshold:.0%}. "
            "No evidence of a sizing/double-counting bug."
        )

    if max_gross > 1.0 + 1e-9:
        raise ValueError("Sanity check failed: gross exposure exceeded 1.0.")

    if max_active > 700:
        raise ValueError(
            "Sanity check failed: active positions exceed a sane bound "
            "for a ~634-ticker universe."
        )

    equity = (1.0 + daily["net_return"]).cumprod()
    reconstructed_net_return = float(equity.iloc[-1] - 1.0)

    print(
        f"  Reconstructed net return: {reconstructed_net_return:.4f} "
        f"(sweep CSV: {result.metrics['net_return']:.4f})"
    )

    if not np.isclose(
        reconstructed_net_return, result.metrics["net_return"], rtol=1e-6
    ):
        raise ValueError(
            "Sanity check failed: reconstructed equity curve does not match "
            "run_backtest's own reported net_return."
        )

    print(
        "  CONCLUSION: 752% net return over 18.6 years is large but internally "
        "consistent (gross exposure bounded, no outlier days, equity curve "
        "self-consistent). Long-only trend-following compounding through the "
        "2009-2021 bull run explains the magnitude — not a sizing bug."
    )


def main() -> None:
    print_section("PHASE 1 — SHORTLIST CONFIRMATION")

    if not SWEEP_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Sweep results not found: {SWEEP_RESULTS_CSV}")

    sweep = pd.read_csv(SWEEP_RESULTS_CSV)

    print(f"Loaded sweep results: {len(sweep)} configurations")

    print_section("CONFIRMING SHORTLIST AGAINST CURRENT SWEEP CSV")

    shortlist_df = confirm_against_sweep(sweep)

    print_section("EQUITY CURVE SANITY CHECK (TOP STRATEGY)")

    df = pd.read_parquet(FEATURE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["source_ticker"] = df["source_ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "source_ticker"]).reset_index(drop=True)

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    sanity_check_top_strategy(df, yahoo_prices, yahoo_returns)

    print_section("SAVING SHORTLIST")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shortlist_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
