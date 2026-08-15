"""
Phase 2 — Walk-forward harness runner.

Runs each distinct (strategy family, mode) line from the Phase 1
shortlist through rolling 5y-IS / 1y-OOS / 1y-step walk-forward
re-optimization, stitches the OOS segments into one continuous curve
per line, and writes results/walkforward_results.csv +
results/walkforward_windows.csv + per-line stitched daily parquet.

IMPORTANT — shortlist collapse: the Phase 1 shortlist has 6 entries,
but two pairs share a (family, mode): the two SMA long-only configs
(50/200 and 10/50) and the two RSI long-only configs. Walk-forward
re-optimizes over the FULL family parameter grid every window, so
re-running the same grid from a different in-sample "starting point"
is deterministic and produces an identical walk-forward path. Running
it twice would be wasted computation, not a different result. So this
script runs 4 distinct walk-forward lines, not 6 — documented here and
in the report rather than silently dropping the duplication.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    DEFAULT_COST_BPS,
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
)
from src.strategy.strategies import (  # noqa: E402
    DONCHIAN_PARAMETER_GRID,
    RSI_PARAMETER_GRID,
    SMA_PARAMETER_GRID,
)
from src.walkforward.harness import run_walk_forward_line  # noqa: E402
from src.walkforward.windows import generate_windows, windows_to_frame  # noqa: E402


FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"

RESULTS_CSV = OUTPUT_DIR / "walkforward_results.csv"
WINDOWS_CSV = OUTPUT_DIR / "walkforward_windows.csv"

COST_BPS = DEFAULT_COST_BPS


# =====================================================================
# The 4 distinct walk-forward lines (collapsed from the 6-entry
# shortlist as documented above).
# =====================================================================

LINES = [
    {
        "line_id": "sma_crossover__long_only",
        "strategy_name": "sma_crossover",
        "mode": "long_only",
        "parameter_grid": SMA_PARAMETER_GRID,
        "shortlist_source": [
            "sma_crossover__fast=50_slow=200__long_only",
            "sma_crossover__fast=10_slow=50__long_only",
        ],
    },
    {
        "line_id": "rsi_mean_reversion__long_only",
        "strategy_name": "rsi_mean_reversion",
        "mode": "long_only",
        "parameter_grid": RSI_PARAMETER_GRID,
        "shortlist_source": [
            "rsi_mean_reversion__entry=40_exit=50_short_entry=60_window=21__long_only",
            "rsi_mean_reversion__entry=30_exit=60_short_entry=70_window=14__long_only",
        ],
    },
    {
        "line_id": "rsi_mean_reversion__long_short",
        "strategy_name": "rsi_mean_reversion",
        "mode": "long_short",
        "parameter_grid": RSI_PARAMETER_GRID,
        "shortlist_source": [
            "rsi_mean_reversion__entry=40_exit=50_short_entry=60_window=21__long_short",
        ],
    },
    {
        "line_id": "donchian_breakout__long_only",
        "strategy_name": "donchian_breakout",
        "mode": "long_only",
        "parameter_grid": DONCHIAN_PARAMETER_GRID,
        "shortlist_source": [
            "donchian_breakout__window=40__long_only",
        ],
    },
]


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("PHASE 2 — WALK-FORWARD HARNESS")

    df = pd.read_parquet(FEATURE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["source_ticker"] = df["source_ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "source_ticker"]).reset_index(drop=True)

    data_start = df["date"].min()
    data_end = df["date"].max()

    print(f"Feature data range: {data_start.date()} -> {data_end.date()}")

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    windows = generate_windows(
        data_start, data_end, is_years=5, oos_years=1, step_years=1
    )

    print(f"Generated {len(windows)} walk-forward windows:")
    for w in windows:
        stub = " (PARTIAL)" if w.is_partial_oos else ""
        print(
            f"  [{w.index}] IS {w.is_start.date()} -> {w.is_end.date()}  "
            f"OOS {w.oos_start.date()} -> {w.oos_end.date()}{stub}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)

    windows_to_frame(windows).to_csv(WINDOWS_CSV, index=False)

    all_window_records = []
    summary_rows = []

    for line in LINES:
        print_section(f"LINE: {line['line_id']}")

        result = run_walk_forward_line(
            line_id=line["line_id"],
            strategy_name=line["strategy_name"],
            mode=line["mode"],
            parameter_grid=line["parameter_grid"],
            df=df,
            windows=windows,
            yahoo_prices=yahoo_prices,
            yahoo_returns=yahoo_returns,
            cost_bps=COST_BPS,
        )

        print(result.window_table.to_string(index=False))

        all_window_records.append(result.window_table)

        result.stitched_daily.to_parquet(
            EQUITY_DIR / f"{line['line_id']}.parquet", index=False
        )

        if result.stitched_signal is not None:
            result.stitched_signal[
                ["date", "source_ticker", "yahoo_ticker", "signal"]
            ].to_parquet(
                EQUITY_DIR / f"{line['line_id']}__signal.parquet", index=False
            )

        metrics = dict(result.stitched_metrics)

        summary_rows.append(
            {
                "line_id": line["line_id"],
                "type": "single",
                "strategy_name": line["strategy_name"],
                "mode": line["mode"],
                "shortlist_source": ";".join(line["shortlist_source"]),
                "n_windows": len(result.window_table),
                "avg_wf_efficiency": result.window_table["wf_efficiency"].mean(),
                "avg_rank_corr_is_vs_oos": result.window_table[
                    "rank_corr_is_vs_oos"
                ].mean(),
                "pct_windows_is_winner_also_oos_best": result.window_table[
                    "is_winner_also_oos_best"
                ].mean(),
                **metrics,
            }
        )

        print(
            f"\n  Stitched OOS Sharpe: {metrics['sharpe']:.4f}   "
            f"Net return: {metrics['net_return']:.4f}   "
            f"Max DD: {metrics['max_drawdown']:.4f}"
        )

    print_section("SAVING RESULTS")

    windows_out = pd.concat(all_window_records, ignore_index=True)
    windows_out.to_csv(OUTPUT_DIR / "walkforward_window_detail.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "sharpe", ascending=False, na_position="last"
    )
    summary_df.to_csv(RESULTS_CSV, index=False)

    print(f"Saved: {RESULTS_CSV}")
    print(f"Saved: {WINDOWS_CSV}")
    print(f"Saved: {OUTPUT_DIR / 'walkforward_window_detail.csv'}")
    print(f"Saved stitched equity parquet for {len(LINES)} lines to {EQUITY_DIR}")

    print_section("SUMMARY (stitched OOS)")
    display_cols = [
        "line_id",
        "mode",
        "sharpe",
        "net_return",
        "max_drawdown",
        "avg_wf_efficiency",
        "avg_rank_corr_is_vs_oos",
        "pct_windows_is_winner_also_oos_best",
    ]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
