from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd


# =====================================================================
# Project paths
# =====================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "processed"
    / "pit_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "processed"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "strategy_sweep_results.parquet"
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "strategy_sweep_results.csv"
)


# =====================================================================
# Project imports
# =====================================================================

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.backtest.backtest import (
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
    run_backtest,
    validate_returns,
)

from src.strategy.strategies import (
    enumerate_strategy_configs,
    generate_strategy_signal,
)


# =====================================================================
# Configuration
# =====================================================================

COST_BPS = 5.0


# =====================================================================
# Helpers
# =====================================================================

def print_section(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    start_time = (
        time.perf_counter()
    )

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    print_section(
        "SWEEP CONFIGURATION"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Feature file:",
        FEATURE_FILE,
    )

    print(
        "Results:",
        RESULTS_FILE,
    )

    print(
        "Transaction cost:",
        f"{COST_BPS} bps / side",
    )

    print(
        "Excluded price tickers:",
        sorted(
            EXCLUDED_PRICE_TICKERS
        ),
    )

    # ---------------------------------------------------------------
    # Validate feature file
    # ---------------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    # ---------------------------------------------------------------
    # Load PIT features ONCE
    # ---------------------------------------------------------------

    print_section(
        "LOADING FEATURE DATA"
    )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Columns:",
        len(df.columns),
    )

    print(
        "Date range:",
        df["date"].min(),
        "→",
        df["date"].max(),
    )

    print(
        "Source securities:",
        df["source_ticker"].nunique(),
    )

    required = {
        "date",
        "source_ticker",
        "adj_close",
        "adj_high",
        "adj_low",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Feature dataset missing columns:\n"
            f"{sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["source_ticker"] = (
        df["source_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = (
        df
        .sort_values(
            [
                "date",
                "source_ticker",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Load Yahoo prices ONCE
    # ---------------------------------------------------------------

    print_section(
        "LOADING PRICE UNIVERSE"
    )

    yahoo_prices = (
        load_yahoo_prices()
    )

    print(
        "Validated Yahoo rows:",
        len(yahoo_prices),
    )

    print(
        "Validated Yahoo tickers:",
        yahoo_prices["ticker"]
        .nunique(),
    )

    # ---------------------------------------------------------------
    # Build next-session returns ONCE
    # ---------------------------------------------------------------

    print_section(
        "BUILDING NEXT-SESSION RETURNS"
    )

    yahoo_returns = (
        build_next_day_returns(
            yahoo_prices
        )
    )

    validate_returns(
        yahoo_returns
    )

    # ---------------------------------------------------------------
    # Strategy configurations
    # ---------------------------------------------------------------

    configs = (
        enumerate_strategy_configs()
    )

    print_section(
        "SWEEP CONFIGURATIONS"
    )

    print(
        "Total configurations:",
        len(configs),
    )

    # ---------------------------------------------------------------
    # Execute sweep
    # ---------------------------------------------------------------

    results = []

    for i, config in enumerate(
        configs,
        start=1,
    ):

        strategy_name = (
            config["strategy_name"]
        )

        family = (
            config["family"]
        )

        parameters = (
            config["parameters"]
        )

        mode = (
            config["mode"]
        )

        print()
        print(
            f"[{i:>3}/{len(configs)}] "
            f"{strategy_name} | "
            f"{mode} | "
            f"{parameters}"
        )

        config_start = (
            time.perf_counter()
        )

        # -----------------------------------------------------------
        # Generate strategy signal
        # -----------------------------------------------------------

        strategy_df = (
            generate_strategy_signal(
                df,
                strategy_name=strategy_name,
                parameters=parameters,
                mode=mode,
            )
        )

        strategy_ids = (
            strategy_df[
                "strategy_id"
            ]
            .dropna()
            .unique()
        )

        if len(strategy_ids) != 1:

            raise ValueError(
                "Expected exactly one "
                "strategy_id, got: "
                f"{strategy_ids}"
            )

        strategy_id = str(
            strategy_ids[0]
        )

        # -----------------------------------------------------------
        # Backtest using PRELOADED price data
        # -----------------------------------------------------------

        backtest_result = (
            run_backtest(
                strategy_df,
                cost_bps=COST_BPS,
                strategy_id=strategy_id,
                yahoo_prices=yahoo_prices,
                yahoo_returns=yahoo_returns,
            )
        )

        metrics = dict(
            backtest_result.metrics
        )

        daily = (
            backtest_result.daily
        )

        # -----------------------------------------------------------
        # Research metadata
        # -----------------------------------------------------------

        row = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "family": family,
            "mode": mode,
        }

        for key, value in (
            parameters.items()
        ):

            row[
                f"param_{key}"
            ] = value

        row.update(
            metrics
        )

        # -----------------------------------------------------------
        # Daily diagnostics
        # -----------------------------------------------------------

        row["start_date"] = (
            daily["date"].min()
            if not daily.empty
            else pd.NaT
        )

        row["end_date"] = (
            daily["date"].max()
            if not daily.empty
            else pd.NaT
        )

        row["trading_days"] = (
            len(daily)
        )

        row["positive_days"] = (
            int(
                (
                    daily["net_return"]
                    > 0
                ).sum()
            )
            if not daily.empty
            else 0
        )

        row["negative_days"] = (
            int(
                (
                    daily["net_return"]
                    < 0
                ).sum()
            )
            if not daily.empty
            else 0
        )

        row["zero_days"] = (
            int(
                (
                    daily["net_return"]
                    == 0
                ).sum()
            )
            if not daily.empty
            else 0
        )

        row["total_transaction_cost"] = (
            float(
                daily[
                    "transaction_cost"
                ].sum()
            )
            if not daily.empty
            else 0.0
        )

        row["total_turnover"] = (
            float(
                daily[
                    "turnover"
                ].sum()
            )
            if not daily.empty
            else 0.0
        )

        row["average_active_positions"] = (
            float(
                daily[
                    "active_positions"
                ].mean()
            )
            if not daily.empty
            else 0.0
        )

        row[
            "active_positions_missing_return"
        ] = metrics.get(
            "active_positions_missing_return",
            0,
        )

        results.append(
            row
        )

        elapsed = (
            time.perf_counter()
            - config_start
        )

        print(
            f"    Sharpe: "
            f"{metrics.get('sharpe', float('nan')):.4f}"
        )

        print(
            f"    Net return: "
            f"{metrics.get('net_return', float('nan')):.4f}"
        )

        print(
            f"    Max DD: "
            f"{metrics.get('max_drawdown', float('nan')):.4f}"
        )

        print(
            f"    Avg turnover: "
            f"{metrics.get('avg_daily_turnover', float('nan')):.4f}"
        )

        print(
            f"    Missing active returns: "
            f"{metrics.get('active_positions_missing_return', 0)}"
        )

        print(
            f"    Time: "
            f"{elapsed:.2f}s"
        )

    # ---------------------------------------------------------------
    # Results DataFrame
    # ---------------------------------------------------------------

    print_section(
        "BUILDING RESULTS TABLE"
    )

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------------
    # Duplicate strategy IDs
    # ---------------------------------------------------------------

    duplicate_ids = (
        results_df[
            "strategy_id"
        ]
        .duplicated()
    )

    if duplicate_ids.any():

        raise ValueError(
            "Duplicate strategy IDs found:\n"
            + results_df.loc[
                duplicate_ids,
                "strategy_id",
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Check for non-finite performance
    # ---------------------------------------------------------------

    for column in [
        "sharpe",
        "sortino",
        "net_return",
        "max_drawdown",
    ]:

        if column not in results_df.columns:
            continue

        bad = ~(
            results_df[column]
            .isna()
            | results_df[column]
            .map(
                pd.api.types.is_number
            )
        )

        if bad.any():

            raise ValueError(
                f"Invalid values found in {column}."
            )

    # ---------------------------------------------------------------
    # Column ordering
    # ---------------------------------------------------------------

    preferred_columns = [
        "strategy_id",
        "strategy_name",
        "family",
        "mode",
    ]

    parameter_columns = sorted(
        [
            column
            for column in results_df.columns
            if column.startswith(
                "param_"
            )
        ]
    )

    metric_columns = [
        "sharpe",
        "sortino",
        "net_return",
        "max_drawdown",
        "months_in_profit",
        "months_count",
        "months_in_profit_pct",
        "avg_monthly_pnl",
        "avg_daily_turnover",
        "avg_gross_exposure",
        "avg_net_exposure",
        "trading_days",
        "positive_days",
        "negative_days",
        "zero_days",
        "total_transaction_cost",
        "total_turnover",
        "average_active_positions",
        "active_positions_missing_return",
        "cost_bps",
        "start_date",
        "end_date",
    ]

    ordered_columns = (
        preferred_columns
        + parameter_columns
        + [
            column
            for column in metric_columns
            if column in results_df.columns
        ]
    )

    remaining = [
        column
        for column in results_df.columns
        if column not in ordered_columns
    ]

    ordered_columns.extend(
        remaining
    )

    results_df = results_df[
        ordered_columns
    ]

    # ---------------------------------------------------------------
    # Sort by Sharpe
    # ---------------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            "sharpe",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        RESULTS_FILE,
        index=False,
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print_section(
        "SWEEP COMPLETE"
    )

    print(
        "Configurations evaluated:",
        len(results_df),
    )

    print(
        "Saved:",
        RESULTS_FILE,
    )

    print(
        "Saved CSV:",
        RESULTS_CSV,
    )

    print()
    print(
        "=== TOP 10 BY SHARPE ==="
    )

    display_columns = [
        "strategy_id",
        "mode",
        "sharpe",
        "sortino",
        "net_return",
        "max_drawdown",
        "months_in_profit_pct",
        "avg_daily_turnover",
        "active_positions_missing_return",
    ]

    print(
        results_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=== WORST 10 BY SHARPE ==="
    )

    print(
        results_df[
            display_columns
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=== FAMILY SUMMARY ==="
    )

    family_summary = (
        results_df
        .groupby(
            [
                "family",
                "mode",
            ]
        )
        .agg(
            strategies=(
                "strategy_id",
                "count",
            ),
            best_sharpe=(
                "sharpe",
                "max",
            ),
            median_sharpe=(
                "sharpe",
                "median",
            ),
            best_net_return=(
                "net_return",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "best_sharpe",
            ascending=False,
        )
    )

    print(
        family_summary.to_string(
            index=False
        )
    )

    elapsed_total = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Total runtime: "
        f"{elapsed_total / 60.0:.2f} minutes"
    )


if __name__ == "__main__":
    main()