from __future__ import annotations

from pathlib import Path

import pandas as pd


PIT_FILE = Path(
    "data/processed/pit_daily.parquet"
)


def main():

    pit = pd.read_parquet(
        PIT_FILE
    )

    pit["date"] = pd.to_datetime(
        pit["date"]
    )

    # ---------------------------------------------------------------
    # Daily breadth
    # ---------------------------------------------------------------

    daily = (
        pit
        .groupby("date")
        .agg(
            stocks=("source_ticker", "nunique"),
            yahoo_tickers=("yahoo_ticker", "nunique"),
            rows=("source_ticker", "size"),
        )
        .reset_index()
    )

    print("\n=== DAILY COVERAGE ===")

    print(
        daily["stocks"].describe()
    )

    print(
        "\nLowest 20 coverage dates:"
    )

    print(
        daily
        .sort_values("stocks")
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nHighest 20 coverage dates:"
    )

    print(
        daily
        .sort_values("stocks", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Yearly coverage
    # ---------------------------------------------------------------

    daily["year"] = (
        daily["date"].dt.year
    )

    yearly = (
        daily
        .groupby("year")["stocks"]
        .agg(
            trading_days="count",
            min="min",
            mean="mean",
            median="median",
            max="max",
        )
    )

    print(
        "\n=== YEARLY DAILY COVERAGE ==="
    )

    print(
        yearly.to_string()
    )

    # ---------------------------------------------------------------
    # Number of securities entering the PIT dataset
    # ---------------------------------------------------------------

    first_seen = (
        pit.groupby(
            "source_ticker"
        )["date"]
        .min()
        .sort_values()
    )

    print(
        "\n=== FIRST 30 SECURITIES IN PIT DATA ==="
    )

    print(
        first_seen.head(30)
        .to_string()
    )

    print(
        "\n=== LAST 30 SECURITIES IN PIT DATA ==="
    )

    print(
        first_seen.tail(30)
        .to_string()
    )

    # ---------------------------------------------------------------
    # Duplicate security/date check
    # ---------------------------------------------------------------

    duplicates = pit.duplicated(
        subset=[
            "date",
            "source_ticker",
        ]
    ).sum()

    print(
        "\nDuplicate (date, source_ticker) rows:",
        duplicates,
    )

    # ---------------------------------------------------------------
    # OHLC validation
    # ---------------------------------------------------------------

    bad_ohlc = (
        (pit["high"] < pit["low"])
        | (pit["open"] > pit["high"])
        | (pit["open"] < pit["low"])
        | (pit["close"] > pit["high"])
        | (pit["close"] < pit["low"])
    )

    print(
        "Invalid OHLC rows:",
        int(bad_ohlc.sum()),
    )

    # ---------------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------------

    print(
        "\n=== MISSING VALUES ==="
    )

    print(
        pit[
            [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        ].isna().sum()
    )


if __name__ == "__main__":
    main()