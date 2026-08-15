from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

YAHOO_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "raw"
    / "yahoo"
)


TICKERS = [
    "EP",
    "EQ",
    "FMCC",
    "GME",
    "GR",
    "HET",
    "HIG",
    "HPC",
    "NCC",
    "NKTR",
    "PBG",
    "RX",
    "STI",
    "CBE",
    "CFC",
    "MEE",
    "TIE",
]


def inspect_ticker(
    ticker: str,
) -> None:

    path = (
        YAHOO_DIR
        / f"{ticker}.parquet"
    )

    print()
    print("=" * 100)
    print(
        f"TICKER: {ticker}"
    )
    print("=" * 100)

    if not path.exists():

        print(
            "FILE NOT FOUND:",
            path,
        )

        return

    df = pd.read_parquet(
        path
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    df["next_date"] = (
        df["date"].shift(-1)
    )

    df["next_adj_close"] = (
        df["adj_close"].shift(-1)
    )

    df["return"] = (
        df["next_adj_close"]
        / df["adj_close"]
        - 1.0
    )

    # ---------------------------------------------------------------
    # Extreme observations
    # ---------------------------------------------------------------

    extreme = (
        df["return"].abs() > 1.0
    )

    suspicious = df[
        extreme
    ].copy()

    print(
        "\nExtreme observations:",
        len(suspicious),
    )

    if suspicious.empty:

        print(
            "None."
        )

        return

    # ---------------------------------------------------------------
    # Print suspicious events
    # ---------------------------------------------------------------

    columns = [
        "date",
        "next_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "next_adj_close",
        "return",
        "volume",
        "dividends",
        "stock_splits",
    ]

    print(
        "\n=== EXTREME RETURNS ==="
    )

    print(
        suspicious[columns]
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Show surrounding observations
    # ---------------------------------------------------------------

    print(
        "\n=== LOCAL CONTEXT ==="
    )

    suspicious_indices = (
        suspicious.index.tolist()
    )

    context_indices = set()

    for idx in suspicious_indices:

        for offset in range(
            -2,
            3,
        ):

            candidate = idx + offset

            if (
                0
                <= candidate
                < len(df)
            ):
                context_indices.add(
                    candidate
                )

    context = (
        df.loc[
            sorted(
                context_indices
            ),
            columns,
        ]
    )

    print(
        context.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Largest absolute return
    # ---------------------------------------------------------------

    max_idx = (
        df["return"]
        .abs()
        .idxmax()
    )

    max_row = df.loc[
        max_idx
    ]

    print(
        "\n=== MAX ABSOLUTE RETURN ==="
    )

    print(
        max_row[columns]
        .to_string()
    )

    # ---------------------------------------------------------------
    # Basic price range
    # ---------------------------------------------------------------

    print(
        "\n=== PRICE RANGE ==="
    )

    print(
        "Minimum adj close:",
        df["adj_close"].min(),
    )

    print(
        "Maximum adj close:",
        df["adj_close"].max(),
    )

    print(
        "Minimum raw close:",
        df["close"].min(),
    )

    print(
        "Maximum raw close:",
        df["close"].max(),
    )


def main() -> None:

    print(
        "Yahoo directory:",
        YAHOO_DIR,
    )

    for ticker in TICKERS:

        inspect_ticker(
            ticker
        )


if __name__ == "__main__":
    main()