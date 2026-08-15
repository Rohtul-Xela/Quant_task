from __future__ import annotations

from pathlib import Path

import pandas as pd


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

YAHOO_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "raw"
    / "yahoo"
)


def load_yahoo_prices() -> pd.DataFrame:
    frames = []

    for path in sorted(
        YAHOO_DIR.glob("*.parquet")
    ):

        try:
            df = pd.read_parquet(path)

            if not {
                "date",
                "ticker",
                "adj_close",
            }.issubset(df.columns):
                continue

            df = df[
                [
                    "date",
                    "ticker",
                    "adj_close",
                ]
            ].copy()

            df["date"] = pd.to_datetime(
                df["date"]
            )

            df["ticker"] = (
                path.stem
                .strip()
                .upper()
            )

            df["adj_close"] = pd.to_numeric(
                df["adj_close"],
                errors="coerce",
            )

            df = (
                df
                .dropna(
                    subset=[
                        "date",
                        "adj_close",
                    ]
                )
                .sort_values("date")
                .drop_duplicates(
                    [
                        "date",
                        "ticker",
                    ],
                    keep="last",
                )
            )

            frames.append(df)

        except Exception:
            continue

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main() -> None:

    print(
        "Loading PIT features..."
    )

    pit = pd.read_parquet(
        FEATURE_FILE
    )

    pit["date"] = pd.to_datetime(
        pit["date"]
    )

    pit["source_ticker"] = (
        pit["source_ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    pit["yahoo_ticker"] = (
        pit["yahoo_ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Use a strategy that produced many active missing returns.
    # SMA 50/200 is the top current candidate.
    from src.strategy.strategies import (
        generate_strategy_signal,
    )

    strategy = generate_strategy_signal(
        pit,
        strategy_name="sma_crossover",
        parameters={
            "fast": 50,
            "slow": 200,
        },
        mode="long_only",
    )

    active = strategy[
        strategy["signal"] != 0
    ].copy()

    print(
        "Active signal rows:",
        len(active),
    )

    print(
        "Loading Yahoo data..."
    )

    yahoo = load_yahoo_prices()

    yahoo["date"] = pd.to_datetime(
        yahoo["date"]
    )

    # ---------------------------------------------------------------
    # Build next observation
    # ---------------------------------------------------------------

    yahoo = (
        yahoo
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    yahoo["next_date"] = (
        yahoo
        .groupby("ticker")["date"]
        .shift(-1)
    )

    yahoo["next_close"] = (
        yahoo
        .groupby("ticker")["adj_close"]
        .shift(-1)
    )

    yahoo["gap_days"] = (
        yahoo["next_date"]
        - yahoo["date"]
    ).dt.days

    yahoo["next_return"] = pd.NA

    valid_gap = (
        yahoo["gap_days"]
        .between(
            1,
            4,
        )
    )

    yahoo.loc[
        valid_gap,
        "next_return",
    ] = (
        yahoo.loc[
            valid_gap,
            "next_close",
        ]
        / yahoo.loc[
            valid_gap,
            "adj_close",
        ]
        - 1.0
    )

    # ---------------------------------------------------------------
    # Join
    # ---------------------------------------------------------------

    merged = active.merge(
        yahoo[
            [
                "date",
                "ticker",
                "next_date",
                "gap_days",
                "next_return",
            ]
        ],
        left_on=[
            "date",
            "yahoo_ticker",
        ],
        right_on=[
            "date",
            "ticker",
        ],
        how="left",
    )

    missing = merged[
        merged["next_return"].isna()
    ].copy()

    print()
    print(
        "=== ACTIVE OBSERVATIONS WITHOUT NEXT RETURN ==="
    )

    print(
        "Count:",
        len(missing),
    )

    # ---------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------

    missing["reason"] = "unknown"

    missing.loc[
        missing["ticker"].isna(),
        "reason",
    ] = "no_yahoo_ticker_history"

    missing.loc[
        missing["ticker"].notna()
        & missing["next_date"].isna(),
        "reason",
    ] = "last_available_yahoo_observation"

    missing.loc[
        missing["ticker"].notna()
        & missing["next_date"].notna()
        & (
            missing["gap_days"] > 4
        ),
        "reason",
    ] = "long_yahoo_gap"

    print(
        "\n=== REASON SUMMARY ==="
    )

    print(
        missing["reason"]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== BY SOURCE TICKER ==="
    )

    by_ticker = (
        missing
        .groupby(
            [
                "source_ticker",
                "yahoo_ticker",
                "reason",
            ]
        )
        .size()
        .reset_index(
            name="missing_rows"
        )
        .sort_values(
            "missing_rows",
            ascending=False,
        )
    )

    print(
        by_ticker
        .head(100)
        .to_string(
            index=False
        )
    )

    print(
        "\n=== FIRST 100 MISSING OBSERVATIONS ==="
    )

    print(
        missing[
            [
                "date",
                "source_ticker",
                "yahoo_ticker",
                "reason",
                "next_date",
                "gap_days",
            ]
        ]
        .head(100)
        .to_string(
            index=False
        )
    )

    output = (
        PROJECT_ROOT
        / "src"
        / "data"
        / "data"
        / "validation"
        / "missing_active_returns.csv"
    )

    missing.to_csv(
        output,
        index=False,
    )

    print(
        "\nSaved:",
        output,
    )


if __name__ == "__main__":
    main()