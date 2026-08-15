from __future__ import annotations

from pathlib import Path

import pandas as pd


INTERVAL_FILE = Path(
    "data/processed/membership_intervals.parquet"
)


def load_intervals() -> pd.DataFrame:
    df = pd.read_parquet(INTERVAL_FILE)

    df["effective_from"] = pd.to_datetime(
        df["effective_from"]
    )

    df["effective_to"] = pd.to_datetime(
        df["effective_to"]
    )

    return df.sort_values(
        ["ticker", "effective_from"]
    ).reset_index(drop=True)


def check_basic_integrity(
    df: pd.DataFrame,
) -> None:

    print("\n=== BASIC INTEGRITY ===")

    print("Rows:", len(df))
    print("Unique tickers:", df["ticker"].nunique())

    print(
        "Null effective_from:",
        df["effective_from"].isna().sum(),
    )

    invalid_dates = (
        df["effective_to"].notna()
        & (
            df["effective_to"]
            <= df["effective_from"]
        )
    )

    print(
        "Invalid intervals:",
        invalid_dates.sum(),
    )


def check_overlaps(
    df: pd.DataFrame,
) -> pd.DataFrame:

    overlaps = []

    for ticker, group in df.groupby("ticker"):

        group = group.sort_values(
            "effective_from"
        ).reset_index(drop=True)

        for i in range(1, len(group)):

            previous_end = group.loc[
                i - 1,
                "effective_to",
            ]

            current_start = group.loc[
                i,
                "effective_from",
            ]

            # Open-ended previous interval.
            if pd.isna(previous_end):

                overlaps.append(
                    {
                        "ticker": ticker,
                        "previous_end": previous_end,
                        "current_start": current_start,
                        "type": "overlap_after_open_interval",
                    }
                )

            elif current_start < previous_end:

                overlaps.append(
                    {
                        "ticker": ticker,
                        "previous_end": previous_end,
                        "current_start": current_start,
                        "type": "overlap",
                    }
                )

    return pd.DataFrame(overlaps)


def check_gaps(
    df: pd.DataFrame,
) -> pd.DataFrame:

    gaps = []

    for ticker, group in df.groupby("ticker"):

        group = group.sort_values(
            "effective_from"
        ).reset_index(drop=True)

        for i in range(1, len(group)):

            previous_end = group.loc[
                i - 1,
                "effective_to",
            ]

            current_start = group.loc[
                i,
                "effective_from",
            ]

            if pd.isna(previous_end):
                continue

            # We intentionally do not treat every one-day
            # difference as necessarily wrong. We report it.
            if current_start > previous_end:

                gaps.append(
                    {
                        "ticker": ticker,
                        "previous_end": previous_end,
                        "current_start": current_start,
                        "gap_days": (
                            current_start
                            - previous_end
                        ).days,
                    }
                )

    return pd.DataFrame(gaps)


def inspect_tickers(
    df: pd.DataFrame,
    tickers: list[str],
) -> None:

    print("\n=== SELECTED TICKERS ===")

    for ticker in tickers:

        subset = df[
            df["ticker"] == ticker
        ]

        print(f"\n{ticker}")

        if subset.empty:
            print("  NOT FOUND")

        else:
            print(
                subset.to_string(index=False)
            )


def main() -> None:

    df = load_intervals()

    check_basic_integrity(df)

    overlaps = check_overlaps(df)

    print("\n=== OVERLAPS ===")

    if overlaps.empty:
        print("None")
    else:
        print(
            overlaps.to_string(index=False)
        )

    gaps = check_gaps(df)

    print("\n=== GAPS ===")

    if gaps.empty:
        print("None")
    else:
        print(
            gaps.to_string(index=False)
        )

    inspect_tickers(
        df,
        [
            "AAPL",
            "AAL",
            "ABC",
            "ABKFQ",
            "BRK-B",
            "BF-B",
        ],
    )


if __name__ == "__main__":
    main()