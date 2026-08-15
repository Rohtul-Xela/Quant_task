from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

YAHOO_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "raw"
    / "yahoo"
    / "CBE.parquet"
)


def main() -> None:

    print(
        "File:",
        YAHOO_FILE,
    )

    if not YAHOO_FILE.exists():
        raise FileNotFoundError(
            f"File not found:\n{YAHOO_FILE}"
        )

    df = pd.read_parquet(
        YAHOO_FILE
    )

    print(
        "\n=== COLUMNS ==="
    )

    print(
        df.columns.tolist()
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Consecutive return
    # ---------------------------------------------------------------

    df["next_date"] = (
        df["date"].shift(-1)
    )

    df["next_close"] = (
        df["adj_close"].shift(-1)
    )

    df["return"] = (
        df["next_close"]
        / df["adj_close"]
        - 1.0
    )

    # ---------------------------------------------------------------
    # Show suspicious rows
    # ---------------------------------------------------------------

    suspicious = df[
        df["return"].abs() > 1.0
    ].copy()

    print(
        "\n=== RETURNS > 100% ==="
    )

    if suspicious.empty:

        print("None")

    else:

        print(
            suspicious[
                [
                    "date",
                    "next_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "next_close",
                    "return",
                    "volume",
                    "dividends",
                    "stock_splits",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Focus on the problematic date
    # ---------------------------------------------------------------

    mask = (
        df["date"].between(
            "2015-12-01",
            "2015-12-10",
        )
    )

    print(
        "\n=== CBE AROUND 2015-12-04 ==="
    )

    print(
        df.loc[
            mask,
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "dividends",
                "stock_splits",
                "next_date",
                "next_close",
                "return",
            ],
        ]
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Largest returns overall
    # ---------------------------------------------------------------

    print(
        "\n=== TOP 20 ABSOLUTE RETURNS ==="
    )

    print(
        df.loc[
            df["return"]
            .abs()
            .sort_values(
                ascending=False
            )
            .head(20)
            .index,
            [
                "date",
                "next_date",
                "adj_close",
                "next_close",
                "return",
                "dividends",
                "stock_splits",
            ],
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()