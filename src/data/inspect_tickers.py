from pathlib import Path

import pandas as pd


TICKER_FILE = Path(
    "data/processed/historical_tickers.csv"
)


def main():

    df = pd.read_csv(TICKER_FILE)

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    print("Total tickers:", len(tickers))

    print("\n=== TICKERS CONTAINING SPACES ===")

    spaces = tickers[
        tickers.str.contains(r"\s", regex=True)
    ]

    print(
        spaces.to_string(index=False)
    )

    print("\n=== TICKERS CONTAINING PARENTHESES ===")

    parentheses = tickers[
        tickers.str.contains(
            r"[\(\)]",
            regex=True,
        )
    ]

    print(
        parentheses.to_string(index=False)
    )

    print("\n=== TICKERS CONTAINING SLASHES ===")

    slashes = tickers[
        tickers.str.contains(
            "/",
            regex=False,
        )
    ]

    print(
        slashes.to_string(index=False)
    )

    print("\n=== TICKERS CONTAINING COMMAS ===")

    commas = tickers[
        tickers.str.contains(
            ",",
            regex=False,
        )
    ]

    print(
        commas.to_string(index=False)
    )


if __name__ == "__main__":
    main()