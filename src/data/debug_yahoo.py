import yfinance as yf
import pandas as pd


def inspect_ticker(ticker: str):

    print(f"\n\n{'=' * 80}")
    print(f"TICKER: {ticker}")
    print(f"{'=' * 80}")

    raw = yf.download(
        ticker,
        start="2008-01-01",
        end="2026-08-13",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )

    print("\nType:")
    print(type(raw))

    print("\nShape:")
    print(raw.shape)

    print("\nColumns:")
    print(raw.columns)

    print("\nDtypes:")
    print(raw.dtypes)

    print("\nHead:")
    print(raw.head().to_string())

    print("\nTail:")
    print(raw.tail().to_string())

    print("\nNull counts:")
    print(raw.isna().sum())


def main():

    for ticker in [
        "RVTY",
        "CVG",
        "MNST",
    ]:
        inspect_ticker(ticker)


if __name__ == "__main__":
    main()