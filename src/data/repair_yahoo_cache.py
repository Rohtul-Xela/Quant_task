from src.data.yahoo import download_ticker


def main():

    tickers = [
        "FDXF",
        "HONA",
        "MER",
        "NFX",
        "RX",
        "SATS",
    ]

    for ticker in tickers:

        print(f"\n=== REPAIRING {ticker} ===")

        result = download_ticker(
            ticker=ticker,
            force=True,
        )

        print(result)


if __name__ == "__main__":
    main()