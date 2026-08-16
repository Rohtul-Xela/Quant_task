from src.data.yahoo import download_universe, load_historical_tickers


def main():
    # SPY is never itself an S&P 500 constituent (it tracks the index), so
    # it never appears in load_historical_tickers()'s membership-derived
    # list -- but src/evaluate/benchmarks.py needs SPY.parquet for the
    # buy-and-hold benchmark. Download it alongside the research universe
    # rather than as an undocumented separate manual step.
    tickers = load_historical_tickers()

    if "SPY" not in tickers:
        tickers = tickers + ["SPY"]

    status = download_universe(
        tickers=tickers,
        force=False,
    )

    print("\n=== DOWNLOAD SUMMARY ===")
    print(
        status["status"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()