from pathlib import Path

import pandas as pd
import yfinance as yf


def main():

    mapping_file = Path(
        "data/processed/security_mapping.csv"
    )

    mapping = pd.read_csv(
        mapping_file
    )

    print("\n=== MAPPINGS ===")
    print(
        mapping.to_string(index=False)
    )

    print("\n=== YAHOO CANDIDATE TEST ===")

    for _, row in mapping.iterrows():

        source = row["source_ticker"]
        yahoo_ticker = row["yahoo_ticker"]

        print(
            f"\n{source} -> {yahoo_ticker}"
        )

        df = yf.download(
            yahoo_ticker,
            start="2008-01-01",
            end="2026-08-13",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            print("NO DATA")
            continue

        print(
            "First:",
            df.index.min(),
        )

        print(
            "Last:",
            df.index.max(),
        )

        print(
            "Rows:",
            len(df),
        )


if __name__ == "__main__":
    main()