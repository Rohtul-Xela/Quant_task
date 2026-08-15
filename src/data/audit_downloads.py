from pathlib import Path

import pandas as pd


STATUS_FILE = Path(
    "data/validation/download_status.csv"
)


def main():
    df = pd.read_csv(STATUS_FILE)

    print("=== OVERALL STATUS ===")
    print(
        df["status"]
        .value_counts()
        .to_string()
    )

    print("\n=== EMPTY TICKERS ===")
    empty = df[df["status"] == "empty"]

    print("Count:", len(empty))

    print(
        empty[
            ["ticker", "status"]
        ].to_string(index=False)
    )

    print("\n=== ERRORS ===")
    errors = df[df["status"] == "error"]

    print("Count:", len(errors))

    print(
        errors[
            ["ticker", "error"]
        ].to_string(index=False)
    )

    print("\n=== DOWNLOADED COVERAGE ===")

    downloaded = df[
        df["status"].isin(
            ["downloaded", "cached"]
        )
    ].copy()

    downloaded["first_date"] = pd.to_datetime(
        downloaded["first_date"]
    )

    downloaded["last_date"] = pd.to_datetime(
        downloaded["last_date"]
    )

    print(
        downloaded[
            [
                "ticker",
                "first_date",
                "last_date",
                "rows",
            ]
        ]
        .sort_values("first_date")
        .head(20)
        .to_string(index=False)
    )

    print("\n=== ROW COUNT SUMMARY ===")

    print(
        downloaded["rows"].describe()
    )


if __name__ == "__main__":
    main()