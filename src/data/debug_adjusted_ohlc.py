from pathlib import Path

import pandas as pd


PRICE_DIR = Path("data/raw/yahoo")


def main():

    all_bad = []

    for path in sorted(
        PRICE_DIR.glob("*.parquet")
    ):

        try:

            df = pd.read_parquet(path)

            required = {
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
            }

            if not required.issubset(
                df.columns
            ):
                continue

            df["date"] = pd.to_datetime(
                df["date"]
            )

            # Ignore zero-close observations for the diagnostic.
            valid_close = df["close"] != 0

            factor = (
                df.loc[valid_close, "adj_close"]
                / df.loc[valid_close, "close"]
            )

            df = df.loc[
                valid_close
            ].copy()

            df["adj_open"] = (
                df["open"] * factor
            )

            df["adj_high"] = (
                df["high"] * factor
            )

            df["adj_low"] = (
                df["low"] * factor
            )

            # -------------------------------------------------------
            # Find adjusted OHLC violations
            # -------------------------------------------------------

            bad = (
                (df["adj_high"] < df["adj_low"])
                | (
                    df["adj_open"]
                    > df["adj_high"]
                )
                | (
                    df["adj_open"]
                    < df["adj_low"]
                )
                | (
                    df["adj_close"]
                    > df["adj_high"]
                )
                | (
                    df["adj_close"]
                    < df["adj_low"]
                )
            )

            if bad.any():

                bad_df = df.loc[
                    bad
                ].copy()

                bad_df["ticker"] = (
                    path.stem
                )

                bad_df["factor"] = factor.loc[
                    bad_df.index
                ]

                all_bad.append(
                    bad_df[
                        [
                            "ticker",
                            "date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "adj_open",
                            "adj_high",
                            "adj_low",
                            "adj_close",
                            "factor",
                        ]
                    ]
                )

        except Exception as exc:

            print(
                f"ERROR {path.name}: {exc}"
            )

    if not all_bad:

        print("No adjusted OHLC violations found.")
        return

    result = pd.concat(
        all_bad,
        ignore_index=True,
    )

    print(
        "\n=== ADJUSTED OHLC VIOLATIONS ==="
    )

    print(
        "Total:",
        len(result),
    )

    print(
        "\nBy ticker:"
    )

    print(
        result["ticker"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print(
        "\nFirst 50 problematic rows:"
    )

    print(
        result.head(50)
        .to_string(index=False)
    )

    print(
        "\nFactor summary:"
    )

    print(
        result["factor"].describe()
    )

    result.to_csv(
        "data/validation/adjusted_ohlc_violations.csv",
        index=False,
    )

    print(
        "\nSaved:"
        " data/validation/adjusted_ohlc_violations.csv"
    )


if __name__ == "__main__":
    main()