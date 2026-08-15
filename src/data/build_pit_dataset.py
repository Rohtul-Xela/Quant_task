from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

MEMBERSHIP_FILE = Path(
    "data/processed/membership_snapshots.parquet"
)

MAPPING_FILE = Path(
    "data/processed/security_mapping.csv"
)

PRICE_DIR = Path(
    "data/raw/yahoo"
)

OUTPUT_FILE = Path(
    "data/processed/pit_daily.parquet"
)

# Fixed research cutoff for reproducibility.
RESEARCH_END_DATE = pd.Timestamp("2026-08-11")

# Known corrupted / unusable Yahoo histories.
EXCLUDED_PRICE_TICKERS = {
    "CBE",
    "CFC",
    "HET",
    "HPC",
    "MEE",
    "MER",
    "NCC",
    "PBG",
    "SATS",
    "TIE",
}


# Floating-point tolerance used for OHLC comparisons.
OHLC_TOLERANCE = 1e-10


# =====================================================================
# Date helpers
# =====================================================================

def normalize_datetime(series: pd.Series) -> pd.Series:
    """
    Normalize timestamps to datetime64[ns].

    This prevents pandas merge_asof failures caused by different
    datetime precisions such as datetime64[ms] vs datetime64[us].
    """

    return pd.to_datetime(
        series,
        errors="raise",
    ).astype("datetime64[ns]")


# =====================================================================
# Load membership
# =====================================================================

def load_membership() -> pd.DataFrame:
    """
    Load normalized point-in-time membership snapshots.

    Expected columns:
        snapshot_date
        ticker
    """

    if not MEMBERSHIP_FILE.exists():
        raise FileNotFoundError(
            f"Membership file not found: {MEMBERSHIP_FILE}"
        )

    df = pd.read_parquet(
        MEMBERSHIP_FILE
    )

    required = {
        "snapshot_date",
        "ticker",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Membership file is missing columns: "
            f"{sorted(missing)}"
        )

    df["snapshot_date"] = normalize_datetime(
        df["snapshot_date"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = (
        df
        .drop_duplicates(
            subset=[
                "snapshot_date",
                "ticker",
            ]
        )
        .sort_values(
            [
                "snapshot_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# =====================================================================
# Load security mappings
# =====================================================================

def load_mapping() -> dict[str, str]:
    """
    Load manually verified source_ticker -> yahoo_ticker mappings.

    Example:
        FB -> META

    Unmapped securities retain their original Yahoo ticker.
    """

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Security mapping file not found: {MAPPING_FILE}"
        )

    mapping = pd.read_csv(
        MAPPING_FILE
    )

    required = {
        "source_ticker",
        "yahoo_ticker",
    }

    missing = required - set(mapping.columns)

    if missing:
        raise ValueError(
            "Security mapping is missing columns: "
            f"{sorted(missing)}"
        )

    mapping["source_ticker"] = (
        mapping["source_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping["yahoo_ticker"] = (
        mapping["yahoo_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    duplicate_sources = (
        mapping.loc[
            mapping["source_ticker"].duplicated(),
            "source_ticker",
        ]
        .unique()
        .tolist()
    )

    if duplicate_sources:
        raise ValueError(
            "Duplicate source_ticker mappings found: "
            f"{duplicate_sources}"
        )

    return dict(
        zip(
            mapping["source_ticker"],
            mapping["yahoo_ticker"],
        )
    )


# =====================================================================
# Membership change points
# =====================================================================

def build_membership_change_points(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse consecutive identical membership snapshots.

    Each retained snapshot is the latest known constituent universe
    starting from that date.
    """

    snapshots = (
        membership
        .groupby("snapshot_date")["ticker"]
        .apply(frozenset)
        .reset_index()
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )

    changed = snapshots["ticker"].ne(
        snapshots["ticker"].shift(1)
    )

    change_points = (
        snapshots.loc[changed]
        .reset_index(drop=True)
    )

    return change_points


# =====================================================================
# Expand membership change points
# =====================================================================

def expand_membership_change_points(
    change_points: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Convert snapshot sets into:

        snapshot_date
        source_ticker
        yahoo_ticker
    """

    rows = []

    for _, row in change_points.iterrows():

        snapshot_date = row["snapshot_date"]

        for source_ticker in row["ticker"]:

            source_ticker = (
                str(source_ticker)
                .strip()
                .upper()
            )

            yahoo_ticker = mapping.get(
                source_ticker,
                source_ticker,
            )

            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "source_ticker": source_ticker,
                    "yahoo_ticker": yahoo_ticker,
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "snapshot_date",
            "source_ticker",
            "yahoo_ticker",
        ],
    )

    if result.empty:
        raise ValueError(
            "No membership rows were created."
        )

    result["snapshot_date"] = normalize_datetime(
        result["snapshot_date"]
    )

    result = (
        result
        .drop_duplicates(
            subset=[
                "snapshot_date",
                "source_ticker",
                "yahoo_ticker",
            ]
        )
        .sort_values(
            [
                "snapshot_date",
                "yahoo_ticker",
                "source_ticker",
            ]
        )
        .reset_index(drop=True)
    )

    return result


# =====================================================================
# Load and prepare Yahoo price data
# =====================================================================

def load_price_data() -> pd.DataFrame:
    """
    Load all usable Yahoo Parquet files and construct adjusted OHLC.

    Required cache columns:

        date
        ticker
        open
        high
        low
        close
        adj_close
        volume
        dividends
        stock_splits

    Constructed columns:

        adj_open
        adj_high
        adj_low
    """

    frames: list[pd.DataFrame] = []

    paths = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    print(
        f"Found {len(paths)} cached Yahoo files."
    )

    for path in paths:

        try:

            df = pd.read_parquet(path)

            if df.empty:
                continue

            required = {
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "dividends",
                "stock_splits",
            }

            missing = required - set(df.columns)

            if missing:
                print(
                    f"[SKIP] {path.name}: "
                    f"missing {sorted(missing)}"
                )
                continue

            # ---------------------------------------------------------
            # Date
            # ---------------------------------------------------------

            df["date"] = normalize_datetime(
                df["date"]
            )

            # ---------------------------------------------------------
            # Research cutoff
            # ---------------------------------------------------------

            df = df[
                df["date"] <= RESEARCH_END_DATE
            ].copy()

            if df.empty:
                continue

            # ---------------------------------------------------------
            # Ticker
            # ---------------------------------------------------------

            df["ticker"] = (
                df["ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            # ---------------------------------------------------------
            # Numeric conversion
            # ---------------------------------------------------------

            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "dividends",
                "stock_splits",
            ]

            for column in numeric_columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            # ---------------------------------------------------------
            # Remove unavailable OHLC observations
            # ---------------------------------------------------------

            df = df.dropna(
                subset=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                ]
            )

            if df.empty:
                continue

            # ---------------------------------------------------------
            # Sort / deduplicate
            # ---------------------------------------------------------

            df = (
                df
                .sort_values("date")
                .drop_duplicates(
                    subset=[
                        "date",
                        "ticker",
                    ],
                    keep="last",
                )
                .reset_index(drop=True)
            )

            # ---------------------------------------------------------
            # Raw OHLC validation with tolerance
            # ---------------------------------------------------------

            tol = OHLC_TOLERANCE

            bad_raw_ohlc = (
                (df["high"] + tol < df["low"])
                | (df["open"] > df["high"] + tol)
                | (df["open"] + tol < df["low"])
                | (df["close"] > df["high"] + tol)
                | (df["close"] + tol < df["low"])
            )

            n_bad_raw = int(
                bad_raw_ohlc.sum()
            )

            if n_bad_raw:

                print(
                    f"[CLEAN] {path.name}: "
                    f"removed {n_bad_raw} invalid raw OHLC rows"
                )

                df = df.loc[
                    ~bad_raw_ohlc
                ].copy()

            if df.empty:
                continue

            # ---------------------------------------------------------
            # Adjustment factor
            #
            # Yahoo adjusted close / raw close.
            #
            # This factor is used to put OHLC onto the same historical
            # adjustment basis as adjusted close.
            # ---------------------------------------------------------

            denominator = (
                df["close"]
                .replace(0, np.nan)
            )

            adjustment_factor = (
                df["adj_close"]
                / denominator
            )

            # ---------------------------------------------------------
            # Validate adjustment factor
            # ---------------------------------------------------------

            bad_factor = (
                ~np.isfinite(
                    adjustment_factor
                )
                | (
                    adjustment_factor
                    <= 0
                )
            )

            n_bad_factor = int(
                bad_factor.sum()
            )

            if n_bad_factor:

                print(
                    f"[CLEAN] {path.name}: "
                    f"removed {n_bad_factor} "
                    "rows with invalid adjustment factors"
                )

                df = df.loc[
                    ~bad_factor
                ].copy()

                adjustment_factor = (
                    df["adj_close"]
                    / df["close"].replace(
                        0, np.nan
                    )
                )

            if df.empty:
                continue

            # ---------------------------------------------------------
            # Construct adjusted OHLC
            # ---------------------------------------------------------

            df["adj_open"] = (
                df["open"]
                * adjustment_factor
            )

            df["adj_high"] = (
                df["high"]
                * adjustment_factor
            )

            df["adj_low"] = (
                df["low"]
                * adjustment_factor
            )

            # ---------------------------------------------------------
            # Adjusted OHLC validation
            # ---------------------------------------------------------

            bad_adj_ohlc = (
                (
                    df["adj_high"]
                    + tol
                    < df["adj_low"]
                )
                | (
                    df["adj_open"]
                    > df["adj_high"]
                    + tol
                )
                | (
                    df["adj_open"]
                    + tol
                    < df["adj_low"]
                )
                | (
                    df["adj_close"]
                    > df["adj_high"]
                    + tol
                )
                | (
                    df["adj_close"]
                    + tol
                    < df["adj_low"]
                )
            )

            n_bad_adj = int(
                bad_adj_ohlc.sum()
            )

            if n_bad_adj:

                print(
                    f"[WARNING] {path.name}: "
                    f"{n_bad_adj} adjusted OHLC rows "
                    "failed validation"
                )

                # Do not silently throw away observations here.
                # Keep them for inspection unless they are truly
                # mathematically invalid.
                df = df.loc[
                    ~bad_adj_ohlc
                ].copy()

            if df.empty:
                continue

            # ---------------------------------------------------------
            # Final columns
            # ---------------------------------------------------------

            frames.append(
                df[
                    [
                        "date",
                        "ticker",
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj_open",
                        "adj_high",
                        "adj_low",
                        "adj_close",
                        "volume",
                        "dividends",
                        "stock_splits",
                    ]
                ]
            )

        except Exception as exc:

            print(
                f"[SKIP] {path.name}: {exc}"
            )

    if not frames:
        raise RuntimeError(
            "No usable Yahoo price files found."
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices["date"] = normalize_datetime(
        prices["date"]
    )

    prices = (
        prices
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    return prices


# =====================================================================
# Build PIT dataset
# =====================================================================

def build_pit_dataset() -> pd.DataFrame:
    """
    Build the daily point-in-time research dataset.

    For each trading date:

        1. Find latest membership snapshot <= date.
        2. Treat it as the PIT constituent universe.
        3. Join matching Yahoo price observations.
        4. Preserve source and Yahoo identifiers.
    """

    # ---------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------

    membership = load_membership()

    mapping = load_mapping()

    prices = load_price_data()

    print(
        "\nMembership snapshot rows:",
        len(membership),
    )

    print(
        "Price rows:",
        len(prices),
    )

    # ---------------------------------------------------------------
    # Membership change points
    # ---------------------------------------------------------------

    change_points = (
        build_membership_change_points(
            membership
        )
    )

    print(
        "Membership change points:",
        len(change_points),
    )

    # ---------------------------------------------------------------
    # Expanded membership
    # ---------------------------------------------------------------

    membership_rows = (
        expand_membership_change_points(
            change_points,
            mapping,
        )
    )

    # ---------------------------------------------------------------
    # Normalize date keys
    # ---------------------------------------------------------------

    prices["date"] = normalize_datetime(
        prices["date"]
    )

    membership_rows[
        "snapshot_date"
    ] = normalize_datetime(
        membership_rows["snapshot_date"]
    )

    change_dates = (
        change_points[
            ["snapshot_date"]
        ]
        .drop_duplicates()
        .sort_values("snapshot_date")
        .rename(
            columns={
                "snapshot_date":
                    "membership_snapshot_date"
            }
        )
        .reset_index(drop=True)
    )

    change_dates[
        "membership_snapshot_date"
    ] = normalize_datetime(
        change_dates[
            "membership_snapshot_date"
        ]
    )

    # ---------------------------------------------------------------
    # Actual price dates
    # ---------------------------------------------------------------

    price_dates = (
        prices[
            ["date"]
        ]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    price_dates["date"] = normalize_datetime(
        price_dates["date"]
    )

    # ---------------------------------------------------------------
    # PIT as-of lookup
    # ---------------------------------------------------------------

    price_dates = pd.merge_asof(
        price_dates.sort_values("date"),
        change_dates.sort_values(
            "membership_snapshot_date"
        ),
        left_on="date",
        right_on="membership_snapshot_date",
        direction="backward",
    )

    # No membership known yet.
    price_dates = price_dates.dropna(
        subset=[
            "membership_snapshot_date"
        ]
    )

    price_dates[
        "membership_snapshot_date"
    ] = normalize_datetime(
        price_dates[
            "membership_snapshot_date"
        ]
    )

    # ---------------------------------------------------------------
    # Attach PIT snapshot to each price row
    # ---------------------------------------------------------------

    prices = prices.merge(
        price_dates[
            [
                "date",
                "membership_snapshot_date",
            ]
        ],
        on="date",
        how="inner",
    )

    # ---------------------------------------------------------------
    # Join prices to the PIT constituent set
    # ---------------------------------------------------------------

    pit = prices.merge(
        membership_rows[
            [
                "snapshot_date",
                "source_ticker",
                "yahoo_ticker",
            ]
        ],
        left_on=[
            "membership_snapshot_date",
            "ticker",
        ],
        right_on=[
            "snapshot_date",
            "yahoo_ticker",
        ],
        how="inner",
    )

    pit = pit.drop(
        columns=[
            "snapshot_date"
        ]
    )

    # ---------------------------------------------------------------
    # Exclude known corrupted Yahoo histories
    # ---------------------------------------------------------------

    pit = pit[
        ~pit["yahoo_ticker"].isin(
            EXCLUDED_PRICE_TICKERS
        )
    ].copy()

    # ---------------------------------------------------------------
    # Final cutoff safety check
    # ---------------------------------------------------------------

    if pit["date"].max() > RESEARCH_END_DATE:
        raise AssertionError(
            "PIT data exceeds research cutoff: "
            f"{pit["date"].max()} > "
            f"{RESEARCH_END_DATE}"
        )

    # ---------------------------------------------------------------
    # Final column order
    # ---------------------------------------------------------------

    final_columns = [
        "date",
        "membership_snapshot_date",
        "source_ticker",
        "yahoo_ticker",

        # Raw OHLC
        "open",
        "high",
        "low",
        "close",

        # Adjusted OHLC
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",

        # Volume / corporate actions
        "volume",
        "dividends",
        "stock_splits",
    ]

    pit = pit[
        [
            column
            for column in final_columns
            if column in pit.columns
        ]
    ]

    # ---------------------------------------------------------------
    # Deduplicate
    # ---------------------------------------------------------------

    pit = (
        pit
        .sort_values(
            [
                "date",
                "source_ticker",
            ]
        )
        .drop_duplicates(
            subset=[
                "date",
                "source_ticker",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # =================================================================
    # Final validation
    # =================================================================

    required_final = {
        "date",
        "membership_snapshot_date",
        "source_ticker",
        "yahoo_ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    }

    missing_final = (
        required_final
        - set(pit.columns)
    )

    if missing_final:
        raise ValueError(
            "Final PIT dataset is missing columns: "
            f"{sorted(missing_final)}"
        )

    # ---------------------------------------------------------------
    # Null validation
    # ---------------------------------------------------------------

    numeric_required = [
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
    ]

    null_counts = (
        pit[numeric_required]
        .isna()
        .sum()
    )

    if null_counts.any():

        raise ValueError(
            "Final PIT dataset contains null price values:\n"
            f"{null_counts[null_counts > 0]}"
        )

    # ---------------------------------------------------------------
    # Duplicate validation
    # ---------------------------------------------------------------

    duplicate_count = int(
        pit.duplicated(
            subset=[
                "date",
                "source_ticker",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            "Final PIT dataset contains "
            f"{duplicate_count} duplicate "
            "(date, source_ticker) rows."
        )

    # ---------------------------------------------------------------
    # Raw OHLC validation
    # ---------------------------------------------------------------

    tol = OHLC_TOLERANCE

    bad_raw_ohlc = (
        (pit["high"] + tol < pit["low"])
        | (pit["open"] > pit["high"] + tol)
        | (pit["open"] + tol < pit["low"])
        | (pit["close"] > pit["high"] + tol)
        | (pit["close"] + tol < pit["low"])
    )

    bad_raw_count = int(
        bad_raw_ohlc.sum()
    )

    if bad_raw_count:

        raise ValueError(
            "Final PIT dataset contains "
            f"{bad_raw_count} invalid raw OHLC rows."
        )

    # ---------------------------------------------------------------
    # Adjusted OHLC validation
    # ---------------------------------------------------------------

    bad_adj_ohlc = (
        (
            pit["adj_high"]
            + tol
            < pit["adj_low"]
        )
        | (
            pit["adj_open"]
            > pit["adj_high"]
            + tol
        )
        | (
            pit["adj_open"]
            + tol
            < pit["adj_low"]
        )
        | (
            pit["adj_close"]
            > pit["adj_high"]
            + tol
        )
        | (
            pit["adj_close"]
            + tol
            < pit["adj_low"]
        )
    )

    bad_adj_count = int(
        bad_adj_ohlc.sum()
    )

    if bad_adj_count:

        raise ValueError(
            "Final PIT dataset contains "
            f"{bad_adj_count} invalid adjusted "
            "OHLC rows after tolerance."
        )

    return pit


# =====================================================================
# Summary
# =====================================================================

def print_summary(
    pit: pd.DataFrame,
) -> None:

    print(
        "\n=== PIT DATASET ==="
    )

    print(
        "Rows:",
        len(pit),
    )

    print(
        "Unique source securities:",
        pit["source_ticker"].nunique(),
    )

    print(
        "Unique Yahoo tickers:",
        pit["yahoo_ticker"].nunique(),
    )

    print(
        "Date range:",
        pit["date"].min(),
        "→",
        pit["date"].max(),
    )

    print(
        "Research as-of date:",
        RESEARCH_END_DATE,
    )

    print(
        "Excluded Yahoo tickers:",
        sorted(EXCLUDED_PRICE_TICKERS),
    )

    # ---------------------------------------------------------------
    # Daily breadth
    # ---------------------------------------------------------------

    daily_breadth = (
        pit
        .groupby("date")[
            "source_ticker"
        ]
        .nunique()
    )

    print(
        "Mean constituents with price data:",
        daily_breadth.mean(),
    )

    print(
        "Minimum daily constituents with price data:",
        daily_breadth.min(),
    )

    print(
        "Maximum daily constituents with price data:",
        daily_breadth.max(),
    )

    # ---------------------------------------------------------------
    # Mapped securities
    # ---------------------------------------------------------------

    mapped = pit[
        pit["source_ticker"]
        != pit["yahoo_ticker"]
    ]

    print(
        "\n=== MAPPED SECURITY COVERAGE ==="
    )

    print(
        "Mapped source/security rows:",
        len(mapped),
    )

    print(
        "Mapped source tickers:",
        mapped[
            "source_ticker"
        ].nunique(),
    )

    # ---------------------------------------------------------------
    # Corporate actions
    # ---------------------------------------------------------------

    print(
        "\n=== CORPORATE ACTIONS ==="
    )

    dividend_rows = int(
        pit["dividends"]
        .fillna(0)
        .ne(0)
        .sum()
    )

    split_rows = int(
        pit["stock_splits"]
        .fillna(0)
        .ne(0)
        .sum()
    )

    print(
        "Rows with dividends:",
        dividend_rows,
    )

    print(
        "Rows with stock splits:",
        split_rows,
    )

    # ---------------------------------------------------------------
    # Date-level mapped security check
    # ---------------------------------------------------------------

    print(
        "\n=== SAMPLE MAPPED ROWS ==="
    )

    if mapped.empty:

        print("None")

    else:

        print(
            mapped.head(10)
            .to_string(index=False)
        )

    # ---------------------------------------------------------------
    # First rows
    # ---------------------------------------------------------------

    display_columns = [
        "date",
        "membership_snapshot_date",
        "source_ticker",
        "yahoo_ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    print(
        "\n=== FIRST 10 ROWS ==="
    )

    print(
        pit[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\n=== LAST 10 ROWS ==="
    )

    print(
        pit[
            display_columns
        ]
        .tail(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Date-level coverage
    # ---------------------------------------------------------------

    print(
        "\n=== DAILY COVERAGE SUMMARY ==="
    )

    print(
        daily_breadth.describe()
    )


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    pit = build_pit_dataset()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pit.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        pit
    )

    print(
        "\nSaved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()