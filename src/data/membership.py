from __future__ import annotations

from pathlib import Path

import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

URL = (
    "https://raw.githubusercontent.com/"
    "chinobing/historical_sp500_constituents/main/"
    "sp_500_historical_components.csv"
)

RAW_DIR = Path("data/raw/membership")
PROCESSED_DIR = Path("data/processed")

RAW_FILE = RAW_DIR / "sp500_historical_components.csv"
SNAPSHOT_FILE = PROCESSED_DIR / "membership_snapshots.parquet"
TICKER_FILE = PROCESSED_DIR / "historical_tickers.csv"

RESEARCH_START = pd.Timestamp("2008-01-01")


# =====================================================================
# Ticker handling
# =====================================================================

def normalize_ticker(ticker: str) -> str:
    """
    Normalize historical S&P tickers to Yahoo Finance symbols.
    """

    ticker = ticker.strip().upper()

    # Historical source alias.
    if ticker == "RVTY (PREVIOUSLY PKI)":
        return "RVTY"

    # Yahoo uses '-' for share-class tickers.
    ticker = ticker.replace(".", "-")

    return ticker


def parse_tickers(value: str) -> list[str]:
    """
    Parse a comma-separated ticker string and normalize each ticker.
    """

    if pd.isna(value):
        return []

    tickers = []

    for ticker in str(value).split(","):

        ticker = ticker.strip()

        if not ticker:
            continue

        tickers.append(
            normalize_ticker(ticker)
        )

    return tickers


# =====================================================================
# Load raw membership data
# =====================================================================

def load_membership() -> pd.DataFrame:
    """
    Download and load the historical S&P 500 constituent snapshots.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading historical S&P 500 membership..."
    )

    df = pd.read_csv(URL)

    print("\n=== RAW DATA ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    # ---------------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------------

    required_columns = {
        "date",
        "tickers",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------------
    # Parse date
    # ---------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    # ---------------------------------------------------------------
    # Basic null validation
    # ---------------------------------------------------------------

    nulls = df[
        ["date", "tickers"]
    ].isna().sum()

    if nulls.any():
        raise ValueError(
            "Membership data contains null values:\n"
            f"{nulls}"
        )

    # ---------------------------------------------------------------
    # Constituent counts
    # ---------------------------------------------------------------

    df["n_constituents"] = (
        df["tickers"]
        .str.split(",")
        .str.len()
    )

    return df


# =====================================================================
# Diagnostics
# =====================================================================

def print_basic_diagnostics(
    df: pd.DataFrame,
) -> None:

    print("\n=== BASIC DIAGNOSTICS ===")

    print(
        "Earliest membership date:",
        df["date"].min(),
    )

    print(
        "Latest membership date:",
        df["date"].max(),
    )

    print(
        "Number of snapshots:",
        len(df),
    )

    print("\nNull values:")
    print(df.isna().sum())

    print("\nConstituent count summary:")
    print(
        df["n_constituents"].describe()
    )


def get_research_period(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict the membership source to the assignment period:
    2008-01-01 onward.
    """

    research = (
        df[
            df["date"] >= RESEARCH_START
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if research.empty:
        raise ValueError(
            "No membership observations found "
            "for the research period."
        )

    return research


def print_research_diagnostics(
    research: pd.DataFrame,
) -> None:

    print("\n=== 2008+ MEMBERSHIP SUMMARY ===")

    print(
        "Snapshots:",
        len(research),
    )

    print(
        "Earliest:",
        research["date"].min(),
    )

    print(
        "Latest:",
        research["date"].max(),
    )

    print("\nConstituent count summary:")
    print(
        research["n_constituents"].describe()
    )

    print("\nConstituent count distribution:")
    print(
        research["n_constituents"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Yearly summary
    # ---------------------------------------------------------------

    yearly = research.copy()

    yearly["year"] = (
        yearly["date"].dt.year
    )

    yearly_summary = (
        yearly
        .groupby("year")["n_constituents"]
        .agg(
            snapshots="count",
            min="min",
            mean="mean",
            max="max",
        )
    )

    print("\nYearly constituent count:")
    print(
        yearly_summary.to_string()
    )


# =====================================================================
# Historical ticker universe
# =====================================================================

def get_unique_historical_tickers(
    research: pd.DataFrame,
) -> set[str]:

    tickers: set[str] = set()

    for value in research["tickers"]:

        tickers.update(
            parse_tickers(value)
        )

    return tickers


def print_ticker_diagnostics(
    research: pd.DataFrame,
) -> set[str]:

    tickers = (
        get_unique_historical_tickers(
            research
        )
    )

    print("\n=== UNIQUE HISTORICAL TICKERS ===")

    print(
        "Unique tickers from 2008-present:",
        len(tickers),
    )

    print("\nFirst 100 tickers:")

    print(
        sorted(tickers)[:100]
    )

    # ---------------------------------------------------------------
    # Original source share-class formatting
    # ---------------------------------------------------------------

    original_share_classes = set()

    for value in research["tickers"]:

        for ticker in str(value).split(","):

            ticker = ticker.strip().upper()

            if "." in ticker:

                original_share_classes.add(
                    ticker
                )

    print(
        "\n=== SHARE-CLASS STYLE TICKERS ==="
    )

    if original_share_classes:

        print(
            sorted(original_share_classes)
        )

    else:

        print("None found.")

    return tickers


# =====================================================================
# Snapshot-date diagnostics
# =====================================================================

def print_date_samples(
    research: pd.DataFrame,
) -> None:

    print(
        "\n=== FIRST 30 MEMBERSHIP DATES "
        "IN 2008 ==="
    )

    dates_2008 = (
        research[
            research["date"].dt.year == 2008
        ]["date"]
        .drop_duplicates()
        .sort_values()
        .head(30)
    )

    print(
        dates_2008.to_string(
            index=False
        )
    )

    print(
        "\n=== FIRST 30 MEMBERSHIP DATES "
        "IN 2026 ==="
    )

    dates_2026 = (
        research[
            research["date"].dt.year == 2026
        ]["date"]
        .drop_duplicates()
        .sort_values()
        .head(30)
    )

    print(
        dates_2026.to_string(
            index=False
        )
    )


# =====================================================================
# Historical ticker growth
# =====================================================================

def print_new_tickers_by_year(
    research: pd.DataFrame,
) -> None:

    print(
        "\n=== UNIQUE TICKERS BY YEAR ==="
    )

    seen: set[str] = set()

    temp = research.copy()

    temp["year"] = (
        temp["date"].dt.year
    )

    for year, group in temp.groupby("year"):

        year_tickers: set[str] = set()

        for value in group["tickers"]:

            year_tickers.update(
                parse_tickers(value)
            )

        new_tickers = (
            year_tickers - seen
        )

        seen.update(year_tickers)

        print(
            f"{year}: "
            f"{len(year_tickers)} present, "
            f"{len(new_tickers)} new cumulative names, "
            f"{len(seen)} cumulative total"
        )


# =====================================================================
# Membership change diagnostics
# =====================================================================

def analyze_membership_changes(
    research: pd.DataFrame,
) -> None:
    """
    Compare consecutive membership snapshots.

    This is diagnostic only. We do not convert these changes into
    membership intervals, because the historical source is sparse
    before recent years.
    """

    research = (
        research
        .sort_values("date")
        .reset_index(drop=True)
    )

    previous_universe: set[str] | None = None

    changes = []

    for _, row in research.iterrows():

        current_universe = set(
            parse_tickers(
                row["tickers"]
            )
        )

        if previous_universe is None:

            added = current_universe
            removed = set()

        else:

            added = (
                current_universe
                - previous_universe
            )

            removed = (
                previous_universe
                - current_universe
            )

        if added or removed:

            changes.append(
                {
                    "date": row["date"],
                    "added": len(added),
                    "removed": len(removed),
                    "turnover": (
                        len(added)
                        + len(removed)
                    ),
                }
            )

        previous_universe = (
            current_universe
        )

    changes_df = pd.DataFrame(
        changes
    )

    print(
        "\n=== MEMBERSHIP CHANGES ==="
    )

    print(
        "Snapshots with changes:",
        len(changes_df),
    )

    if changes_df.empty:

        print("No membership changes found.")

        return

    print("\nFirst 20 changes:")

    print(
        changes_df
        .head(20)
        .to_string(index=False)
    )

    print("\nChange summary:")

    print(
        changes_df[
            [
                "added",
                "removed",
                "turnover",
            ]
        ].describe()
    )


# =====================================================================
# Build PIT membership snapshots
# =====================================================================

def build_membership_snapshots(
    research: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand the source into:

        snapshot_date | ticker

    Each row represents what the source explicitly tells us about
    membership on that snapshot date.

    We intentionally DO NOT forward-fill membership between sparse
    historical snapshots.
    """

    rows = []

    for _, row in research.iterrows():

        snapshot_date = row["date"]

        tickers = parse_tickers(
            row["tickers"]
        )

        for ticker in tickers:

            rows.append(
                {
                    "snapshot_date": (
                        snapshot_date
                    ),
                    "ticker": ticker,
                }
            )

    snapshots = pd.DataFrame(
        rows,
        columns=[
            "snapshot_date",
            "ticker",
        ],
    )

    # ---------------------------------------------------------------
    # Defensive cleanup
    # ---------------------------------------------------------------

    snapshots = (
        snapshots
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

    if snapshots.empty:
        raise ValueError(
            "No membership snapshot rows were created."
        )

    return snapshots


# =====================================================================
# Save processed data
# =====================================================================

def save_processed_data(
    raw_df: pd.DataFrame,
    snapshots: pd.DataFrame,
    historical_tickers: set[str],
) -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Raw source
    # ---------------------------------------------------------------

    raw_df.to_csv(
        RAW_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Normalized PIT snapshots
    # ---------------------------------------------------------------

    snapshots.to_parquet(
        SNAPSHOT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Unique historical ticker universe
    # ---------------------------------------------------------------

    ticker_df = pd.DataFrame(
        {
            "ticker": sorted(
                historical_tickers
            )
        }
    )

    ticker_df.to_csv(
        TICKER_FILE,
        index=False,
    )

    print(
        f"\nSaved raw membership to: "
        f"{RAW_FILE}"
    )

    print(
        "Saved PIT snapshots to: "
        f"{SNAPSHOT_FILE}"
    )

    print(
        "Saved historical ticker "
        f"universe to: {TICKER_FILE}"
    )


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    # ---------------------------------------------------------------
    # 1. Load source
    # ---------------------------------------------------------------

    df = load_membership()

    # ---------------------------------------------------------------
    # 2. Complete source diagnostics
    # ---------------------------------------------------------------

    print_basic_diagnostics(df)

    # ---------------------------------------------------------------
    # 3. Restrict to 2008+
    # ---------------------------------------------------------------

    research = get_research_period(df)

    print_research_diagnostics(
        research
    )

    # ---------------------------------------------------------------
    # 4. Historical ticker universe
    # ---------------------------------------------------------------

    historical_tickers = (
        print_ticker_diagnostics(
            research
        )
    )

    # ---------------------------------------------------------------
    # 5. Date diagnostics
    # ---------------------------------------------------------------

    print_date_samples(
        research
    )

    # ---------------------------------------------------------------
    # 6. Historical ticker growth
    # ---------------------------------------------------------------

    print_new_tickers_by_year(
        research
    )

    # ---------------------------------------------------------------
    # 7. Membership changes
    # ---------------------------------------------------------------

    analyze_membership_changes(
        research
    )

    # ---------------------------------------------------------------
    # 8. Build normalized PIT snapshots
    # ---------------------------------------------------------------

    snapshots = (
        build_membership_snapshots(
            research
        )
    )

    print(
        "\n=== MEMBERSHIP SNAPSHOTS ==="
    )

    print(
        snapshots.head(20)
        .to_string(index=False)
    )

    print(
        "\nSnapshot rows:",
        len(snapshots),
    )

    print(
        "Historical unique tickers:",
        len(historical_tickers),
    )

    # ---------------------------------------------------------------
    # 9. Save
    # ---------------------------------------------------------------

    save_processed_data(
        raw_df=df,
        snapshots=snapshots,
        historical_tickers=historical_tickers,
    )


if __name__ == "__main__":
    main()