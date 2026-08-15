from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.indicators import calculate_indicators


# =====================================================================
# Project paths
# =====================================================================

# Current project structure:
#
# StockhuntTask/
# ├── .venv/
# └── src/
#     ├── data/
#     │   └── data/
#     │       └── processed/
#     │           └── pit_daily.parquet
#     │
#     └── features/
#         ├── indicators.py
#         └── build_features.py
#
# build_features.py:
# StockhuntTask/src/features/build_features.py
#
# parents[0] = features
# parents[1] = src
# parents[2] = StockhuntTask

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
)

INPUT_FILE = (
    DATA_DIR
    / "processed"
    / "pit_daily.parquet"
)

OUTPUT_FILE = (
    DATA_DIR
    / "processed"
    / "pit_features.parquet"
)


# =====================================================================
# Helpers
# =====================================================================

def print_section(title: str) -> None:
    """Print a readable console section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    print_section(
        "PROJECT PATHS"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Data directory:",
        DATA_DIR,
    )

    print(
        "Input:",
        INPUT_FILE,
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "\nPIT dataset not found.\n"
            f"Expected:\n{INPUT_FILE}"
        )

    # ---------------------------------------------------------------
    # Load PIT dataset
    # ---------------------------------------------------------------

    print_section(
        "LOADING PIT DATASET"
    )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        "Input rows:",
        len(df),
    )

    print(
        "Input columns:",
        len(df.columns),
    )

    print(
        "Date range:",
        df["date"].min(),
        "→",
        df["date"].max(),
    )

    print(
        "Source securities:",
        df["source_ticker"].nunique(),
    )

    # ---------------------------------------------------------------
    # Validate required input columns
    # ---------------------------------------------------------------

    required_columns = {
        "date",
        "source_ticker",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "PIT dataset is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------------
    # Standardize input
    # ---------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["source_ticker"] = (
        df["source_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ---------------------------------------------------------------
    # Check date/security uniqueness
    # ---------------------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=[
                "date",
                "source_ticker",
            ]
        ).sum()
    )

    print(
        "Duplicate (date, source_ticker) rows:",
        duplicate_count,
    )

    if duplicate_count:

        raise ValueError(
            "Input PIT dataset contains "
            f"{duplicate_count} duplicate "
            "(date, source_ticker) rows."
        )

    # ---------------------------------------------------------------
    # Sort before indicator calculation
    # ---------------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "source_ticker",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Calculate indicators
    # ---------------------------------------------------------------

    print_section(
        "CALCULATING INDICATORS"
    )

    features = calculate_indicators(
        df
    )

    # ---------------------------------------------------------------
    # Verify row count
    # ---------------------------------------------------------------

    if len(features) != len(df):

        raise ValueError(
            "Indicator calculation changed the number of rows:\n"
            f"Input:  {len(df)}\n"
            f"Output: {len(features)}"
        )

    # ---------------------------------------------------------------
    # Verify date/security identity survived
    # ---------------------------------------------------------------

    if not (
        features["date"].equals(
            df["date"]
        )
        and features["source_ticker"].equals(
            df["source_ticker"]
        )
    ):

        raise ValueError(
            "Indicator calculation changed the "
            "date/source_ticker alignment."
        )

    # ---------------------------------------------------------------
    # Identify feature columns
    # ---------------------------------------------------------------

    original_columns = set(
        df.columns
    )

    indicator_columns = [
        column
        for column in features.columns
        if column not in original_columns
    ]

    print(
        "Indicator columns:",
        len(indicator_columns),
    )

    for column in indicator_columns:

        print(
            " -",
            column,
        )

    # ---------------------------------------------------------------
    # Indicator NaN summary
    # ---------------------------------------------------------------

    print_section(
        "INDICATOR NaN SUMMARY"
    )

    nan_summary = (
        features[indicator_columns]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for column, count in nan_summary.items():

        print(
            f"{column}: {int(count)}"
        )

    # ---------------------------------------------------------------
    # Indicator coverage
    # ---------------------------------------------------------------

    print_section(
        "INDICATOR COVERAGE"
    )

    coverage = (
        features[indicator_columns]
        .notna()
        .mean()
        * 100.0
    )

    coverage = coverage.sort_values()

    for column, pct in coverage.items():

        print(
            f"{column}: {pct:.2f}%"
        )

    # ---------------------------------------------------------------
    # Infinite value check
    #
    # NaN is expected during indicator warm-up.
    # +inf / -inf is not.
    # ---------------------------------------------------------------

    print_section(
        "INFINITE VALUE CHECK"
    )

    infinite_mask = (
        features[indicator_columns]
        .isin(
            [
                float("inf"),
                float("-inf"),
            ]
        )
    )

    infinite_counts = (
        infinite_mask
        .sum()
        .sort_values(
            ascending=False
        )
    )

    infinite_counts = (
        infinite_counts[
            infinite_counts > 0
        ]
    )

    if infinite_counts.empty:

        print(
            "No infinite indicator values found."
        )

    else:

        print(
            infinite_counts.to_string()
        )

        raise ValueError(
            "Infinite indicator values detected. "
            "The feature dataset will not be saved."
        )

    # ---------------------------------------------------------------
    # Final indicator sanity check
    #
    # At least one indicator must have valid observations.
    # ---------------------------------------------------------------

    if not indicator_columns:

        raise ValueError(
            "No indicator columns were generated."
        )

    valid_indicator_count = int(
        features[indicator_columns]
        .notna()
        .any()
        .sum()
    )

    if valid_indicator_count == 0:

        raise ValueError(
            "All indicator columns contain only NaN values."
        )

    # ---------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------

    print_section(
        "SAVING FEATURE DATASET"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Final confirmation
    # ---------------------------------------------------------------

    print_section(
        "FEATURE DATASET CREATED"
    )

    print(
        "Rows:",
        len(features),
    )

    print(
        "Columns:",
        len(features.columns),
    )

    print(
        "Indicator columns:",
        len(indicator_columns),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "\nFeature calculation completed successfully."
    )


if __name__ == "__main__":
    main()