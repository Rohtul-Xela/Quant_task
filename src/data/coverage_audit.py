from __future__ import annotations

from pathlib import Path

import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

MEMBERSHIP_FILE = Path(
    "data/processed/membership_snapshots.parquet"
)

STATUS_FILE = Path(
    "data/validation/download_status.csv"
)

MAPPING_FILE = Path(
    "data/processed/security_mapping.csv"
)

PRICE_DIR = Path(
    "data/raw/yahoo"
)

OUTPUT_FILE = Path(
    "data/processed/coverage_audit.parquet"
)


# =====================================================================
# Load membership
# =====================================================================

def load_membership() -> pd.DataFrame:
    """
    Load normalized PIT membership snapshots.

    Columns:
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

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Membership file missing columns: "
            f"{sorted(missing)}"
        )

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


# =====================================================================
# Load security mappings
# =====================================================================

def load_security_mapping() -> pd.DataFrame:
    """
    Load manually verified source_ticker -> yahoo_ticker mappings.

    Tickers without a mapping remain unchanged.
    """

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Security mapping file not found: {MAPPING_FILE}"
        )

    df = pd.read_csv(
        MAPPING_FILE
    )

    required = {
        "source_ticker",
        "yahoo_ticker",
        "mapping_type",
        "confidence",
        "reason",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Security mapping missing columns: "
            f"{sorted(missing)}"
        )

    df["source_ticker"] = (
        df["source_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["yahoo_ticker"] = (
        df["yahoo_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Prevent ambiguous mappings.
    duplicate_sources = (
        df["source_ticker"]
        [df["source_ticker"].duplicated()]
        .unique()
    )

    if len(duplicate_sources) > 0:
        raise ValueError(
            "Duplicate source_ticker mappings found: "
            f"{duplicate_sources.tolist()}"
        )

    return df


# =====================================================================
# Build mapping lookup
# =====================================================================

def build_mapping_lookup(
    mapping: pd.DataFrame,
) -> dict[str, dict]:
    """
    Convert the mapping dataframe into a dictionary keyed by
    source ticker.
    """

    lookup: dict[str, dict] = {}

    for _, row in mapping.iterrows():

        source = row["source_ticker"]

        lookup[source] = {
            "yahoo_ticker": row["yahoo_ticker"],
            "mapping_type": row["mapping_type"],
            "mapping_confidence": row["confidence"],
            "mapping_reason": row["reason"],
        }

    return lookup


# =====================================================================
# Membership summary
# =====================================================================

def build_membership_summary(
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per source ticker with its PIT membership coverage.
    """

    summary = (
        membership
        .groupby("ticker")
        .agg(
            membership_first_date=(
                "snapshot_date",
                "min",
            ),
            membership_last_date=(
                "snapshot_date",
                "max",
            ),
            membership_snapshots=(
                "snapshot_date",
                "nunique",
            ),
        )
        .reset_index()
    )

    return summary


# =====================================================================
# Load Yahoo price coverage
# =====================================================================

def load_price_summary() -> pd.DataFrame:
    """
    Read all cached Yahoo Parquet files.

    Returns one row per Yahoo ticker.
    """

    rows = []

    for path in sorted(
        PRICE_DIR.glob("*.parquet")
    ):

        yahoo_ticker = path.stem

        try:

            df = pd.read_parquet(path)

            if df.empty:
                continue

            if "date" not in df.columns:
                print(
                    f"Skipping {path}: "
                    "missing date column"
                )
                continue

            df["date"] = pd.to_datetime(
                df["date"]
            )

            # Only rows with an actual date count
            # toward coverage.
            df = df.dropna(
                subset=["date"]
            )

            if df.empty:
                continue

            rows.append(
                {
                    "yahoo_ticker": yahoo_ticker,
                    "price_first_date": (
                        df["date"].min()
                    ),
                    "price_last_date": (
                        df["date"].max()
                    ),
                    "price_rows": len(df),
                }
            )

        except Exception as exc:

            print(
                f"Could not read {path}: {exc}"
            )

    return pd.DataFrame(rows)


# =====================================================================
# Build mapping-aware audit
# =====================================================================

def build_audit() -> pd.DataFrame:

    # ---------------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------------

    membership = load_membership()

    mapping = load_security_mapping()

    mapping_lookup = (
        build_mapping_lookup(
            mapping
        )
    )

    price_summary = (
        load_price_summary()
    )

    membership_summary = (
        build_membership_summary(
            membership
        )
    )

    # ---------------------------------------------------------------
    # Apply mapping to each source ticker
    # ---------------------------------------------------------------

    audit = membership_summary.copy()

    audit["yahoo_ticker"] = (
        audit["ticker"]
        .map(
            lambda ticker:
                mapping_lookup
                .get(
                    ticker,
                    {
                        "yahoo_ticker": ticker,
                        "mapping_type": "direct",
                        "mapping_confidence": None,
                        "mapping_reason": None,
                    },
                )["yahoo_ticker"]
        )
    )

    audit["mapping_type"] = (
        audit["ticker"]
        .map(
            lambda ticker:
                mapping_lookup
                .get(
                    ticker,
                    {
                        "mapping_type": "direct"
                    },
                )["mapping_type"]
        )
    )

    audit["mapping_confidence"] = (
        audit["ticker"]
        .map(
            lambda ticker:
                mapping_lookup
                .get(
                    ticker,
                    {
                        "mapping_confidence": None
                    },
                )["mapping_confidence"]
        )
    )

    audit["mapping_reason"] = (
        audit["ticker"]
        .map(
            lambda ticker:
                mapping_lookup
                .get(
                    ticker,
                    {
                        "mapping_reason": None
                    },
                )["mapping_reason"]
        )
    )

    # ---------------------------------------------------------------
    # Join against Yahoo prices using the mapped Yahoo ticker.
    # ---------------------------------------------------------------

    audit = audit.merge(
        price_summary,
        on="yahoo_ticker",
        how="left",
    )

    # ---------------------------------------------------------------
    # Availability flags
    # ---------------------------------------------------------------

    audit["has_price_data"] = (
        audit["price_rows"]
        .fillna(0)
        .gt(0)
    )

    # Broad range overlap.
    audit["price_overlaps_membership"] = (
        audit["has_price_data"]
        & (
            audit["price_last_date"]
            >= audit["membership_first_date"]
        )
        & (
            audit["price_first_date"]
            <= audit["membership_last_date"]
        )
    )

    # Mapping-aware classification.
    audit["coverage_type"] = "unavailable"

    audit.loc[
        audit["has_price_data"]
        & (
            audit["mapping_type"]
            == "direct"
        ),
        "coverage_type",
    ] = "direct"

    audit.loc[
        audit["has_price_data"]
        & (
            audit["mapping_type"]
            != "direct"
        ),
        "coverage_type",
    ] = "mapped"

    # Final broad research usability flag.
    audit["usable_for_research"] = (
        audit["price_overlaps_membership"]
    )

    return (
        audit
        .sort_values("ticker")
        .reset_index(drop=True)
    )


# =====================================================================
# Reporting
# =====================================================================

def print_summary(
    audit: pd.DataFrame,
) -> None:

    total = len(audit)

    direct = int(
        (
            audit["coverage_type"]
            == "direct"
        ).sum()
    )

    mapped = int(
        (
            audit["coverage_type"]
            == "mapped"
        ).sum()
    )

    unavailable = int(
        (
            audit["coverage_type"]
            == "unavailable"
        ).sum()
    )

    overlapping = int(
        audit[
            "price_overlaps_membership"
        ].sum()
    )

    print("\n=== COVERAGE AUDIT ===")

    print(
        "Historical tickers:",
        total,
    )

    print(
        "Direct Yahoo coverage:",
        direct,
    )

    print(
        "Mapped Yahoo coverage:",
        mapped,
    )

    print(
        "Unavailable Yahoo coverage:",
        unavailable,
    )

    print(
        "Price overlaps membership:",
        overlapping,
    )

    print(
        "Usable for research:",
        overlapping,
    )

    print(
        "Direct ticker coverage:",
        f"{direct / total:.2%}",
    )

    print(
        "Coverage including mappings:",
        f"{(direct + mapped) / total:.2%}",
    )

    print(
        "Usable research coverage:",
        f"{overlapping / total:.2%}",
    )

    # ---------------------------------------------------------------
    # Mapping details
    # ---------------------------------------------------------------

    mapped_df = audit[
        audit["mapping_type"] != "direct"
    ]

    print(
        "\n=== MAPPED SECURITIES ==="
    )

    if mapped_df.empty:

        print("None")

    else:

        print(
            mapped_df[
                [
                    "ticker",
                    "yahoo_ticker",
                    "mapping_type",
                    "mapping_confidence",
                    "price_first_date",
                    "price_last_date",
                    "price_rows",
                    "price_overlaps_membership",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------------
    # No price data
    # ---------------------------------------------------------------

    missing = audit[
        ~audit["has_price_data"]
    ]

    print(
        "\n=== NO YAHOO PRICE DATA ==="
    )

    print(
        "Count:",
        len(missing),
    )

    if not missing.empty:

        print(
            missing[
                [
                    "ticker",
                    "membership_first_date",
                    "membership_last_date",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------------
    # Price data but no overlap
    # ---------------------------------------------------------------

    no_overlap = audit[
        audit["has_price_data"]
        & ~audit[
            "price_overlaps_membership"
        ]
    ]

    print(
        "\n=== PRICE DATA BUT NO "
        "MEMBERSHIP OVERLAP ==="
    )

    print(
        "Count:",
        len(no_overlap),
    )

    if not no_overlap.empty:

        print(
            no_overlap[
                [
                    "ticker",
                    "yahoo_ticker",
                    "mapping_type",
                    "membership_first_date",
                    "membership_last_date",
                    "price_first_date",
                    "price_last_date",
                ]
            ].to_string(index=False)
        )


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    audit = build_audit()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        audit
    )

    print(
        "\nSaved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()