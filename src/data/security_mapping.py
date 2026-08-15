from __future__ import annotations

from pathlib import Path

import pandas as pd


MAPPING_FILE = Path(
    "data/processed/security_mapping.csv"
)


def load_security_mapping() -> pd.DataFrame:
    """
    Load manually verified historical security mappings.

    The mapping file is deliberately explicit.
    We do not infer mappings automatically.
    """

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_FILE}"
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
            f"Missing mapping columns: {sorted(missing)}"
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

    return df


def apply_mapping(
    tickers: pd.Series,
    mapping: pd.DataFrame,
) -> pd.Series:
    """
    Apply verified source_ticker -> yahoo_ticker mappings.

    Tickers without an explicit mapping remain unchanged.
    """

    lookup = dict(
        zip(
            mapping["source_ticker"],
            mapping["yahoo_ticker"],
        )
    )

    return (
        tickers
        .astype(str)
        .str.upper()
        .str.strip()
        .map(
            lambda x: lookup.get(x, x)
        )
    )