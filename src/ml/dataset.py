"""
Phase 4 — ML feature/label dataset.

Builds the feature matrix directly from the existing
`pit_features.parquet` (no new indicator computation) and the label
from the existing `build_next_day_returns` (no reimplementation of
return construction) — this module is purely a join + label + light
feature hygiene.

Feature hygiene note: 5 of the 35 raw indicator columns are dropped
because they are unbounded/price-level quantities whose already
scale-free counterparts exist elsewhere in the same feature set —
carrying both would just be redundant, differently-scaled copies of
the same information, which hurts a pooled cross-sectional model more
than it helps:
  - donchian_upper_20 / donchian_lower_20 (raw price levels)
    -> donchian_position_20 already encodes this scale-free.
  - bb_upper_20_2.0 / bb_lower_20_2.0 (raw price levels)
    -> bb_zscore_20_2.0 / bb_width_20_2.0 already encode this
       scale-free.
  - obv (unbounded cumulative volume, not comparable across tickers
    or time, no scale-free counterpart in the feature set).
This is a minimal hygiene step, not new feature engineering — every
column used is one that already exists in build_features.py's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

_DROPPED_RAW_LEVEL_COLUMNS = {
    "donchian_upper_20",
    "donchian_lower_20",
    "bb_upper_20_2.0",
    "bb_lower_20_2.0",
    "obv",
}

_NON_FEATURE_COLUMNS = {
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


def load_feature_frame() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["source_ticker"] = df["source_ticker"].astype(str).str.strip().str.upper()
    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).str.strip().str.upper()
    return df.sort_values(["date", "source_ticker"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        c
        for c in df.columns
        if c not in _NON_FEATURE_COLUMNS and c not in _DROPPED_RAW_LEVEL_COLUMNS
    ]
    return sorted(candidates)


def build_ml_dataset(
    df: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (dataset, feature_cols).

    dataset columns: date, source_ticker, yahoo_ticker, <feature_cols>,
    next_return, label.

    label = 1 if next_return > 0, 0 if next_return < 0. Rows with a
    missing or exactly-zero next_return are dropped — a missing return
    means there is nothing tradable to label, and an exact zero is an
    ambiguous up/down call that would just inject label noise.
    """

    cols = feature_columns(df)

    merged = df.merge(
        yahoo_returns[["date", "ticker", "next_date", "next_return"]],
        left_on=["date", "yahoo_ticker"],
        right_on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["ticker"])

    merged = merged[merged["next_return"].notna() & (merged["next_return"] != 0.0)]

    merged = merged.dropna(subset=cols)

    merged["label"] = (merged["next_return"] > 0.0).astype(int)

    keep = (
        ["date", "source_ticker", "yahoo_ticker", "adj_close"]
        + cols
        + ["next_return", "label"]
    )

    dataset = merged[keep].sort_values(["date", "source_ticker"]).reset_index(drop=True)

    return dataset, cols
