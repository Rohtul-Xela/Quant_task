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

Cross-sectional point-in-time percentile normalization: every feature
column is replaced by its rank, as a percentile in [0, 1], among all
tickers with a valid value for that column ON THE SAME DATE — e.g. a
raw RSI of 75 becomes "this ticker's RSI is higher than 82% of the
universe today." This is point-in-time by construction (each date's
ranks use only that date's own already-causal indicator values, never
a future or full-sample statistic) and makes features comparable
across tickers and across time regardless of a feature's absolute
scale or a market-wide level shift (e.g. a volatility regime change
shifting every ticker's ATR up together no longer moves the *ranks*).
Replaces raw indicator values as the model input entirely — not an
additional column set.
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


def cross_sectional_percentile_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Replace each of `cols` with its same-date cross-sectional percentile
    rank in [0, 1]. `pct=True` ranking within a `groupby("date")` uses
    only that date's own rows, so this cannot look forward in time; NaNs
    are left as NaN (excluded from that date's ranking, not imputed).
    """

    result = df.copy()

    result[cols] = result.groupby("date")[cols].rank(pct=True)

    return result


def build_ml_dataset(
    df: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
    use_percentile_rank: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (dataset, feature_cols).

    dataset columns: date, source_ticker, yahoo_ticker, <feature_cols>,
    next_return, label.

    label = 1 if next_return > 0, 0 if next_return < 0. Rows with a
    missing or exactly-zero next_return are dropped — a missing return
    means there is nothing tradable to label, and an exact zero is an
    ambiguous up/down call that would just inject label noise.

    use_percentile_rank=True applies `cross_sectional_percentile_rank`
    (see above) instead of raw feature values. Opt-in, default False:
    tried as an ablation (`src/pipeline/run_ml_pit_percentile_ablation.py`)
    and found to decouple the model's predicted-probability calibration
    from the fixed 0.70/0.30 trading-rule threshold (percentile-ranked
    features are uniform on [0,1] with no fat tails, so predicted
    probabilities compress toward 0.5 and the threshold stops firing) —
    a real, diagnosed finding, not adopted as the default because it
    would require re-deriving the threshold rather than reusing the one
    that deliberately echoes the product's own "70% confidence" framing.
    See report.md for the full writeup.
    """

    cols = feature_columns(df)

    if use_percentile_rank:
        df = cross_sectional_percentile_rank(df, cols)

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
