# src/data/validate.py

from __future__ import annotations

import pandas as pd


def validate_prices(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []

    if df.empty:
        problems.append("empty dataframe")
        return problems

    if df["date"].duplicated().any():
        problems.append("duplicate dates")

    if not df["date"].is_monotonic_increasing:
        problems.append("dates not sorted")

    if df[["open", "high", "low", "close"]].isna().any().any():
        problems.append("missing OHLC")

    if (df["volume"] < 0).any():
        problems.append("negative volume")

    if (df["high"] < df["low"]).any():
        problems.append("high < low")

    if (df["open"] > df["high"]).any():
        problems.append("open > high")

    if (df["open"] < df["low"]).any():
        problems.append("open < low")

    if (df["close"] > df["high"]).any():
        problems.append("close > high")

    if (df["close"] < df["low"]).any():
        problems.append("close < low")

    return problems