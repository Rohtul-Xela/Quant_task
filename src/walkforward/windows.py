"""
Rolling walk-forward window generation.

5-year in-sample / 1-year out-of-sample / 1-year step, calendar-year
aligned. The final OOS segment is a partial stub (the research dataset
ends 2026-08-11, mid-year) — it is kept, not dropped, and flagged via
`is_partial_oos` so downstream code/reporting can call it out rather
than silently averaging it in as if it were a full year.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Window:
    index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_partial_oos: bool


def generate_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    is_years: int = 5,
    oos_years: int = 1,
    step_years: int = 1,
) -> list[Window]:

    data_start = pd.Timestamp(data_start)
    data_end = pd.Timestamp(data_end)

    if is_years <= 0 or oos_years <= 0 or step_years <= 0:
        raise ValueError("is_years, oos_years and step_years must be positive.")

    windows: list[Window] = []

    is_start = pd.Timestamp(year=data_start.year, month=1, day=1)
    index = 0

    while True:

        is_end = (
            is_start
            + pd.DateOffset(years=is_years)
            - pd.Timedelta(days=1)
        )

        oos_start = is_end + pd.Timedelta(days=1)

        oos_end_full = (
            oos_start
            + pd.DateOffset(years=oos_years)
            - pd.Timedelta(days=1)
        )

        if oos_start > data_end:
            break

        oos_end = min(oos_end_full, data_end)

        windows.append(
            Window(
                index=index,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_partial_oos=oos_end < oos_end_full,
            )
        )

        index += 1
        is_start = is_start + pd.DateOffset(years=step_years)

    if not windows:
        raise ValueError(
            "No walk-forward windows generated — check data_start/data_end "
            "against is_years/oos_years."
        )

    return windows


def windows_to_frame(windows: list[Window]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_index": w.index,
                "is_start": w.is_start,
                "is_end": w.is_end,
                "oos_start": w.oos_start,
                "oos_end": w.oos_end,
                "is_partial_oos": w.is_partial_oos,
            }
            for w in windows
        ]
    )
