"""
Purged & embargoed walk-forward CV.

Reuses the exact same rolling windows as the rule-based walk-forward
harness (`src.walkforward.windows.generate_windows`) so the ML
evaluation is directly comparable to Phase 2/3 — same IS/OOS periods,
same stitching.

Within each window, train = IS range, test = OOS range (never
overlapping, train always strictly chronologically before test — this
is what makes it "walk-forward" rather than k-fold). Two buffers are
removed from the END of the IS range, immediately before the OOS
range starts, in trading days (not calendar days):

- purge (default 1 day): the label at IS's last date is `next-day
  return`, i.e. it depends on the first OOS session's price. Any
  training row whose label horizon crosses into the test period must
  be dropped, or the model would be trained on a label that is
  partly test-period information.
- embargo (default 5 days): additional conservative buffer for the
  rolling-window features in this dataset (e.g. 20-day volatility,
  Bollinger, ATR) whose lookback could span the IS/OOS boundary. Not
  strictly required for a pure chronological walk-forward (train is
  never re-used from after a test fold here, unlike k-fold CV), but
  applied anyway as the standard extra safety margin.

No `train_test_split` / `KFold` anywhere in this module or the code
that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.walkforward.windows import Window


@dataclass(frozen=True)
class MLFold:
    window: Window
    train_mask: np.ndarray
    test_mask: np.ndarray
    purge_embargo_cutoff: pd.Timestamp


def purged_embargoed_folds(
    dates: pd.Series,
    windows: list[Window],
    purge_days: int = 1,
    embargo_days: int = 5,
) -> list[MLFold]:

    trading_days = pd.Series(pd.to_datetime(dates.unique())).sort_values()
    trading_days = trading_days.reset_index(drop=True)

    gap_days = purge_days + embargo_days

    folds = []

    for window in windows:

        oos_start_pos = trading_days.searchsorted(window.oos_start, side="left")

        cutoff_pos = max(oos_start_pos - gap_days, 0)
        purge_embargo_cutoff = (
            trading_days.iloc[cutoff_pos]
            if cutoff_pos < len(trading_days)
            else window.oos_start
        )

        train_mask = (
            (dates >= window.is_start) & (dates < purge_embargo_cutoff)
        ).to_numpy()

        test_mask = (
            (dates >= window.oos_start) & (dates <= window.oos_end)
        ).to_numpy()

        folds.append(
            MLFold(
                window=window,
                train_mask=train_mask,
                test_mask=test_mask,
                purge_embargo_cutoff=purge_embargo_cutoff,
            )
        )

    return folds
