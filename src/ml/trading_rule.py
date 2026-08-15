"""
Phase 4 — ML probability -> trading signal.

P(up) >= 0.70 -> long, P(up) <= 0.30 -> short, otherwise flat. The
0.70 threshold intentionally echoes the "70% confidence layer"
language from the product's own stated goal.

Output schema matches exactly what `run_backtest` requires (date,
source_ticker, yahoo_ticker, adj_close, signal), so ML predictions can
be run through the identical walk-forward stitching/cost machinery as
the rule-based strategies in Phase 2/3 — no separate backtest path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LONG_THRESHOLD = 0.70
SHORT_THRESHOLD = 0.30


def predictions_to_signal(
    test_df: pd.DataFrame,
    proba_up: np.ndarray,
    long_threshold: float = LONG_THRESHOLD,
    short_threshold: float = SHORT_THRESHOLD,
) -> pd.DataFrame:

    if len(test_df) != len(proba_up):
        raise ValueError("test_df and proba_up must be the same length.")

    result = test_df[["date", "source_ticker", "yahoo_ticker", "adj_close"]].copy()

    result["proba_up"] = np.asarray(proba_up)

    result["signal"] = np.select(
        [result["proba_up"] >= long_threshold, result["proba_up"] <= short_threshold],
        [1.0, -1.0],
        default=0.0,
    )

    return result
