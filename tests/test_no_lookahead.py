"""
Regression guard for the project's #1 automatic-fail condition: a
signal computed for date t must never change depending on data that
only exists after t. This is inherent to how pandas rolling windows
work, but is worth pinning down as an explicit test given how central
it is to the task brief.
"""

import numpy as np
import pandas as pd

from src.strategy.strategies import sma_crossover_signal


def _synthetic_prices(n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    price = 100.0 + np.cumsum(rng.normal(0, 1, size=n_days))

    return pd.DataFrame(
        {
            "date": dates,
            "source_ticker": "AAA",
            "yahoo_ticker": "AAA",
            "adj_high": price + 0.5,
            "adj_low": price - 0.5,
            "adj_close": price,
        }
    )


def test_signal_at_t_is_unchanged_by_truncating_future_data():
    full = _synthetic_prices(300)

    cutoff_idx = 250
    cutoff_date = full.loc[cutoff_idx, "date"]

    truncated = full.iloc[: cutoff_idx + 1].copy()

    full_signal = sma_crossover_signal(full, fast=10, slow=50, mode="long_only")
    truncated_signal = sma_crossover_signal(truncated, fast=10, slow=50, mode="long_only")

    full_value = full_signal.loc[full_signal["date"] == cutoff_date, "signal"].iloc[0]
    truncated_value = truncated_signal.loc[
        truncated_signal["date"] == cutoff_date, "signal"
    ].iloc[0]

    assert full_value == truncated_value


def test_signal_history_before_cutoff_is_identical():
    full = _synthetic_prices(300)

    cutoff_idx = 250
    truncated = full.iloc[: cutoff_idx + 1].copy()

    full_signal = sma_crossover_signal(full, fast=10, slow=50, mode="long_only")
    truncated_signal = sma_crossover_signal(truncated, fast=10, slow=50, mode="long_only")

    full_history = full_signal.iloc[: cutoff_idx + 1]["signal"].to_numpy()
    truncated_history = truncated_signal["signal"].to_numpy()

    np.testing.assert_array_equal(full_history, truncated_history)
