"""
Smoke tests + boundary checks for the 4 new rule-based strategies
(MACD, Stochastic, Bollinger mean-reversion, MFI mean-reversion) and
the 2 regime gates, run on synthetic data before committing to a full
sweep re-run.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.strategies import (
    _bb_target_state_long_only,
    _bb_target_state_long_short,
    adx_trend_regime_signal,
    bollinger_mean_reversion_signal,
    low_volatility_regime_signal,
    macd_crossover_signal,
    mfi_mean_reversion_signal,
    stochastic_crossover_signal,
)


def _synthetic_ohlcv(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n_days))
    high = close + rng.uniform(0.1, 1.0, size=n_days)
    low = close - rng.uniform(0.1, 1.0, size=n_days)
    volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype(float)

    return pd.DataFrame(
        {
            "date": dates,
            "source_ticker": "AAA",
            "yahoo_ticker": "AAA",
            "adj_high": high,
            "adj_low": low,
            "adj_close": close,
            "volume": volume,
        }
    )


@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_macd_crossover_smoke(mode):
    df = _synthetic_ohlcv()
    result = macd_crossover_signal(df, mode=mode)

    assert result["signal"].notna().all()
    allowed = {0.0, 1.0} if mode == "long_only" else {-1.0, 0.0, 1.0}
    assert set(result["signal"].unique()).issubset(allowed)


@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_stochastic_crossover_smoke(mode):
    df = _synthetic_ohlcv()
    result = stochastic_crossover_signal(df, mode=mode)

    assert result["signal"].notna().all()
    allowed = {0.0, 1.0} if mode == "long_only" else {-1.0, 0.0, 1.0}
    assert set(result["signal"].unique()).issubset(allowed)


@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_bollinger_mean_reversion_smoke(mode):
    df = _synthetic_ohlcv()
    result = bollinger_mean_reversion_signal(df, mode=mode)

    assert result["signal"].notna().all()
    allowed = {0.0, 1.0} if mode == "long_only" else {-1.0, 0.0, 1.0}
    assert set(result["signal"].unique()).issubset(allowed)


def test_bollinger_invalid_thresholds_raise():
    df = _synthetic_ohlcv()
    with pytest.raises(ValueError):
        bollinger_mean_reversion_signal(df, entry=1.0, exit=0.0, short_entry=2.0)


def test_bb_state_machine_enters_and_exits_long_only():
    # oversold -> stays long until reversion
    zscore = pd.Series([0.0, -2.5, -2.5, -1.0, 0.5, 0.5])
    signal = _bb_target_state_long_only(zscore, entry=-2.0, exit=0.0)
    assert list(signal) == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_bb_state_machine_long_short_mirrors_around_zero():
    zscore = pd.Series([0.0, 2.5, 2.5, 1.0, -0.5, -0.5])
    signal = _bb_target_state_long_short(zscore, entry=-2.0, exit=0.0, short_entry=2.0)
    assert list(signal) == [0.0, -1.0, -1.0, -1.0, 0.0, 0.0]


@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_mfi_mean_reversion_smoke(mode):
    df = _synthetic_ohlcv()
    result = mfi_mean_reversion_signal(df, mode=mode)

    assert result["signal"].notna().all()
    allowed = {0.0, 1.0} if mode == "long_only" else {-1.0, 0.0, 1.0}
    assert set(result["signal"].unique()).issubset(allowed)


def test_mfi_requires_volume_column():
    df = _synthetic_ohlcv().drop(columns=["volume"])
    with pytest.raises(ValueError):
        mfi_mean_reversion_signal(df)


def test_adx_regime_gate_is_binary():
    df = _synthetic_ohlcv()
    result = adx_trend_regime_signal(df)
    assert set(result["signal"].unique()).issubset({0.0, 1.0})


def test_low_volatility_regime_gate_is_binary():
    df = _synthetic_ohlcv()
    result = low_volatility_regime_signal(df)
    assert set(result["signal"].unique()).issubset({0.0, 1.0})


@pytest.mark.parametrize(
    "signal_fn",
    [
        macd_crossover_signal,
        stochastic_crossover_signal,
        bollinger_mean_reversion_signal,
        mfi_mean_reversion_signal,
        adx_trend_regime_signal,
        low_volatility_regime_signal,
    ],
)
def test_no_lookahead_for_new_strategies(signal_fn):
    full = _synthetic_ohlcv(300)

    cutoff_idx = 250
    cutoff_date = full.loc[cutoff_idx, "date"]
    truncated = full.iloc[: cutoff_idx + 1].copy()

    full_signal = signal_fn(full)
    truncated_signal = signal_fn(truncated)

    full_value = full_signal.loc[full_signal["date"] == cutoff_date, "signal"].iloc[0]
    truncated_value = truncated_signal.loc[
        truncated_signal["date"] == cutoff_date, "signal"
    ].iloc[0]

    assert full_value == truncated_value
