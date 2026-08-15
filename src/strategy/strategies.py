from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "processed"
    / "pit_features.parquet"
)


# =====================================================================
# Strategy metadata
# =====================================================================

@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    description: str
    signal_function: Callable
    parameter_grid: list[dict]


# =====================================================================
# Helpers
# =====================================================================

def _validate_input(df: pd.DataFrame) -> None:

    required = {
        "date",
        "source_ticker",
        "adj_high",
        "adj_low",
        "adj_close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Strategy input missing columns: {sorted(missing)}"
        )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:

    _validate_input(df)

    result = (
        df.copy()
        .sort_values(
            ["source_ticker", "date"]
        )
        .reset_index(drop=True)
    )

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result["source_ticker"] = (
        result["source_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return result


def _group(df: pd.DataFrame):

    return df.groupby(
        "source_ticker",
        sort=False,
        group_keys=False,
    )


# =====================================================================
# SMA crossover
# =====================================================================

def sma_crossover_signal(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    mode: str = "long_only",
) -> pd.DataFrame:

    if fast <= 0 or slow <= 0:
        raise ValueError(
            "SMA windows must be positive."
        )

    if fast >= slow:
        raise ValueError(
            "SMA fast window must be smaller than slow."
        )

    if mode not in {
        "long_only",
        "long_short",
    }:
        raise ValueError(
            "mode must be 'long_only' or 'long_short'."
        )

    result = _prepare(df)

    close = _group(result)["adj_close"]

    fast_sma = close.transform(
        lambda x: x.rolling(
            fast,
            min_periods=fast,
        ).mean()
    )

    slow_sma = close.transform(
        lambda x: x.rolling(
            slow,
            min_periods=slow,
        ).mean()
    )

    strength = (
        fast_sma / slow_sma - 1.0
    )

    if mode == "long_only":

        signal = np.where(
            strength > 0.0,
            1.0,
            0.0,
        )

    else:

        signal = np.select(
            [
                strength > 0.0,
                strength < 0.0,
            ],
            [
                1.0,
                -1.0,
            ],
            default=0.0,
        )

    signal = pd.Series(
        signal,
        index=result.index,
        dtype=float,
    )

    signal[
        fast_sma.isna()
        | slow_sma.isna()
    ] = 0.0

    result["signal"] = signal
    result["signal_strength"] = strength
    result["strategy"] = "sma_crossover"
    result["strategy_family"] = "trend"

    return result


# =====================================================================
# RSI
# =====================================================================

def _rsi_series(
    close: pd.Series,
    window: int,
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1.0 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    # Handle zero-loss periods explicitly.
    # If average loss = 0 and gain > 0, RSI is 100.
    # If both are zero, RSI is undefined.
    rsi = pd.Series(
        np.nan,
        index=close.index,
        dtype=float,
    )

    positive_loss = avg_loss > 0

    rsi.loc[positive_loss] = (
        100.0
        - (
            100.0
            / (
                1.0
                + avg_gain.loc[positive_loss]
                / avg_loss.loc[positive_loss]
            )
        )
    )

    zero_loss_positive_gain = (
        avg_loss.eq(0)
        & avg_gain.gt(0)
    )

    rsi.loc[
        zero_loss_positive_gain
    ] = 100.0

    return rsi


# =====================================================================
# RSI state machine
# =====================================================================

def _rsi_target_state_long_only(
    rsi: pd.Series,
    entry: float,
    exit: float,
) -> pd.Series:
    """
    Stateful long-only mean reversion.

    Enter when RSI <= entry.
    Once long, remain long until RSI >= exit.
    """

    signal = pd.Series(
        0.0,
        index=rsi.index,
    )

    state = 0.0

    for i in range(len(rsi)):

        value = rsi.iloc[i]

        if pd.isna(value):
            signal.iloc[i] = 0.0
            continue

        if state == 0.0:

            if value <= entry:
                state = 1.0

        else:

            if value >= exit:
                state = 0.0

        signal.iloc[i] = state

    return signal


def _rsi_target_state_long_short(
    rsi: pd.Series,
    entry: float,
    exit: float,
    short_entry: float,
) -> pd.Series:
    """
    Stateful long/short RSI mean reversion.

    Flat:
        RSI <= entry     -> long
        RSI >= short_entry -> short

    Long:
        RSI >= exit      -> flat

    Short:
        RSI <= (100-exit) -> flat
    """

    signal = pd.Series(
        0.0,
        index=rsi.index,
    )

    state = 0.0

    short_exit = 100.0 - exit

    for i in range(len(rsi)):

        value = rsi.iloc[i]

        if pd.isna(value):
            signal.iloc[i] = 0.0
            continue

        if state == 0.0:

            if value <= entry:

                state = 1.0

            elif value >= short_entry:

                state = -1.0

        elif state == 1.0:

            if value >= exit:
                state = 0.0

        elif state == -1.0:

            if value <= short_exit:
                state = 0.0

        signal.iloc[i] = state

    return signal


def rsi_mean_reversion_signal(
    df: pd.DataFrame,
    window: int = 14,
    entry: float = 30.0,
    exit: float = 50.0,
    short_entry: float = 70.0,
    mode: str = "long_only",
) -> pd.DataFrame:

    if window <= 0:
        raise ValueError(
            "RSI window must be positive."
        )

    if not (
        0.0
        < entry
        < exit
        < short_entry
        < 100.0
    ):
        raise ValueError(
            "Require 0 < entry < exit < short_entry < 100."
        )

    if mode not in {
        "long_only",
        "long_short",
    }:
        raise ValueError(
            "mode must be 'long_only' or 'long_short'."
        )

    result = _prepare(df)

    rsi = (
        _group(result)["adj_close"]
        .transform(
            lambda x: _rsi_series(
                x,
                window,
            )
        )
    )

    if mode == "long_only":

        signal = (
            result
            .groupby(
                "source_ticker",
                sort=False,
                group_keys=False,
            )
            .apply(
                lambda g: _rsi_target_state_long_only(
                    g[f"rsi_{window}"]
                    if f"rsi_{window}" in g.columns
                    else pd.Series(
                        rsi.loc[g.index].values,
                        index=g.index,
                    ),
                    entry,
                    exit,
                )
            )
        )

        # Align back to original index.
        signal = signal.reset_index(
            level=0,
            drop=True,
        )
        signal = signal.reindex(
            result.index
        )

    else:

        pieces = []

        for _, group in result.groupby(
            "source_ticker",
            sort=False,
        ):

            group_rsi = rsi.loc[
                group.index
            ]

            group_signal = (
                _rsi_target_state_long_short(
                    group_rsi,
                    entry,
                    exit,
                    short_entry,
                )
            )

            pieces.append(
                group_signal
            )

        signal = pd.concat(
            pieces
        ).reindex(
            result.index
        )

    signal = signal.astype(float)

    result[f"rsi_{window}"] = rsi
    result["signal"] = signal

    result["signal_strength"] = (
        50.0 - rsi
    )

    result["strategy"] = (
        "rsi_mean_reversion"
    )

    result["strategy_family"] = (
        "momentum"
    )

    return result


# =====================================================================
# Donchian breakout
# =====================================================================

def donchian_breakout_signal(
    df: pd.DataFrame,
    window: int = 20,
    mode: str = "long_only",
) -> pd.DataFrame:

    if window <= 1:
        raise ValueError(
            "Donchian window must be > 1."
        )

    if mode not in {
        "long_only",
        "long_short",
    }:
        raise ValueError(
            "mode must be 'long_only' or 'long_short'."
        )

    result = _prepare(df)

    high = _group(result)["adj_high"]
    low = _group(result)["adj_low"]

    upper = high.transform(
        lambda x: x.shift(1).rolling(
            window,
            min_periods=window,
        ).max()
    )

    lower = low.transform(
        lambda x: x.shift(1).rolling(
            window,
            min_periods=window,
        ).min()
    )

    long_condition = (
        result["adj_close"] > upper
    )

    short_condition = (
        result["adj_close"] < lower
    )

    if mode == "long_only":

        signal = np.where(
            long_condition,
            1.0,
            0.0,
        )

    else:

        signal = np.select(
            [
                long_condition,
                short_condition,
            ],
            [
                1.0,
                -1.0,
            ],
            default=0.0,
        )

    signal = pd.Series(
        signal,
        index=result.index,
        dtype=float,
    )

    signal[
        upper.isna()
        | lower.isna()
    ] = 0.0

    width = (
        upper - lower
    ).replace(
        0,
        np.nan,
    )

    result[
        f"donchian_upper_{window}"
    ] = upper

    result[
        f"donchian_lower_{window}"
    ] = lower

    result[
        f"donchian_position_{window}"
    ] = (
        result["adj_close"] - lower
    ) / width

    result["signal"] = signal

    result["signal_strength"] = (
        result["adj_close"]
        - (upper + lower) / 2.0
    )

    result["strategy"] = (
        "donchian_breakout"
    )

    result["strategy_family"] = (
        "trend"
    )

    return result


# =====================================================================
# Parameter grids
# =====================================================================

SMA_PARAMETER_GRID = [
    {"fast": 10, "slow": 30},
    {"fast": 10, "slow": 50},
    {"fast": 20, "slow": 50},
    {"fast": 20, "slow": 100},
    {"fast": 30, "slow": 100},
    {"fast": 50, "slow": 200},
]

RSI_PARAMETER_GRID = [
    {
        "window": 7,
        "entry": 20,
        "exit": 50,
        "short_entry": 80,
    },
    {
        "window": 14,
        "entry": 20,
        "exit": 50,
        "short_entry": 80,
    },
    {
        "window": 14,
        "entry": 30,
        "exit": 50,
        "short_entry": 70,
    },
    {
        "window": 14,
        "entry": 30,
        "exit": 60,
        "short_entry": 70,
    },
    {
        "window": 21,
        "entry": 30,
        "exit": 50,
        "short_entry": 70,
    },
    {
        "window": 21,
        "entry": 40,
        "exit": 50,
        "short_entry": 60,
    },
]

DONCHIAN_PARAMETER_GRID = [
    {"window": 10},
    {"window": 20},
    {"window": 40},
    {"window": 60},
    {"window": 100},
]


# =====================================================================
# Registry
# =====================================================================

STRATEGIES = {

    "sma_crossover": StrategySpec(
        name="sma_crossover",
        family="trend",
        description=(
            "Fast/slow simple moving-average "
            "trend-following crossover."
        ),
        signal_function=sma_crossover_signal,
        parameter_grid=SMA_PARAMETER_GRID,
    ),

    "rsi_mean_reversion": StrategySpec(
        name="rsi_mean_reversion",
        family="momentum",
        description=(
            "Stateful RSI mean-reversion strategy."
        ),
        signal_function=rsi_mean_reversion_signal,
        parameter_grid=RSI_PARAMETER_GRID,
    ),

    "donchian_breakout": StrategySpec(
        name="donchian_breakout",
        family="trend",
        description=(
            "Prior-window Donchian breakout."
        ),
        signal_function=donchian_breakout_signal,
        parameter_grid=DONCHIAN_PARAMETER_GRID,
    ),
}


# =====================================================================
# Generate strategy signal
# =====================================================================

def generate_strategy_signal(
    df: pd.DataFrame,
    strategy_name: str,
    parameters: dict,
    mode: str = "long_only",
) -> pd.DataFrame:

    if strategy_name not in STRATEGIES:
        raise KeyError(
            f"Unknown strategy: {strategy_name}"
        )

    spec = STRATEGIES[
        strategy_name
    ]

    params = dict(
        parameters
    )

    params["mode"] = mode

    result = spec.signal_function(
        df,
        **params,
    )

    parameter_string = "_".join(
        f"{key}={value}"
        for key, value in sorted(
            parameters.items()
        )
    )

    result["strategy_id"] = (
        f"{strategy_name}"
        f"__{parameter_string}"
        f"__{mode}"
    )

    result["mode"] = mode

    return result


# =====================================================================
# Enumerate configurations
# =====================================================================

def enumerate_strategy_configs(
    modes: tuple[str, ...] = (
        "long_only",
        "long_short",
    ),
) -> list[dict]:

    configs = []

    for strategy_name, spec in STRATEGIES.items():

        for parameters in spec.parameter_grid:

            for mode in modes:

                configs.append(
                    {
                        "strategy_name": strategy_name,
                        "family": spec.family,
                        "parameters": dict(
                            parameters
                        ),
                        "mode": mode,
                    }
                )

    return configs


# =====================================================================
# Smoke test
# =====================================================================

def main() -> None:

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Feature file:",
        FEATURE_FILE,
    )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    print(
        "\nLoading feature dataset..."
    )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Columns:",
        len(df.columns),
    )

    print(
        "Date range:",
        df["date"].min(),
        "→",
        df["date"].max(),
    )

    print(
        "Source securities:",
        df["source_ticker"].nunique(),
    )

    # ---------------------------------------------------------------
    # Small sample
    # ---------------------------------------------------------------

    test_tickers = (
        df["source_ticker"]
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    sample = df[
        df["source_ticker"].isin(
            test_tickers
        )
    ].copy()

    print(
        "\nSample securities:",
        test_tickers,
    )

    print(
        "Sample rows:",
        len(sample),
    )

    # ---------------------------------------------------------------
    # Config count
    # ---------------------------------------------------------------

    configs = enumerate_strategy_configs()

    print(
        "\nTotal strategy configurations:",
        len(configs),
    )

    # ---------------------------------------------------------------
    # Smoke tests
    # ---------------------------------------------------------------

    tests = [
        {
            "strategy_name": "sma_crossover",
            "parameters": {
                "fast": 20,
                "slow": 50,
            },
        },
        {
            "strategy_name": "rsi_mean_reversion",
            "parameters": {
                "window": 14,
                "entry": 30,
                "exit": 50,
                "short_entry": 70,
            },
        },
        {
            "strategy_name": "donchian_breakout",
            "parameters": {
                "window": 20,
            },
        },
    ]

    for mode in [
        "long_only",
        "long_short",
    ]:

        for test in tests:

            result = generate_strategy_signal(
                sample,
                strategy_name=test[
                    "strategy_name"
                ],
                parameters=test[
                    "parameters"
                ],
                mode=mode,
            )

            strategy_id = (
                result["strategy_id"].iloc[0]
            )

            print()
            print("=" * 70)

            print(
                "Strategy:",
                strategy_id,
            )

            print(
                "\nSignal counts:"
            )

            print(
                result["signal"]
                .value_counts()
                .sort_index()
                .to_string()
            )

            null_signals = int(
                result["signal"]
                .isna()
                .sum()
            )

            print(
                "\nNull signals:",
                null_signals,
            )

            if null_signals:
                raise ValueError(
                    f"{strategy_id} generated "
                    "null signals."
                )

    print()
    print("=" * 70)
    print(
        "STRATEGY SMOKE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()