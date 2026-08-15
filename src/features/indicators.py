from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# Utility functions
# =====================================================================

def _validate_input(df: pd.DataFrame) -> None:
    required = {
        "date",
        "source_ticker",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def _groupby_ticker(df: pd.DataFrame):
    return df.groupby(
        "source_ticker",
        sort=False,
        group_keys=False,
    )


# =====================================================================
# Trend indicators
# =====================================================================

def sma_crossover(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
) -> pd.DataFrame:
    """
    SMA fast - SMA slow.

    Positive -> fast trend above slow trend.
    """

    result = df.copy()

    close = result["adj_close"]

    result[f"sma_{fast}"] = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.rolling(
                fast,
                min_periods=fast,
            ).mean()
        )
    )

    result[f"sma_{slow}"] = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.rolling(
                slow,
                min_periods=slow,
            ).mean()
        )
    )

    result[f"sma_cross_{fast}_{slow}"] = (
        result[f"sma_{fast}"]
        / result[f"sma_{slow}"]
        - 1.0
    )

    return result


def ema_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
) -> pd.DataFrame:

    result = df.copy()

    result[f"ema_{fast}"] = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.ewm(
                span=fast,
                adjust=False,
                min_periods=fast,
            ).mean()
        )
    )

    result[f"ema_{slow}"] = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.ewm(
                span=slow,
                adjust=False,
                min_periods=slow,
            ).mean()
        )
    )

    result[f"ema_cross_{fast}_{slow}"] = (
        result[f"ema_{fast}"]
        / result[f"ema_{slow}"]
        - 1.0
    )

    return result


def price_sma_distance(
    df: pd.DataFrame,
    window: int = 50,
) -> pd.DataFrame:

    result = df.copy()

    sma = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
    )

    result[f"price_sma_dist_{window}"] = (
        result["adj_close"] / sma - 1.0
    )

    return result


def ema_slope(
    df: pd.DataFrame,
    window: int = 20,
    slope_periods: int = 5,
) -> pd.DataFrame:

    result = df.copy()

    ema = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.ewm(
                span=window,
                adjust=False,
                min_periods=window,
            ).mean()
        )
    )

    result[f"ema_slope_{window}_{slope_periods}"] = (
        ema / ema.groupby(result["source_ticker"]).shift(slope_periods)
        - 1.0
    )

    return result


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:

    result = df.copy()

    grouped = _groupby_ticker(result)["adj_close"]

    ema_fast = grouped.transform(
        lambda x: x.ewm(
            span=fast,
            adjust=False,
            min_periods=fast,
        ).mean()
    )

    ema_slow = grouped.transform(
        lambda x: x.ewm(
            span=slow,
            adjust=False,
            min_periods=slow,
        ).mean()
    )

    macd_line = ema_fast - ema_slow

    macd_signal = (
        macd_line
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.ewm(
                span=signal,
                adjust=False,
                min_periods=signal,
            ).mean()
        )
    )

    result["macd"] = macd_line
    result["macd_signal"] = macd_signal
    result["macd_hist"] = (
        macd_line - macd_signal
    )

    return result


def adx(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Wilder-style ADX approximation using rolling means.
    """

    result = df.copy()

    groups = _groupby_ticker(result)

    prev_close = groups["adj_close"].shift(1)
    prev_high = groups["adj_high"].shift(1)
    prev_low = groups["adj_low"].shift(1)

    up_move = (
        result["adj_high"] - prev_high
    )

    down_move = (
        prev_low - result["adj_low"]
    )

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=result.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=result.index,
    )

    tr = pd.concat(
        [
            result["adj_high"]
            - result["adj_low"],

            (result["adj_high"] - prev_close).abs(),

            (result["adj_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        tr.groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
    )

    plus_di = (
        100
        * plus_dm.groupby(
            result["source_ticker"]
        ).transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
        / atr
    )

    minus_di = (
        100
        * minus_dm.groupby(
            result["source_ticker"]
        ).transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
        / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di)
    )

    result[f"adx_{window}"] = (
        dx.groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
    )

    result[f"plus_di_{window}"] = plus_di
    result[f"minus_di_{window}"] = minus_di

    return result


def donchian_breakout(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:

    result = df.copy()

    groups = _groupby_ticker(result)

    upper = (
        groups["adj_high"]
        .transform(
            lambda x: x.shift(1).rolling(
                window,
                min_periods=window,
            ).max()
        )
    )

    lower = (
        groups["adj_low"]
        .transform(
            lambda x: x.shift(1).rolling(
                window,
                min_periods=window,
            ).min()
        )
    )

    result[f"donchian_upper_{window}"] = upper
    result[f"donchian_lower_{window}"] = lower

    channel_width = (
            upper - lower
    ).replace(
        0,
        np.nan,
    )

    result[
        f"donchian_position_{window}"
    ] = (
                result["adj_close"] - lower
        ) / channel_width

    return result


# =====================================================================
# Momentum indicators
# =====================================================================

def rsi(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:

    result = df.copy()

    def calc_rsi(x: pd.Series) -> pd.Series:

        delta = x.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = gain.ewm(
            alpha=1 / window,
            adjust=False,
            min_periods=window,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / window,
            adjust=False,
            min_periods=window,
        ).mean()

        rs = avg_gain / avg_loss

        return 100 - (
            100 / (1 + rs)
        )

    result[f"rsi_{window}"] = (
        _groupby_ticker(result)["adj_close"]
        .transform(calc_rsi)
    )

    return result


def stochastic(
    df: pd.DataFrame,
    window: int = 14,
    smooth: int = 3,
) -> pd.DataFrame:

    result = df.copy()

    groups = _groupby_ticker(result)

    lowest = (
        groups["adj_low"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).min()
        )
    )

    highest = (
        groups["adj_high"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).max()
        )
    )

    denominator = (
        highest - lowest
    )

    percent_k = (
        100
        * (result["adj_close"] - lowest)
        / denominator
    )

    percent_d = (
        percent_k
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                smooth,
                min_periods=smooth,
            ).mean()
        )
    )

    result[
        f"stoch_k_{window}"
    ] = percent_k

    result[
        f"stoch_d_{window}_{smooth}"
    ] = percent_d

    return result


def roc(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:

    result = df.copy()

    result[f"roc_{window}"] = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.pct_change(
                window
            )
        )
    )

    return result


def williams_r(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:

    result = df.copy()

    groups = _groupby_ticker(result)

    highest = (
        groups["adj_high"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).max()
        )
    )

    lowest = (
        groups["adj_low"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).min()
        )
    )

    result[f"williams_r_{window}"] = (
        -100
        * (
            highest
            - result["adj_close"]
        )
        / (highest - lowest)
    )

    return result


def cci(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:

    result = df.copy()

    typical_price = (
        result["adj_high"]
        + result["adj_low"]
        + result["adj_close"]
    ) / 3.0

    groups = result.groupby(
        "source_ticker",
        sort=False,
    )

    rolling_mean = (
        typical_price
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
    )

    mean_deviation = (
        typical_price
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).apply(
                lambda y: np.mean(
                    np.abs(
                        y - np.mean(y)
                    )
                ),
                raw=True,
            )
        )
    )

    result[f"cci_{window}"] = (
        (
            typical_price
            - rolling_mean
        )
        / (
            0.015
            * mean_deviation
        )
    )

    return result


# =====================================================================
# Volatility indicators
# =====================================================================

def atr(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:

    result = df.copy()

    groups = _groupby_ticker(result)

    prev_close = (
        groups["adj_close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            result["adj_high"]
            - result["adj_low"],

            (
                result["adj_high"]
                - prev_close
            ).abs(),

            (
                result["adj_low"]
                - prev_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result[f"atr_{window}"] = (
        tr
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
    )

    result[f"atr_pct_{window}"] = (
        result[f"atr_{window}"]
        / result["adj_close"]
    )

    return result


def bollinger(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:

    result = df.copy()

    groups = _groupby_ticker(result)

    middle = (
        groups["adj_close"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).mean()
        )
    )

    std = (
        groups["adj_close"]
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).std()
        )
    )

    upper = middle + num_std * std
    lower = middle - num_std * std

    result[
        f"bb_upper_{window}_{num_std}"
    ] = upper

    result[
        f"bb_lower_{window}_{num_std}"
    ] = lower

    result[
        f"bb_zscore_{window}_{num_std}"
    ] = (
        result["adj_close"] - middle
    ) / std

    result[
        f"bb_width_{window}_{num_std}"
    ] = (
        (upper - lower)
        / middle
    )

    return result


def rolling_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:

    result = df.copy()

    log_return = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: np.log(
                x / x.shift(1)
            )
        )
    )

    volatility = (
        log_return
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).std()
        )
    )

    if annualize:
        volatility = (
            volatility
            * np.sqrt(252)
        )

    result[
        f"volatility_{window}"
    ] = volatility

    return result


def keltner_position(
    df: pd.DataFrame,
    ema_window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:

    result = df.copy()

    ema = (
        _groupby_ticker(result)["adj_close"]
        .transform(
            lambda x: x.ewm(
                span=ema_window,
                adjust=False,
                min_periods=ema_window,
            ).mean()
        )
    )

    prev_close = (
        _groupby_ticker(result)["adj_close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            result["adj_high"]
            - result["adj_low"],

            (
                result["adj_high"]
                - prev_close
            ).abs(),

            (
                result["adj_low"]
                - prev_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_value = (
        tr
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                atr_window,
                min_periods=atr_window,
            ).mean()
        )
    )

    upper = ema + multiplier * atr_value
    lower = ema - multiplier * atr_value

    channel_width = (
            upper - lower
    ).replace(
        0,
        np.nan,
    )

    result[
        f"keltner_position_{ema_window}_{atr_window}"
    ] = (
            (
                    result["adj_close"]
                    - lower
            )
            / channel_width
    )

    return result


# =====================================================================
# Volume indicators
# =====================================================================

def obv(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    def calc_obv(group: pd.DataFrame) -> pd.Series:

        close = group["adj_close"]
        volume = group["volume"]

        direction = np.sign(
            close.diff()
        )

        return (
            direction
            .fillna(0)
            * volume
        ).cumsum()

    result["obv"] = (
        result
        .groupby(
            "source_ticker",
            sort=False,
            group_keys=False,
        )
        .apply(
            calc_obv,
            include_groups=False,
        )
        .reset_index(
            level=0,
            drop=True,
        )
    )

    return result


def mfi(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:

    result = df.copy()

    typical_price = (
        result["adj_high"]
        + result["adj_low"]
        + result["adj_close"]
    ) / 3.0

    raw_money_flow = (
        typical_price
        * result["volume"]
    )

    previous_tp = (
        typical_price
        .groupby(result["source_ticker"])
        .shift(1)
    )

    positive_flow = pd.Series(
        np.where(
            typical_price > previous_tp,
            raw_money_flow,
            0.0,
        ),
        index=result.index,
    )

    negative_flow = pd.Series(
        np.where(
            typical_price < previous_tp,
            raw_money_flow,
            0.0,
        ),
        index=result.index,
    )

    positive_sum = (
        positive_flow
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).sum()
        )
    )

    negative_sum = (
        negative_flow
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).sum()
        )
    )

    money_ratio = (
        positive_sum
        / negative_sum.replace(
            0,
            np.nan,
        )
    )

    result[f"mfi_{window}"] = (
        100
        - (
            100
            / (1 + money_ratio)
        )
    )

    return result


def cmf(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:

    result = df.copy()

    denominator = (
        result["adj_high"]
        - result["adj_low"]
    )

    money_flow_multiplier = (
        (
            (
                result["adj_close"]
                - result["adj_low"]
            )
            - (
                result["adj_high"]
                - result["adj_close"]
            )
        )
        / denominator.replace(
            0,
            np.nan,
        )
    )

    money_flow_volume = (
        money_flow_multiplier
        * result["volume"]
    )

    numerator = (
        money_flow_volume
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).sum()
        )
    )

    denominator_volume = (
        result["volume"]
        .groupby(result["source_ticker"])
        .transform(
            lambda x: x.rolling(
                window,
                min_periods=window,
            ).sum()
        )
    )

    result[f"cmf_{window}"] = (
        numerator
        / denominator_volume.replace(
            0,
            np.nan,
        )
    )

    return result


def volume_roc(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:

    result = df.copy()

    result[
        f"volume_roc_{window}"
    ] = (
        _groupby_ticker(result)["volume"]
        .transform(
            lambda x: (
                x
                / x.shift(window).replace(
                    0,
                    np.nan,
                )
                - 1.0
            )
        )
    )

    return result


# =====================================================================
# Indicator registry
# =====================================================================

DEFAULT_INDICATORS = {
    "sma_crossover": sma_crossover,
    "ema_crossover": ema_crossover,
    "price_sma_distance": price_sma_distance,
    "ema_slope": ema_slope,
    "macd": macd,
    "adx": adx,
    "donchian_breakout": donchian_breakout,

    "rsi": rsi,
    "stochastic": stochastic,
    "roc": roc,
    "williams_r": williams_r,
    "cci": cci,

    "atr": atr,
    "bollinger": bollinger,
    "rolling_volatility": rolling_volatility,
    "keltner_position": keltner_position,

    "obv": obv,
    "mfi": mfi,
    "cmf": cmf,
    "volume_roc": volume_roc,
}


# =====================================================================
# Apply all default indicators
# =====================================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the complete default indicator library.

    Input must be the PIT dataset.

    All calculations are performed independently within each
    source_ticker, preventing cross-security contamination.
    """

    _validate_input(df)

    result = (
        df
        .sort_values(
            [
                "source_ticker",
                "date",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    # Trend
    result = sma_crossover(
        result,
        fast=20,
        slow=50,
    )

    result = ema_crossover(
        result,
        fast=12,
        slow=26,
    )

    result = price_sma_distance(
        result,
        window=50,
    )

    result = ema_slope(
        result,
        window=20,
        slope_periods=5,
    )

    result = macd(
        result,
        fast=12,
        slow=26,
        signal=9,
    )

    result = adx(
        result,
        window=14,
    )

    result = donchian_breakout(
        result,
        window=20,
    )

    # Momentum
    result = rsi(
        result,
        window=14,
    )

    result = stochastic(
        result,
        window=14,
        smooth=3,
    )

    result = roc(
        result,
        window=20,
    )

    result = williams_r(
        result,
        window=14,
    )

    result = cci(
        result,
        window=20,
    )

    # Volatility
    result = atr(
        result,
        window=14,
    )

    result = bollinger(
        result,
        window=20,
        num_std=2.0,
    )

    result = rolling_volatility(
        result,
        window=20,
    )

    result = keltner_position(
        result,
        ema_window=20,
        atr_window=10,
        multiplier=2.0,
    )

    # Volume
    result = obv(result)

    result = mfi(
        result,
        window=14,
    )

    result = cmf(
        result,
        window=20,
    )

    result = volume_roc(
        result,
        window=20,
    )
    # Replace any accidental non-finite values with NaN.
    # NaN is acceptable during indicator warm-up or undefined
    # denominator conditions; +/-inf is not.
    indicator_columns = [
        column
        for column in result.columns
        if column not in df.columns
    ]

    result[indicator_columns] = (
        result[indicator_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )
    return result