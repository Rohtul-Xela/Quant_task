from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Configuration
# =====================================================================

DEFAULT_COST_BPS = 5.0

# Fixed research cutoff for reproducibility.
# Raw Yahoo files may contain data beyond this date, but the research
# dataset must not use it.
RESEARCH_END_DATE = pd.Timestamp("2026-08-11")

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "processed"
    / "pit_features.parquet"
)

YAHOO_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "raw"
    / "yahoo"
)

# These histories were classified as unusable for research:
# repeated/corrupt price regimes or insufficient usable history.
EXCLUDED_PRICE_TICKERS = {
    "CBE",
    "CFC",
    "HET",
    "HPC",
    "MEE",
    "MER",
    "NCC",
    "PBG",
    "SATS",
    "TIE",
}


# =====================================================================
# Result container
# =====================================================================

@dataclass
class BacktestResult:
    strategy_id: str
    daily: pd.DataFrame
    metrics: dict


# =====================================================================
# Strategy input validation
# =====================================================================

def _validate_strategy_input(
    df: pd.DataFrame,
) -> None:

    required = {
        "date",
        "source_ticker",
        "yahoo_ticker",
        "adj_close",
        "signal",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            "Strategy input missing columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Strategy input is empty."
        )


# =====================================================================
# Load complete Yahoo price history
# =====================================================================

def load_yahoo_prices(
    yahoo_dir: Path = YAHOO_DIR,
    as_of_date: str | pd.Timestamp = RESEARCH_END_DATE,
) -> pd.DataFrame:
    """
    Load all usable Yahoo price histories.

    The PIT dataset determines:
        - eligible securities
        - signal(t)

    The Yahoo history determines:
        - actual price(t+1)
        - realized return(t -> t+1)

    Research data is explicitly capped at as_of_date so that the
    backtest is reproducible and does not depend on later Yahoo updates.
    """

    as_of_date_ts = pd.Timestamp(
        as_of_date
    ).normalize()

    if not yahoo_dir.exists():
        raise FileNotFoundError(
            f"Yahoo directory not found:\n{yahoo_dir}"
        )

    paths = sorted(
        yahoo_dir.glob("*.parquet")
    )

    if not paths:
        raise FileNotFoundError(
            f"No Yahoo parquet files found in:\n{yahoo_dir}"
        )

    print(
        f"Loading {len(paths)} Yahoo price files..."
    )

    frames = []

    for path in paths:

        ticker = path.stem.upper()

        if ticker in EXCLUDED_PRICE_TICKERS:
            continue

        try:

            data = pd.read_parquet(
                path
            )

            if data.empty:
                continue

            required = {
                "date",
                "ticker",
                "adj_close",
            }

            missing = (
                required
                - set(data.columns)
            )

            if missing:
                print(
                    f"[SKIP] {path.name}: "
                    f"missing {sorted(missing)}"
                )
                continue

            data = data[
                [
                    "date",
                    "ticker",
                    "adj_close",
                ]
            ].copy()

            data["date"] = pd.to_datetime(
                data["date"],
                errors="coerce",
            )

            data["ticker"] = (
                data["ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            # Use filename as the authoritative ticker for the cache.
            data["ticker"] = ticker

            data["adj_close"] = pd.to_numeric(
                data["adj_close"],
                errors="coerce",
            )

            data = data.dropna(
                subset=[
                    "date",
                    "adj_close",
                ]
            )

            # -------------------------------------------------------
            # Research as-of cutoff
            # -------------------------------------------------------

            data = data[
                data["date"] <= as_of_date_ts
            ].copy()

            if data.empty:
                continue

            data = (
                data
                .sort_values("date")
                .drop_duplicates(
                    subset=[
                        "date",
                        "ticker",
                    ],
                    keep="last",
                )
            )

            frames.append(
                data
            )

        except Exception as exc:

            print(
                f"[SKIP] {path.name}: {exc}"
            )

    if not frames:
        raise RuntimeError(
            "No usable Yahoo price data found "
            f"through {as_of_date_ts.date()}."
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices = (
        prices
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Final cutoff safety check
    # ---------------------------------------------------------------

    if prices["date"].max() > as_of_date_ts:
        raise AssertionError(
            "Yahoo prices exceed research as-of date: "
            f"{prices['date'].max()} > {as_of_date_ts}"
        )

    print(
        "Yahoo price rows:",
        len(prices),
    )

    print(
        "Yahoo tickers:",
        prices["ticker"].nunique(),
    )

    print(
        "Excluded tickers:",
        sorted(
            EXCLUDED_PRICE_TICKERS
        ),
    )

    print(
        "Yahoo research date range:",
        prices["date"].min(),
        "→",
        prices["date"].max(),
    )

    print(
        "Research as-of date:",
        as_of_date_ts,
    )

    return prices


# =====================================================================
# Build valid next-session returns
# =====================================================================

def _build_nyse_sessions(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """
    Return the official NYSE trading sessions between two dates.
    """

    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    return schedule.index.tz_localize(None)


def build_next_day_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate t -> next valid NYSE trading-session returns.

    The next observation is considered valid only when it is the
    immediately following NYSE session for that ticker.

    This avoids using an arbitrary calendar-day threshold and correctly
    handles extended exchange closures such as Hurricane Sandy.

    Genuine Yahoo data gaps remain excluded.
    """

    result = (
        prices
        .copy()
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    result["date"] = pd.to_datetime(
        result["date"]
    )

    # ---------------------------------------------------------------
    # Build official NYSE trading-session calendar
    # ---------------------------------------------------------------

    calendar_start = result["date"].min()
    calendar_end = result["date"].max()

    sessions = _build_nyse_sessions(
        calendar_start,
        calendar_end,
    )

    session_position = pd.Series(
        np.arange(len(sessions)),
        index=sessions,
    )

    # ---------------------------------------------------------------
    # Next Yahoo observation
    # ---------------------------------------------------------------

    result["next_date"] = (
        result
        .groupby(
            "ticker",
            sort=False,
        )["date"]
        .shift(-1)
    )

    result["next_adj_close"] = (
        result
        .groupby(
            "ticker",
            sort=False,
        )["adj_close"]
        .shift(-1)
    )

    result["calendar_gap_days"] = (
        result["next_date"]
        - result["date"]
    ).dt.days

    # ---------------------------------------------------------------
    # Determine whether next_date is the immediately following
    # NYSE session.
    # ---------------------------------------------------------------

    current_session_position = (
        result["date"].map(
            session_position
        )
    )

    next_session_position = (
        result["next_date"].map(
            session_position
        )
    )

    valid_next_session = (
        current_session_position.notna()
        & next_session_position.notna()
        & (
            next_session_position
            == current_session_position + 1
        )
    )

    # ---------------------------------------------------------------
    # Calculate return only for genuine next-session observations
    # ---------------------------------------------------------------

    result["next_return"] = np.nan

    result.loc[
        valid_next_session,
        "next_return",
    ] = (
        result.loc[
            valid_next_session,
            "next_adj_close",
        ]
        / result.loc[
            valid_next_session,
            "adj_close",
        ]
        - 1.0
    )

    return result[
        [
            "date",
            "ticker",
            "next_date",
            "calendar_gap_days",
            "next_adj_close",
            "next_return",
        ]
    ]


# =====================================================================
# Return validation
# =====================================================================

def validate_returns(
    returns: pd.DataFrame,
) -> None:
    """
    Validate the price return universe.

    Large returns are reported, not silently clipped.
    """

    valid = returns[
        returns["next_return"].notna()
    ].copy()

    if valid.empty:
        raise ValueError(
            "No valid next-session returns."
        )

    finite = np.isfinite(
        valid["next_return"]
    )

    if not finite.all():

        count = int(
            (~finite).sum()
        )

        raise ValueError(
            f"{count} non-finite returns detected."
        )

    max_abs = (
        valid["next_return"]
        .abs()
        .max()
    )

    max_row = valid.loc[
        valid["next_return"]
        .abs()
        .idxmax()
    ]

    print()
    print(
        "=== RETURN SANITY CHECK ==="
    )

    print(
        "Valid return rows:",
        len(valid),
    )

    print(
        "Maximum absolute asset return:",
        max_abs,
    )

    print(
        "Ticker:",
        max_row["ticker"],
    )

    print(
        "Date:",
        max_row["date"],
    )

    print(
        "Next date:",
        max_row["next_date"],
    )

    print(
        "Return:",
        max_row["next_return"],
    )

    if max_abs > 2.0:

        print(
            "WARNING: a single-period return "
            "exceeds 200%. This is retained and "
            "requires review; it is not clipped."
        )

    # ---------------------------------------------------------------
    # Report remaining invalid next-session observations
    # ---------------------------------------------------------------

    invalid_next_session = returns[
        returns["next_return"].isna()
        & returns["next_date"].notna()
    ]

    print(
        "Yahoo observations with no valid "
        "next NYSE session:",
        len(invalid_next_session),
    )


# =====================================================================
# Prepare strategy data
# =====================================================================

def _prepare_strategy(
    df: pd.DataFrame,
) -> pd.DataFrame:

    _validate_strategy_input(
        df
    )

    result = (
        df
        .copy()
        .sort_values(
            [
                "date",
                "source_ticker",
            ]
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

    result["yahoo_ticker"] = (
        result["yahoo_ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result["signal"] = pd.to_numeric(
        result["signal"],
        errors="coerce",
    ).fillna(0.0)

    duplicate_count = int(
        result.duplicated(
            subset=[
                "date",
                "source_ticker",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            "Duplicate strategy observations: "
            f"{duplicate_count}"
        )

    return result


# =====================================================================
# Portfolio weights
# =====================================================================

def build_portfolio_weights(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert signals to equal-weight target portfolio weights.
    """

    required = {
        "date",
        "source_ticker",
        "signal",
        "portfolio_signal",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Portfolio weight input missing columns: "
            f"{sorted(missing)}"
        )

    result = df[
        [
            "date",
            "source_ticker",
            "signal",
            "portfolio_signal",
        ]
    ].copy()

    long_count = (
        result["portfolio_signal"]
        .gt(0)
        .groupby(result["date"])
        .transform("sum")
    )

    short_count = (
        result["portfolio_signal"]
        .lt(0)
        .groupby(result["date"])
        .transform("sum")
    )

    has_long = (
        long_count > 0
    )

    has_short = (
        short_count > 0
    )

    both_sides = (
        has_long & has_short
    )

    long_gross = np.where(
        both_sides,
        0.5,
        1.0,
    )

    short_gross = np.where(
        both_sides,
        0.5,
        1.0,
    )

    result["target_weight"] = 0.0

    long_mask = (
        result["portfolio_signal"] > 0
    )

    short_mask = (
        result["portfolio_signal"] < 0
    )

    result.loc[
        long_mask,
        "target_weight",
    ] = (
        long_gross[long_mask]
        / long_count[long_mask]
    )

    result.loc[
        short_mask,
        "target_weight",
    ] = (
        -short_gross[short_mask]
        / short_count[short_mask]
    )

    gross = (
        result
        .groupby("date")[
            "target_weight"
        ]
        .apply(
            lambda x: float(
                np.abs(x).sum()
            )
        )
    )

    if (
        gross > 1.0 + 1e-10
    ).any():

        raise ValueError(
            "Gross exposure exceeded 1.0."
        )

    return result


def align_target_weights(
    weights: pd.DataFrame,
) -> pd.DataFrame:

    return (
        weights
        .pivot(
            index="date",
            columns="source_ticker",
            values="target_weight",
        )
        .sort_index()
        .fillna(0.0)
    )


# =====================================================================
# Metrics
# =====================================================================

def calculate_metrics(
    daily: pd.DataFrame,
) -> dict:

    if daily.empty:

        return {
            "sharpe": np.nan,
            "sortino": np.nan,
            "net_return": np.nan,
            "max_drawdown": np.nan,
            "months_in_profit": 0,
            "months_count": 0,
            "months_in_profit_pct": np.nan,
            "avg_monthly_pnl": np.nan,
            "avg_daily_turnover": np.nan,
            "avg_gross_exposure": np.nan,
            "avg_net_exposure": np.nan,
        }

    returns = (
        daily["net_return"]
        .dropna()
    )

    equity = (
        1.0 + returns
    ).cumprod()

    net_return = (
        equity.iloc[-1]
        - 1.0
    )

    daily_std = (
        returns.std(
            ddof=1
        )
    )

    if (
        pd.notna(daily_std)
        and daily_std > 0
    ):

        sharpe = (
            returns.mean()
            / daily_std
            * np.sqrt(252.0)
        )

    else:

        sharpe = np.nan

    downside = returns[
        returns < 0
    ]

    if len(downside) > 1:

        downside_std = (
            downside.std(
                ddof=1
            )
        )

    else:

        downside_std = np.nan

    if (
        pd.notna(downside_std)
        and downside_std > 0
    ):

        sortino = (
            returns.mean()
            / downside_std
            * np.sqrt(252.0)
        )

    else:

        sortino = np.nan

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_drawdown = (
        drawdown.min()
    )

    monthly_returns = (
        daily
        .set_index("date")[
            "net_return"
        ]
        .resample("ME")
        .apply(
            lambda x: (
                1.0 + x
            ).prod() - 1.0
        )
        .dropna()
    )

    months_count = len(
        monthly_returns
    )

    months_in_profit = int(
        (
            monthly_returns > 0
        ).sum()
    )

    if months_count > 0:

        months_in_profit_pct = (
            months_in_profit
            / months_count
        )

        avg_monthly_pnl = (
            monthly_returns.mean()
        )

    else:

        months_in_profit_pct = np.nan
        avg_monthly_pnl = np.nan

    return {
        "sharpe": float(sharpe)
        if pd.notna(sharpe)
        else np.nan,

        "sortino": float(sortino)
        if pd.notna(sortino)
        else np.nan,

        "net_return": float(
            net_return
        ),

        "max_drawdown": float(
            max_drawdown
        ),

        "months_in_profit": (
            months_in_profit
        ),

        "months_count": (
            months_count
        ),

        "months_in_profit_pct": float(
            months_in_profit_pct
        )
        if pd.notna(
            months_in_profit_pct
        )
        else np.nan,

        "avg_monthly_pnl": float(
            avg_monthly_pnl
        )
        if pd.notna(
            avg_monthly_pnl
        )
        else np.nan,

        "avg_daily_turnover": float(
            daily["turnover"].mean()
        ),

        "avg_gross_exposure": float(
            daily["gross_exposure"].mean()
        ),

        "avg_net_exposure": float(
            daily["net_exposure"].mean()
        ),
    }


# =====================================================================
# Backtest
# =====================================================================

def run_backtest(
    df: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    strategy_id: str | None = None,
    yahoo_prices: pd.DataFrame | None = None,
    yahoo_returns: pd.DataFrame | None = None,
) -> BacktestResult:

    if cost_bps < 0:
        raise ValueError(
            "cost_bps must be >= 0."
        )

    strategy = _prepare_strategy(
        df
    )

    # ---------------------------------------------------------------
    # Use preloaded Yahoo data when supplied.
    # ---------------------------------------------------------------

    if yahoo_prices is None:

        yahoo_prices = (
            load_yahoo_prices()
        )

    else:

        yahoo_prices = (
            yahoo_prices.copy()
        )

    # ---------------------------------------------------------------
    # Apply research exclusions consistently.
    # ---------------------------------------------------------------

    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(
            EXCLUDED_PRICE_TICKERS
        )
    ].copy()

    # ---------------------------------------------------------------
    # Build next-session returns once if needed.
    # ---------------------------------------------------------------

    if yahoo_returns is None:

        yahoo_returns = (
            build_next_day_returns(
                yahoo_prices
            )
        )

    # ---------------------------------------------------------------
    # Join strategy observations to Yahoo returns.
    # ---------------------------------------------------------------

    strategy = strategy.merge(
        yahoo_returns[
            [
                "date",
                "ticker",
                "next_date",
                "calendar_gap_days",
                "next_return",
            ]
        ],
        left_on=[
            "date",
            "yahoo_ticker",
        ],
        right_on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    strategy = strategy.drop(
        columns=[
            "ticker",
        ]
    )

    # ---------------------------------------------------------------
    # Active positions without a valid return
    # ---------------------------------------------------------------

    missing_active = strategy[
        strategy["signal"].ne(0)
        & strategy["next_return"].isna()
    ]

    missing_active_count = len(
        missing_active
    )

    if not missing_active.empty:
        print(
            "\nACTIVE POSITIONS WITHOUT VALID NEXT RETURN:"
        )

        print(
            missing_active[
                [
                    "date",
                    "source_ticker",
                    "yahoo_ticker",
                    "signal",
                    "next_date",
                    "calendar_gap_days",
                    "next_return",
                ]
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Portfolio eligibility
    #
    # A signal may exist on a date, but it is only tradable if a valid
    # next-session return exists. Preserve the original signal for
    # diagnostics, but exclude non-tradable observations from portfolio
    # construction, exposure and turnover.
    # ---------------------------------------------------------------

    strategy["portfolio_signal"] = np.where(
        strategy["next_return"].notna(),
        strategy["signal"],
        0.0,
    )

    # ---------------------------------------------------------------
    # Target weights
    # ---------------------------------------------------------------

    weights = (
        build_portfolio_weights(
            strategy
        )
    )

    strategy = strategy.merge(
        weights,
        on=[
            "date",
            "source_ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    strategy["target_weight"] = (
        strategy["target_weight"]
        .fillna(0.0)
    )

    # ---------------------------------------------------------------
    # Cross-sectional portfolio weights
    # ---------------------------------------------------------------

    weight_matrix = (
        align_target_weights(
            weights
        )
    )

    previous_weights = (
        weight_matrix
        .shift(1)
        .fillna(0.0)
    )

    turnover_matrix = (
        weight_matrix
        - previous_weights
    ).abs()

    daily_turnover = (
        turnover_matrix
        .sum(axis=1)
    )

    # ---------------------------------------------------------------
    # Gross portfolio contribution
    # ---------------------------------------------------------------

    strategy["gross_contribution"] = (
        strategy["target_weight"]
        * strategy["next_return"]
    )

    valid_return = (
        strategy["next_return"]
        .notna()
    )

    gross_daily = (
        strategy.loc[
            valid_return
        ]
        .groupby("date")[
            "gross_contribution"
        ]
        .sum()
    )

    return_dates = (
        strategy.loc[
            valid_return,
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )

    if return_dates.empty:

        raise ValueError(
            "No valid next-session returns "
            "for this strategy."
        )

    # ---------------------------------------------------------------
    # Daily portfolio series
    # ---------------------------------------------------------------

    daily = pd.DataFrame(
        index=return_dates
    )

    daily.index.name = "date"

    daily["gross_return"] = (
        gross_daily
        .reindex(return_dates)
        .fillna(0.0)
    )

    daily["turnover"] = (
        daily_turnover
        .reindex(return_dates)
        .fillna(0.0)
    )

    cost_rate = (
        cost_bps / 10000.0
    )

    daily["transaction_cost"] = (
        daily["turnover"]
        * cost_rate
    )

    daily["net_return"] = (
        daily["gross_return"]
        - daily["transaction_cost"]
    )

    # ---------------------------------------------------------------
    # Exposure
    # ---------------------------------------------------------------

    gross_exposure = (
        weight_matrix
        .abs()
        .sum(axis=1)
    )

    net_exposure = (
        weight_matrix
        .sum(axis=1)
    )

    active_positions = (
        weight_matrix
        .ne(0)
        .sum(axis=1)
    )

    daily["active_positions"] = (
        active_positions
        .reindex(return_dates)
        .fillna(0)
        .astype(int)
    )

    daily["gross_exposure"] = (
        gross_exposure
        .reindex(return_dates)
        .fillna(0.0)
    )

    daily["net_exposure"] = (
        net_exposure
        .reindex(return_dates)
        .fillna(0.0)
    )

    daily = (
        daily
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Strategy metadata
    # ---------------------------------------------------------------

    if strategy_id is None:

        if "strategy_id" in df.columns:

            ids = (
                df["strategy_id"]
                .dropna()
                .unique()
            )

            if len(ids) == 1:

                strategy_id = str(
                    ids[0]
                )

            else:

                strategy_id = "unknown"

        else:

            strategy_id = "unknown"

    daily["strategy_id"] = (
        strategy_id
    )

    # ---------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------

    max_gross = (
        daily["gross_exposure"]
        .max()
    )

    if max_gross > 1.0 + 1e-10:

        raise ValueError(
            "Gross exposure exceeded 1.0: "
            f"{max_gross}"
        )

    max_turnover = (
        daily["turnover"]
        .max()
    )

    if max_turnover > 2.0 + 1e-10:

        raise ValueError(
            "Turnover exceeded theoretical maximum 2.0: "
            f"{max_turnover}"
        )

    if not np.isfinite(
        daily["net_return"]
    ).all():

        raise ValueError(
            "Non-finite portfolio return detected."
        )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    metrics = calculate_metrics(
        daily
    )

    metrics["strategy_id"] = (
        strategy_id
    )

    metrics["cost_bps"] = (
        cost_bps
    )

    metrics[
        "active_positions_missing_return"
    ] = missing_active_count

    return BacktestResult(
        strategy_id=strategy_id,
        daily=daily,
        metrics=metrics,
    )


# =====================================================================
# Smoke test
# =====================================================================

def main() -> None:

    from src.strategy.strategies import (
        generate_strategy_signal,
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Feature file:",
        FEATURE_FILE,
    )

    print(
        "Yahoo directory:",
        YAHOO_DIR,
    )

    print(
        "Research end date:",
        RESEARCH_END_DATE,
    )

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    print(
        "\nLoading feature dataset..."
    )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    print(
        "Full dataset rows:",
        len(df),
    )

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
        "Test securities:",
        test_tickers,
    )

    print(
        "Sample rows:",
        len(sample),
    )

    # ---------------------------------------------------------------
    # Load and validate Yahoo universe ONCE
    # ---------------------------------------------------------------

    yahoo_prices = (
        load_yahoo_prices()
    )

    yahoo_returns = (
        build_next_day_returns(
            yahoo_prices
        )
    )

    validate_returns(
        yahoo_returns
    )

    # ---------------------------------------------------------------
    # Smoke-test strategies
    # ---------------------------------------------------------------

    tests = [
        (
            "sma_crossover",
            {
                "fast": 20,
                "slow": 50,
            },
        ),
        (
            "rsi_mean_reversion",
            {
                "window": 14,
                "entry": 30,
                "exit": 50,
                "short_entry": 70,
            },
        ),
        (
            "donchian_breakout",
            {
                "window": 20,
            },
        ),
    ]

    for mode in (
        "long_only",
        "long_short",
    ):

        for strategy_name, parameters in tests:

            print()
            print("=" * 80)

            print(
                "Testing:",
                strategy_name,
                "|",
                mode,
            )

            strategy_df = (
                generate_strategy_signal(
                    sample,
                    strategy_name=strategy_name,
                    parameters=parameters,
                    mode=mode,
                )
            )

            strategy_id = (
                strategy_df[
                    "strategy_id"
                ]
                .iloc[0]
            )

            result = run_backtest(
                strategy_df,
                cost_bps=5.0,
                strategy_id=strategy_id,
                yahoo_prices=yahoo_prices,
                yahoo_returns=yahoo_returns,
            )

            print(
                "\nStrategy ID:",
                strategy_id,
            )

            print(
                "\nMetrics:"
            )

            for key, value in (
                result.metrics.items()
            ):

                print(
                    f"{key}: {value}"
                )

            print(
                "\nLast 10 daily observations:"
            )

            print(
                result.daily.tail(10)
                .to_string(
                    index=False
                )
            )

    print()
    print("=" * 80)

    print(
        "BACKTEST SMOKE TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()