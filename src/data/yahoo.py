from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "data" / "data"

START_DATE = "2008-01-01"
# yfinance's `end` is exclusive (verified empirically against the pinned
# yfinance==1.5.2), so this must be one day past the intended inclusive
# research cutoff (RESEARCH_END_DATE in build_pit_dataset.py, "2026-08-11")
# for the download to actually reach that date.
END_DATE = "2026-08-12"

PRICE_DIR = DATA_DIR / "raw" / "yahoo"
STATUS_FILE = DATA_DIR / "validation" / "download_status.csv"


def _flatten_yfinance_columns(
    df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Normalize yfinance MultiIndex output for one ticker."""

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    if df.columns.nlevels == 2:
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)

        unique_level_1 = {
            str(x) for x in level_1
        }

        if unique_level_1 == {ticker}:
            df.columns = level_0
            return df

    df.columns = [
        str(col[0]) if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]

    return df


def _validate_and_clean(
    df: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Clean and validate downloaded Yahoo data."""

    warnings: list[str] = []

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{ticker}: missing columns {missing}"
        )

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # Remove rows with missing OHLC.
    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    invalid_price = df[
        price_columns
    ].isna().any(axis=1)

    n_invalid = int(
        invalid_price.sum()
    )

    if n_invalid:
        warnings.append(
            f"removed {n_invalid} rows with missing OHLC"
        )

        df = df.loc[
            ~invalid_price
        ].copy()

    # Sort and deduplicate.
    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Validate OHLC relationships.
    bad_ohlc = (
        (df["high"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    )

    n_bad_ohlc = int(
        bad_ohlc.sum()
    )

    if n_bad_ohlc:
        warnings.append(
            f"removed {n_bad_ohlc} rows with invalid OHLC"
        )

        df = df.loc[
            ~bad_ohlc
        ].copy()

    if len(df) < 100:
        raise ValueError(
            f"{ticker}: only {len(df)} usable price rows"
        )

    df["ticker"] = ticker

    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    df = df[
        [
            col for col in columns
            if col in df.columns
        ]
    ]

    return df, warnings


def download_ticker(
    ticker: str,
    output_dir: str | Path = PRICE_DIR,
    start: str = START_DATE,
    end: str | None = END_DATE,
    force: bool = False,
) -> dict:
    """
    Download one Yahoo ticker.

    The raw cache may contain data beyond the research cutoff if it was
    downloaded previously. The research layer should apply AS_OF_DATE
    when loading data.

    force=True rebuilds the cache file.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / f"{ticker}.parquet"
    )

    # ---------------------------------------------------------------
    # Cache
    # ---------------------------------------------------------------

    if (
        output_path.exists()
        and not force
    ):
        try:
            df = pd.read_parquet(
                output_path
            )

            required_cache = {
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "dividends",
                "stock_splits",
            }

            if not required_cache.issubset(
                df.columns
            ):
                raise ValueError(
                    "old cache format"
                )

            return {
                "ticker": ticker,
                "status": "cached",
                "first_date": df["date"].min(),
                "last_date": df["date"].max(),
                "rows": len(df),
                "error": None,
            }

        except Exception:
            output_path.unlink(
                missing_ok=True
            )

    # ---------------------------------------------------------------
    # Download
    # ---------------------------------------------------------------

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )

        if raw.empty:
            return {
                "ticker": ticker,
                "status": "empty",
                "first_date": None,
                "last_date": None,
                "rows": 0,
                "error": None,
            }

        raw = _flatten_yfinance_columns(
            raw,
            ticker,
        )

        raw = raw.reset_index()

        raw = raw.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits",
            }
        )

        clean, warnings = (
            _validate_and_clean(
                raw,
                ticker,
            )
        )

        clean.to_parquet(
            output_path,
            index=False,
        )

        return {
            "ticker": ticker,
            "status": (
                "downloaded_cleaned"
                if warnings
                else "downloaded"
            ),
            "first_date": clean["date"].min(),
            "last_date": clean["date"].max(),
            "rows": len(clean),
            "error": (
                "; ".join(warnings)
                if warnings
                else None
            ),
        }

    except Exception as exc:
        return {
            "ticker": ticker,
            "status": "invalid_data",
            "first_date": None,
            "last_date": None,
            "rows": 0,
            "error": repr(exc),
        }

    finally:
        time.sleep(0.2)


def load_historical_tickers() -> list[str]:
    """Load canonical historical ticker universe."""

    path = DATA_DIR / "processed" / "historical_tickers.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Ticker universe not found: {path}"
        )

    df = pd.read_csv(path)

    if "ticker" not in df.columns:
        raise ValueError(
            "historical_tickers.csv must contain 'ticker'"
        )

    return sorted(
        set(
            df["ticker"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
    )


def download_universe(
    tickers: list[str] | None = None,
    force: bool = False,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    """Download the entire historical ticker universe."""

    if tickers is None:
        tickers = load_historical_tickers()

    STATUS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total = len(tickers)

    print(
        f"Downloading {total} historical tickers..."
    )

    for i, ticker in enumerate(
        tickers,
        start=1,
    ):
        print(
            f"[{i:>3}/{total}] {ticker}"
        )

        result = download_ticker(
            ticker=ticker,
            end=end,
            force=force,
        )

        results.append(result)

        # Save progress after every ticker.
        pd.DataFrame(results).to_csv(
            STATUS_FILE,
            index=False,
        )

    return pd.DataFrame(results)