# src/data/download.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .yahoo import download_ticker


DATA_DIR = Path("data")
PRICE_DIR = DATA_DIR / "raw" / "yahoo"


def download_universe(tickers: list[str]) -> pd.DataFrame:

    results = []

    for i, ticker in enumerate(tickers, start=1):
        print(
            f"[{i:>4}/{len(tickers)}] "
            f"{ticker}"
        )

        result = download_ticker(
            ticker=ticker,
            output_dir=PRICE_DIR,
            start="2008-01-01",
        )

        results.append(result)

    status = pd.DataFrame(results)

    DATA_DIR.joinpath("validation").mkdir(
        parents=True,
        exist_ok=True,
    )

    status.to_csv(
        DATA_DIR / "validation" / "download_status.csv",
        index=False,
    )

    return status