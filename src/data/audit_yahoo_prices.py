from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

YAHOO_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "raw"
    / "yahoo"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "data"
    / "validation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "yahoo_price_quality.csv"
)


def audit_ticker(
    path: Path,
) -> dict:

    ticker = path.stem

    try:
        df = pd.read_parquet(
            path
        )

        required = {
            "date",
            "adj_close",
        }

        if not required.issubset(
            df.columns
        ):
            return {
                "ticker": ticker,
                "status": "missing_columns",
            }

        df = df[
            [
                "date",
                "adj_close",
            ]
        ].copy()

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df["adj_close"] = pd.to_numeric(
            df["adj_close"],
            errors="coerce",
        )

        df = (
            df
            .dropna(
                subset=[
                    "date",
                    "adj_close",
                ]
            )
            .sort_values("date")
            .drop_duplicates(
                "date",
                keep="last",
            )
            .reset_index(drop=True)
        )

        if len(df) < 2:

            return {
                "ticker": ticker,
                "status": "insufficient_data",
            }

        previous = (
            df["adj_close"]
            .shift(1)
        )

        returns = (
            df["adj_close"]
            / previous
            - 1.0
        )

        extreme = (
            returns.abs() > 1.0
        )

        extreme_count = int(
            extreme.sum()
        )

        # ----------------------------------------------------------
        # Detect alternating price regimes.
        #
        # Example:
        # 0.005 -> 170 -> 0.005
        #
        # This produces two consecutive extreme returns with
        # opposite direction.
        # ----------------------------------------------------------

        extreme_return = (
            returns.where(extreme)
        )

        next_extreme_return = (
            extreme_return.shift(-1)
        )

        alternating = (
            extreme
            & next_extreme_return.notna()
            & (
                np.sign(
                    extreme_return
                )
                != np.sign(
                    next_extreme_return
                )
            )
        )

        alternating_count = int(
            alternating.sum()
        )

        # ----------------------------------------------------------
        # Price ratio diagnostic
        # ----------------------------------------------------------

        price_ratio = (
            df["adj_close"]
            / previous
        )

        massive_jump = (
            price_ratio > 100.0
        ) | (
            price_ratio < 0.01
        )

        massive_jump_count = int(
            massive_jump.sum()
        )

        # ----------------------------------------------------------
        # Decision
        # ----------------------------------------------------------

        if alternating_count >= 2:

            status = (
                "exclude_alternating_extremes"
            )

        elif massive_jump_count >= 3:

            status = (
                "review_repeated_extremes"
            )

        elif extreme_count > 0:

            status = (
                "review_extremes"
            )

        else:

            status = "clean"

        return {
            "ticker": ticker,
            "status": status,
            "rows": len(df),
            "first_date": df["date"].min(),
            "last_date": df["date"].max(),
            "extreme_return_count": extreme_count,
            "alternating_extreme_count": alternating_count,
            "massive_jump_count": massive_jump_count,
            "max_abs_return": float(
                returns.abs().max()
            ),
        }

    except Exception as exc:

        return {
            "ticker": ticker,
            "status": "error",
            "error": repr(exc),
        }


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        YAHOO_DIR.glob("*.parquet")
    )

    print(
        "Yahoo files:",
        len(files),
    )

    results = []

    for i, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{i}/{len(files)}] {path.name}"
        )

        results.append(
            audit_ticker(path)
        )

    audit = pd.DataFrame(
        results
    )

    audit = (
        audit
        .sort_values(
            [
                "status",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "=== STATUS SUMMARY ==="
    )

    print(
        audit["status"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "=== SUSPICIOUS TICKERS ==="
    )

    suspicious = audit[
        audit["status"] != "clean"
    ]

    if suspicious.empty:

        print("None")

    else:

        print(
            suspicious.to_string(
                index=False
            )
        )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()