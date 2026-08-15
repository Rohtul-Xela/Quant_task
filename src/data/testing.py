from src.backtest.backtest import (
    load_yahoo_prices,
    build_next_day_returns,
)
import pandas as pd
import pandas_market_calendars as mcal


prices = load_yahoo_prices()
returns = build_next_day_returns(prices)


# ============================================================
# 1. Basic return statistics
# ============================================================

print("\n" + "=" * 80)
print("RETURN DATA QUALITY")
print("=" * 80)

print("Total Yahoo observations:", len(prices))
print("Total return observations:", len(returns))

print(
    "Valid next-session returns:",
    returns["next_return"].notna().sum(),
)

print(
    "Missing next-session returns:",
    returns["next_return"].isna().sum(),
)


# ============================================================
# 2. Long gaps
# ============================================================

long_gaps = returns[
    returns["calendar_gap_days"] > 4
].copy()

print("\nLong calendar gaps (>4 days):", len(long_gaps))

print("\nLargest gaps:")
print(
    long_gaps[
        [
            "ticker",
            "date",
            "next_date",
            "calendar_gap_days",
            "next_return",
        ]
    ]
    .sort_values(
        "calendar_gap_days",
        ascending=False,
    )
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 3. Check whether rejected observations skipped an actual
#    NYSE trading session.
# ============================================================

nyse = mcal.get_calendar("NYSE")

schedule = nyse.schedule(
    start_date=prices["date"].min(),
    end_date=prices["date"].max(),
)

sessions = (
    schedule.index
    .tz_localize(None)
)


session_set = set(sessions)


def skipped_nyse_sessions(row):

    if pd.isna(row["next_date"]):
        return 0

    start = row["date"]
    end = row["next_date"]

    between = sessions[
        (sessions > start)
        & (sessions < end)
    ]

    return len(between)


long_gaps["skipped_nyse_sessions"] = (
    long_gaps.apply(
        skipped_nyse_sessions,
        axis=1,
    )
)


# ============================================================
# 4. Classify long gaps
# ============================================================

print("\n" + "=" * 80)
print("LONG GAP CLASSIFICATION")
print("=" * 80)

print(
    long_gaps[
        "skipped_nyse_sessions"
    ]
    .value_counts()
    .sort_index()
)


print("\nLong gaps with skipped NYSE sessions:")

print(
    long_gaps[
        long_gaps["skipped_nyse_sessions"] > 0
    ][
        [
            "ticker",
            "date",
            "next_date",
            "calendar_gap_days",
            "skipped_nyse_sessions",
            "next_return",
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# ============================================================
# 5. Largest gaps where NO NYSE sessions were skipped
# ============================================================

print("\n" + "=" * 80)
print("SUSPICIOUS GAPS")
print("=" * 80)

suspicious = long_gaps[
    long_gaps["skipped_nyse_sessions"] == 0
]

print(
    "Long gaps with zero skipped NYSE sessions:",
    len(suspicious),
)

print(
    suspicious[
        [
            "ticker",
            "date",
            "next_date",
            "calendar_gap_days",
            "next_return",
        ]
    ]
    .sort_values(
        "calendar_gap_days",
        ascending=False,
    )
    .head(50)
    .to_string(index=False)
)