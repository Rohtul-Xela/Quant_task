import numpy as np
import pandas as pd
import pytest

from src.ml.dataset import cross_sectional_percentile_rank


def _synthetic_frame():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01"] * 4 + ["2020-01-02"] * 4
            ),
            "source_ticker": ["A", "B", "C", "D"] * 2,
            "x": [10, 20, 30, 40, 100, 200, 300, 400],
        }
    )


def test_ranks_are_in_zero_one():
    df = cross_sectional_percentile_rank(_synthetic_frame(), ["x"])
    assert df["x"].between(0, 1).all()


def test_rank_only_uses_same_date_cross_section():
    df = cross_sectional_percentile_rank(_synthetic_frame(), ["x"])

    day_1 = df[df["date"] == "2020-01-01"].set_index("source_ticker")["x"]
    day_2 = df[df["date"] == "2020-01-02"].set_index("source_ticker")["x"]

    # Both days have the identical *within-day* relative ordering
    # (10<20<30<40 and 100<200<300<400), so ranks should be identical
    # across days even though the raw scale is 10x different — proving
    # the second day's ranking wasn't influenced by the first day's
    # absolute values.
    pd.testing.assert_series_equal(day_1, day_2, check_names=False)

    # Smallest value on each day -> lowest rank (0.25 of 4 names).
    assert day_1["A"] == 0.25
    assert day_1["D"] == 1.0


def test_nan_excluded_from_ranking_not_imputed():
    df = _synthetic_frame()
    df.loc[df["source_ticker"] == "B", "x"] = np.nan

    result = cross_sectional_percentile_rank(df, ["x"])

    b_rows = result[result["source_ticker"] == "B"]
    assert b_rows["x"].isna().all()

    # Remaining 3 tickers on day 1 (A=10,C=30,D=40) still rank among
    # themselves, not diluted by B's missing value.
    day_1 = result[
        (result["date"] == "2020-01-01") & (result["source_ticker"] != "B")
    ].set_index("source_ticker")["x"]
    assert day_1["A"] == pytest.approx(1 / 3)
