import pandas as pd
import pytest

from src.walkforward.windows import generate_windows


def test_windows_are_contiguous_with_no_gap_or_overlap():
    windows = generate_windows(
        pd.Timestamp("2008-01-02"),
        pd.Timestamp("2020-06-30"),
        is_years=5,
        oos_years=1,
        step_years=1,
    )

    assert len(windows) > 1

    for w in windows:
        assert w.is_start < w.is_end < w.oos_start <= w.oos_end

    for a, b in zip(windows, windows[1:]):
        # step_years == oos_years, so each window's OOS should pick up
        # exactly where the previous one's OOS left off.
        assert b.oos_start == a.oos_end + pd.Timedelta(days=1)


def test_final_window_is_marked_partial_when_data_ends_mid_year():
    windows = generate_windows(
        pd.Timestamp("2008-01-02"),
        pd.Timestamp("2026-08-11"),
        is_years=5,
        oos_years=1,
        step_years=1,
    )

    assert windows[-1].is_partial_oos
    assert windows[-1].oos_end == pd.Timestamp("2026-08-11")
    assert not any(w.is_partial_oos for w in windows[:-1])


def test_invalid_years_raise():
    with pytest.raises(ValueError):
        generate_windows(
            pd.Timestamp("2008-01-02"),
            pd.Timestamp("2020-01-01"),
            is_years=0,
        )


def test_no_windows_raises():
    with pytest.raises(ValueError):
        generate_windows(
            pd.Timestamp("2008-01-02"),
            pd.Timestamp("2009-01-01"),
            is_years=5,
            oos_years=1,
        )
