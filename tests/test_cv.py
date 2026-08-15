import pandas as pd

from src.ml.cv import purged_embargoed_folds
from src.walkforward.windows import generate_windows


def _synthetic_trading_days(start, end):
    return pd.bdate_range(start, end)


def test_train_and_test_never_overlap():
    trading_days = _synthetic_trading_days("2008-01-01", "2020-12-31")
    dates = pd.Series(trading_days)

    windows = generate_windows(
        trading_days.min(), trading_days.max(), is_years=5, oos_years=1, step_years=1
    )

    folds = purged_embargoed_folds(dates, windows, purge_days=1, embargo_days=5)

    for fold in folds:
        assert not (fold.train_mask & fold.test_mask).any()


def test_purge_embargo_gap_removes_dates_immediately_before_oos():
    trading_days = _synthetic_trading_days("2008-01-01", "2020-12-31")
    dates = pd.Series(trading_days)

    windows = generate_windows(
        trading_days.min(), trading_days.max(), is_years=5, oos_years=1, step_years=1
    )

    purge_days = 1
    embargo_days = 5

    folds = purged_embargoed_folds(
        dates, windows, purge_days=purge_days, embargo_days=embargo_days
    )

    fold = folds[0]

    train_dates = dates[fold.train_mask]
    oos_start = fold.window.oos_start

    # No training date should fall within the combined purge+embargo
    # trading-day gap immediately before OOS start.
    gap_start_pos = trading_days.searchsorted(oos_start) - (purge_days + embargo_days)
    if gap_start_pos >= 0:
        gap_dates = set(trading_days[gap_start_pos : trading_days.searchsorted(oos_start)])
        assert not (set(train_dates) & gap_dates)


def test_test_mask_matches_oos_range_exactly():
    trading_days = _synthetic_trading_days("2008-01-01", "2020-12-31")
    dates = pd.Series(trading_days)

    windows = generate_windows(
        trading_days.min(), trading_days.max(), is_years=5, oos_years=1, step_years=1
    )

    folds = purged_embargoed_folds(dates, windows)

    for fold in folds:
        expected = (dates >= fold.window.oos_start) & (dates <= fold.window.oos_end)
        assert (fold.test_mask == expected.to_numpy()).all()
