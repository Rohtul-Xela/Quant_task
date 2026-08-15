import pandas as pd
import pytest

from src.walkforward.combos import combine_signals


def _make_signal(values: dict, mode_values="long_only"):
    rows = []
    for (date, ticker), sig in values.items():
        rows.append(
            {
                "date": pd.Timestamp(date),
                "source_ticker": ticker,
                "yahoo_ticker": ticker,
                "adj_close": 100.0,
                "signal": sig,
            }
        )
    return pd.DataFrame(rows)


CASES_LONG_ONLY = [
    # (sig_a, sig_b) -> (and, or, weighted_vote[0.6/0.4])
    ((0, 0), (0, 0, 0)),
    ((1, 0), (0, 1, 1)),   # weighted: 0.6*1 = 0.6 >= 0.5 -> long
    ((0, 1), (0, 1, 0)),   # weighted: 0.4*1 = 0.4 < 0.5 -> flat
    ((1, 1), (1, 1, 1)),
]


@pytest.mark.parametrize("sig_a,expected", CASES_LONG_ONLY)
def test_long_only_truth_table(sig_a, expected):
    a, b = sig_a
    exp_and, exp_or, exp_weighted = expected

    df_a = _make_signal({("2020-01-01", "AAA"): a})
    df_b = _make_signal({("2020-01-01", "AAA"): b})

    assert combine_signals(df_a, df_b, "and")["signal"].iloc[0] == exp_and
    assert combine_signals(df_a, df_b, "or")["signal"].iloc[0] == exp_or
    assert (
        combine_signals(df_a, df_b, "weighted_vote", weight_a=0.6)["signal"].iloc[0]
        == exp_weighted
    )


def test_or_conflict_resolves_to_flat_for_long_short():
    df_a = _make_signal({("2020-01-01", "AAA"): 1})
    df_b = _make_signal({("2020-01-01", "AAA"): -1})

    result = combine_signals(df_a, df_b, "or")
    assert result["signal"].iloc[0] == 0.0


def test_and_requires_same_sign_for_long_short():
    df_a = _make_signal({("2020-01-01", "AAA"): 1})
    df_b = _make_signal({("2020-01-01", "AAA"): -1})

    result = combine_signals(df_a, df_b, "and")
    assert result["signal"].iloc[0] == 0.0

    df_b_agree = _make_signal({("2020-01-01", "AAA"): 1})
    result_agree = combine_signals(df_a, df_b_agree, "and")
    assert result_agree["signal"].iloc[0] == 1.0


def test_invalid_method_raises():
    df_a = _make_signal({("2020-01-01", "AAA"): 1})
    df_b = _make_signal({("2020-01-01", "AAA"): 1})

    with pytest.raises(ValueError):
        combine_signals(df_a, df_b, "xor")
