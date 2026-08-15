"""
Signal combination for Phase 3 (strategy pairs).

Two binary/ternary signals (long_only: {0,1}, long_short: {-1,0,1})
combined three ways:

- "and": trade only when both legs agree and are nonzero. Conservative.
- "or":  trade when either leg is nonzero. If both are nonzero but
  disagree in sign, the conflict resolves to flat (0) — documented,
  not silently averaged into a fractional position.
- "weighted_vote": weight_a/weight_b weighted sum of the two signals,
  thresholded at +/-0.5. With unequal weights (default 0.6/0.4) this
  is a genuine third option, not a relabeling of AND/OR: leg A alone
  can trigger a position, but leg B alone cannot — a middle ground
  between "both required" (AND) and "either sufficient" (OR). With
  equal weights on two binary legs, weighted_vote degenerates to OR
  at the >=0.5 threshold — which is why unequal weights are the
  default here, not an afterthought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_METHODS = {"and", "or", "weighted_vote"}


def combine_signals(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    method: str,
    weight_a: float = 0.6,
) -> pd.DataFrame:

    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_METHODS)}")

    required = {"date", "source_ticker", "yahoo_ticker", "adj_close", "signal"}

    for name, frame in (("sig_a", sig_a), ("sig_b", sig_b)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    left = sig_a[["date", "source_ticker", "yahoo_ticker", "adj_close", "signal"]]
    right = sig_b[["date", "source_ticker", "signal"]]

    merged = left.merge(
        right,
        on=["date", "source_ticker"],
        how="outer",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )

    merged["signal_a"] = merged["signal_a"].fillna(0.0)
    merged["signal_b"] = merged["signal_b"].fillna(0.0)

    a = merged["signal_a"]
    b = merged["signal_b"]

    if method == "and":
        same_sign_nonzero = (a != 0) & (b != 0) & (np.sign(a) == np.sign(b))
        combined = np.where(same_sign_nonzero, np.sign(a), 0.0)

    elif method == "or":
        both_nonzero_conflict = (a != 0) & (b != 0) & (np.sign(a) != np.sign(b))
        combined = np.select(
            [
                both_nonzero_conflict,
                a != 0,
                b != 0,
            ],
            [
                0.0,
                a,
                b,
            ],
            default=0.0,
        )

    else:  # weighted_vote
        if not (0.0 < weight_a < 1.0):
            raise ValueError("weight_a must be strictly between 0 and 1.")
        weight_b = 1.0 - weight_a
        score = weight_a * a + weight_b * b
        combined = np.select(
            [score >= 0.5, score <= -0.5],
            [1.0, -1.0],
            default=0.0,
        )

    merged["signal"] = combined

    result = merged[
        ["date", "source_ticker", "yahoo_ticker", "adj_close", "signal"]
    ].copy()

    result = result.sort_values(["source_ticker", "date"]).reset_index(drop=True)

    return result
