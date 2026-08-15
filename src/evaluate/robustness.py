"""
Phase 7 — robustness.

- Block bootstrap Sharpe confidence interval (respects daily-return
  autocorrelation, unlike an i.i.d. bootstrap).
- Bootstrap-based multiple-testing correction (Benjamini-Hochberg over
  the bootstrap p-values of every finalist actually walk-forward
  tested) — the task brief explicitly allows a bootstrap-based
  correction rather than requiring deflated Sharpe specifically.
- Parameter-stability plateau, reusing the EXISTING in-sample sweep
  grid (no new backtests) — a heatmap showing the shortlisted point
  sits on a plateau, not an isolated spike.
- Breadth: fraction of the research universe a strategy is
  net-profitable on over the stitched OOS period. Reuses
  `build_portfolio_weights` from backtest.py rather than
  reimplementing weight construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import build_portfolio_weights  # noqa: E402


def block_bootstrap_sharpe(
    daily_returns: np.ndarray,
    block_size: int = 20,
    n_boot: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Circular block bootstrap: resample overlapping blocks of
    `block_size` consecutive daily returns (with wraparound) until the
    resample matches the original length, compute the annualized
    Sharpe of each resample. Block bootstrap (vs. i.i.d. resampling of
    individual days) preserves short-run autocorrelation structure.
    """

    returns = np.asarray(daily_returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n < block_size * 2:
        raise ValueError(
            f"Too few observations ({n}) for block_size={block_size}."
        )

    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_size))

    boot_sharpes = np.empty(n_boot)

    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = [
            returns[np.arange(s, s + block_size) % n] for s in starts
        ]
        resample = np.concatenate(blocks)[:n]

        std = resample.std(ddof=1)
        boot_sharpes[i] = (
            resample.mean() / std * np.sqrt(252.0) if std > 0 else np.nan
        )

    return boot_sharpes


def bootstrap_sharpe_ci(
    daily_returns: np.ndarray,
    block_size: int = 20,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:

    boot = block_bootstrap_sharpe(daily_returns, block_size, n_boot, random_state)
    boot = boot[~np.isnan(boot)]

    alpha = 1.0 - ci
    lower = np.quantile(boot, alpha / 2.0)
    upper = np.quantile(boot, 1.0 - alpha / 2.0)

    p_value_le_zero = float((boot <= 0).mean())

    returns = np.asarray(daily_returns)
    returns = returns[~np.isnan(returns)]
    std = returns.std(ddof=1)
    point_sharpe = returns.mean() / std * np.sqrt(252.0) if std > 0 else np.nan

    return {
        "point_sharpe": point_sharpe,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value_sharpe_le_0": p_value_le_zero,
        "n_boot": len(boot),
    }


def benjamini_hochberg(p_values: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """
    Standard BH step-up procedure. Returns a frame with the adjusted
    p-value and a significant-at-alpha flag, indexed the same as the
    input.
    """

    n = len(p_values)
    order = p_values.sort_values().index

    ranked = p_values.loc[order]
    ranks = np.arange(1, n + 1)

    adjusted = ranked.to_numpy() * n / ranks
    # Enforce monotonicity (BH adjusted p-values must be non-decreasing
    # when read from the largest p-value down to the smallest).
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    result = pd.DataFrame(
        {
            "p_value": ranked.to_numpy(),
            "bh_adjusted_p": adjusted,
            "significant_at_alpha": adjusted <= alpha,
        },
        index=order,
    )

    return result.loc[p_values.index]


def parameter_stability_pivot(
    sweep_df: pd.DataFrame,
    strategy_name: str,
    mode: str,
) -> pd.DataFrame:

    subset = sweep_df[
        (sweep_df["strategy_name"] == strategy_name) & (sweep_df["mode"] == mode)
    ]

    if strategy_name == "sma_crossover":
        pivot = subset.pivot(index="param_fast", columns="param_slow", values="sharpe")
    elif strategy_name == "donchian_breakout":
        pivot = subset.set_index("param_window")[["sharpe"]]
    else:
        raise ValueError(f"No pivot layout defined for {strategy_name!r}.")

    return pivot.sort_index()


def per_ticker_breadth(
    signal_df: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstructs per-ticker cumulative P&L contribution using the same
    target-weight construction as `run_backtest`
    (`build_portfolio_weights`, reused not reimplemented), summed over
    whatever date range `signal_df` already covers (callers pass the
    stitched-OOS-only signal). Only tickers that were ever actively
    weighted are counted — flat-the-whole-period names would otherwise
    dilute the breadth statistic without ever having been "tested" by
    the strategy.
    """

    strategy = signal_df.merge(
        yahoo_returns[["date", "ticker", "next_return"]],
        left_on=["date", "yahoo_ticker"],
        right_on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["ticker"])

    strategy["portfolio_signal"] = np.where(
        strategy["next_return"].notna(), strategy["signal"], 0.0
    )

    weights = build_portfolio_weights(strategy)

    merged = strategy.merge(
        weights[["date", "source_ticker", "target_weight"]],
        on=["date", "source_ticker"],
        how="left",
    )

    merged["contribution"] = merged["target_weight"] * merged["next_return"].fillna(0.0)

    per_ticker = (
        merged.groupby("source_ticker")
        .agg(
            cumulative_contribution=("contribution", "sum"),
            active_days=("target_weight", lambda s: int((s != 0).sum())),
        )
        .reset_index()
    )

    traded = per_ticker[per_ticker["active_days"] > 0].copy()
    traded["profitable"] = traded["cumulative_contribution"] > 0

    return traded


def breadth_summary(traded: pd.DataFrame) -> dict:
    if traded.empty:
        return {"n_traded": 0, "n_profitable": 0, "breadth_pct": np.nan}

    n_traded = len(traded)
    n_profitable = int(traded["profitable"].sum())

    return {
        "n_traded": n_traded,
        "n_profitable": n_profitable,
        "breadth_pct": n_profitable / n_traded,
    }
