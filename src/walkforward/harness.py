"""
Walk-forward harness.

Reuses `generate_strategy_signal`, `run_backtest` and `calculate_metrics`
from the existing codebase unmodified — this module only handles window
slicing, per-window in-sample parameter re-optimization, and stitching
the resulting out-of-sample segments into one continuous curve.

Design note (see plan / report "where weakest" section): indicators
only look backward, so a strategy's signal can be computed once over
the FULL history and then sliced by date per window — slicing never
introduces look-ahead, it only restricts which already-computed rows
enter a given backtest call. `run_backtest` computes turnover from a
pivot of *only the rows it is given*, so the first day of every
sliced window is treated as a from-flat entry. This is a known,
documented mechanical side-effect of stitching, not a leak.
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    BacktestResult,
    calculate_metrics,
    run_backtest,
)
from src.strategy.strategies import generate_strategy_signal  # noqa: E402

from src.walkforward.windows import Window  # noqa: E402


def param_key(parameters: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted(parameters.items()))


def precompute_signal_frames(
    df: pd.DataFrame,
    strategy_name: str,
    mode: str,
    parameter_grid: list[dict],
) -> dict[str, pd.DataFrame]:
    """
    Compute each parameter combo's signal ONCE over the full history.
    Walk-forward windows then just slice these by date — the indicator
    values themselves never change with the window.
    """

    frames = {}

    for parameters in parameter_grid:
        key = param_key(parameters)

        signal_df = generate_strategy_signal(
            df,
            strategy_name=strategy_name,
            parameters=parameters,
            mode=mode,
        )

        frames[key] = signal_df

    return frames


def _slice_dates(
    signal_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (signal_df["date"] >= start) & (signal_df["date"] <= end)
    return signal_df.loc[mask]


def _safe_backtest(
    sliced_df: pd.DataFrame,
    strategy_id: str,
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
    cost_bps: float,
) -> BacktestResult | None:

    if sliced_df.empty:
        return None

    try:
        # run_backtest prints a full, unbounded diagnostic table whenever
        # any active position lacks a valid next return — routine on
        # window boundaries with long-only trend signals. At hundreds of
        # calls per walk-forward line that I/O dominates runtime, so it
        # is suppressed here (harness-only; run_backtest's own behavior
        # and return value are unchanged, still checked via
        # active_positions_missing_return in the metrics).
        with contextlib.redirect_stdout(io.StringIO()):
            return run_backtest(
                sliced_df,
                cost_bps=cost_bps,
                strategy_id=strategy_id,
                yahoo_prices=yahoo_prices,
                yahoo_returns=yahoo_returns,
            )
    except ValueError as exc:
        warnings.warn(f"Backtest skipped for {strategy_id}: {exc}")
        return None


@dataclass
class WalkForwardLineResult:
    line_id: str
    strategy_name: str
    mode: str
    window_table: pd.DataFrame
    stitched_daily: pd.DataFrame
    stitched_metrics: dict
    stitched_signal: pd.DataFrame | None = None
    per_param_daily: dict[str, pd.DataFrame] = field(default_factory=dict)


def run_walk_forward_line(
    line_id: str,
    strategy_name: str,
    mode: str,
    parameter_grid: list[dict],
    df: pd.DataFrame,
    windows: list[Window],
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
    cost_bps: float,
) -> WalkForwardLineResult:

    signal_frames = precompute_signal_frames(
        df, strategy_name, mode, parameter_grid
    )

    window_records = []
    oos_daily_frames = []
    oos_signal_frames = []

    for window in windows:

        is_sharpes: dict[str, float] = {}
        oos_sharpes: dict[str, float] = {}
        oos_results: dict[str, BacktestResult | None] = {}
        oos_slices: dict[str, pd.DataFrame] = {}

        for key, signal_df in signal_frames.items():

            is_slice = _slice_dates(signal_df, window.is_start, window.is_end)
            is_result = _safe_backtest(
                is_slice,
                f"{line_id}__{key}__is{window.index}",
                yahoo_prices,
                yahoo_returns,
                cost_bps,
            )
            is_sharpes[key] = (
                is_result.metrics["sharpe"] if is_result is not None else np.nan
            )

            oos_slice = _slice_dates(signal_df, window.oos_start, window.oos_end)
            oos_slices[key] = oos_slice
            oos_result = _safe_backtest(
                oos_slice,
                f"{line_id}__{key}__oos{window.index}",
                yahoo_prices,
                yahoo_returns,
                cost_bps,
            )
            oos_results[key] = oos_result
            oos_sharpes[key] = (
                oos_result.metrics["sharpe"] if oos_result is not None else np.nan
            )

        # -------------------------------------------------------------
        # Pick the IS winner (highest IS Sharpe; NaN treated as -inf).
        # -------------------------------------------------------------

        ranked_by_is = sorted(
            is_sharpes.items(),
            key=lambda kv: kv[1] if pd.notna(kv[1]) else -np.inf,
            reverse=True,
        )

        chosen_key, chosen_is_sharpe = ranked_by_is[0]
        chosen_oos_result = oos_results[chosen_key]
        chosen_oos_sharpe = oos_sharpes[chosen_key]

        if chosen_oos_result is not None:
            oos_daily_frames.append(chosen_oos_result.daily)
            oos_signal_frames.append(oos_slices[chosen_key])

        # -------------------------------------------------------------
        # Ranking stability: Spearman correlation between IS rank and
        # OOS rank across all candidates in this window (only defined
        # when >= 2 candidates have both a valid IS and OOS Sharpe).
        # -------------------------------------------------------------

        keys_both_valid = [
            k for k in signal_frames
            if pd.notna(is_sharpes[k]) and pd.notna(oos_sharpes[k])
        ]

        if len(keys_both_valid) >= 2:
            is_vals = [is_sharpes[k] for k in keys_both_valid]
            oos_vals = [oos_sharpes[k] for k in keys_both_valid]
            rank_corr, _ = spearmanr(is_vals, oos_vals)
        else:
            rank_corr = np.nan

        did_is_winner_also_win_oos = (
            chosen_key
            == max(
                keys_both_valid,
                key=lambda k: oos_sharpes[k],
                default=None,
            )
            if keys_both_valid
            else np.nan
        )

        wf_efficiency = (
            chosen_oos_sharpe / chosen_is_sharpe
            if pd.notna(chosen_is_sharpe)
            and chosen_is_sharpe > 0
            and pd.notna(chosen_oos_sharpe)
            else np.nan
        )

        window_records.append(
            {
                "line_id": line_id,
                "window_index": window.index,
                "is_start": window.is_start,
                "is_end": window.is_end,
                "oos_start": window.oos_start,
                "oos_end": window.oos_end,
                "is_partial_oos": window.is_partial_oos,
                "n_candidates": len(signal_frames),
                "chosen_params": chosen_key,
                "is_sharpe": chosen_is_sharpe,
                "oos_sharpe": chosen_oos_sharpe,
                "wf_efficiency": wf_efficiency,
                "rank_corr_is_vs_oos": rank_corr,
                "is_winner_also_oos_best": did_is_winner_also_win_oos,
            }
        )

    if not oos_daily_frames:
        raise ValueError(f"No OOS results produced for line {line_id!r}.")

    stitched_daily = (
        pd.concat(oos_daily_frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    stitched_metrics = calculate_metrics(stitched_daily)
    stitched_metrics["strategy_id"] = line_id
    stitched_metrics["cost_bps"] = cost_bps

    stitched_signal = (
        pd.concat(oos_signal_frames, ignore_index=True)
        .sort_values(["source_ticker", "date"])
        .reset_index(drop=True)
    )

    return WalkForwardLineResult(
        line_id=line_id,
        strategy_name=strategy_name,
        mode=mode,
        window_table=pd.DataFrame(window_records),
        stitched_daily=stitched_daily,
        stitched_metrics=stitched_metrics,
        stitched_signal=stitched_signal,
    )


def run_fixed_signal_walk_forward(
    line_id: str,
    signal_df: pd.DataFrame,
    windows: list[Window],
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
    cost_bps: float,
) -> WalkForwardLineResult:
    """
    Same OOS-slicing-and-stitching as `run_walk_forward_line`, but with
    NO per-window in-sample parameter search — for combo strategies
    (Phase 3), whose parameters are already fixed from the Phase 1
    shortlist, and for cost-sensitivity re-runs (Phase 6) of an
    already-chosen configuration. There is nothing to "optimize" here,
    so is_sharpe/wf_efficiency/rank_corr columns are omitted rather
    than faked.
    """

    window_records = []
    oos_daily_frames = []
    oos_signal_frames = []

    for window in windows:

        oos_slice = _slice_dates(signal_df, window.oos_start, window.oos_end)
        oos_result = _safe_backtest(
            oos_slice,
            f"{line_id}__oos{window.index}",
            yahoo_prices,
            yahoo_returns,
            cost_bps,
        )

        if oos_result is not None:
            oos_daily_frames.append(oos_result.daily)
            oos_signal_frames.append(oos_slice)

        window_records.append(
            {
                "line_id": line_id,
                "window_index": window.index,
                "oos_start": window.oos_start,
                "oos_end": window.oos_end,
                "is_partial_oos": window.is_partial_oos,
                "oos_sharpe": (
                    oos_result.metrics["sharpe"] if oos_result is not None else np.nan
                ),
            }
        )

    if not oos_daily_frames:
        raise ValueError(f"No OOS results produced for line {line_id!r}.")

    stitched_daily = (
        pd.concat(oos_daily_frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    stitched_metrics = calculate_metrics(stitched_daily)
    stitched_metrics["strategy_id"] = line_id
    stitched_metrics["cost_bps"] = cost_bps

    stitched_signal = (
        pd.concat(oos_signal_frames, ignore_index=True)
        .sort_values(["source_ticker", "date"])
        .reset_index(drop=True)
    )

    return WalkForwardLineResult(
        line_id=line_id,
        strategy_name=line_id,
        mode="n/a",
        window_table=pd.DataFrame(window_records),
        stitched_daily=stitched_daily,
        stitched_metrics=stitched_metrics,
        stitched_signal=stitched_signal,
    )
