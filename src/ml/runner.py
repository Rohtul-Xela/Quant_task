"""
Shared ML walk-forward runner.

Factored out of the Phase 4 pipeline so Phase 6 (cost sensitivity) can
re-run the same model/fold logic at different cost_bps without
duplicating — and risking drift from — the Phase 4 loop.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass

import pandas as pd

from src.backtest.backtest import calculate_metrics, run_backtest
from src.ml.cv import MLFold
from src.ml.models import fit_predict_proba
from src.ml.trading_rule import predictions_to_signal


@dataclass
class MLLineResult:
    label: str
    model_name: str
    fold_table: pd.DataFrame
    stitched_daily: pd.DataFrame
    stitched_metrics: dict
    stitched_signal: pd.DataFrame | None = None


def run_ml_line(
    label: str,
    model_name: str,
    dataset: pd.DataFrame,
    feature_cols: list[str],
    folds: list[MLFold],
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
    cost_bps: float,
) -> MLLineResult:

    oos_daily_frames = []
    oos_signal_frames = []
    fold_records = []

    for fold in folds:

        train_df = dataset.loc[fold.train_mask]
        test_df = dataset.loc[fold.test_mask]

        if train_df.empty or test_df.empty:
            warnings.warn(
                f"Skipping window {fold.window.index} for {label}: "
                f"train={len(train_df)} test={len(test_df)}"
            )
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]

        _, proba = fit_predict_proba(model_name, X_train, y_train, X_test)

        signal_df = predictions_to_signal(test_df, proba)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_backtest(
                    signal_df,
                    cost_bps=cost_bps,
                    strategy_id=f"{label}__oos{fold.window.index}__cost{cost_bps}",
                    yahoo_prices=yahoo_prices,
                    yahoo_returns=yahoo_returns,
                )
        except ValueError as exc:
            warnings.warn(f"Backtest skipped for window {fold.window.index}: {exc}")
            continue

        oos_daily_frames.append(result.daily)
        oos_signal_frames.append(signal_df)

        fold_records.append(
            {
                "line_id": label,
                "window_index": fold.window.index,
                "oos_start": fold.window.oos_start,
                "oos_end": fold.window.oos_end,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "oos_sharpe": result.metrics["sharpe"],
            }
        )

    if not oos_daily_frames:
        raise ValueError(f"No OOS results produced for ML line {label!r}.")

    stitched_daily = (
        pd.concat(oos_daily_frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    stitched_metrics = calculate_metrics(stitched_daily)
    stitched_metrics["strategy_id"] = label
    stitched_metrics["cost_bps"] = cost_bps

    stitched_signal = (
        pd.concat(oos_signal_frames, ignore_index=True)
        .sort_values(["source_ticker", "date"])
        .reset_index(drop=True)
    )

    return MLLineResult(
        label=label,
        model_name=model_name,
        fold_table=pd.DataFrame(fold_records),
        stitched_daily=stitched_daily,
        stitched_metrics=stitched_metrics,
        stitched_signal=stitched_signal,
    )
