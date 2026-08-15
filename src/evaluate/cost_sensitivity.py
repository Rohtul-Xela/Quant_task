"""
Phase 6 — cost sensitivity.

Re-runs an already-chosen finalist's FIXED stitched OOS signal (saved
by Phase 2/3/4 as `{line_id}__signal.parquet`) through
`run_fixed_signal_walk_forward` at several cost levels. Holding the
strategy/parameter choice fixed and varying only cost isolates the
cost effect from window-to-window re-optimization noise — a cleaner
experiment for "where does the edge die" than re-running full
in-sample re-optimization at each cost level, and for the ML lines it
avoids retraining six times over (retraining doesn't depend on cost,
only the backtest step does).

0 bps is used ONLY at this single point, never elsewhere in the
project (hard constraint from the task brief).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.walkforward.harness import run_fixed_signal_walk_forward  # noqa: E402
from src.walkforward.windows import Window  # noqa: E402

DEFAULT_COST_LEVELS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def cost_sensitivity_sweep(
    line_id: str,
    signal_df: pd.DataFrame,
    windows: list[Window],
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
    cost_levels: tuple[float, ...] = DEFAULT_COST_LEVELS,
) -> pd.DataFrame:

    rows = []

    for cost_bps in cost_levels:

        result = run_fixed_signal_walk_forward(
            line_id=f"{line_id}__cost{cost_bps}",
            signal_df=signal_df,
            windows=windows,
            yahoo_prices=yahoo_prices,
            yahoo_returns=yahoo_returns,
            cost_bps=cost_bps,
        )

        metrics = dict(result.stitched_metrics)

        rows.append(
            {
                "line_id": line_id,
                "cost_bps": cost_bps,
                "sharpe": metrics["sharpe"],
                "net_return": metrics["net_return"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)
