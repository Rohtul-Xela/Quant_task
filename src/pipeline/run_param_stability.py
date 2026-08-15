"""
Denser parameter-stability grids for the 5 finalists with genuine
positive Sharpe (SMA, Bollinger, MFI, RSI, MACD) — Donchian (negative,
deliberate weak comparator), Stochastic (~0, not a real winner), and
the long-short RSI line (not significant) are skipped; a plateau check
is only meaningful for an actual winner.

Unlike the existing sparse 6-point SMA grid reused in Phase 7 (a
by-product of the original sweep's design, not built for this
purpose), each grid here is purpose-built: a dense 5x5 local
neighborhood around the winning parameter pair, holding any other
parameters fixed at their winning values. In-sample only (same
methodology as the existing sweep, `run_sweep.py`) — no new
walk-forward needed, this is purely about whether the winning point
sits on a plateau or a spike.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    DEFAULT_COST_BPS,
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
    run_backtest,
)
from src.strategy.strategies import generate_strategy_signal  # noqa: E402

FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = OUTPUT_DIR / "charts"

COST_BPS = DEFAULT_COST_BPS


# =====================================================================
# Grids: dense 5x5 neighborhoods around each winning parameter pair
# (winners from results/shortlist.csv), holding other parameters fixed.
# =====================================================================

FINALISTS = [
    {
        "strategy_name": "sma_crossover",
        "mode": "long_only",
        "axis_1": ("fast", [40, 45, 50, 55, 60]),
        "axis_2": ("slow", [180, 190, 200, 210, 220]),
        "fixed": {},
    },
    {
        "strategy_name": "macd_crossover",
        "mode": "long_only",
        "axis_1": ("fast", [14, 17, 19, 21, 24]),
        "axis_2": ("slow", [34, 37, 39, 41, 44]),
        "fixed": {"signal": 9},
    },
    {
        "strategy_name": "rsi_mean_reversion",
        "mode": "long_only",
        "axis_1": ("window", [14, 17, 21, 25, 28]),
        "axis_2": ("entry", [30, 35, 40, 45, 49]),
        "fixed": {"exit": 50, "short_entry": 60},
    },
    {
        "strategy_name": "bollinger_mean_reversion",
        "mode": "long_only",
        "axis_1": ("window", [14, 17, 20, 23, 26]),
        "axis_2": ("entry", [-2.5, -2.0, -1.5, -1.0, -0.5]),
        "fixed": {"num_std": 1.5, "exit": 0.0, "short_entry": 1.5},
    },
    {
        "strategy_name": "mfi_mean_reversion",
        "mode": "long_only",
        "axis_1": ("window", [10, 12, 14, 17, 21]),
        "axis_2": ("entry", [20, 25, 30, 35, 40]),
        "fixed": {"exit": 50, "short_entry": 70},
    },
]


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def run_grid(
    finalist: dict,
    df: pd.DataFrame,
    yahoo_prices: pd.DataFrame,
    yahoo_returns: pd.DataFrame,
) -> pd.DataFrame:

    strategy_name = finalist["strategy_name"]
    mode = finalist["mode"]
    axis_1_name, axis_1_values = finalist["axis_1"]
    axis_2_name, axis_2_values = finalist["axis_2"]
    fixed = finalist["fixed"]

    rows = []

    for v1 in axis_1_values:
        for v2 in axis_2_values:

            parameters = dict(fixed)
            parameters[axis_1_name] = v1
            parameters[axis_2_name] = v2

            try:
                signal_df = generate_strategy_signal(
                    df, strategy_name=strategy_name, parameters=parameters, mode=mode
                )

                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_backtest(
                        signal_df,
                        cost_bps=COST_BPS,
                        strategy_id=f"{strategy_name}__grid",
                        yahoo_prices=yahoo_prices,
                        yahoo_returns=yahoo_returns,
                    )

                sharpe = result.metrics["sharpe"]

            except ValueError:
                sharpe = float("nan")

            rows.append({axis_1_name: v1, axis_2_name: v2, "sharpe": sharpe})

    return pd.DataFrame(rows)


def plot_heatmap(pivot: pd.DataFrame, strategy_name: str, axis_1_name: str, axis_2_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(axis_2_name)
    ax.set_ylabel(axis_1_name)
    ax.set_title(f"{strategy_name} (long_only) in-sample Sharpe — dense local grid")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"param_stability_{strategy_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print_section("DENSE PARAMETER-STABILITY GRIDS")

    df = pd.read_parquet(FEATURE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["source_ticker"] = df["source_ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "source_ticker"]).reset_index(drop=True)

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for finalist in FINALISTS:

        strategy_name = finalist["strategy_name"]
        axis_1_name, _ = finalist["axis_1"]
        axis_2_name, _ = finalist["axis_2"]

        print_section(f"GRID: {strategy_name}")

        grid_df = run_grid(finalist, df, yahoo_prices, yahoo_returns)

        pivot = grid_df.pivot(index=axis_1_name, columns=axis_2_name, values="sharpe")
        print(pivot.to_string())

        pivot.to_csv(OUTPUT_DIR / f"param_stability_{strategy_name}_dense.csv")
        plot_heatmap(pivot, strategy_name, axis_1_name, axis_2_name)

        best = grid_df.loc[grid_df["sharpe"].idxmax()]
        median_sharpe = grid_df["sharpe"].median()

        print(
            f"  Grid best: {axis_1_name}={best[axis_1_name]}, "
            f"{axis_2_name}={best[axis_2_name]}, Sharpe={best['sharpe']:.4f}"
        )
        print(f"  Grid median Sharpe: {median_sharpe:.4f}")
        print(
            f"  Spread (max-min): {grid_df['sharpe'].max() - grid_df['sharpe'].min():.4f}"
        )

    print_section("DONE")
    print(f"Saved dense grids + charts to {OUTPUT_DIR} / {CHARTS_DIR}")


if __name__ == "__main__":
    main()
