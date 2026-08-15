"""
Phase 6 — cost sensitivity runner.

Picks the best rule-based finalist (single or combo, by stitched OOS
Sharpe) and the best ML finalist from Phase 2/3/4's results, and runs
each through the cost sensitivity sweep at 0/5/10/15/20/25 bps.
"""

from __future__ import annotations

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
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
)
from src.evaluate.cost_sensitivity import cost_sensitivity_sweep  # noqa: E402
from src.walkforward.windows import generate_windows  # noqa: E402


FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"
CHARTS_DIR = OUTPUT_DIR / "charts"

RESULTS_CSV = OUTPUT_DIR / "cost_sensitivity.csv"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def pick_best(csv_path: Path, label: str) -> str:
    if not csv_path.exists():
        raise FileNotFoundError(f"{label} results not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["sharpe"].notna()]

    if df.empty:
        raise ValueError(f"No rows with a valid Sharpe in {csv_path}")

    best = df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"Best {label}: {best['line_id']}  (Sharpe={best['sharpe']:.4f})")
    return str(best["line_id"])


def main() -> None:
    print_section("PHASE 6 — COST SENSITIVITY")

    best_rule_id = pick_best(OUTPUT_DIR / "walkforward_results.csv", "rule-based (single+combo)")
    best_ml_id = pick_best(OUTPUT_DIR / "ml_results.csv", "ML model")

    df = pd.read_parquet(FEATURE_FILE, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    data_start = df["date"].min()
    data_end = df["date"].max()

    windows = generate_windows(
        data_start, data_end, is_years=5, oos_years=1, step_years=1
    )

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    all_rows = []

    for line_id in [best_rule_id, best_ml_id]:

        print_section(f"COST SWEEP: {line_id}")

        signal_path = EQUITY_DIR / f"{line_id}__signal.parquet"

        if not signal_path.exists():
            raise FileNotFoundError(
                f"Stitched signal not found for {line_id}: {signal_path}. "
                "Phase 2/3/4 must save `{line_id}__signal.parquet`."
            )

        signal_df = pd.read_parquet(signal_path)
        signal_df["adj_close"] = 1.0  # not used numerically by run_backtest

        sweep_df = cost_sensitivity_sweep(
            line_id, signal_df, windows, yahoo_prices, yahoo_returns
        )

        print(sweep_df.to_string(index=False))

        all_rows.append(sweep_df)

    print_section("SAVING RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    result_df = pd.concat(all_rows, ignore_index=True)
    result_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved: {RESULTS_CSV}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for line_id, group in result_df.groupby("line_id"):
        axes[0].plot(group["cost_bps"], group["sharpe"], marker="o", label=line_id)
        axes[1].plot(group["cost_bps"], group["net_return"], marker="o", label=line_id)

    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xlabel("Cost (bps / side)")
    axes[0].set_ylabel("Stitched OOS Sharpe")
    axes[0].set_title("Sharpe vs. transaction cost")
    axes[0].legend(fontsize=7)

    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("Cost (bps / side)")
    axes[1].set_ylabel("Stitched OOS net return")
    axes[1].set_title("Net return vs. transaction cost")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "cost_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {CHARTS_DIR / 'cost_sensitivity.png'}")


if __name__ == "__main__":
    main()
