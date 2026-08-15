"""
Phase 7 — robustness runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
)
from src.evaluate.robustness import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    breadth_summary,
    parameter_stability_pivot,
    per_ticker_breadth,
)

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"
CHARTS_DIR = OUTPUT_DIR / "charts"

SWEEP_RESULTS_CSV = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "strategy_sweep_results.csv"
)

N_SWEEP_CONFIGS = 34  # Phase 1 in-sample sweep — see confirm_shortlist.py


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_finalists() -> pd.DataFrame:
    frames = []

    wf_path = OUTPUT_DIR / "walkforward_results.csv"
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        frames.append(wf[["line_id", "sharpe"]])

    ml_path = OUTPUT_DIR / "ml_results.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        frames.append(ml[["line_id", "sharpe"]])

    if not frames:
        raise FileNotFoundError("No walkforward_results.csv or ml_results.csv found.")

    return pd.concat(frames, ignore_index=True).dropna(subset=["sharpe"])


def main() -> None:
    print_section("PHASE 7 — ROBUSTNESS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Bootstrap Sharpe CI + multiple-testing correction
    # -----------------------------------------------------------------

    print_section("BLOCK BOOTSTRAP SHARPE CI + MULTIPLE-TESTING CORRECTION")

    finalists = load_finalists()
    n_walkforward_tested = len(finalists)
    total_configs_tested = N_SWEEP_CONFIGS + n_walkforward_tested

    print(
        f"Configurations tested: {N_SWEEP_CONFIGS} in-sample sweep + "
        f"{n_walkforward_tested} walk-forward/OOS finalists (single + combo + ML) "
        f"= {total_configs_tested} total. BH correction below is computed over "
        f"the {n_walkforward_tested} finalists that have an actual OOS bootstrap "
        f"p-value; the in-sample sweep is disclosed for context, not re-tested."
    )

    bootstrap_rows = []

    for _, row in finalists.iterrows():
        line_id = row["line_id"]
        equity_path = EQUITY_DIR / f"{line_id}.parquet"

        if not equity_path.exists():
            continue

        daily = pd.read_parquet(equity_path)
        ci = bootstrap_sharpe_ci(daily["net_return"].to_numpy())
        ci["line_id"] = line_id
        bootstrap_rows.append(ci)

        print(
            f"  {line_id:<45s} Sharpe={ci['point_sharpe']:.4f} "
            f"95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
            f"p(Sharpe<=0)={ci['p_value_sharpe_le_0']:.4f}"
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows).set_index("line_id")

    bh = benjamini_hochberg(bootstrap_df["p_value_sharpe_le_0"], alpha=0.05)
    bootstrap_df = bootstrap_df.join(bh[["bh_adjusted_p", "significant_at_alpha"]])
    bootstrap_df = bootstrap_df.reset_index().sort_values(
        "point_sharpe", ascending=False
    )

    bootstrap_df.to_csv(OUTPUT_DIR / "robustness_bootstrap.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'robustness_bootstrap.csv'}")

    # -----------------------------------------------------------------
    # Parameter-stability plateau (reuses existing sweep grid)
    # -----------------------------------------------------------------

    print_section("PARAMETER-STABILITY PLATEAU (SMA, long_only)")

    sweep_df = pd.read_csv(SWEEP_RESULTS_CSV)
    sma_pivot = parameter_stability_pivot(sweep_df, "sma_crossover", "long_only")
    sma_pivot.to_csv(OUTPUT_DIR / "param_stability_sma_long_only.csv")

    print(sma_pivot.to_string())

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(sma_pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(sma_pivot.columns)))
    ax.set_xticklabels(sma_pivot.columns)
    ax.set_yticks(range(len(sma_pivot.index)))
    ax.set_yticklabels(sma_pivot.index)
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    ax.set_title("SMA crossover (long_only) in-sample Sharpe")
    for i in range(sma_pivot.shape[0]):
        for j in range(sma_pivot.shape[1]):
            val = sma_pivot.to_numpy()[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "param_stability_sma.png", dpi=150)
    plt.close(fig)

    print(f"Saved: {CHARTS_DIR / 'param_stability_sma.png'}")

    # -----------------------------------------------------------------
    # Breadth (best rule-based + best ML finalist)
    # -----------------------------------------------------------------

    print_section("BREADTH")

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    wf_df = pd.read_csv(OUTPUT_DIR / "walkforward_results.csv").dropna(subset=["sharpe"])
    ml_df = pd.read_csv(OUTPUT_DIR / "ml_results.csv").dropna(subset=["sharpe"])

    best_rule_id = wf_df.sort_values("sharpe", ascending=False).iloc[0]["line_id"]
    best_ml_id = ml_df.sort_values("sharpe", ascending=False).iloc[0]["line_id"]

    breadth_rows = []

    for line_id in [best_rule_id, best_ml_id]:
        signal_path = EQUITY_DIR / f"{line_id}__signal.parquet"

        if not signal_path.exists():
            print(f"  SKIP {line_id}: no stitched signal parquet found.")
            continue

        signal_df = pd.read_parquet(signal_path)
        traded = per_ticker_breadth(signal_df, yahoo_returns)
        summary = breadth_summary(traded)
        summary["line_id"] = line_id
        breadth_rows.append(summary)

        traded.to_csv(OUTPUT_DIR / f"breadth_{line_id}.csv", index=False)

        print(
            f"  {line_id}: {summary['n_profitable']}/{summary['n_traded']} "
            f"tickers profitable ({summary['breadth_pct']:.1%})"
        )

    pd.DataFrame(breadth_rows).to_csv(OUTPUT_DIR / "breadth_summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'breadth_summary.csv'}")


if __name__ == "__main__":
    main()
