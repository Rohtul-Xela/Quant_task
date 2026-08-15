"""
Phase 8 — charts + return-correlation matrix.

Produces:
  - results/charts/equity_curves.png: stitched OOS equity curves for
    every finalist (single/combo/ML) vs. SPY and the equal-weight
    basket benchmark.
  - results/correlation_matrix.csv + results/charts/correlation_matrix.png:
    pairwise correlation of each finalist's daily OOS RETURN STREAM
    (not raw indicator values).
  - results/charts/ml_feature_importance.png: top permutation-importance
    features per ML model.

Cost-sensitivity and parameter-stability charts are produced directly
by run_cost_sensitivity.py / run_robustness.py alongside their data,
since each is a one-shot output of that phase's own computation.
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

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"
CHARTS_DIR = OUTPUT_DIR / "charts"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_all_equity_curves() -> dict[str, pd.DataFrame]:
    curves = {}
    for path in sorted(EQUITY_DIR.glob("*.parquet")):
        if path.name.endswith("__signal.parquet"):
            continue
        line_id = path.stem
        curves[line_id] = pd.read_parquet(path)[["date", "net_return"]]
    return curves


def plot_equity_curves(curves: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    benchmark_ids = {"benchmark_spy_buy_hold", "benchmark_equal_weight_basket"}

    for line_id, daily in curves.items():
        daily = daily.sort_values("date")
        equity = (1.0 + daily["net_return"]).cumprod()

        is_benchmark = line_id in benchmark_ids
        ax.plot(
            daily["date"],
            equity,
            label=line_id,
            linewidth=2.2 if is_benchmark else 1.2,
            linestyle="--" if is_benchmark else "-",
            alpha=0.95 if is_benchmark else 0.8,
        )

    ax.axhline(1.0, color="gray", linewidth=0.6)
    ax.set_ylabel("Stitched OOS growth of $1")
    ax.set_title("Walk-forward stitched OOS equity curves vs. benchmarks")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)


def build_correlation_matrix(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = {}
    for line_id, daily in curves.items():
        s = daily.set_index("date")["net_return"]
        s = s[~s.index.duplicated(keep="last")]
        series[line_id] = s

    wide = pd.DataFrame(series)
    corr = wide.corr()
    return corr


def plot_correlation_matrix(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title("Daily OOS return-stream correlation")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(path: Path, top_n: int = 15) -> None:
    if not path.exists():
        print(f"  SKIP: {path} not found.")
        return

    importance = pd.read_csv(path)

    models = importance["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        subset = (
            importance[importance["model"] == model]
            .sort_values("importance_mean", ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        ax.barh(subset["feature"], subset["importance_mean"], xerr=subset["importance_std"])
        ax.set_title(model)
        ax.set_xlabel("Permutation importance (ROC AUC drop)")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ml_feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print_section("PHASE 8 — CHARTS + CORRELATION MATRIX")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    curves = load_all_equity_curves()
    print(f"Loaded {len(curves)} equity curves: {sorted(curves)}")

    plot_equity_curves(curves)
    print(f"Saved: {CHARTS_DIR / 'equity_curves.png'}")

    corr = build_correlation_matrix(curves)
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"Saved: {OUTPUT_DIR / 'correlation_matrix.csv'}")

    plot_correlation_matrix(corr)
    print(f"Saved: {CHARTS_DIR / 'correlation_matrix.png'}")

    plot_feature_importance(OUTPUT_DIR / "ml_feature_importance.csv")
    print(f"Saved: {CHARTS_DIR / 'ml_feature_importance.png'}")


if __name__ == "__main__":
    main()
