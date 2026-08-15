"""
Ablation: point-in-time cross-sectional percentile-normalized ML features.

Not part of `run_all.py`'s main chain and does not overwrite
`results/ml_results.csv` — this is a documented side experiment, not a
replacement for the verified primary ML result.

Finding: replacing raw indicator values with same-date cross-sectional
percentile ranks (`build_ml_dataset(..., use_percentile_rank=True)`,
see `src/ml/dataset.py`) is methodologically well-motivated — it makes
features comparable across tickers regardless of absolute scale or a
market-wide regime shift — but percentile ranks are uniform on [0, 1]
by construction, with no fat tails. Logistic regression's predicted
P(up) compresses toward 0.5 as a result, and the fixed 0.70/0.30
trading-rule threshold (calibrated implicitly against the raw-feature
probability distribution) stops firing almost entirely. This script
quantifies exactly that, rather than asserting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtest import (  # noqa: E402
    DEFAULT_COST_BPS,
    EXCLUDED_PRICE_TICKERS,
    build_next_day_returns,
    load_yahoo_prices,
)
from src.ml.cv import purged_embargoed_folds  # noqa: E402
from src.ml.dataset import build_ml_dataset, load_feature_frame  # noqa: E402
from src.ml.models import MODEL_BUILDERS, fit_predict_proba, gbm_library_name  # noqa: E402
from src.ml.runner import run_ml_line  # noqa: E402
from src.walkforward.windows import generate_windows  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results"
COST_BPS = DEFAULT_COST_BPS


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("ABLATION — PIT CROSS-SECTIONAL PERCENTILE FEATURES")

    gbm_name = gbm_library_name()
    model_labels = {
        "logistic_regression": "ablation_pct_logistic_regression",
        "gbm": f"ablation_pct_{gbm_name}",
    }

    df = load_feature_frame()

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    dataset, feature_cols = build_ml_dataset(
        df, yahoo_returns, use_percentile_rank=True
    )

    print(f"Dataset: {len(dataset)} rows, {len(feature_cols)} features")
    print(
        "Feature value range check (should be ~[0,1], confirming the "
        f"transform applied): min={dataset[feature_cols].min().min():.4f}, "
        f"max={dataset[feature_cols].max().max():.4f}"
    )

    windows = generate_windows(
        df["date"].min(), df["date"].max(), is_years=5, oos_years=1, step_years=1
    )
    folds = purged_embargoed_folds(dataset["date"], windows, purge_days=1, embargo_days=5)

    summary_rows = []

    for model_name in MODEL_BUILDERS:

        label = model_labels[model_name]
        print_section(f"MODEL: {label}")

        # -----------------------------------------------------------
        # Probability-calibration diagnostic on the first fold: does
        # the predicted P(up) distribution ever reach the 0.70/0.30
        # trading threshold under percentile-ranked features?
        # -----------------------------------------------------------

        first_fold = folds[0]
        train_df = dataset.loc[first_fold.train_mask]
        test_df = dataset.loc[first_fold.test_mask]

        _, proba = fit_predict_proba(
            model_name, train_df[feature_cols], train_df["label"], test_df[feature_cols]
        )

        n_long = int((proba >= 0.70).sum())
        n_short = int((proba <= 0.30).sum())

        print(
            f"  First-fold P(up): min={proba.min():.4f} max={proba.max():.4f} "
            f"mean={proba.mean():.4f}"
        )
        print(
            f"  First-fold trade triggers at 0.70/0.30 threshold: "
            f"{n_long} long, {n_short} short, out of {len(proba)} predictions "
            f"({(n_long + n_short) / len(proba):.4%})"
        )

        # -----------------------------------------------------------
        # Full walk-forward run (same harness as the primary ML
        # pipeline) for the headline Sharpe under this ablation.
        # -----------------------------------------------------------

        result = run_ml_line(
            label=label,
            model_name=model_name,
            dataset=dataset,
            feature_cols=feature_cols,
            folds=folds,
            yahoo_prices=yahoo_prices,
            yahoo_returns=yahoo_returns,
            cost_bps=COST_BPS,
        )

        metrics = result.stitched_metrics
        sharpe = metrics.get("sharpe", np.nan)

        print(
            f"  Stitched OOS Sharpe: {sharpe if pd.notna(sharpe) else 'NaN (no trades)'}   "
            f"Net return: {metrics['net_return']:.4f}"
        )

        summary_rows.append(
            {
                "line_id": label,
                "model": model_name,
                "first_fold_proba_min": proba.min(),
                "first_fold_proba_max": proba.max(),
                "first_fold_trade_trigger_pct": (n_long + n_short) / len(proba),
                **metrics,
            }
        )

    print_section("SAVING RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "ml_pit_percentile_ablation.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {summary_path}")
    print(
        "\nThis is a documented ablation, not a replacement for "
        "results/ml_results.csv — the primary ML result (raw features, "
        "fixed 0.70/0.30 threshold) is unchanged."
    )


if __name__ == "__main__":
    main()
