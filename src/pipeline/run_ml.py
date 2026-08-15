"""
Phase 4 — ML pipeline runner.

Trains LogisticRegression and a gradient-boosted model (LightGBM,
falling back to XGBoost) on purged/embargoed walk-forward folds
(the SAME rolling windows as Phase 2/3), converts each fold's P(up)
predictions into a signal via the 0.70/0.30 threshold rule, and runs
that signal through the identical `run_backtest` + cost model as
every other strategy in this project. Stitches OOS segments into one
continuous curve per model -> results/ml_results.csv.

No train_test_split / KFold anywhere in this script (see src/ml/cv.py).
The per-model/per-fold walk-forward loop itself lives in
src/ml/runner.py so Phase 6 (cost sensitivity) can reuse it exactly
rather than re-deriving it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

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
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"

RESULTS_CSV = OUTPUT_DIR / "ml_results.csv"
IMPORTANCE_CSV = OUTPUT_DIR / "ml_feature_importance.csv"

COST_BPS = DEFAULT_COST_BPS

MODEL_LABELS = {
    "logistic_regression": "ml_logistic_regression",
    "gbm": None,  # filled in with the actual library name at runtime
}


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("PHASE 4 — ML PIPELINE")

    try:
        gbm_name = gbm_library_name()
    except ImportError as exc:
        raise SystemExit(f"No gradient-boosted model library available: {exc}")

    MODEL_LABELS["gbm"] = f"ml_{gbm_name}"
    print(f"Gradient-boosted model library: {gbm_name}")

    df = load_feature_frame()
    print(f"Feature frame: {len(df)} rows, {df['source_ticker'].nunique()} tickers")

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    dataset, feature_cols = build_ml_dataset(df, yahoo_returns)

    print(
        f"ML dataset: {len(dataset)} rows, {len(feature_cols)} features "
        f"(dropped 5 raw-level columns, see src/ml/dataset.py docstring)"
    )
    print(f"Label balance: {dataset['label'].mean():.4f} positive")

    data_start = df["date"].min()
    data_end = df["date"].max()

    windows = generate_windows(
        data_start, data_end, is_years=5, oos_years=1, step_years=1
    )

    folds = purged_embargoed_folds(dataset["date"], windows, purge_days=1, embargo_days=5)

    print(f"Generated {len(folds)} purged/embargoed walk-forward folds")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    importance_rows = []

    for model_name in MODEL_BUILDERS:

        label = MODEL_LABELS[model_name]

        print_section(f"MODEL: {label}")

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

        print(result.fold_table.to_string(index=False))

        result.stitched_daily.to_parquet(EQUITY_DIR / f"{label}.parquet", index=False)
        result.fold_table.to_csv(OUTPUT_DIR / f"{label}_folds.csv", index=False)

        if result.stitched_signal is not None:
            result.stitched_signal[
                ["date", "source_ticker", "yahoo_ticker", "signal"]
            ].to_parquet(EQUITY_DIR / f"{label}__signal.parquet", index=False)

        summary_rows.append(
            {
                "line_id": label,
                "type": "ml",
                "model": model_name,
                "n_windows": len(result.fold_table),
                **result.stitched_metrics,
            }
        )

        metrics = result.stitched_metrics
        print(
            f"\n  Stitched OOS Sharpe: {metrics['sharpe']:.4f}   "
            f"Net return: {metrics['net_return']:.4f}   "
            f"Max DD: {metrics['max_drawdown']:.4f}"
        )

        # -------------------------------------------------------------
        # Permutation feature importance on the LAST fold's OOS test
        # set (most recent, largest information set) — descriptive
        # evidence about which indicators the model relies on, not a
        # walk-forward performance number, so computed once per model
        # rather than per fold.
        # -------------------------------------------------------------

        last_window_index = result.fold_table["window_index"].max()
        last_fold = next(f for f in folds if f.window.index == last_window_index)

        train_df = dataset.loc[last_fold.train_mask]
        test_df = dataset.loc[last_fold.test_mask]

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test = test_df[feature_cols]
        y_test = test_df["label"]

        print("  Refitting last fold's model for permutation importance...")

        model, _ = fit_predict_proba(model_name, X_train, y_train, X_test)

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
        )

        for i, feature in enumerate(feature_cols):
            importance_rows.append(
                {
                    "model": label,
                    "feature": feature,
                    "importance_mean": perm.importances_mean[i],
                    "importance_std": perm.importances_std[i],
                }
            )

    print_section("SAVING RESULTS")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "sharpe", ascending=False, na_position="last"
    )
    summary_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved: {RESULTS_CSV}")

    importance_df = pd.DataFrame(importance_rows).sort_values(
        ["model", "importance_mean"], ascending=[True, False]
    )
    importance_df.to_csv(IMPORTANCE_CSV, index=False)
    print(f"Saved: {IMPORTANCE_CSV}")

    print_section("SUMMARY (stitched OOS)")
    display_cols = ["line_id", "model", "n_windows", "sharpe", "net_return", "max_drawdown"]
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
