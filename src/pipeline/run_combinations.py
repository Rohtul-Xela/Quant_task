"""
Phase 3 — strategy combinations.

Three pairs (SMA+RSI, SMA+Donchian, RSI+Donchian), each using the
Phase 1 shortlist's top long_only parameters for that family — no new
parameter search. Each pair is combined three ways (AND / OR /
weighted_vote, see src/walkforward/combos.py for the exact rules) and
run through the walk-forward harness's OOS-only stitching (no IS
re-optimization: the legs' parameters are already fixed, so there is
nothing to re-optimize per window — only the combination *method*
varies).

weighted_vote weights 0.6 toward the first-named leg in each pair
(SMA in SMA+RSI/SMA+Donchian, RSI in RSI+Donchian) — the generally
stronger family in the Phase 1 in-sample sweep (both SMA and RSI
comfortably out-ranked Donchian; SMA is named first in SMA+RSI as a
coin-flip between two similarly strong legs).

Two regime-filter pairs, in addition to the AND/OR/weighted_vote grid
above: SMA gated by an ADX trending-market filter (classic pairing —
trend-following should only fire during a confirmed trend), and RSI
mean-reversion gated by a low-volatility-regime filter (mean reversion
classically works better in calm markets). These use a base trading
signal + a regime gate (not two trading legs), so they are combined
once each rather than crossed with METHODS.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from src.strategy.strategies import (  # noqa: E402
    adx_trend_regime_signal,
    generate_strategy_signal,
    low_volatility_regime_signal,
)
from src.walkforward.combos import combine_signals  # noqa: E402
from src.walkforward.harness import run_fixed_signal_walk_forward  # noqa: E402
from src.walkforward.windows import generate_windows  # noqa: E402


FEATURE_FILE = (
    PROJECT_ROOT / "src" / "data" / "data" / "processed" / "pit_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
EQUITY_DIR = OUTPUT_DIR / "walkforward_equity"
RESULTS_CSV = OUTPUT_DIR / "walkforward_results.csv"

COST_BPS = DEFAULT_COST_BPS

# Shortlisted long_only legs (see src/pipeline/confirm_shortlist.py).
LEGS = {
    "sma": {
        "strategy_name": "sma_crossover",
        "parameters": {"fast": 50, "slow": 200},
        "mode": "long_only",
    },
    "rsi": {
        "strategy_name": "rsi_mean_reversion",
        "parameters": {"entry": 40, "exit": 50, "short_entry": 60, "window": 21},
        "mode": "long_only",
    },
    "donchian": {
        "strategy_name": "donchian_breakout",
        "parameters": {"window": 40},
        "mode": "long_only",
    },
}

PAIRS = [("sma", "rsi"), ("sma", "donchian"), ("rsi", "donchian")]
METHODS = ["and", "or", "weighted_vote"]

REGIME_PAIRS = [
    {
        "combo_name": "sma_adxtrend",
        "base_leg": "sma",
        "gate_function": adx_trend_regime_signal,
        "gate_name": "adx_trend_regime",
    },
    {
        "combo_name": "rsi_lowvol",
        "base_leg": "rsi",
        "gate_function": low_volatility_regime_signal,
        "gate_name": "low_volatility_regime",
    },
]


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    print_section("PHASE 3 — STRATEGY COMBINATIONS")

    df = pd.read_parquet(FEATURE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["source_ticker"] = df["source_ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["date", "source_ticker"]).reset_index(drop=True)

    data_start = df["date"].min()
    data_end = df["date"].max()

    yahoo_prices = load_yahoo_prices()
    yahoo_prices = yahoo_prices[
        ~yahoo_prices["ticker"].isin(EXCLUDED_PRICE_TICKERS)
    ].copy()
    yahoo_returns = build_next_day_returns(yahoo_prices)

    windows = generate_windows(
        data_start, data_end, is_years=5, oos_years=1, step_years=1
    )

    print("Precomputing leg signals...")
    leg_signals = {}
    for leg_key, leg in LEGS.items():
        leg_signals[leg_key] = generate_strategy_signal(
            df,
            strategy_name=leg["strategy_name"],
            parameters=leg["parameters"],
            mode=leg["mode"],
        )
        print(f"  {leg_key}: {leg['strategy_name']} {leg['parameters']}")

    print("Precomputing regime gates...")
    gate_signals = {}
    for regime in REGIME_PAIRS:
        gate_signals[regime["gate_name"]] = regime["gate_function"](df)
        print(f"  {regime['gate_name']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)

    new_rows = []

    for leg_a_key, leg_b_key in PAIRS:
        for method in METHODS:

            line_id = f"combo__{leg_a_key}_{leg_b_key}__{method}"
            print_section(f"COMBO: {line_id}")

            combined = combine_signals(
                leg_signals[leg_a_key],
                leg_signals[leg_b_key],
                method=method,
                weight_a=0.6,
            )

            result = run_fixed_signal_walk_forward(
                line_id=line_id,
                signal_df=combined,
                windows=windows,
                yahoo_prices=yahoo_prices,
                yahoo_returns=yahoo_returns,
                cost_bps=COST_BPS,
            )

            result.stitched_daily.to_parquet(
                EQUITY_DIR / f"{line_id}.parquet", index=False
            )

            if result.stitched_signal is not None:
                result.stitched_signal[
                    ["date", "source_ticker", "yahoo_ticker", "signal"]
                ].to_parquet(EQUITY_DIR / f"{line_id}__signal.parquet", index=False)

            metrics = dict(result.stitched_metrics)

            new_rows.append(
                {
                    "line_id": line_id,
                    "type": "combo",
                    "strategy_name": f"{leg_a_key}+{leg_b_key}",
                    "mode": "long_only",
                    "shortlist_source": (
                        f"{leg_a_key}={LEGS[leg_a_key]['parameters']};"
                        f"{leg_b_key}={LEGS[leg_b_key]['parameters']}"
                    ),
                    "n_windows": len(result.window_table),
                    "combo_method": method,
                    **metrics,
                }
            )

            print(
                f"  Stitched OOS Sharpe: {metrics['sharpe']:.4f}   "
                f"Net return: {metrics['net_return']:.4f}   "
                f"Max DD: {metrics['max_drawdown']:.4f}"
            )

    for regime in REGIME_PAIRS:

        base_leg_key = regime["base_leg"]
        line_id = f"combo__{regime['combo_name']}__regime_filter"
        print_section(f"COMBO: {line_id}")

        combined = combine_signals(
            leg_signals[base_leg_key],
            gate_signals[regime["gate_name"]],
            method="regime_filter",
        )

        result = run_fixed_signal_walk_forward(
            line_id=line_id,
            signal_df=combined,
            windows=windows,
            yahoo_prices=yahoo_prices,
            yahoo_returns=yahoo_returns,
            cost_bps=COST_BPS,
        )

        result.stitched_daily.to_parquet(
            EQUITY_DIR / f"{line_id}.parquet", index=False
        )

        if result.stitched_signal is not None:
            result.stitched_signal[
                ["date", "source_ticker", "yahoo_ticker", "signal"]
            ].to_parquet(EQUITY_DIR / f"{line_id}__signal.parquet", index=False)

        metrics = dict(result.stitched_metrics)

        new_rows.append(
            {
                "line_id": line_id,
                "type": "combo",
                "strategy_name": f"{base_leg_key}+{regime['gate_name']}",
                "mode": "long_only",
                "shortlist_source": (
                    f"{base_leg_key}={LEGS[base_leg_key]['parameters']};"
                    f"gate={regime['gate_name']}"
                ),
                "n_windows": len(result.window_table),
                "combo_method": "regime_filter",
                **metrics,
            }
        )

        print(
            f"  Stitched OOS Sharpe: {metrics['sharpe']:.4f}   "
            f"Net return: {metrics['net_return']:.4f}   "
            f"Max DD: {metrics['max_drawdown']:.4f}"
        )

    print_section("SAVING RESULTS")

    new_df = pd.DataFrame(new_rows)

    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        # Concat and keep the newest row per line_id, so re-running this
        # script updates its own combo rows without duplicating them or
        # disturbing Phase 2's single-strategy rows.
        combined_df = pd.concat([existing, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset="line_id", keep="last")
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values(
        "sharpe", ascending=False, na_position="last"
    )
    combined_df.to_csv(RESULTS_CSV, index=False)

    print(f"Saved: {RESULTS_CSV} ({len(combined_df)} total rows)")

    print_section("SUMMARY (combos, stitched OOS)")
    print(
        new_df[["line_id", "combo_method", "sharpe", "net_return", "max_drawdown"]]
        .sort_values("sharpe", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
