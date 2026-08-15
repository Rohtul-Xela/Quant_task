# Stockhunt Test Task

Point-in-time S&P 500 backtesting research: 20 technical indicators (7
wired as rule-based strategies spanning trend/momentum/volatility/volume,
the rest available as ML features), walk-forward validation, strategy
combinations (AND/OR/weighted-vote/regime-filter), a purged/embargoed ML
layer, benchmarks, cost sensitivity, and robustness evidence, over the
~634-ticker research universe from 2008-01-02 through 2026-08-11.

## Foundation (built before this phase, ~6-8 hours)

The PIT data/backtest foundation this build extends was hand-built, not
generated: point-in-time S&P 500 membership reconstruction, Yahoo price
download and ticker mapping (including renamed/delisted securities),
corporate-action and OHLC-consistency validation, and investigation of
missing/corrupted price histories. Several decisions here directly protect
the no-look-ahead and cost-realism guarantees the rest of this project
relies on:

- A fixed 2026-08-11 research cutoff (`RESEARCH_END_DATE`), so raw Yahoo
  data updating after that date can never leak into the research dataset.
- NYSE-session-aware next-day return construction (`build_next_day_returns`
  in `src/backtest/backtest.py`), including correct handling of extended
  exchange closures (e.g. Hurricane Sandy, Oct 2012) — not a naive
  calendar-day shift.
- `EXCLUDED_PRICE_TICKERS`: known-corrupted price histories excluded by
  name, documented, not silently dropped.
- Extreme returns are flagged and retained, never clipped — preserving
  genuine tail events rather than making the data look tamer than it was.
- Portfolio construction treats a signal without a valid next-session
  return as untradable, not silently zero — no fabricated fills.

This foundation was validated with an initial 34-configuration in-sample
sweep (long-only and long/short, 5bps costs, `run_sweep.py`) before any of
the work described in the rest of this README began.

## One-command run

```bash
pip install -r requirements.txt
python run_all.py
```

This assumes the PIT data pipeline has already been run once (the
`src/data/data/processed/pit_features.parquet` and
`strategy_sweep_results.csv` files exist — these are the pre-existing
data/backtest foundation this build was handed, not part of this
task's own scope). If they don't exist yet, run the data pipeline
first: `download_all.py` → `build_pit_dataset.py` → `build_features.py`
→ `run_sweep.py` (see `src/data/` and `src/features/`).

`run_all.py` runs each phase as its own subprocess, in order, and
stops at the first failure (every later phase depends on an earlier
phase's output files). Run a subset with:

```bash
python run_all.py --only run_benchmarks,run_cost_sensitivity
```

Each phase is also independently runnable, e.g.
`python -m src.pipeline.run_walkforward`.

**Expected runtime**: Phase 2 (walk-forward, 8 strategy lines × 14
rolling windows) and Phase 4 (ML, 2 models × 14 purged/embargoed folds
over ~1M+ row training sets) are the dominant cost. Budget roughly
45-75 minutes end-to-end on a single machine. (`strategy_sweep_results.csv` is gitignored — regenerate it with
`run_sweep.py` first if it's missing, see above; that adds ~25-30 more
minutes for its 66 configs.)

## What each phase produces

| Phase | Script | Output |
|---|---|---|
| 1 | `src/pipeline/confirm_shortlist.py` | `results/shortlist.csv` — 6 curated + 4 programmatically-selected (best-in-sample-long-only per new indicator family) shortlisted configs, re-verified against the live sweep CSV, plus an equity-curve sanity check |
| 2 | `src/pipeline/run_walkforward.py` | `results/walkforward_results.csv` (rows with `type=single`, 8 distinct lines), `results/walkforward_window_detail.csv`, `results/walkforward_equity/*.parquet` |
| 3 | `src/pipeline/run_combinations.py` | Appends `type=combo` rows to `results/walkforward_results.csv` — 3 pairs × {AND, OR, weighted-vote} plus 2 regime-filter pairs |
| 4 | `src/pipeline/run_ml.py` | `results/ml_results.csv`, `results/ml_feature_importance.csv`, `results/ml_*_folds.csv` |
| 5 | `src/pipeline/run_benchmarks.py` | `results/benchmark_results.csv` (SPY buy-hold, equal-weight basket) |
| 6 | `src/pipeline/run_cost_sensitivity.py` | `results/cost_sensitivity.csv`, `results/charts/cost_sensitivity.png` |
| 7 | `src/pipeline/run_robustness.py` | `results/robustness_bootstrap.csv`, `results/param_stability_sma_long_only.csv`, `results/breadth_summary.csv`, charts |
| 8 | `src/pipeline/build_charts.py` | `results/correlation_matrix.csv`, `results/charts/*.png` |

## Column glossary (results/walkforward_results.csv and results/ml_results.csv)

- `line_id`: unique strategy identifier. Rule-based lines are named
  `{family}__{mode}`; combos are `combo__{leg_a}_{leg_b}__{method}`;
  ML lines are `ml_{model_library}`.
- `type`: `single` / `combo` / `ml`.
- `sharpe`, `sortino`, `net_return`, `max_drawdown`: computed on the
  **stitched out-of-sample** daily return series only (never
  in-sample), annualized with `sqrt(252)`.
- `months_in_profit` / `months_count` / `months_in_profit_pct`: monthly
  P&L breakdown of the stitched OOS curve.
- `avg_wf_efficiency` (single lines only): mean of
  `OOS Sharpe / IS Sharpe` across windows, where the IS Sharpe is
  positive (undefined otherwise — reported as NaN, not zero).
- `avg_rank_corr_is_vs_oos` (single lines only): mean Spearman
  correlation, across windows, between each window's in-sample
  parameter ranking and its out-of-sample ranking — a ranking-stability
  diagnostic, not a performance number.
- `cost_bps`: transaction cost applied (5 bps/side everywhere except
  the Phase 6 cost-sensitivity sweep, which is the only place 0 bps
  appears).

## Repo layout added by this build

```
src/walkforward/   window generation, walk-forward harness, signal combination
src/ml/            feature/label dataset, purged+embargoed CV, models, trading rule
src/evaluate/      benchmarks, cost sensitivity, bootstrap/robustness
src/pipeline/      one script per phase, orchestrated by run_all.py
tests/             pytest: window correctness, purge/embargo boundaries,
                   combo truth table, no-look-ahead regression guard
results/           all CSVs, parquet equity curves, charts (generated, gitignored-worthy)
reports/report.md  the final write-up
```

## Known limitations (see `reports/report.md` for the full list)

- **7 of the 20 available indicators have rule-based trading strategies**
  (`sma_crossover`, `donchian_breakout`, `macd_crossover` — trend;
  `rsi_mean_reversion`, `stochastic_crossover` — momentum;
  `bollinger_mean_reversion` — volatility; `mfi_mean_reversion` — volume).
  All four required families are represented, but not all 20 indicators
  have rules — the remaining 13 are still used as ML features (Phase 4).
- Slicing a strategy's full-history signal to a window makes day 1 of
  every OOS segment look like a from-flat entry to `run_backtest`'s
  turnover calculation (it only sees the rows it's given) — a small,
  documented mechanical artifact of stitching, not a look-ahead leak.
- The Phase 1 shortlist was screened by in-sample Sharpe from the
  sweep, not from inside the walk-forward loop itself — a stated,
  deliberate time-constrained compromise (see report §2, §10 — the
  in-sample winner, Bollinger, actually lost to SMA out-of-sample).
- Breadth is computed for every tested strategy
  (`results/breadth_summary.csv`, 21 rows) and parameter-stability for
  the 5 finalists with genuine positive Sharpe — dense 5x5 local grids
  (`src/pipeline/run_param_stability.py`), all showing real plateaus.
  Survivorship-bias direction is quantified
  (`src/pipeline/quantify_survivorship_bias.py`,
  `results/survivorship_bias_quantification.csv`): using today's S&P
  500 list instead of the PIT-reconstructed universe overstates Sharpe
  by +0.084 and net return by +108pp.
- The stateful mean-reversion strategies (RSI, MFI, Bollinger) use
  per-ticker Python loops for their entry/exit state machine, not
  vectorized — correct but slow; the dense parameter-stability grids
  for Bollinger/MFI took the bulk of that phase's runtime for this
  reason. A real performance gap, not yet addressed.
