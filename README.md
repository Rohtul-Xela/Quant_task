# Stockhunt Test Task

Point-in-time S&P 500 backtesting research: 20 technical indicators (7
wired as rule-based strategies spanning trend/momentum/volatility/volume,
the rest available as ML features), walk-forward validation, strategy
combinations (AND/OR/weighted-vote/regime-filter), a purged/embargoed ML
layer, benchmarks, cost sensitivity, and robustness evidence, over the
~631-ticker research universe from 2008-01-02 through 2026-08-11.

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

That's genuinely it, on a genuinely clean clone. `run_all.py` runs all 16
phases below in order, each as its own subprocess, stopping at the first
failure (every later phase depends on an earlier phase's output files):
phases 1-5 are the PIT data/backtest foundation (download membership,
download prices, build the PIT dataset, build features, run the in-sample
sweep — ~6-8 hours' worth of work the first time, network-bound and
rate-limit-prone since it pulls ~650+ tickers from Yahoo Finance), and
phases 6-16 are the walk-forward/ML/benchmark build, including three
gap-closing analyses cited in the report (~80-120 minutes, dominated by
phase 14's dense parameter-stability grids). The
only input phase 1-5 can't produce itself is
`src/data/data/processed/security_mapping.csv` (hand-verified
renamed/delisted ticker mappings, e.g. `FB -> META`; not regenerable by
any script, see `src/data/security_mapping.py`) — that one file **is**
committed to git specifically so the clone alone provides it.

If the data foundation (phases 1-5's output) already exists on disk —
e.g. this repo's own dev environment — skip straight to phase 6:

```bash
python run_all.py --skip-data
```

Run an arbitrary subset with `--only` (matches by module suffix,
independent of `--skip-data`):

```bash
python run_all.py --only run_benchmarks,run_cost_sensitivity
```

Each phase is also independently runnable, e.g.
`python -m src.pipeline.run_walkforward` or
`python -m src.data.download_all`.

## What each phase produces

| Phase | Script | Output |
|---|---|---|
| 1 | `src/data/membership.py` | `src/data/data/processed/membership_snapshots.parquet`, `historical_tickers.csv` — downloads historical S&P 500 constituent data from a public GitHub CSV and reconstructs PIT snapshots |
| 2 | `src/data/download_all.py` | `src/data/data/raw/yahoo/*.parquet` — daily OHLCV for every ticker phase 1 found |
| 3 | `src/data/build_pit_dataset.py` | `src/data/data/processed/pit_daily.parquet` — phase 1's snapshots + `security_mapping.csv` + phase 2's prices, joined into a daily PIT panel |
| 4 | `src/features/build_features.py` | `src/data/data/processed/pit_features.parquet` — the 20 technical indicators |
| 5 | `src/backtest/run_sweep.py` | `src/data/data/processed/strategy_sweep_results.csv` — 66-config in-sample sweep |
| 6 | `src/pipeline/confirm_shortlist.py` | `results/shortlist.csv` — 6 curated + 4 programmatically-selected (best-in-sample-long-only per new indicator family) shortlisted configs, re-verified against the live sweep CSV, plus an equity-curve sanity check |
| 7 | `src/pipeline/run_walkforward.py` | `results/walkforward_results.csv` (rows with `type=single`, 8 distinct lines), `results/walkforward_window_detail.csv`, `results/walkforward_equity/*.parquet` |
| 8 | `src/pipeline/run_combinations.py` | Appends `type=combo` rows to `results/walkforward_results.csv` — 3 pairs × {AND, OR, weighted-vote} plus 2 regime-filter pairs |
| 9 | `src/pipeline/run_ml.py` | `results/ml_results.csv`, `results/ml_feature_importance.csv`, `results/ml_*_folds.csv` |
| 10 | `src/pipeline/run_benchmarks.py` | `results/benchmark_results.csv` (SPY buy-hold, equal-weight basket) |
| 11 | `src/pipeline/run_cost_sensitivity.py` | `results/cost_sensitivity.csv`, `results/charts/cost_sensitivity.png` |
| 12 | `src/pipeline/run_robustness.py` | `results/robustness_bootstrap.csv`, `results/param_stability_sma_long_only.csv`, `results/breadth_summary.csv`, charts |
| 13 | `src/pipeline/quantify_survivorship_bias.py` | `results/survivorship_bias_quantification.csv` — direction/magnitude of survivorship bias, re-running `combo__sma_rsi__or`'s signal restricted to today's-list-only vs. the full PIT universe |
| 14 | `src/pipeline/run_param_stability.py` | `results/param_stability_{sma_crossover,macd_crossover,rsi_mean_reversion,bollinger_mean_reversion,mfi_mean_reversion}_dense.csv` + charts — dense 5×5 local grids around each of the 5 positive-Sharpe finalists. By far the slowest added phase (30-45 min) — Bollinger/MFI's per-ticker Python loops, not vectorized |
| 15 | `src/pipeline/run_ml_pit_percentile_ablation.py` | `results/ml_pit_percentile_ablation.csv` — feature-representation ablation (§5 of the report), not a replacement for `ml_results.csv` |
| 16 | `src/pipeline/build_charts.py` | `results/correlation_matrix.csv`, `results/charts/*.png` |

**Expected runtime**: phases 1-2 (network-bound Yahoo download) dominate
the data-foundation cost and carry real rate-limit risk on a from-scratch
run. Within the research build, phase 7 (walk-forward, 8 strategy lines ×
14 rolling windows), phase 9 (ML, 2 models × 14 purged/embargoed folds
over ~1M+ row training sets), and phase 14 (dense parameter-stability
grids, 30-45 minutes alone) are the dominant costs — budget roughly
80-120 minutes for phases 6-16 alone once the data foundation exists.

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
  the phase 11 cost-sensitivity sweep, which is the only place 0 bps
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
reports/report.md  the final write-up (source of truth)
reports/report.pdf the same content, rendered — this is the actual 5-page-max deliverable
```

The report itself (`reports/report.md`) is hand-curated analysis, not machine-regenerated from
a template, so it isn't part of `run_all.py`. After editing it, rebuild the PDF with:

```bash
python -m reports.build_report_pdf
```

## Known limitations (see `reports/report.md` for the full list)

- **7 of the 20 available indicators have rule-based trading strategies**
  (`sma_crossover`, `donchian_breakout`, `macd_crossover` — trend;
  `rsi_mean_reversion`, `stochastic_crossover` — momentum;
  `bollinger_mean_reversion` — volatility; `mfi_mean_reversion` — volume).
  All four required families are represented, but not all 20 indicators
  have rules — the remaining 13 are still used as ML features (phase 9).
- Slicing a strategy's full-history signal to a window makes day 1 of
  every OOS segment look like a from-flat entry to `run_backtest`'s
  turnover calculation (it only sees the rows it's given) — a small,
  documented mechanical artifact of stitching, not a look-ahead leak.
- The phase 6 shortlist was screened by in-sample Sharpe from the
  sweep, not from inside the walk-forward loop itself — a stated,
  deliberate time-constrained compromise (see report §2, §10 — the
  in-sample winner, Bollinger, actually lost to SMA out-of-sample).
- Combination strategies (`run_combinations.py`) use each leg's single
  parameter set already chosen by the walk-forward loop; the combination
  itself (which legs, AND/OR/weighted-vote/regime-filter, vote weights,
  regime-filter thresholds) is fixed once and reused across all 14
  windows, not re-optimized per window the way single-strategy parameters
  are. A methodological limitation, not a failure to test the four
  combination types (all four are tested — see report §4).
- Breadth is computed for every tested strategy
  (`results/breadth_summary.csv`, 21 rows) and parameter-stability for
  the 5 finalists with genuine positive Sharpe — dense 5x5 local grids
  (`src/pipeline/run_param_stability.py`), all showing real plateaus.
  Survivorship-bias direction is quantified
  (`src/pipeline/quantify_survivorship_bias.py`,
  `results/survivorship_bias_quantification.csv`): using today's S&P
  500 list instead of the PIT-reconstructed universe overstates Sharpe
  by +0.086 and net return by +110pp.
- The stateful mean-reversion strategies (RSI, MFI, Bollinger) use
  per-ticker Python loops for their entry/exit state machine, not
  vectorized — correct but slow; the dense parameter-stability grids
  for Bollinger/MFI took the bulk of that phase's runtime for this
  reason. A real performance gap, not yet addressed.
