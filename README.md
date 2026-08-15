# Stockhunt Test Task

Point-in-time S&P 500 backtesting research: 21 technical indicators,
rule-based strategies, walk-forward validation, strategy combinations,
a purged/embargoed ML layer, benchmarks, cost sensitivity, and
robustness evidence, over the ~634-ticker research universe from
2008-01-02 through 2026-08-11.

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

**Expected runtime**: Phase 2 (walk-forward, 4 strategy families × 14
rolling windows) and Phase 4 (ML, 2 models × 14 purged/embargoed
folds over ~1M+ row training sets) are the dominant cost. Budget
roughly 30-60 minutes end-to-end on a single machine, most of it in
Phases 2 and 4.

## What each phase produces

| Phase | Script | Output |
|---|---|---|
| 1 | `src/pipeline/confirm_shortlist.py` | `results/shortlist.csv` — the 6 shortlisted in-sample configs, re-verified against the live sweep CSV, plus an equity-curve sanity check |
| 2 | `src/pipeline/run_walkforward.py` | `results/walkforward_results.csv` (rows with `type=single`), `results/walkforward_window_detail.csv`, `results/walkforward_equity/*.parquet` |
| 3 | `src/pipeline/run_combinations.py` | Appends `type=combo` rows to `results/walkforward_results.csv` |
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

- Slicing a strategy's full-history signal to a window makes day 1 of
  every OOS segment look like a from-flat entry to `run_backtest`'s
  turnover calculation (it only sees the rows it's given) — a small,
  documented mechanical artifact of stitching, not a look-ahead leak.
- The Phase 1 shortlist was screened by in-sample Sharpe from the
  existing sweep, not from inside the walk-forward loop itself — a
  stated, deliberate time-constrained compromise (see report).
- Breadth-across-the-universe and the parameter-stability surface are
  the first two things cut under time pressure per the task's own
  contingency plan, if they didn't make it into this run.
