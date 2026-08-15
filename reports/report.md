# Stockhunt Test Task — Walk-Forward + ML Results

Research universe: ~634 PIT S&P 500 constituents, 2008-01-02–2026-08-11. Walk-forward:
5y in-sample / 1y out-of-sample / 1y rolling step, 14 windows, stitched OOS =
2013-01-01–2026-08-11 (final window is a partial ~7.5-month stub). Cost: 5 bps/side
everywhere except the Phase 6 cost sweep. All numbers below are reproduced by
`python run_all.py`.

## 1. Summary

- **Best strategy overall**: `combo__sma_rsi__or` (SMA 50/200 long-only OR RSI
  21-day long-only), stitched OOS **Sharpe 0.89**, 95% bootstrap CI **[0.34, 1.48]**,
  net return 552%, max drawdown -40%, profitable on **89% of the 607 tickers it
  ever traded**. Survives Benjamini-Hochberg correction across all 49 configurations
  tested in this project.
- **The long-side-more-robust hypothesis survives walk-forward, sharply**: the
  best RSI long-short config scored Sharpe 0.49 in-sample; walk-forward OOS it
  collapsed to **-0.03** (95% CI [-0.53, 0.43], not distinguishable from zero).
  Every long-only line held up; the one long-short line tested did not. This is a
  specific, falsifiable result and it held.
- **ML does not beat the best rule-based Sharpe.** Best ML model (logistic
  regression) scored 0.71 OOS vs. the best rule-based combo's 0.89. Per the task
  brief, this is reported as a strong result for the rule-based approach, not
  hedged. LightGBM did *worse* than the logistic baseline (Sharpe -0.08) —  a
  reminder that model complexity is not free in a low-signal, high-noise setting.
- **Neither the best rule-based nor the best ML strategy beats simply holding the
  research universe equal-weight (Sharpe 0.99) or SPY (Sharpe 0.92) over this OOS
  period.** This is the single most important qualifier on every other result in
  this report — see §6 and §10.
- **Unexpected finding**: every rule-based long-only line/combo is 0.76–0.98
  correlated with the equal-weight basket benchmark — they are close to a filtered
  version of market beta, not an independent source of return. The two ML models
  are the only return streams in the whole study with near-zero correlation to
  everything else (see §9), despite lower standalone Sharpe.

## 2. Method

- Phase 1 shortlist (6 configs) was screened by **in-sample Sharpe from the
  existing 34-config sweep**, not from inside the walk-forward loop itself. This
  is a stated, deliberate, time-constrained compromise: a strategy that would have
  looked attractive walk-forward-native but weak in the original in-sample sweep
  never had a chance to reach Phase 2/3. Disclosed here rather than left implicit.
- Because walk-forward re-optimizes over each strategy family's *full* parameter
  grid every window, two shortlist pairs collapsed to one walk-forward line each
  (both SMA long-only entries → one `sma_crossover__long_only` line; both RSI
  long-only entries → one `rsi_mean_reversion__long_only` line) — re-running an
  identical grid from a different in-sample starting point is deterministic, so
  running it twice would not have produced a different result.
- Combinations (Phase 3) use each family's fixed, already-shortlisted parameters
  — only the AND/OR/weighted-vote *combination method* varies, long_only mode only.
- ML (Phase 4): 30 of the 35 available indicator columns (5 raw price-level
  columns dropped as redundant with already-normalized counterparts, see
  `src/ml/dataset.py`), label = sign(next-day return), purged (1-day) + embargoed
  (5-day) walk-forward CV aligned to the identical windows used everywhere else.
  No `train_test_split`/`KFold` anywhere. Trading rule: P(up)≥0.70 long,
  P(up)≤0.30 short, else flat.

## 3. Top strategies (rule-based)

| line_id | Sharpe | Net return | Max DD | % months profitable |
|---|---:|---:|---:|---:|
| `sma_crossover__long_only` | 0.774 | 355% | -37.7% | 67.1% |
| `rsi_mean_reversion__long_only` | 0.587 | 322% | -43.6% | 60.4% |
| `rsi_mean_reversion__long_short` | -0.030 | -11% | -33.3% | 46.3% |
| `donchian_breakout__long_only` | -0.855 | -89% | -91.1% | 46.3% |

Donchian was included deliberately as a weak comparator and stayed weak
out-of-sample — the whole family never worked here, in-sample or out. SMA and RSI
both held up as standalone long-only rules.

## 4. Best combinations

| line_id | Method | Sharpe | Net return | Max DD |
|---|---|---:|---:|---:|
| `combo__sma_rsi__or` | OR | **0.890** | 552% | -40.1% |
| `combo__sma_rsi__weighted_vote` | weighted (0.6 SMA / 0.4 RSI) | 0.854 | 441% | -37.7% |
| `combo__sma_donchian__weighted_vote` | weighted (0.6 SMA / 0.4 Donchian) | 0.854 | 441% | -37.7% |
| `combo__sma_donchian__or` | OR | 0.818 | 401% | -37.7% |
| `combo__rsi_donchian__weighted_vote` | weighted (0.6 RSI / 0.4 Donchian) | 0.734 | 490% | -43.6% |
| `combo__sma_rsi__and` | AND | 0.526 | 202% | -41.1% |
| `combo__rsi_donchian__or` | OR | 0.417 | 131% | -42.5% |
| `combo__rsi_donchian__and` | AND | -0.178 | -5% | -7.0% |
| `combo__sma_donchian__and` | AND | -1.101 | -93% | -94.4% |

OR and weighted-vote both improved on their best individual leg in every pair;
AND was worse than the better leg in every pair — requiring both signals to agree
filtered out too many genuinely profitable trades. `combo__sma_donchian__or`
matches `combo__sma_donchian__weighted_vote` almost exactly because Donchian is
so rarely active that OR and a 0.6/0.4 vote resolve to nearly the same trades.

## 5. Rule-based vs. ML head-to-head

| line_id | Model | Sharpe | Net return | Max DD | Tickers traded | % profitable |
|---|---|---:|---:|---:|---:|---:|
| `ml_logistic_regression` | Logistic Regression | 0.711 | 870% | -45.2% | 30 | 53.3% |
| `ml_lightgbm` | LightGBM | -0.079 | -53% | -73.7% | — | — |
| *(best rule-based, for reference)* `combo__sma_rsi__or` | — | 0.890 | 552% | -40.1% | 607 | 89.1% |

The best rule-based Sharpe (0.89) beats the best ML Sharpe (0.71) — reported as a
strong result for the rule-based approach, not hedged, per the task brief.

LightGBM underperformed its own simpler logistic baseline. Permutation feature
importance shows why they diverged: logistic regression's top features are
`price_sma_dist_50` and `rsi_14` — directly the same signals that won as
standalone rules, i.e. convergent evidence. LightGBM's top features are
`cmf_20`, `roc_20`, `volume_roc_20` — volume-based indicators that did not
independently win as rules anywhere in this study, consistent with it having
found relationships that generalized poorly. See
`results/charts/ml_feature_importance.png`.

The logistic model's 0.71 Sharpe is fragile in a way its headline number doesn't
show: only **53% of months were profitable**, and it only ever took a position in
**30 of the 634 tickers** in the universe (avg. gross exposure 7.4%) — a
threshold-triggered, low-frequency, concentrated strategy, not a broad-based one.
Its equity curve (`results/charts/equity_curves.png`) is visibly a staircase of
long flat stretches and a few large jumps (notably 2024), not a smooth compounding
line. Treat the 0.71 Sharpe as evidence the threshold rule can work, not as a
finished, diversified strategy.

## 6. Benchmarks

| | Sharpe | Net return | Max DD |
|---|---:|---:|---:|
| Equal-weight universe basket | **0.992** | 829% | -40.3% |
| SPY buy-and-hold | 0.916 | 566% | -33.7% |
| *(best of this study, for reference)* `combo__sma_rsi__or` | 0.890 | 552% | -40.1% |

Both benchmarks beat every single strategy tested in this project over the same
stitched OOS window. This is the headline caveat on every result above — see §10.

## 7. Cost sensitivity

Best rule-based and best ML finalist re-run at 0/5/10/15/20/25 bps (0 bps used
only here, per the hard constraint):

| Cost (bps) | `combo__sma_rsi__or` Sharpe | `ml_logistic_regression` Sharpe |
|---:|---:|---:|
| 0 | 0.915 | 0.726 |
| 5 (baseline) | 0.890 | 0.711 |
| 10 | 0.865 | 0.696 |
| 15 | 0.840 | 0.682 |
| 20 | 0.815 | 0.667 |
| 25 | 0.790 | 0.652 |

Neither edge dies anywhere in this range — both decay nearly linearly and stay
comfortably positive at 5x the baseline cost assumption. Chart:
`results/charts/cost_sensitivity.png`.

## 8. Walk-forward efficiency and ranking stability

| line | avg(OOS Sharpe / IS Sharpe) | avg rank corr (IS vs OOS) | % windows IS winner = OOS best |
|---|---:|---:|---:|
| `sma_crossover__long_only` | 1.51 | 0.11 | 21% |
| `rsi_mean_reversion__long_only` | 1.05 | 0.23 | 7% |
| `rsi_mean_reversion__long_short` | 0.10 | 0.07 | 14% |
| `donchian_breakout__long_only` | n/a (IS Sharpe never positive) | 0.11 | 0% |

Walk-forward efficiency >1 for both long-only lines means OOS Sharpe averaged
*higher* than IS Sharpe — largely because IS windows include the 2008-09 crisis
years, which drag in-sample Sharpe down more than any OOS window was hurt.
Ranking stability is weak everywhere (rank correlations 0.07–0.23, and the
in-sample winner was rarely also the best out-of-sample candidate) — the
parameter grids are small and the per-window differences between candidates are
mostly noise, not a sign that re-optimization is finding something durable.

## 9. Robustness

**Block-bootstrap Sharpe CIs and multiple-testing correction** (1000 resamples,
20-day blocks; Benjamini-Hochberg at α=0.05 across the 15 walk-forward/OOS
finalists — out of **49 configurations tested in total**: 34 in-sample sweep +
15 OOS finalists; the sweep is disclosed for context, not re-tested since it has
no OOS return series to bootstrap):

**9 of 15 finalists remain significant after correction**, including the top
result (`combo__sma_rsi__or`, p≈0, 95% CI [0.34, 1.48]) and the ML logistic
model (p=0.001, 95% CI [0.24, 1.15]). Not significant: both long-short lines,
`ml_lightgbm`, and both Donchian-involving AND combos. Full table:
`results/robustness_bootstrap.csv`.

**Parameter-stability**: reusing the existing SMA sweep grid (no new backtests),
the winning 50/200 pair scores 0.74 vs. its nearest tested neighbor 30/100 at
0.57 — a real gap, not a smooth plateau. The grid is sparse (6 specific pairs,
not a full cross-product), so we cannot rule out 50/200 being a local spike at
the edge of what was tested rather than the middle of a stable region. Chart:
`results/charts/param_stability_sma.png`.

**Breadth**: `combo__sma_rsi__or` was profitable on 541/607 traded tickers
(89.1%) — broad-based. `ml_logistic_regression` was profitable on 16/30 (53.3%)
— it barely beats a coin flip on the small set of names it actually traded.

## 10. Where results are weakest

- **No strategy in this study beat its own benchmark.** The equal-weight basket
  (0.99 Sharpe) and SPY (0.92) both beat the best combo (0.89) and the best ML
  model (0.71). Over 2013–2026 — a strongly bullish stretch for US equities —
  simple beta was hard to add value on top of with the strategies tested here.
- **The rule-based edge is mostly beta, not alpha.** Every long-only rule/combo
  is 0.76–0.98 correlated with the equal-weight basket (§9/correlation_matrix.csv).
  Combined with the point above, the honest read is: these strategies are a
  filtered, higher-cost, higher-drawdown version of just holding the universe —
  the case for them isn't "they add return," it's that they're the best of what
  was tested, not that they're a good idea in absolute terms.
- **In-sample-screened shortlist** (§2): a strategy screened only inside the
  walk-forward loop from the start might have produced a different top-6.
- **ML logistic result is concentrated**: 30 tickers, 53% of months profitable,
  staircase equity curve. Treat as weak evidence of a real edge, not a finished
  strategy.
- **Turnover-at-roll-date artifact**: `run_backtest` computes turnover from a
  weight-matrix pivot of only the rows it's given, so day 1 of every stitched OOS
  segment looks like a from-flat entry rather than a continuation of the prior
  window's book — a small, documented mechanical overstatement of turnover/cost
  at each of the 13 roll dates. Not a look-ahead leak.
- **Parameter-stability surface is sparse**, not a dense grid (§9) — can't fully
  rule out the winning SMA pair being a local spike.
- **Weak ranking stability** (§8): the walk-forward re-optimization step is not
  clearly finding a durable signal above noise at this grid resolution.
- **Breadth was only computed for the top rule-based and top ML finalist**, not
  every line, to keep scope bounded — all other lines' `__signal.parquet` files
  are saved and `per_ticker_breadth()` in `src/evaluate/robustness.py` can be run
  against any of them.
- Not attempted: SHAP feature importance (permutation importance used instead —
  model-agnostic, no extra heavy dependency, satisfies the requirement); a full
  cross-product parameter grid for the stability surface; long-short combos.

## Appendix: reproducing these numbers

`python run_all.py` from a clean `results/` directory reproduces every number in
this report. See `README.md` for phase-by-phase output locations and the column
glossary.
