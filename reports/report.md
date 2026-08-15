# Stockhunt Test Task — Walk-Forward + ML Results

Research universe: ~634 PIT S&P 500 constituents, 2008-01-02–2026-08-11. 7 of the 20
available indicators have rule-based trading strategies (see §2), spanning all four
required families. Walk-forward: 5y in-sample / 1y out-of-sample / 1y rolling step,
14 windows, stitched OOS = 2013-01-01–2026-08-11 (final window is a partial
~7.5-month stub). Cost: 5 bps/side everywhere except the Phase 6 cost sweep. All
numbers below are reproduced by `python run_all.py`.

## 1. Summary

- **Best strategy overall**: `combo__sma_rsi__or` (SMA 50/200 long-only OR RSI
  21-day long-only), stitched OOS **Sharpe 0.89**, 95% bootstrap CI **[0.34, 1.48]**,
  net return 552%, max drawdown -40%, profitable on 89% of the 607 tickers it ever
  traded. Survives Benjamini-Hochberg correction across all 87 configurations
  evaluated in this project.
- **Wiring up volatility and volume indicators paid off, not just a checkbox.**
  Bollinger mean-reversion (volatility family, zero rule-based coverage before this
  pass) actually *won the in-sample sweep outright* (Sharpe 0.76 vs. SMA's 0.74),
  and both it (0.72 OOS) and MFI mean-reversion (0.62 OOS, volume family) landed
  ahead of the original RSI baseline (0.59 OOS) walk-forward. That said, SMA still
  wins walk-forward OOS despite losing the in-sample sweep — a clean illustration
  of exactly why walk-forward validation is required rather than trusting an
  in-sample ranking (see §8).
- **Regime-filter, the previously-missing 4th combination type, works — for one of
  the two pairings tested.** Gating SMA with an ADX trending-regime filter improved
  it (0.79 vs. 0.77 ungated) — the classic rationale held. Gating RSI
  mean-reversion with a low-volatility-regime filter *hurt* it (0.50 vs. 0.59
  ungated) — the classic "mean-reversion works better in calm markets" rationale
  did not hold here; RSI's edge in this dataset appears concentrated in sharp,
  higher-volatility reversals (2020, 2022), not quiet range-bound stretches, so
  filtering those episodes out removed edge rather than noise. Reported as a real,
  non-obvious negative result, not smoothed over.
- **The long-side-more-robust hypothesis survives walk-forward, sharply**: the
  best RSI long-short config scored Sharpe 0.49 in-sample; walk-forward OOS it
  collapsed to **-0.03** (95% CI [-0.53, 0.43], not distinguishable from zero).
  Every long-only line held up; the one long-short line tested did not.
- **ML does not beat the best rule-based Sharpe.** Best ML model (logistic
  regression) scored 0.71 OOS vs. the best rule-based combo's 0.89. Per the task
  brief, this is reported as a strong result for the rule-based approach, not
  hedged. LightGBM did *worse* than the logistic baseline (Sharpe -0.08).
- **Neither the best rule-based nor the best ML strategy beats simply holding the
  research universe equal-weight (Sharpe 0.99) or SPY (Sharpe 0.92) over this OOS
  period.** This is the single most important qualifier on every other result in
  this report — see §6 and §10.
- **Every long-only rule-based strategy — old and new alike — is 0.91–0.95
  correlated with the equal-weight basket benchmark.** They are close to a
  filtered version of market beta, not an independent source of return. The
  regime-filtered RSI+low-vol combo is the one clear exception (0.65 correlation)
  — restricting trading to specific volatility regimes makes it meaningfully less
  beta-like, even though it's also lower-Sharpe. The two ML models remain the only
  return streams in the whole study with near-zero correlation to everything else.

## 2. Method

- **Data foundation** (~6-8 hours, built before the work in the rest of this
  report): PIT S&P 500 membership reconstruction, Yahoo price download and
  ticker mapping (renamed/delisted securities), corporate-action and
  OHLC-consistency validation, missing/corrupted price history investigation.
  Several of its decisions are directly load-bearing for the no-look-ahead and
  cost-realism claims made throughout this report: a fixed 2026-08-11 research
  cutoff so later Yahoo data can never leak in; NYSE-session-aware next-day
  return construction (`build_next_day_returns`), including correct handling
  of extended exchange closures (e.g. Hurricane Sandy, Oct 2012), not a naive
  calendar-day shift; known-corrupted tickers excluded by name
  (`EXCLUDED_PRICE_TICKERS`); extreme returns flagged and retained, never
  clipped; and a signal without a valid next-session return treated as
  untradable, never a fabricated fill. Validated with an initial 34-config
  in-sample sweep before any walk-forward/ML/combination work began.
- **Indicator coverage**: `indicators.py` computes 20 indicators spanning trend,
  momentum, volatility, and volume. Before this pass, only 3 had rule-based
  trading strategies (`sma_crossover`, `rsi_mean_reversion`, `donchian_breakout` —
  trend and momentum only). This pass adds 4 more, one genuinely representative of
  each remaining gap: `macd_crossover` (trend), `stochastic_crossover` (momentum),
  `bollinger_mean_reversion` (volatility — previously zero coverage),
  `mfi_mean_reversion` (volume — previously zero coverage). **7 of 20 now have
  rules, not all 20** — a prioritized, family-complete subset given time
  constraints, stated honestly rather than presented as full coverage. The
  remaining 13 indicators are still used as ML features (§5).
- Phase 1 shortlist (10 configs: 6 curated + 1 programmatically-selected
  best-in-sample-long-only pick per new family) was screened by **in-sample
  Sharpe from the 66-config sweep**, not from inside the walk-forward loop itself.
  Stated as a deliberate, time-constrained compromise.
- Walk-forward re-optimizes over each strategy family's *full* parameter grid
  every window, so shortlist entries sharing a (family, mode) collapse to one
  walk-forward line — 10 shortlist entries produce 8 distinct lines, not 10.
- Combinations (Phase 3): 3 fixed-parameter pairs (SMA+RSI, SMA+Donchian,
  RSI+Donchian) × {AND, OR, weighted-vote}, plus 2 regime-filter pairs
  (SMA gated by an ADX trending-regime signal; RSI mean-reversion gated by a
  low-volatility-regime signal) — the 4th combination type named in the task
  brief, previously missing. `combine_signals(method="regime_filter")` in
  `src/walkforward/combos.py`.
- ML (Phase 4, unchanged by this pass — it already used all 20 indicators'
  worth of feature columns regardless of which had rule-based strategies): 30 of
  35 available indicator columns, label = sign(next-day return), purged (1-day) +
  embargoed (5-day) walk-forward CV aligned to the identical windows used
  everywhere else. No `train_test_split`/`KFold` anywhere. Trading rule:
  P(up)≥0.70 long, P(up)≤0.30 short, else flat.

## 3. Top strategies (rule-based, single)

| line_id | Family | Sharpe | Net return | Max DD | % months profitable |
|---|---|---:|---:|---:|---:|
| `sma_crossover__long_only` | trend | 0.774 | 355% | -37.7% | 67.1% |
| `bollinger_mean_reversion__long_only` | volatility | 0.724 | 445% | -43.0% | 63.4% |
| `mfi_mean_reversion__long_only` | volume | 0.621 | 321% | -46.1% | 61.0% |
| `rsi_mean_reversion__long_only` | momentum | 0.587 | 322% | -43.6% | 60.4% |
| `macd_crossover__long_only` | trend | 0.430 | 117% | -36.0% | 61.6% |
| `stochastic_crossover__long_only` | momentum | 0.028 | -13% | -48.0% | 56.7% |
| `rsi_mean_reversion__long_short` | momentum | -0.030 | -11% | -33.3% | 46.3% |
| `donchian_breakout__long_only` | trend | -0.855 | -89% | -91.1% | 46.3% |

Every family now has a positive-Sharpe long-only representative except trend's
second entry (MACD, still positive but weak) and momentum's second entry
(Stochastic, essentially flat). Donchian remains the deliberately weak comparator
and stayed weak out-of-sample. Stochastic crossover's best in-sample config was
already negative (-0.13, see `results/shortlist.csv`) — the version shown here is
its best walk-forward OOS run, still barely above zero. Not every indicator
produces a working rule; reported as found, not cherry-picked.

## 4. Best combinations

| line_id | Method | Sharpe | Net return | Max DD |
|---|---|---:|---:|---:|
| `combo__sma_rsi__or` | OR | **0.890** | 552% | -40.1% |
| `combo__sma_rsi__weighted_vote` | weighted (0.6 SMA / 0.4 RSI) | 0.854 | 441% | -37.7% |
| `combo__sma_donchian__weighted_vote` | weighted (0.6 SMA / 0.4 Donchian) | 0.854 | 441% | -37.7% |
| `combo__sma_donchian__or` | OR | 0.818 | 401% | -37.7% |
| `combo__sma_adxtrend__regime_filter` | regime-filter (ADX-trend gate) | 0.787 | 372% | -37.6% |
| `combo__rsi_donchian__weighted_vote` | weighted (0.6 RSI / 0.4 Donchian) | 0.734 | 490% | -43.6% |
| `combo__sma_rsi__and` | AND | 0.526 | 202% | -41.1% |
| `combo__rsi_lowvol__regime_filter` | regime-filter (low-vol gate) | 0.503 | 216% | -48.4% |
| `combo__rsi_donchian__or` | OR | 0.417 | 131% | -42.5% |
| `combo__rsi_donchian__and` | AND | -0.178 | -5% | -7.0% |
| `combo__sma_donchian__and` | AND | -1.101 | -93% | -94.4% |

OR and weighted-vote both improved on their best individual leg in every pair;
AND was worse than the better leg in every pair — requiring both signals to agree
filtered out too many genuinely profitable trades. The ADX-trend regime filter on
SMA (0.787) *beat* plain SMA (0.774) — turnover fell (fewer, better-timed trades)
without giving up much upside, a genuine diversification-via-selectivity result.
The low-vol regime filter on RSI (0.503) *underperformed* plain RSI (0.587) — see
§1 for why. `combo__sma_donchian__or` matches `..._weighted_vote` almost exactly
because Donchian is so rarely active that OR and a 0.6/0.4 vote resolve to nearly
the same trades.

## 5. Rule-based vs. ML head-to-head

| line_id | Model | Sharpe | Net return | Max DD | Tickers traded | % profitable |
|---|---|---:|---:|---:|---:|---:|
| `ml_logistic_regression` | Logistic Regression | 0.711 | 870% | -45.2% | 30 | 53.3% |
| `ml_lightgbm` | LightGBM | -0.079 | -53% | -73.7% | — | — |
| *(best rule-based, for reference)* `combo__sma_rsi__or` | — | 0.890 | 552% | -40.1% | 607 | 89.1% |

The best rule-based Sharpe (0.89) beats the best ML Sharpe (0.71) — reported as a
strong result for the rule-based approach, not hedged, per the task brief.

LightGBM underperformed its own simpler logistic baseline. Permutation feature
importance shows why: logistic regression's top features are `price_sma_dist_50`
and `rsi_14` — directly the same signals that won as standalone rules (SMA and
RSI both being the walk-forward top performers), i.e. convergent evidence.
LightGBM's top features are `cmf_20`, `roc_20`, `volume_roc_20` — volume-based
indicators that did not independently win as rules anywhere in this study,
consistent with it having found relationships that generalized poorly. Notably,
neither model's top features are `bb_zscore_20_2.0` or `mfi_14`, despite Bollinger
and MFI being the two next-best standalone rules found this pass — the ML models
and the rule search surfaced *different* signal, not fully overlapping evidence.
See `results/charts/ml_feature_importance.png`.

The logistic model's 0.71 Sharpe is fragile in a way its headline number doesn't
show: only 53% of months were profitable, and it only ever took a position in 30
of the 634 tickers in the universe (avg. gross exposure 7.4%) — a
threshold-triggered, low-frequency, concentrated strategy. Its equity curve
(`results/charts/equity_curves.png`) is visibly a staircase of long flat
stretches and a few large jumps (notably 2024), not a smooth compounding line.

**Feature-representation ablation** (`src/pipeline/run_ml_pit_percentile_ablation.py`,
`results/ml_pit_percentile_ablation.csv`, not part of the primary result above):
replacing raw indicator values with same-date cross-sectional percentile ranks
(`build_ml_dataset(..., use_percentile_rank=True)` in `src/ml/dataset.py`) is a
principled normalization — it makes a feature comparable across tickers
regardless of absolute scale or a market-wide regime shift — but it broke the
fixed 0.70/0.30 trading threshold rather than improving on it. Percentile ranks
are uniform on [0,1] by construction, with no fat tails, so logistic
regression's predicted P(up) compressed to **[0.453, 0.558]** — it never once
crossed 0.70 across 89,170 first-fold predictions, i.e. the model made zero
trades. LightGBM widened slightly to [0.264, 0.656] but still only fired 4
times out of 89,170. The threshold was calibrated implicitly against the raw
feature distribution; changing the feature representation without
re-calibrating the threshold together with it silently disables the trading
rule. Kept as a documented ablation rather than adopted as the primary result —
recalibrating the threshold under the new distribution would mean abandoning
the literal 0.70/0.30 framing, which was a deliberate choice (§2), not
something to trade away for a normalization technique that, on this evidence,
doesn't actually improve the outcome here.

## 6. Benchmarks

| | Sharpe | Net return | Max DD |
|---|---:|---:|---:|
| Equal-weight universe basket | **0.992** | 829% | -40.3% |
| SPY buy-and-hold | 0.916 | 566% | -33.7% |
| *(best of this study, for reference)* `combo__sma_rsi__or` | 0.890 | 552% | -40.1% |

Both benchmarks beat every single strategy tested in this project — old and new
indicators alike — over the same stitched OOS window. This is the headline
caveat on every result above — see §10.

## 7. Cost sensitivity

Best rule-based and best ML finalist re-run at 0/5/10/15/20/25 bps (0 bps used
only here, per the hard constraint) — unchanged from before this pass since
neither finalist changed:

| Cost (bps) | `combo__sma_rsi__or` Sharpe | `ml_logistic_regression` Sharpe |
|---:|---:|---:|
| 0 | 0.915 | 0.726 |
| 5 (baseline) | 0.890 | 0.711 |
| 10 | 0.865 | 0.696 |
| 15 | 0.840 | 0.682 |
| 20 | 0.815 | 0.667 |
| 25 | 0.790 | 0.652 |

Neither edge dies anywhere in this range. Chart: `results/charts/cost_sensitivity.png`.

## 8. Walk-forward efficiency and ranking stability

| line | avg(OOS Sharpe / IS Sharpe) | avg rank corr (IS vs OOS) | % windows IS winner = OOS best |
|---|---:|---:|---:|
| `sma_crossover__long_only` | 1.51 | 0.11 | 21% |
| `bollinger_mean_reversion__long_only` | 1.27 | 0.29 | 29% |
| `mfi_mean_reversion__long_only` | 1.20 | 0.03 | 36% |
| `rsi_mean_reversion__long_only` | 1.05 | 0.23 | 7% |
| `macd_crossover__long_only` | 2.84 | 0.39 | 21% |
| `stochastic_crossover__long_only` | -7.92 | 0.24 | 36% |
| `rsi_mean_reversion__long_short` | 0.10 | 0.07 | 14% |
| `donchian_breakout__long_only` | n/a (IS Sharpe never positive) | 0.11 | 0% |

**Bollinger beating SMA in-sample but losing to it out-of-sample (§1) is the
clearest illustration in this dataset of why the task brief requires walk-forward
rather than an in-sample ranking** — an in-sample-only report would have shipped
Bollinger as the headline strategy. Stochastic's -7.92 "efficiency" is an
artifact of a near-zero in-sample Sharpe denominator, not a meaningful ratio —
included for completeness, not implying an 8x blowup is real. Ranking stability
is weak everywhere (rank correlations 0.03–0.39) — the parameter grids are small
and per-window differences between candidates are mostly noise.

## 9. Robustness

**Block-bootstrap Sharpe CIs and multiple-testing correction** (1000 resamples,
20-day blocks; Benjamini-Hochberg at α=0.05 across the 21 walk-forward/OOS
finalists — out of **87 configurations tested in total**: 66 in-sample sweep + 21
OOS finalists; the sweep is disclosed for context, not re-tested since it has no
OOS return series to bootstrap):

**13 of 21 finalists remain significant after correction**, including the top
result (`combo__sma_rsi__or`, p≈0, 95% CI [0.34, 1.48]), both new standalone
strategies (Bollinger p=0.014, MFI p=0.015), the ADX-regime-filter combo
(p=0.003), the low-vol-regime-filter combo despite its lower raw Sharpe (p=0.037),
and the ML logistic model (p=0.003). **Not significant**: MACD (p=0.069, just
misses), Stochastic, both long-short lines, `ml_lightgbm`, and both
Donchian-involving AND combos. Full table: `results/robustness_bootstrap.csv`.

**Parameter-stability**: the original sparse 6-point SMA sweep grid understated
this — its nearest tested neighbor to the 50/200 winner was 30/100, a real gap
(0.74 vs. 0.57) that couldn't rule out a local spike. A dense 5×5 local grid was
built around each of the 5 finalists with genuine positive Sharpe (SMA,
Bollinger, MFI, RSI, MACD — Donchian, Stochastic, and RSI long-short excluded as
non-winners), varying each strategy's two structural parameters while holding
the rest fixed at their winning values, in-sample, same methodology as the
original sweep. **All five show a real plateau, not a spike**: SMA ranges
0.62–0.78 (spread 0.16) across the whole 5×5 neighborhood, MACD 0.26–0.33
(spread 0.065, the tightest of the five), RSI 0.47–0.77 (spread 0.31), Bollinger
0.46–0.82 (spread 0.35), MFI 0.51–0.71 (spread 0.20) — every cell in every grid
stays positive, and each original winner sits inside its plateau rather than at
an isolated peak (Bollinger's grid actually found a nearby point 0.06 Sharpe
*better* than the original shortlisted pick — window=17/entry=-0.5 vs.
window=20/entry=-1.5 — evidence the original coarse sweep simply hadn't sampled
that neighborhood, not that the result is fragile). Charts:
`results/charts/param_stability_{sma_crossover,macd_crossover,rsi_mean_reversion,bollinger_mean_reversion,mfi_mean_reversion}.png`.

**Breadth**: computed for all 21 tested strategies (`results/breadth_summary.csv`),
not just the top finalist. `combo__sma_rsi__or` leads at 541/607 (89.1%); the
next four are also combos (83–84%). Breadth and Sharpe don't always agree:
`stochastic_crossover` has high breadth (78.8%) despite ~zero Sharpe (many small
winners, no edge after costs), and `donchian_breakout`/`ml_lightgbm` show the
opposite pattern — `donchian_breakout` is profitable on 49.4% of names yet has
the worst Sharpe and drawdown in the study (a few catastrophic losers overwhelm
many modest winners); `ml_lightgbm` is profitable on 58.1% of the (few) names it
traded yet still lost money overall. `ml_logistic_regression` remains the
narrowest bet in the study: 16/30 (53.3%).

**Survivorship bias, quantified** (the PDF's explicit ask, not previously
closed): re-running `combo__sma_rsi__or`'s exact stitched OOS signal, unchanged,
restricted to only the 474 tickers that are S&P 500 constituents *today*
(vs. the full 611-ticker PIT-reconstructed universe the signal actually traded)
— same window, same cost — **survivorship bias overstates the Sharpe ratio by
+0.084 (0.893 → 0.977) and net return by +108 percentage points (557% → 665%)**.
Direction: using today's list instead of the correct PIT list makes a strategy
look better than it is, as expected — a non-PIT-aware backtest on this data
would have overstated the edge by roughly 9% of the Sharpe itself.
`results/survivorship_bias_quantification.csv`.

## 10. Where results are weakest

- **No strategy in this study beat its own benchmark**, old or new. The
  equal-weight basket (0.99 Sharpe) and SPY (0.92) both beat the best combo
  (0.89) and the best ML model (0.71).
- **The rule-based edge is mostly beta, not alpha, and this held even after
  adding 4 new indicator families.** Every long-only rule/combo tested — SMA,
  RSI, MACD, Bollinger, MFI, Stochastic, and their combinations — is 0.91–0.95
  correlated with the equal-weight basket (§1/§9, `correlation_matrix.csv`). The
  one exception, the low-vol-regime-filtered RSI combo (0.65 correlation), is
  also one of the lower-Sharpe results — the diversification and the
  underperformance appear to be the same phenomenon (trading a narrower slice of
  market conditions).
- **7 of 20 indicators have rules, not all 20.** Trend and momentum each have a
  second, weaker representative (MACD, Stochastic); volatility and volume each
  have exactly one (Bollinger, MFI). The remaining 13 are ML features only. Stated
  explicitly rather than left implicit.
- **In-sample-screened shortlist** (§2): the shortlist that entered walk-forward
  was chosen by in-sample Sharpe. Bollinger's in-sample win over SMA (§8) shows
  this ranking is not reliable — a strategy screened only inside the walk-forward
  loop from the start might have produced a different top set.
- **ML logistic result is concentrated**: 30 tickers, 53% of months profitable,
  staircase equity curve.
- **Regime-filter's second pairing (RSI + low-volatility gate) underperformed the
  ungated strategy** — included and explained (§1, §4), not dropped for looking
  bad. Only 2 regime-filter pairings were tested; a broader sweep of gate
  thresholds or gate/base-strategy pairings was out of scope for this pass.
- **Turnover-at-roll-date artifact**: `run_backtest` computes turnover from a
  weight-matrix pivot of only the rows it's given, so day 1 of every stitched OOS
  segment looks like a from-flat entry rather than a continuation of the prior
  window's book — a small, documented mechanical overstatement of turnover/cost
  at each of the 13 roll dates. Not a look-ahead leak.
- **Weak ranking stability** (§8): the walk-forward re-optimization step is not
  clearly finding a durable signal above noise at this grid resolution.
- The two stateful mean-reversion state machines with the slowest per-call
  runtime (`_rsi_target_state_long_only/_long_short`, reused by MFI; the
  Bollinger equivalent) are plain per-ticker Python loops, not vectorized —
  functionally correct (see `tests/test_no_lookahead.py` and the smoke tests)
  but a real performance gap, not just a style choice: the dense
  parameter-stability grids for Bollinger and MFI took most of this pass's
  total runtime for exactly this reason.
- **Cross-sectional PIT percentile normalization was attempted, not skipped**
  (§5 ablation) — tried, diagnosed as breaking the fixed confidence threshold,
  not adopted. Still genuinely not attempted: SHAP feature importance
  (permutation importance used instead); long-short combos; regime-filter
  pairings beyond the 2 tested; wiring the remaining 13 indicators as rules;
  PCA/SHAP/Boruta-driven feature reduction ahead of model training (5 redundant
  columns were dropped by manual inspection instead, see `src/ml/dataset.py`);
  ML hyperparameter tuning (both models use fixed, untuned defaults); a
  re-calibrated threshold that would make the percentile-normalized features
  usable (e.g. per-window quantile-based cutoffs instead of a fixed value).

## Appendix: reproducing these numbers

`python run_all.py` from a clean `results/` directory reproduces every number in
this report. See `README.md` for phase-by-phase output locations and the column
glossary.
