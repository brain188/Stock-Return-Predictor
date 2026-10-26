# Phase 6 verification

Seven checks on the Phase 6 backtest, run on the development period
(2005-07-22 to 2019-12-31). The 2020-2024 holdout was not used by any check.
One earlier run of the cost reconciliation read notebook 4's data to 2024, as
the notebook itself does; that is disclosed under point 2, and the script now
cuts the data at 2019-12-31.

> **Note added in Phase 8 (2026-10-26).** The figures below come from the runs
> before the boundary fix. Phase 8 cut the panel at `validation.test_start`
> before features and labels are built, refitted research and evaluation, and
> logged the refits as new trials (107 to 133). The development set lost the 11
> rows dated 2019-12-31, whose labels needed the return dated 2020-01-02. Path
> returns now end 2019-12-30. Net Sharpe ratios moved by at most 0.009 (ML
> books) and 0.002 (benchmarks), and the pairs selection did not change. No
> conclusion here changes. The current figures are in `reports/results/` and
> `reports/research_report.md`.

Evidence files:

- `tests/test_timing.py`, `tests/test_evaluate.py`, `tests/test_pairs.py`,
  `tests/test_performance.py`, `tests/test_engine.py`
- `reports/methodology/cost_reconciliation.py` and `benchmark_validity.py`
  (both runnable)
- `reports/results/evaluation/` (`comparison.csv`, `regimes.csv`, `path_check.csv`)

## 1. Execution timing

The chain, from forecast to profit and loss:

| When | What happens | Where |
| --- | --- | --- |
| Close t | Features use data up to t (checked by the truncation test in `build_features`); the forecast is about the return from t to t + 1 | `features/build.py`, `labelling/fixed_horizon.py` |
| Close t | `targets()` turns the forecast into a target weight | `backtest/strategy.py` |
| Close t + 1 | `positions()` delays it by `execution_lag` = 1 bar: first held after this close | `Strategy.positions`, `delay` |
| Close t + 2 | The engine credits the holding with the return dated t + 2 | `engine.gross_returns`: weights at t − 1 times returns at t |

So a forecast made at the close of t never earns the return it forecast (dated
t + 1). Evidence:

- **A hand-worked example.** Two assets, six days, drift included: the
  engine's gross returns match `[0, 0, −0.01, −0.004798, 0.009502, −0.028279]`
  to 1e-6 (`test_gross_returns_match_a_hand_calculation`).
- **Perfect foresight.** A signal equal to tomorrow's return earns an annual
  Sharpe below 0.5 with the delay on its own (below 1 through the full CPCV
  pipeline, on a short synthetic panel), and above 10 without the delay in both
  cases (`test_perfect_foresight_*`).
- **Matching notebook 4's engine.** On identical long-or-flat inputs, the Phase 6
  engine reproduces notebook 4's own `backtest()` gross returns and costs, one
  row apart, as the different conventions require
  (`test_reproduces_notebook_4s_backtest_on_the_same_inputs`).

## 2. Transaction costs: notebook 4's 0.90 against Phase 6's 0.06

**The 0.90 is not a Sharpe ratio.** Notebook 4's `cost_drag` column is
`book["cost"].sum()`: the sum of daily costs, about 90% of capital paid in fees
over 2008-2024. The Sharpe its long/short rule lost to costs was
0.5901 − 0.2787 = **0.31**. The Phase 6 summary misquoted this as a 0.90
Sharpe loss, and `task.md` section 2 phrases it ambiguously.

The reconciliation below runs on development years only (from
`cost_reconciliation.py`; the notebook's bars are cut at 2019-12-31). Each
step changes one thing:

| Step | Gross Sharpe | Net Sharpe | Lost to costs | Turnover / yr | Cost / yr |
| --- | --- | --- | --- | --- | --- |
| 0 Notebook 4's own code (2008-2019) | 0.326 | −0.007 | 0.333 | 130.1 | 6.50% |
| 1 Same positions, Phase 6 engine | 0.323 | −0.008 | 0.331 | 130.2 | 6.51% |
| 2 + borrow 0.5%/yr on shorts | 0.323 | −0.018 | 0.340 | 130.2 | 6.69% |
| 3 + the one-bar delay | 0.529 | 0.188 | 0.342 | 130.1 | 6.69% |
| 4 Phase 6 forecasts, SPY only | 0.421 | 0.386 | 0.035 | 12.6 | 0.64% |
| 5 Phase 6 book, 11 tickers (path 0) | 0.805 | 0.740 | 0.065 | 19.9 | 1.01% |

What the steps show:

- **Same cost model.** On the same positions the two engines charge the same
  cost to the basis point (steps 0 and 1): 5 bps a side in both, split 2 bps
  commission and 3 bps slippage in Phase 6. Slippage is meant to cover the
  half-spread; there is no separate spread charge in either.
- **The small gross difference is drift.** In steps 0 and 1 the gross returns
  differ only on days that began with a short (largest gap 0.0097; on long
  days exactly 0.0) and on the first day. The notebook rebalances a short back
  to −1 every day for free; Phase 6 lets it drift.
- **Borrow is minor:** +0.01 Sharpe lost. In the Phase 6 book it costs 0.016% a
  year, against 0.40% commission and 0.60% slippage.
- **Timing doesn't change cost.** It changes gross return: on these years the
  delay raises the notebook model's gross Sharpe (0.33 → 0.53). An earlier full
  2008-2024 run (now withdrawn) showed the opposite sign, so the notebook
  model's edge depends on the sample period rather than being stable.
- **The gap is turnover, and turnover is the forecast.** The notebook's SPY
  forecast changes sign on 25.8% of days and is positive 64.7% of the time. The
  Phase 6 forecasts change sign on 2.5% of days for SPY (3.9% on average across
  tickers) and are positive 96.8% of the time. At the same price per trade, 130
  turns a year cost 6.5%; 13 to 20 turns cost 0.6% to 1.0%.

**Disclosure.** The first run of this script executed notebook 4's code
unmodified, so its walk-forward read data to 2024. Its only output used here
was reproducing the notebook's already-published figures (0.5901 / 0.2787 /
0.9035). No parameter, strategy or decision was based on it. The script now cuts
the notebook's bars at 2019-12-31 and asserts that no later date reaches it.

## 3. Benchmark validity

**How the universe was chosen.** The config describes the 11 tickers as fixed as
of the 2005 start date, but neither the config nor the notebooks record a
selection rule (for example "the 10 largest S&P 500 members on 2005-01-03").
They are well-known US large caps that all survived and still trade, chosen with
2020s knowledge. GE and INTC, two later laggards, partly offset this, but
nothing removes it. Without a point-in-time constituent list the bias can be
sized, not fixed.

**Its size** (from `benchmark_validity.py`, development period):

| Equal-weight book | Net Sharpe | Annual return | Growth of 1 |
| --- | --- | --- | --- |
| All 11 names | 0.859 | 15.8% | 8.31 |
| Without AAPL | 0.630 | 9.6% | 3.75 |
| Without AAPL and MSFT | 0.575 | 8.5% | 3.23 |
| SPY alone | 0.560 | 9.1% | 3.50 |

- **Two names are the whole edge.** AAPL alone grew 54-fold. Equal weight's
  entire edge over SPY comes from AAPL and MSFT, the two names a hindsight-
  chosen list is most likely to contain.
- **No other name moves it.** Leaving out any other single ticker shifts the
  Sharpe by at most 0.031.

**What the benchmark can support:**

- **Relative comparisons within this universe.** The ML strategies, momentum and
  pairs trade the same 11 names, so equal weight holds the same survivors and is
  the fair passive reference. "No strategy beat holding these names" is
  supported.

**What it cannot support:**

- **Absolute returns or Sharpe ratios as achievable.** They are inflated by the
  choice of names.
- **Any claim of beating the market.** Every "beats SPY" result is mostly the
  universe holding AAPL: the ML long/short and long-only books are 93-96% net
  long.
- **Generalising to an investable universe** or to a later period.
- **Fine distinctions between strategies.** A strategy's edge over equal weight
  could itself depend on how it treated AAPL; that is not measured here.

## 4. CPCV paths

N = 6 groups of about 606 dates; k = 2 test groups per split; 15 splits; 5 paths.

**How paths are built.** Group g on path p takes its predictions from the p-th
split (in lexicographic order) that tests g. So:

- **Every split fills exactly two group slots** (15 × 2 = 30 = 5 paths × 6
  groups).
- **Five splits have neighbouring groups** ((0,1), (1,2), (2,3), (3,4), (4,5))
  and put both of their groups on the same path. The other ten put one group on
  each of two paths.
- **Every pair of paths shares exactly one split's model.**

**Verified on every run** (`verify_paths`, run inside `build_context`; saved as
`path_check.csv`):

- **Full coverage.** Each path covers all 3,636 development dates, with 0
  duplicates and 0 missing.
- **Groups come from the right splits.** Each group comes from a split that
  tested it and did not train on it; ML forecasts cover every row.
- **Breaks are caught.** A deliberately broken assignment raises `PathError`.
- **No silent gaps at backtest time.** `backtest_paths` refuses positions that
  duplicate or omit a development date.

**Paths and splits are not independent samples.** They share models (as above),
share most of their training data, and all cover the same market dates, so a
bad year hurts every path at once. Consequences:

- **The path spread is not a standard error.** The spread of the net Sharpe
  across paths (0.04 to 0.06 for the ML books) measures sensitivity to which
  model predicted which group, not sampling error. It is far smaller than the
  Sharpe standard error of about 0.26.
- **Path counts are not binomial evidence.** "5 of 5 paths beat SPY" is not a
  test.
- **Rule-based strategies have identical paths** (spread 0), so the paths carry
  no information about their uncertainty.

For these reasons the comparison now reports, next to every path summary, the
Sharpe standard error and a paired bootstrap interval (point 6).

## 5. Pairs on non-contiguous training dates

A split's training dates can be up to four disconnected blocks. Before this
check:

- **The ADF inside Engle–Granger** differenced the residuals across the gaps,
  treating a months-long jump as one day's change.
- **The half-life regression** did the same.
- **The first return after each gap** was measured from a close inside a test
  block.

Now:

- **No differencing across gaps.** `contiguous_segments()` labels each block.
  The ADF (`adf_statistic(..., segments=)`) and the half-life only use
  differences and lags within a block. The levels regression in Engle–Granger
  needs no adjacency and keeps every training date.
- **No test-period returns.** The return opening each later block is dropped
  from the correlation screen.
- **Rolling windows never see the gaps.** The rolling hedge ratio and z-score
  run on the full, unbroken trading calendar, using only trailing real prices.
  `pair_targets` refuses a calendar that is not sorted and unique.
- **Tests.**
  - The segment-aware ADF gives the same answer whatever order the blocks come
    in (the naive one does not).
  - The half-life matches a hand OLS that drops the straddling change.
  - Prices inside a gap cannot change the selection.
- **Counted as new trials.** The method change gets a new fingerprint
  (`selection_method: segment_aware`), so the pairs trials were re-run; the
  earlier ones stay counted. Selection barely moved: 13 pairs traded both
  times, with JNJ–PG replaced by KO–PG, and the pairs book is still negative
  (net Sharpe −0.17).

**Remaining limitations:**

- **Future training data.** CPCV training data includes dates after the test
  block, so a pair is sometimes selected with knowledge of later years. This is
  standard for CPCV, but optimistic for a regime-sensitive choice like
  cointegration.
- **Asymptotic critical values.** Engle–Granger uses MacKinnon's asymptotic
  values, not finite-sample ones.
- **A single intercept across blocks.** The cointegrating regression fits one
  intercept to all blocks, so a genuine level break between blocks is treated
  as part of the spread.

## 6. Regimes and uncertainty

**Labels use only the past.** A date's label comes from the benchmark's 21-day
realised volatility against the terciles of its own earlier history
(`volatility_regimes`; tested: changing late returns leaves earlier labels
unchanged).

**The timing bug.** The label of date t includes that day's own return, and it
was being used to bucket the return earned on day t. That let each day's own
move help decide its regime: big down days pushed themselves out of "low". The
breakdown now buckets each return by the label of the previous close. The effect
was large:

| Low-volatility net Sharpe | Same-day label (before) | Previous-close label (now) |
| --- | --- | --- |
| Equal-weight hold | 1.87 | 1.12 |
| SPY hold | 1.58 | 0.65 |
| ML long/short | 1.52 | 0.76 |

**Uncertainty is now shown everywhere:**

- **Sharpe standard error** (Mertens 2002, allowing for skew and fat tails):
  about 0.26 over the full period, and 0.44 to 0.51 within a regime (1,013 to
  1,319 days).
- **A paired 95% interval** for the Sharpe difference against equal weight,
  from a 21-day block bootstrap. Within a regime the blocks are cut from the
  regime's days strung together, an approximation.

Net Sharpe against equal weight, development period:

| Strategy | Net Sharpe | ± SE | vs equal weight | 95% interval | Resamples ahead |
| --- | --- | --- | --- | --- | --- |
| ML long/short | 0.750 | 0.26 | −0.11 | [−0.38, +0.15] | 22% |
| ML long only | 0.749 | 0.26 | −0.11 | [−0.33, +0.11] | 17% |
| ML thresholded | 0.694 | 0.26 | −0.17 | [−0.45, +0.11] | 14% |
| Momentum | 0.050 | 0.26 | −0.81 | [−1.39, −0.28] | 0% |
| Pairs | −0.172 | 0.26 | −1.03 | [−1.73, −0.37] | 0% |
| SPY hold | 0.560 | 0.26 | −0.30 | [−0.56, −0.03] | 2% |

**What the intervals support:**

- **The ML books are indistinguishable from equal weight.** Their intervals
  include zero.
- **Momentum, pairs and SPY are distinguishably worse** than equal weight.

**What they don't support:**

- **No regime difference is meaningful.** Every within-regime difference
  between an ML book and equal weight has an interval that includes zero.
  "The ML books did relatively better in high volatility" (+0.06 to +0.07
  against equal weight) is noise.

## 7. Trial accounting

`trial_accounting.md` sets out how Phase 8 will count N and estimate V[SR]:

- **N:** 42 notebook trials, plus every unique research and backtest trial that
  produced a result, superseded versions included. Benchmarks and diagnostics
  are excluded, with the reasons recorded.
- **V[SR]:** from unique backtest trials. Whichever variance estimate gives the
  higher luck threshold is used.
- **Correlated trials:** reported as an effective N beside the raw-N headline.
- **Freezing:** N and V[SR] are frozen before the holdout is run.

Current log: 91 executions, of which 1 unique research trial and 32 unique
backtest trials, 2 benchmarks and 6 diagnostics; every rerun agrees with its
earlier executions.

## Unresolved

- **No investable benchmark** can be built from this universe without
  point-in-time constituents. Absolute results stay inflated.
- **The bootstrap is approximate within regimes.** It strings together dates
  that are not consecutive.
- **Pairs selection in CPCV** can use training data from after the test block.
- **Trial correlation is not yet modelled** (Phase 8).
- **Not yet built:** Monte Carlo resampling and the `--monte-carlo-runs` option
  (Phase 7).
