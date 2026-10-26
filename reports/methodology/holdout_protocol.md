# Holdout evaluation protocol: proposal

**Status: proposed, not approved, not run.** `task.md` does not define a holdout
step, so nothing here is authorised until it is approved. The holdout is
2020-01-02 to 2024-12-30 and has not been evaluated by any pipeline.

## Why a protocol first

The holdout can be used once. Every choice that could be bent after seeing it
(which strategies, which metrics, what counts as success) is fixed here
beforehand. Anything decided after opening it is exploratory and labelled so.

## 1. Preconditions: freeze before opening

1. **The deflated Sharpe is built and tested.** Phase 8.2,
   `src/evaluation/deflated_sharpe.py`, currently empty.
2. **The code is committed.** The holdout run must record a clean
   `code_version`; every development run so far was `-dirty`.
3. **The config is frozen.** Its hash goes in the freeze file. No parameter
   changes after this point.
4. **The trial count is frozen.** N, V[SR] (with and without the allocation
   trials) and the full trial list are written to
   `reports/results/holdout_freeze.json`, following `trial_accounting.md`.
5. **The candidates are declared** in the freeze file before any holdout number
   exists:
   - **Primary candidate:** one strategy, chosen on development evidence only.
   - **Secondary candidates:** every other `backtest` strategy family (ML long/
     short, long only, thresholded, momentum, pairs book, the four allocation
     rules), reported in full whatever their results.
6. **A lock.** The run script refuses to run if the freeze file is missing, or
   if a holdout result file already exists.

## 2. What is fitted, and on what

- **The ML model:** refitted once on all development rows. The fracdiff order
  (0.60), the features, labels and model settings are as frozen.
- **The pairs:** selected once on all development dates, using the
  segment-aware method.
- **Momentum and the allocation rules** use trailing windows. Their first
  holdout decisions may use late-2019 prices as history, which is legitimate
  because those are past prices.
- **Nothing is tuned.**

**The boundary.** The development label for 2019-12-31 already used the return
dated 2020-01-02 (see `phase7_verification.md`). Holdout metrics therefore use
returns dated **2020-01-06 onward**: the first position comes from a signal at
the 2020-01-02 close, traded under the one-bar delay at the 2020-01-03 close,
earning the return dated 2020-01-06. The return dated 2020-01-02 is never
counted as out of sample.

## 3. Costs and execution (same as development)

- **Costs:** 2 bps commission and 3 bps slippage per side, plus 0.5% a year
  borrow on shorts.
- **Execution:** one-bar lag; gross leverage at most 1; each position at most 1.
- **Cash:** earns 0, consistent with the zero risk-free rate in the Sharpe.

## 4. Predeclared metrics

**Primary:** the annualised net Sharpe of the primary candidate. Its deflated
Sharpe ratio uses the frozen N and V[SR], and the holdout's own length, skew and
kurtosis.

**Also reported:**

- **Returns:** net and gross Sharpe, annual return, volatility, maximum
  drawdown, turnover and annual cost.
- **Uncertainty:** the Sharpe standard error (Mertens), and the paired 21-day
  block-bootstrap interval of the net Sharpe difference against each benchmark.
- **Resampling:** the share of paired resamples beating each benchmark.
- **The decision rule's own inputs:** the PSR against 0, and the DSR.

**Benchmarks:** SPY buy and hold, equal-weight buy and hold of the 11 names, and
monthly-rebalanced equal weight.

## 5. Decision rule (fixed in advance)

The primary candidate is reported as **showing evidence of an edge** only if:

1. its DSR exceeds 0.95, and
2. its paired 95% interval for net Sharpe minus equal-weight hold lies entirely
   above 0.

Otherwise the conclusion is **"no evidence of an edge"**, whatever its point
estimate. With about 1,250 holdout days the Sharpe standard error is about 0.45,
so only a large difference can pass.

## 6. Reporting rules

- **Report everything:** every declared candidate and benchmark, including the
  failures.
- **Run once.** No rerun with changed settings after seeing results. If a bug is
  found afterwards, both the original and the corrected results are reported,
  and the original stays in the trial log as kind `holdout`.
- **Keep follow-ups apart.** Any analysis invented after opening the holdout is
  labelled exploratory and kept out of the headline.
- **Scope of the result.** The holdout result applies to this hindsight-chosen
  11-stock universe only (survivorship). It is not evidence about an investable
  universe.
