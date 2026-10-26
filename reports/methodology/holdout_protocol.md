# Holdout evaluation protocol

**Status: approved 2026-10-26 by the project owner. Not run.** The holdout
(2020 to 2024) has not been evaluated. This text is pinned by its SHA-256
(LF line endings) in `reports/results/holdout_declaration.json`. Changing it
afterwards makes every preflight fail until a replacement declaration is
written (section 8).

The amendment record at the end shows how this differs from the earlier
proposal.

## 1. Why a protocol first

The holdout can be used once. Every choice that could be bent after seeing it
is fixed here beforehand: which strategies, which metrics, what counts as
success. Anything decided after opening it is exploratory and labelled so.

## 2. What is frozen, and where

| Item | Where | Value |
| --- | --- | --- |
| Trial count, V[SR], trial list | `reports/results/holdout_freeze.json` | Written by `make freeze` from clean commit `20cdb7a`. The file itself was committed in `65263c7`. Never rewritten. |
| Config | Freeze `config_hash` | `f33d4859198d`, a hash of `config/config.yaml` with line endings normalised |
| Trial history | Freeze `trial_log_rows`, `trial_log_sha256` | 175 rows. The SHA-256 covers the bytes as written (CRLF) |
| Candidates, timing, costs, statistics, this protocol's hash | `reports/results/holdout_declaration.json` | Written once (section 8) |
| Final model, pairs selection | `reports/results/holdout_artifacts/` and its `manifest.json` | **Not yet prepared.** Built in a separately authorised step (section 5) |

**Code version.** The freeze records `20cdb7a`, the commit whose code computed
N and V[SR]. The freeze file can only be committed after that commit, and the
runner after that again, so HEAD will never equal `20cdb7a`. The run therefore
checks four things instead:

- `20cdb7a` is an ancestor of HEAD;
- `git diff 20cdb7a HEAD` touches only an explicit allowlist (section 7);
- the freeze file equals its committed copy in `65263c7`;
- the working tree is clean.

Each attempt records both the frozen code version and the runner's own commit.

**Trial-log line endings.** `trial_log_sha256` was computed on the Windows copy
of the trial log, which has CRLF line endings because the csv module writes
them; git stores LF. Both representations reproduce exactly from the trial log
committed in `65263c7`, and both are recorded in the declaration. The check
accepts either one, for exactly the first 175 rows, and nothing else. Any row
after them must be of kind `holdout`.

## 3. Candidates and benchmarks

**Primary candidate:** `allocation_minimum_variance`, the minimum-variance
allocation. It is long only, uses a Ledoit-Wolf covariance over the trailing
252 days, and is rebalanced at each month end.

It was chosen on development evidence only. It had the highest development net
Sharpe (0.785) among the 22 current candidates of the frozen variance scenario.
It also needs no trained model or pair selection.

**The evidence is inconclusive:**

- its frozen DSR is 0.207;
- its paired interval against monthly equal weight (0.724) included zero;
- its lead over `ml_long_short` (0.759) compares one walk-forward path with an
  average over CPCV paths, and is within noise.

**Secondary candidates (eight, reported in full, descriptive only):**

- the allocations `allocation_inverse_volatility`, `allocation_risk_parity` and
  `allocation_maximum_sharpe`;
- the ML books `ml_long_short`, `ml_long_only` and `ml_thresholded`;
- `momentum` and `pairs`.

**Benchmarks:**

| Name | What it is | Role |
| --- | --- | --- |
| `buy_and_hold` | SPY bought once and held | Descriptive |
| `equal_weight_hold` | Equal weight in the 11 tickers, bought once, then left to drift | **The decision reference** |
| `equal_weight` | Equal weight, rebalanced at each month end | The primary's development reference |

The primary was compared with `equal_weight` during development. The holdout
decision compares it with `equal_weight_hold`, as this protocol fixed before
the declaration. The comparison with `equal_weight` is still reported, as
descriptive only. Neither reference may be swapped after the holdout is opened.

**Excluded:**

- `ml_long_short_same_close`, a diagnostic of notebook 4's timing;
- the individual `pair_<A>_<B>` development trials.

Secondary candidates and benchmarks never replace the primary or change the
decision rule.

## 4. Timing and execution

The engine's convention (`src/backtest/engine.py`): `returns[t]` runs from the
close of t − 1 to the close of t. A weight held after the close of t earns
`returns[t + 1]`. A trade at the close of t is charged at t.

- **Calendar.** The calendar is the trading dates of the cleaned panel, so
  weekends and holidays have no rows.
- **Decision date D0.** D0 is the first trading date on or after
  `validation.test_start` (2020-01-01), expected to be 2020-01-02.
  - Every book's target at D0 is what its own development code asks for at the
    close of D0, using prices up to D0 only.
- **Starting state.** Every book is flat (all cash) until D1 = D0 + one bar,
  expected to be 2020-01-03.
  - At the close of D1 every book trades from flat to its D0 target and pays the
    entry cost.
  - D1 is the first row of the window. Its gross return is zero for every book,
    and its net return is minus the entry cost.
- **First return earned.** The bar after D1, expected to be 2020-01-06.
- **Last date.** The last trading date on or before `universe.end_date`
  (2024-12-31) on which every ticker has a price.
- **One common window.** Every candidate and benchmark is evaluated on the same
  rows, from D1 to the last date.
  - The window starts at D1 only if every book can be evaluated there. The
    development history before D0 must cover every warm-up (`required_history`),
    and every book's D0 target must be defined: allocation weights, a momentum
    signal, a z score for each selected pair, a forecast for every ticker.
  - Otherwise the run fails. The window is never moved.
- **December 2019 weights are not carried over.**
  - The development books' 2019 positions belong to the development period.
  - A monthly book's D0 target is its weights from the last scheduled
    rebalance on or before D0. That is 2019-12-31, estimated on the trailing
    window ending that day, so on development data only.
  - That target is entered at D1, from flat.
- **Rebalancing.**
  - Allocations and momentum rebalance on the last trading date of each month
    and trade one bar later. Allocations are forced back to target on each
    rebalance, as in development.
  - The other books trade whenever their target changes.
- **Warm-up.**
  - Signals may use any price up to the decision date, so development prices
    serve as warm-up history.
  - No holdout return is used to fit, select or tune anything.
- **ML forecasts.**
  - Forecasts exist only from D0. The model was trained on every earlier date,
    so earlier forecasts would be in-sample.
  - The thresholded rule's held state therefore starts flat at D0.
- **Pairs state.** The spread rule runs on the unbroken price history, as in
  development. A trade open at the close of D0, decided from past prices, is
  entered at D1.
- **Missing prices.** As in development:
  - the cleaned panel forward-fills at most three days;
  - a return still missing is treated as zero;
  - the number of missing cells in the window is recorded.
- **Costs:**
  - 2 bps commission and 3 bps slippage per side, on the traded weight;
  - 0.5% a year to borrow on shorts;
  - cash earns 0, consistent with the zero risk-free rate in the Sharpe.
- **Limits:**
  - each position is at most 1;
  - gross exposure is at most 1;
  - shorts are allowed for the strategy books;
  - allocations are long only.
  - Strategy targets are clipped and scaled to the limits, as in development.
    An allocation that breaks a limit stops the run.

## 5. What is fitted, and on what (the preparation step)

`python entrypoint/holdout.py prepare --authorised-by NAME` runs once, and only
after separate authorisation. It cuts the panel before 2020-01-01 before
anything is built, so nothing after the boundary can reach either artifact.

- **The ML model.** One random forest, fitted on every labelled development
  row with the development sample weights. It uses:
  - the frozen settings (regression, fixed-horizon labels, the frozen features);
  - the fracdiff order chosen on development data.
- **The pairs.** Selected once, by `find_pairs` on every development date (one
  contiguous segment), with the frozen screen:
  - daily log-return correlation of at least 0.5;
  - Engle-Granger cointegration at 5%;
  - a half-life above 0 and at most 126 days.
- **How the pairs book trades.** It trades every selected pair with an equal
  share of the gross budget:
  - a rolling 252-day hedge ratio and a 63-day z score;
  - entry at 2, exit at 0.5, stop at 4.

  If no pair is selected, the book is flat, reported as such, and its Sharpe
  is undefined.
- **Momentum and the allocation rules** learn nothing in advance. They use
  trailing windows, as in development.

Preparation writes `manifest.json`, which records:

- the SHA-256 of every file;
- the declaration it was prepared under;
- the training window and the fracdiff order;
- the selected pairs;
- who authorised it.

**Nothing is tuned.** The step logs no trial: it produces no result.

## 6. Statistics and the decision rule

- **Primary metric.** The annualised net Sharpe ratio of the primary over the
  window.
- **DSR of the primary.**
  - N = 98, from the freeze's `headline.n_trials`.
  - V[SR] = 0.15815 a year, which is 0.000628 a day, from
    `variance_scenarios.strategies_and_allocations` (the headline's
    `variance_used`).
  - The holdout's own length, skew and raw kurtosis come from the primary's
    daily net returns.

  Using the full frozen N for one predeclared primary is a **deliberately
  conservative choice**. The selection happened on development data, and this
  rule charges the holdout result for it again. No other N or V[SR] may be
  substituted after opening.
- **PSR against zero.** Reported for every book, separately from the DSR.
  Neither the PSR nor the DSR is a probability of future profit. Each is the
  probability, under the method's assumptions, that the true Sharpe ratio over
  the holdout period exceeded a benchmark: zero, or the luck threshold for N
  trials.
- **Paired bootstrap.** Every book's net Sharpe minus each benchmark's, with
  the same blocks of dates for both series
  (`sharpe_difference_interval`).
  - The block length is 21 trading days.
  - There are 1,000 resamples, with seed 0, the seed of every development
    comparison.
  - The interval is 95%.
- **Also reported, descriptive:**
  - net and gross Sharpe, annual return and volatility;
  - maximum drawdown and Sortino;
  - the Mertens standard error of the Sharpe;
  - turnover, annual cost and mean exposure;
  - the share of resamples beating each benchmark.

**Decision rule (fixed).** The primary shows **evidence of an edge** only if
both of these hold, each strictly:

1. its DSR exceeds 0.95, and
2. its paired 95% interval for net Sharpe minus `equal_weight_hold` lies
   entirely above 0.

Otherwise the conclusion is **"no convincing evidence of an edge"**, whatever
the point estimate. A DSR of exactly 0.95, or an interval bound of exactly 0,
fails. With about 1,250 holdout days the Sharpe standard error is about 0.45.
Rule 1 alone needs a holdout Sharpe of roughly 1.7 or more.

## 7. Safeguards before the holdout is opened

`python entrypoint/holdout.py check` (or `make holdout-check`) runs these
checks and stops. It reads git, the freeze, the declaration, this protocol,
the trial log, the config and the prepared artifacts, and **never reads a
price**. `run` repeats them, and calls the price loader only after they all
pass and `STARTED.json` has been created.

1. **Clean tree.** The working tree is clean, untracked files included.
2. **Ancestry.** `20cdb7a` is an ancestor of HEAD.
3. **Freeze.** The freeze file is identical (LF-normalised) to its copy in
   `65263c7`.
   - It records `20cdb7a` and holdout start 2020-01-01.
   - Its headline used `strategies_and_allocations`.
4. **Config.** The config hash equals the frozen one.
5. **Trial-log rows.** The trial log starts with the frozen 175 rows (trials 1 to
   175).
6. **Trial-log hash.**
   - Those rows hash, raw or LF-normalised, to the values the declaration
     recorded.
   - Every later row is of kind `holdout`.
7. **Declaration.** The declaration, or its latest replacement:
   - is approved;
   - cites `65263c7`, `20cdb7a` and the committed freeze file's SHA-256;
   - cites this protocol's current SHA-256;
   - matches the frozen N, V[SR] and config hash;
   - declares no book twice;
   - names a decision reference that is a declared benchmark.
8. **Books.** Every declared book is one the runner supports.
9. **Artifacts.**
   - The artifact manifest exists and was prepared under a declaration in the
     chain.
   - Every file matches its recorded SHA-256.
   - The model's training window ends before 2020-01-01.
10. **Attempt.** The attempt is allowed (section 8), and no holdout row exists
    without an attempt folder.
11. **Changed paths.** `git diff 20cdb7a HEAD` touches only:
    - the freeze, the declaration and its replacements;
    - the artifacts and the attempt folders;
    - the trial log, whose rows are checked by items 5 and 6;
    - this protocol and `trial_accounting.md`;
    - `src/pipelines/holdout.py`, `entrypoint/holdout.py`, `tests/test_holdout.py`
      and the `Makefile`;
    - and one exact edit to `src/evaluation/trial_log.py`, which adds the
      `holdout` kind.

    Data, config and every module that computes a result must be unchanged.

## 8. Running once, and what happens if it fails

- **Declaration.** Written once, by
  `python entrypoint/holdout.py declare --approved-by NAME --approved-on DATE`.
  It is never overwritten.
  - A change before opening needs `declare --replace REASON`. That writes
    `holdout_declaration.r<k>.json`, which cites the SHA-256 of the file it
    replaces. The original is kept.
  - Once any attempt exists, no replacement is accepted.
- **Attempts.** Each attempt is a folder `reports/results/holdout/attempt-<k>/`.
  - `STARTED.json` is created exclusively before the loader is called. It holds:
    - every preflight result and hash;
    - the runner commit and the frozen code version;
    - the frozen N and V[SR];
    - the bootstrap settings, the candidates and the costs.
  - The outputs are `daily_returns.csv`, `summary.csv`, `comparisons.csv`,
    `evaluation.json` and `report.md`. Each is created exclusively.
  - One `holdout` trial-log row is appended per book.
  - `COMPLETE.json` records the SHA-256 of every output, the trial ids, and a
    re-check that the frozen 175 rows are unchanged.
- **Failure.** `FAILED.json` records the error, and a failed `holdout` row is
  logged.
  - An attempt with only `STARTED.json` (interrupted) is **stale**, and is
    treated like a failed one.
  - Nothing in a failed or stale attempt is deleted or rewritten.
- **A further attempt** needs `run --new-attempt REASON`. The earlier folder
  must be committed first, so the clean-tree check (item 1) passes.
  - The new attempt records every earlier attempt's state.
  - Every attempt is reported.
  - If a bug is found after a complete attempt, the original and the corrected
    result are both reported, and the first complete attempt remains the
    primary record.
- **Keep follow-ups apart.** Any analysis invented after opening is labelled
  exploratory and kept out of the headline.

## 9. Scope of the result

The universe is 11 tickers chosen with hindsight about which companies
survived (survivorship bias, acknowledged in config). The holdout result
applies to this universe only. It is not evidence about an investable
universe.

The report states this, along with the single-period, single-path nature of
the test and the fact that costs are a model.

## Amendment record

- **Before 2026-10-26: proposal.** The earlier text was marked "proposed, not
  approved, not run". Its section 1 said the freeze "has not been run yet".
  That was true when written; the freeze was then created at `20cdb7a` and
  committed in `65263c7`.
- **2026-10-26: approved version (this text).** It resolves the ambiguities the
  readiness audit found:
  - **Candidates.** The secondary candidates and benchmarks are named exactly.
    The pairs book is one strategy, and the individual pair trials are not
    candidates.
  - **Timing.** The window, the starting state and the handling of December
    2019 weights are now defined for monthly books. Before, they were defined
    only for daily signals.
  - **Line endings.** The rule for the trial-log hash is stated.
  - **Statistics.** The DSR fields and the bootstrap settings are named.
  - **Procedures.** The declaration, preparation, preflight and attempt
    procedures are written out.

  The decision rule and the benchmarks are unchanged from the proposal. The
  previous version stays in git history.
