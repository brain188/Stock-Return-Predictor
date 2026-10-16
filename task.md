# Stock Return Predictor: Implementation Task List

> **File path:** `task.md` (repo root)

Everything that still needs to be built, in the order it should be built, with the
reason each piece exists. The notebooks are finished and the data is already
pulled, so this document starts from production code.

---

## Table of contents

1. [Where the project stands](#1-where-the-project-stands)
2. [What the notebooks already told me](#2-what-the-notebooks-already-told-me)
3. [How to read this task list](#3-how-to-read-this-task-list)
4. [Phase 1: Financial data collection](#phase-1-financial-data-collection)
5. [Phase 2: Exploratory data analysis](#phase-2-exploratory-data-analysis)
6. [Phase 3: Feature engineering](#phase-3-feature-engineering)
7. [Phase 4: Return prediction models](#phase-4-return-prediction-models)
8. [Phase 5: Time series model evaluation](#phase-5-time-series-model-evaluation)
9. [Phase 6: Trading strategy and backtesting](#phase-6-trading-strategy-and-backtesting)
10. [Phase 7: Portfolio and risk analysis](#phase-7-portfolio-and-risk-analysis)
11. [Phase 8: Experiment tracking and reporting](#phase-8-experiment-tracking-and-reporting)
12. [Phase 9: Dashboard](#phase-9-dashboard-built-last)
13. [Build order](#13-build-order)
14. [Complete folder structure](#14-complete-folder-structure)
15. [Definition of done](#15-definition-of-done)
16. [Stitch design prompt for the dashboard](#16-stitch-design-prompt-for-the-dashboard)
17. [What the README should contain](#17-what-the-readme-should-contain)

---

## 1. Where the project stands

### Already built and working

| Piece | File | State |
|---|---|---|
| Typed config | `src/config.py` | Done, 11 validation guards, tested |
| Config values | `config/config.yaml` | Done, every section present |
| Logging | `src/logger.py` | Done, `log_step`, `log_frame`, `log_config` |
| Data loaders | `src/data/loaders.py` | Done, yfinance primary, Stooq fallback, parquet cache |
| Fetch CLI | `entrypoint/fetch.py` | Done |
| Dependencies | `pyproject.toml` | Done, uv managed, ruff and pytest config inside |
| Tests | `tests/test_config.py`, `tests/test_loaders.py` | Done, 25 passing |
| Notebook 1 | `notebooks/01_data_quality.ipynb` | Done, 52 cells |
| Notebook 2 | `notebooks/02_stationarity.ipynb` | Done, 43 cells |
| Notebook 3 | `notebooks/03_feature_research.ipynb` | Done, 40 cells |
| Notebook 4 | `notebooks/04_backtest.ipynb` | Done, 36 cells |

### Data already on disk

11 tickers (SPY, AAPL, MSFT, JNJ, JPM, XOM, PG, KO, WMT, GE, INTC), 2005-01-01 to
2024-12-31, 5,032 bars each, cached as parquet under `data/01-raw/`.

**Data collection is finished. Nothing in this task list re-pulls data.**

### What this task list covers

Everything from `src/data/universe.py` onwards. The notebooks were experiments
where I worked things out by hand with no imports from `src/`. Now the same logic
gets written properly: typed, tested, logged, importable, and reusable.

---

## 2. What the notebooks already told me

These findings are the spine of the whole project. Every module below exists to
either confirm them or measure them more carefully, so they belong at the top.

**Daily equity direction is close to unpredictable.** The model found a positive
and consistent Information Coefficient (+0.0279, same sign in 5 of 5 folds) but a
negative edge over simply always predicting up (-0.0357, in 5 of 5 folds). IC
measures ranking quality. Hit rate measures sign. My features rank correctly and
point the wrong way, because they are contrarian in a market that rises on 55% of
days.

**Nothing beat buy and hold once costs were charged.**

| Strategy | Gross Sharpe | Net Sharpe | Monte Carlo paths beating hold |
|---|---|---|---|
| Buy and hold | 0.85 | 0.85 | baseline |
| ML long only | 0.78 | 0.70 | 22.1% |
| ML binary long/short | 0.59 | 0.28 | far below |

**Costs are the dominant term, not the model.** At 5 bps a side, daily position
flipping costs about 5,063 times what buy and hold costs. The cost drag of 0.90
removed roughly half the gross Sharpe of the long/short rule.

**Both ML strategies fail the luck threshold.** At 42 trials the expected maximum
Sharpe from pure luck is around 0.46. Neither strategy clears it.

**2020 was the only regime where the model won** (0.97 and 0.99 against buy and
hold's 0.67). One winning regime out of four is not an edge, it is a coincidence
worth reporting honestly.

**So the deliverable is the methodology, not a profitable strategy.** The honest
headline for the README and the dashboard is this: daily equity direction is close
to unpredictable, the small edge that exists does not survive transaction costs,
and the machinery that establishes this rigorously is the thing worth showing.

Every number the production code computes must be allowed to disagree with the
notebooks. If it does, the notebook was wrong or the module is wrong, and finding
out which is the point of writing both.

---

## 3. How to read this task list

Each task has:

- **`[ ]`** a checkbox
- **File** the exact path to create
- **What it does** in one or two lines
- **Tests** what `tests/` must assert
- **Why it matters** the quant finance reason, because this is my first quant
  project and I want to understand each piece rather than copy it

Rules that apply to every task, no exceptions:

1. **No look ahead.** Any statistic computed from data must only use data that
   existed at that moment. Rolling windows, expanding means, fitted scalers, all
   of them.
2. **Costs from the first backtest.** Never "add costs later". A strategy that
   looks good gross and dies net was never a strategy.
3. **Every experiment goes in the trial log.** The deflated Sharpe needs the trial
   count and it cannot be reconstructed afterwards.
4. **One line comments, Google style docstrings.** Same standard as the freight
   project.
5. **Tests before the next module.** A module without tests is not done.
6. **Log the shape and date range of every frame that crosses a module boundary.**
   Silent row loss is the most common bug in a pipeline like this.

---

## Phase 1: Financial Data Collection

**Status: mostly done.** Data is pulled and cached. Two modules remain so the
pipeline is complete and the universe decision is written down in code rather than
living in my head.

### 1.1 Universe definition

- [ ] **File** `src/data/universe.py`
- **What it does** Resolves the ticker list from config, records the start date each
  ticker becomes eligible, and exposes the survivorship bias acknowledgement as a
  function rather than a comment.
- **Functions**
  - `resolve_universe(config) -> list[str]`
  - `eligible_from(ticker, config) -> pd.Timestamp`
  - `survivorship_note(config) -> str` returns the text the report and dashboard both print
  - `validate_universe(panel, config) -> UniverseReport` checks every ticker has the expected bar count
- **Tests** A ticker missing from the panel raises. The note is non empty. A ticker
  with fewer bars than the threshold is flagged, not silently accepted.
- **Why it matters** My universe was fixed as of the 2005 start date, which means
  every company in it survived to 2024. Survivorship bias inflates annual return by
  roughly 1% to 4%. I cannot remove it without a point in time constituent list,
  but I can state its size and direction so no reader is misled.

### 1.2 Cleaning

- [ ] **File** `src/data/cleaning.py`
- **What it does** Turns raw pulls into trustworthy bars. Everything notebook 1
  found by hand, now as code.
- **Functions**
  - `drop_zero_volume(frame, config)` zero volume days are usually holidays or halts
  - `flag_price_gaps(frame, config)` detects unadjusted splits as returns beyond a threshold
  - `winsorise_returns(frame, config)` caps extremes without deleting the row
  - `align_calendar(panel, benchmark)` forces every ticker onto the benchmark's trading days
  - `clean_panel(panel, config) -> tuple[pd.DataFrame, CleaningReport]`
- **Tests** A synthetic 2 for 1 split is flagged. A zero volume day is removed. The
  cleaned panel has exactly the benchmark's index. The report row counts add up to
  the input count.
- **Why it matters** Notebook 1 found that stacking a masked frame before filtering
  kept the NaNs and inflated the extreme move count from 0 to 55,341. Cleaning bugs
  do not announce themselves, they quietly become signal. The report object exists
  so I can see how many rows each rule touched.

---

## Phase 2: Exploratory Data Analysis

**Status: done in the notebooks.** The remaining work is making the figures
reproducible so the report and dashboard read them from disk instead of me
exporting screenshots by hand.

### 2.1 Figure generation

- [ ] **File** `src/reporting/figures.py`
- **What it does** Rebuilds every EDA figure from the cleaned panel and saves it to
  `reports/figures/`.
- **Figures to produce**
  - Price and cumulative return per ticker
  - Return distribution with normal overlay, skew and kurtosis annotated
  - Rolling 21 day and 63 day volatility, showing clustering
  - Return autocorrelation at lags 1 to 21 with confidence bands
  - Correlation heatmap across the universe
  - Volatility regime shading over the 2005 to 2024 window
- **Tests** Every function returns a figure object and writes a file. Running twice
  overwrites rather than erroring.
- **Why it matters** A figure I cannot regenerate is a figure I cannot defend. If a
  reviewer asks why I claim volatility clusters, I should be able to re run one
  command.

### 2.2 Stationarity report

- [ ] **File** `src/reporting/stationarity.py`
- **What it does** Runs the ADF test across the universe and writes a table of the
  minimum fractional differentiation order per ticker.
- **Tests** A random walk fails the test. White noise passes. An AR(1) with
  coefficient 0.5 passes. These three cases pin the implementation.
- **Why it matters** Notebook 2 hand rolled the ADF regression
  `Δy_t = α + γ·y_{t-1} + Σ δ_i·Δy_{t-i} + ε_t` and compared γ's t statistic against
  MacKinnon's critical values (-3.43 at 1%, -2.86 at 5%, -2.57 at 10%). Having the
  three known cases as tests is what makes me trust the number on real data.

---

## Phase 3: Feature Engineering

The brief asks for moving averages, lagged returns, RSI and rolling volatility.
Those are the core. Fractional differentiation goes alongside them because it is
the one feature that solves the stationarity and memory trade off properly.

### 3.1 Technical features

- [ ] **File** `src/features/technical.py`
- **What it does** The whole indicator set, each as a standalone function taking a
  series and returning a series, so each one is testable on its own.
- **Functions**
  - `lagged_returns(close, lags=[1,2,3,5,10,21])` yesterday's return, last week's, last month's
  - `moving_average(close, window)` and `ma_ratio(close, window)` the ratio, not the raw level, because the level is not stationary
  - `rsi(close, window=14)` Wilder's smoothing, not a simple mean
  - `rolling_volatility(returns, window)` annualised by `sqrt(252)`
  - `momentum(close, window)` cumulative return over the window
  - `volatility_ratio(returns, short, long)` short vol over long vol, a regime proxy
  - `distance_from_high(close, window)` drawdown from the rolling peak
  - `volume_ratio(volume, window)` today's volume against its average
- **Tests** RSI is exactly 100 on a monotonically rising series and 0 on a falling
  one. A constant series gives zero volatility. Every function's output has the
  same index as its input. Every function's first `window - 1` values are NaN and
  never 0.
- **Why it matters** Two traps here. First, NaN filled with 0 is a lie that the
  model will happily learn from, which is why the tests check for NaN explicitly.
  Second, `ma_ratio` is used instead of the raw moving average because a price level
  from 2005 means nothing next to one from 2024, and a tree will happily split on
  "price above 400" and call it a signal.

### 3.2 Fractional differentiation

- [ ] **File** `src/features/fracdiff.py`
- **What it does** Differentiates a series by a fractional order, making it
  stationary while keeping memory.
- **Functions**
  - `fracdiff_weights(d, threshold)` the recursion `ω_0 = 1`, `ω_k = -ω_{k-1}·(d-k+1)/k`
  - `frac_diff(series, d, threshold)` fixed width window version
  - `min_ffd_order(series, candidates, config)` searches for the smallest `d` passing ADF
  - `adf_statistic(series, max_lags)` the hand rolled test
- **Tests** `d=0` returns the original series exactly. `d=1` returns the first
  difference exactly. Weights sum to the expected value. A random walk needs
  `d > 0`. A stationary series returns `d = 0`.
- **Why it matters** This is López de Prado's pitfall 4, integer differentiation.
  Raw prices are not stationary so a model cannot use them. First differences are
  stationary but throw away all memory, which is why returns look like noise. A
  fractional order between 0 and 1 is the compromise, and finding the minimum `d`
  that clears the ADF threshold keeps the maximum memory possible. This is the most
  distinctive single thing in the project.
- **Note** Notebook 2 flagged a `d = 0` trap. If the search returns 0 for a price
  series, the ADF test is almost certainly being handed returns rather than prices.
  Keep that guard in the production version.

### 3.3 Feature matrix assembly

- [ ] **File** `src/features/build.py`
- **What it does** Builds the full feature matrix across the panel with no look
  ahead anywhere.
- **Functions**
  - `build_features(panel, config) -> pd.DataFrame` MultiIndex on (date, ticker)
  - `drop_warmup(frame, config)` trims the first `max_lookback` rows per ticker
  - `assert_no_lookahead(frame, panel)` the leak check from notebook 3, as an assertion
  - `feature_names(config) -> list[str]` the single source of truth for column order
- **Tests** Shifting the input by one day shifts the output by exactly one day
  (a leak would break this). No feature correlates above 0.99 with the target.
  Warmup rows are dropped per ticker, not globally. Column order is stable across
  runs.
- **Why it matters** Three leak checks caught problems in notebook 3. The shift test
  is the strongest of them: if a feature at time `t` uses data from `t+1`, shifting
  the whole input changes the relationship and the test fails. Running this as an
  assertion inside the build means the leak can never reach a model silently.

### 3.4 Labelling: fixed horizon

- [ ] **File** `src/labelling/fixed_horizon.py`
- **What it does** The simple label the brief asks for: next day return, and its
  sign for classification.
- **Functions**
  - `forward_return(close, horizon=1)`
  - `label_sign(forward)` with a zero return rule that is stated, not implied
  - `label_fixed_horizon(panel, config) -> pd.Series`
- **Tests** The label at time `t` equals the return from `t` to `t+horizon`. The last
  `horizon` rows are NaN. The class balance on real data lands near 55% up, matching
  the notebook.
- **Why it matters** This is the baseline label and the one the brief specifies. It
  is also pitfall 5, fixed time horizon labelling: a fixed 1 day horizon ignores
  that a 1% move in a calm market and a 1% move in a panic are not the same event.
  Triple barrier labelling fixes that, which is why both exist.

### 3.5 Labelling: triple barrier

- [ ] **File** `src/labelling/triple_barrier.py`
- **What it does** Labels each observation by which of three barriers it hits first:
  a profit target, a stop loss, or a time limit.
- **Functions**
  - `daily_volatility(close, span)` exponentially weighted, the barrier scale
  - `apply_triple_barrier(close, events, barriers, vertical)` returns which barrier hit and when
  - `label_triple_barrier(panel, config) -> pd.DataFrame` with `label`, `touch_time`, `return`
- **Tests** A series rising straight up hits the profit barrier. A series falling
  straight down hits the stop. A flat series hits the time barrier. Barrier widths
  scale with volatility, so the same label means the same thing in 2008 and 2017.
- **Why it matters** Barriers scaled by volatility make labels comparable across
  regimes. A 2% target in 2017 (low vol) and in 2008 (high vol) are completely
  different bets, and a fixed percentage target quietly makes the model learn the
  volatility regime instead of the signal.

### 3.6 Sample weights

- [ ] **File** `src/labelling/sample_weights.py`
- **What it does** Down weights overlapping labels so the model does not treat
  correlated samples as independent evidence.
- **Functions**
  - `concurrency(close_index, touch_times)` how many labels are live on each day
  - `average_uniqueness(touch_times, concurrency)` per sample uniqueness
  - `time_decay_weights(uniqueness, decay)` older samples count less
  - `sample_weights(labels, config) -> pd.Series`
- **Tests** Non overlapping labels all get weight 1. Fully overlapping labels get
  weight `1/n`. Weights are strictly positive and sum to a documented total.
- **Why it matters** Pitfall 7, non IID sample weighting. Every machine learning
  library assumes samples are independent. Financial labels built on overlapping
  windows are not, so the model sees the same information several times and becomes
  overconfident. For my 1 day fixed horizon the labels do not overlap, but the
  features do, through rolling windows, which is exactly why the embargo is
  mandatory even when purging is not.

---

## Phase 4: Return Prediction Models

The brief says Random Forest or LSTM. Random Forest is primary because 5,000 bars
per ticker is far too little data for an LSTM to beat a tree, and I want to be able
to defend that choice rather than apologise for it. LightGBM goes alongside as the
stronger gradient boosting comparison. An LSTM is included as an optional
comparison, built honestly so that losing is a real result.

### 4.1 Baselines

- [ ] **File** `src/models/baseline.py`
- **What it does** The things a model must beat before it is interesting.
- **Classes**
  - `BuyAndHold` the benchmark that is genuinely hard to beat
  - `AlwaysUp` always predicts up, which on this data is right about 55% of the time
  - `RandomForecaster` coin flip, to establish what luck looks like
  - `LastSignPersistence` tomorrow matches today, the naive momentum rule
- **Tests** `AlwaysUp` scores the empirical up rate exactly. `RandomForecaster` with
  a fixed seed is reproducible. Every baseline exposes the same `fit` and `predict`
  interface as the real models.
- **Why it matters** Notebook 3 established 53.4% as the random hit rate on this
  data. Knowing what a coin flip looks like before seeing a model output is the
  single cheapest protection against fooling myself. `AlwaysUp` is the harder bar,
  and it is the one my model lost to.

### 4.2 Estimator wrapper

- [ ] **File** `src/models/estimator.py`
- **What it does** One interface over RandomForest and LightGBM so swapping them
  changes a config value and nothing else.
- **Classes and functions**
  - `ReturnPredictor` with `fit(X, y, sample_weight)`, `predict`, `predict_proba`
  - `build_estimator(config)` reads `model.type` from config
  - `feature_importance(model, names)` both split based and permutation based
  - `fit_with_weights(model, X, y, weights)` passes sample weights through properly
- **Tests** Fitting twice with the same seed gives identical predictions. Sample
  weights actually change the fit. Feature importance names line up with the input
  columns. The classification and regression paths both work.
- **Why it matters** Random Forest on a panel of 11 tickers is not the same as
  Random Forest on 11 separate series. Decide and document which one this is. I am
  pooling, which assumes the signal is shared across tickers, and that assumption
  needs to be in the docstring.

### 4.3 Sequence model, optional

- [ ] **File** `src/models/sequence.py`
- **What it does** An LSTM on windowed sequences, as a comparison rather than a
  centrepiece.
- **Functions**
  - `make_sequences(X, y, lookback)` builds the 3D tensor with no leak across the split
  - `build_lstm(config)` a deliberately small network, since the dataset is small
  - `fit_lstm(model, sequences, config)` early stopping on a chronological validation slice
- **Tests** Sequence windows never cross a train/test boundary. Output shape matches
  the input length minus the lookback. A fixed seed is reproducible.
- **Why it matters** The honest expectation is that the LSTM does not beat the
  Random Forest, because 5,000 daily bars is a tiny dataset for a recurrent network
  and daily returns have almost no autocorrelation to exploit. Running it anyway and
  reporting that is a better result than skipping it, as long as I give it a fair
  chance and count it as a trial.

### 4.4 Persistence

- [ ] **File** `src/models/persistence.py`
- **What it does** Saves and loads a model with everything needed to reproduce its
  predictions.
- **Functions**
  - `save_model(model, config, metrics, path)` writes the model and a metadata JSON
  - `load_model(path) -> ModelBundle`
  - `model_metadata(model, config, data_window)` records the training window, feature list, library versions, git commit
- **Tests** A round trip gives identical predictions. Loading a bundle whose feature
  list does not match the current config raises rather than predicting nonsense.
- **Why it matters** The freight project taught me this: every prediction must be
  traceable to the run that produced it. The feature list check is the guard that
  catches the quiet failure where I add a feature and an old model happily predicts
  on the wrong columns.

---

## Phase 5: Time Series Model Evaluation

This is the phase that separates a portfolio project from a tutorial. The brief
asks for a time series split with no shuffling. That is the minimum. Purged cross
validation and CPCV are what make the numbers defensible.

### 5.1 Purged cross validation

- [ ] **File** `src/validation/purged_cv.py`
- **What it does** A time series splitter that purges training samples overlapping
  the test set and applies an embargo after it.
- **Classes and functions**
  - `PurgedKFold` with `split(X, y, groups)` matching the scikit-learn interface
  - `purge_train_indices(train, test, touch_times)` removes overlap
  - `apply_embargo(train, test, embargo_pct)` removes the gap after the test set
  - `train_test_split_chronological(X, y, test_start)` the plain split the brief asks for
  - `assert_no_overlap(train, test, lookback)` raises on any leak
- **Tests** The embargo gap is exactly `embargo_pct * n` bars wide. No train index
  falls inside a test window. No train index falls inside the embargo. A test date
  never precedes a train date. The simple chronological split has zero overlap.
- **Why it matters** Pitfall 8, cross validation leakage. Standard k fold shuffles,
  which lets the model train on next week and test on last week. Even an unshuffled
  split leaks if a feature at the last training day uses a 21 day rolling window
  that reaches into the test set. Purging removes training samples whose labels
  overlap the test set. The embargo removes the gap after it, where features still
  share data. For my 1 day horizon the labels do not overlap so purging does little,
  but the features do overlap, so the embargo does the real work. That distinction
  belongs in the docstring.

### 5.2 Combinatorial purged cross validation

- [ ] **File** `src/validation/cpcv.py`
- **What it does** Splits the data into groups and tests on every combination, giving
  many backtest paths rather than one.
- **Functions**
  - `combinatorial_splits(n_groups=6, n_test_groups=2)` yields the 15 combinations
  - `CombinatorialPurgedCV` purges and embargoes each combination
  - `backtest_paths(splits)` reassembles the combinations into distinct paths
  - `path_count(n_groups, n_test_groups)` how many paths the configuration produces
- **Tests** 6 groups taking 2 at a time gives exactly 15 splits. Every split is
  purged. Every path covers the full timeline exactly once. No two paths are
  identical.
- **Why it matters** Pitfall 9, walk forward backtesting. A single walk forward gives
  one equity curve, one Sharpe, and no way to know whether that Sharpe was luck.
  CPCV gives a distribution of Sharpes from the same data, and the spread of that
  distribution is the honest measure of uncertainty. This is what makes the Monte
  Carlo evaluation the brief asks for meaningful rather than decorative.

### 5.3 Metrics

- [ ] **File** `src/validation/metrics.py`
- **What it does** Every metric that matters for a return prediction, not just
  accuracy.
- **Functions**
  - `information_coefficient(predicted, actual)` Spearman rank correlation
  - `ic_by_period(predicted, actual, freq)` IC over time, to check stability
  - `hit_rate(predicted, actual)` and `edge_over_always_up(predicted, actual)`
  - `directional_accuracy(predicted, actual)`
  - `mae`, `rmse`, `r2` for the regression view
  - `metric_table(predicted, actual, baselines)` one frame comparing everything
- **Tests** Perfect prediction gives IC 1.0. Inverted prediction gives IC -1.0.
  Random prediction gives IC near 0 within the Monte Carlo error. Hit rate on a
  known vector matches a hand computed value.
- **Why it matters** IC and hit rate measure different things and mine disagreed:
  IC +0.0279 (consistent, positive) against edge over always up -0.0357
  (consistent, negative). IC says the ranking is right. Hit rate says the sign is
  wrong. The reason is that my features are contrarian in a market that rises 55% of
  days. Reporting both and explaining the conflict is more valuable than reporting
  only the flattering one. For context: an IC of 0.02 to 0.05 is tradeable at scale,
  0.1 is exceptional.

### 5.4 Research pipeline

- [ ] **File** `src/pipelines/research.py`
- **What it does** Runs feature research end to end: builds features, labels, fits,
  and reports importance and IC across regimes. Does not backtest.
- **Functions**
  - `run_research(config) -> ResearchResult`
  - `importance_across_folds(model, splits)` stability of importance, not a single ranking
  - `ic_across_regimes(predictions, regimes)` does the signal survive a regime change
  - `feature_redundancy(features)` clustered correlation
- **Tests** The pipeline runs end to end on a small synthetic panel. Every
  configuration it tries appends a trial log row.
- **Why it matters** Pitfall 2, research through backtesting. The temptation is to
  run a backtest, see a bad Sharpe, tweak a feature, and run it again. Twenty
  iterations of that and the strategy is fitted to the test set with no way to
  measure the damage. Feature research is done on importance and IC. The backtest is
  run once the features are settled.

### 5.5 Research CLI

- [ ] **File** `entrypoint/research.py`
- **What it does** Command line entry point for the research pipeline.
- **Arguments** `--config`, `--refresh`, `--log-level`, `--label-method`, `--model`,
  `--output`
- **Tests** `--help` works. An invalid config path fails with a clear message, not a
  traceback.

---

## Phase 6: Trading Strategy and Backtesting

The brief asks for a long/short strategy backtested against buy and hold, plus
momentum strategy backtesting and mean reversion pairs trading. All three go here.
Costs are charged from the first line of code.

### 6.1 Costs

- [ ] **File** `src/backtest/costs.py`
- **What it does** Charges commission, slippage and borrow. Built first, deliberately,
  so no backtest can exist without it.
- **Functions**
  - `commission(turnover, bps)` 2 bps per side from config
  - `slippage(turnover, bps)` 3 bps per side from config
  - `borrow_cost(short_exposure, annual_rate, days)` shorts are not free
  - `total_cost(positions, config) -> pd.Series` the full per period drag
  - `turnover(positions)` the absolute change in position, the thing costs scale with
  - `breakeven_return(turnover, config)` what the strategy needs just to stand still
- **Tests** Holding a position with no change charges zero. A full flip charges the
  round trip (10 bps). Turnover of a constant position is zero. `breakeven_return`
  on a daily flipping strategy returns a figure near 25% annual, matching the
  arithmetic from notebook 4.
- **Why it matters** This is the finding that killed every ML strategy in notebook
  4. Daily flipping costs about 5,063 times what buy and hold costs. A strategy
  trading daily at 5 bps a side needs roughly 25% gross annual return before it
  breaks even. Having `breakeven_return` as a function means I can print that number
  next to every strategy rather than discovering it at the end.

### 6.2 Strategy rules

- [ ] **File** `src/backtest/strategy.py`
- **What it does** Converts a signal into a position. Every strategy the project
  tests lives here behind one interface.
- **Strategies**
  - `MLLongShort` sign of the prediction, the brief's main strategy
  - `MLLongOnly` long when positive, flat otherwise, which cut the cost drag in the notebook
  - `MLThresholded` only trade when the predicted magnitude clears a threshold, the cheapest cost control
  - `Momentum` long the top `k` by trailing `n` month return, short the bottom `k`
  - `MeanReversionPairs` the pairs strategy, detailed in 6.5
  - `BuyAndHold` the benchmark
- **Shared interface**
  - `positions(signal, prices, config) -> pd.DataFrame` one column per ticker, values in `[-1, 1]`
  - `name` and `parameters` for the trial log
- **Tests** Every strategy returns positions bounded by the leverage limit. Position
  changes happen on the bar after the signal, never the same bar. A flat signal gives
  flat positions. Each strategy is reproducible with a fixed seed.
- **Why it matters** Three traps. First, the one bar delay: a signal computed from
  today's close cannot be traded at today's close, and skipping that is the single
  most common backtest error. Second, position sizing is where the cost battle is
  won, so the three ML variants exist to measure it. Third, pitfall 6, learning side
  and size simultaneously: the model calls direction, the strategy decides size,
  and keeping them separate is what makes meta labelling possible later.

### 6.3 Backtest engine

- [ ] **File** `src/backtest/engine.py`
- **What it does** Runs positions against prices and produces an equity curve, gross
  and net.
- **Functions**
  - `run_backtest(positions, returns, config) -> BacktestResult`
  - `gross_returns(positions, returns)`
  - `net_returns(gross, positions, config)` costs subtracted
  - `equity_curve(returns, initial)` cumulative
  - `BacktestResult` dataclass holding gross, net, positions, turnover, costs, metadata
- **Tests** A flat position gives zero return. A constant long position reproduces
  buy and hold exactly, gross. Net is always less than or equal to gross. The
  position at time `t` multiplies the return from `t` to `t+1`, never from `t-1` to
  `t`. An equity curve from known returns matches a hand computed value.
- **Why it matters** The constant long test is the most important one in the project.
  If a strategy that is always fully long does not exactly reproduce buy and hold
  gross, the engine has an alignment bug and every number it has ever produced is
  wrong. Keep that test at the top of the file.

### 6.4 Performance measurement

- [ ] **File** `src/backtest/performance.py`
- **What it does** Every statistic needed to judge a strategy.
- **Functions**
  - `annualised_return(returns)`, `annualised_volatility(returns)`
  - `sharpe_ratio(returns, risk_free)` and `sortino_ratio(returns)`
  - `max_drawdown(equity)` and `drawdown_series(equity)`
  - `calmar_ratio(returns, equity)`
  - `hit_rate_trades(positions, returns)`
  - `turnover_annual(positions)`
  - `exposure(positions)` long, short and net exposure over time
  - `performance_table(results, benchmark) -> pd.DataFrame` the strategy comparison
- **Tests** Sharpe of a constant positive return series is infinite or guarded.
  Max drawdown of a monotonically rising curve is zero. Max drawdown of a known
  series matches a hand computed value. Annualisation uses 252 days consistently
  everywhere.
- **Why it matters** `performance_table` is the single table that goes in the README
  and the dashboard. It must show gross and net side by side, because the gap between
  them is the most informative number in the whole project.

### 6.5 Mean reversion pairs trading

- [ ] **File** `src/backtest/pairs.py`
- **What it does** The pairs strategy. Finds cointegrated pairs, trades the spread
  when it diverges, exits when it reverts.
- **Functions**
  - `find_pairs(panel, config)` correlation screen then cointegration test
  - `engle_granger_test(a, b)` regress one on the other, ADF the residual
  - `hedge_ratio(a, b, window)` rolling OLS beta, so it adapts
  - `spread_zscore(a, b, hedge_ratio, window)` the trading signal
  - `pairs_positions(zscore, entry_z, exit_z, stop_z)` long one leg, short the other
  - `half_life(spread)` how fast the spread reverts, which sets the holding period
- **Tests** Two series built to be cointegrated are found. Two independent random
  walks are rejected. The z score of a stationary spread has mean near 0 and
  standard deviation near 1. Entry at `z > 2` and exit at `z < 0.5` fires on a
  synthetic diverging spread. The hedge ratio is computed on a rolling window and
  never uses future data.
- **Why it matters** This is the one strategy in the project with a real economic
  story behind it, which makes it a useful contrast to the ML signal. Two companies
  in the same sector share risk factors, so their price ratio should be mean
  reverting even though each price is a random walk. The natural pair in my universe
  is XOM with another energy name, or JPM with another financial. Three warnings
  worth writing into the docstring: cointegration found in sample often breaks out of
  sample, the hedge ratio must be rolling because a static one silently decays, and
  the half life tells me whether the holding period is days or months, which decides
  whether costs destroy it. Every pair I test counts as a trial.

### 6.6 Momentum strategy backtesting

- [ ] **File** `src/backtest/momentum.py`
- **What it does** The classic cross sectional momentum strategy, as a reference
  point with decades of published evidence behind it.
- **Functions**
  - `trailing_return(panel, lookback_months)` the ranking signal, typically 12 months
  - `rank_cross_section(signal, date)` ranks tickers on each date
  - `momentum_positions(ranks, top_k, bottom_k, config)` long the winners, short the losers
  - `skip_month(signal)` skips the most recent month, the standard correction for short term reversal
  - `rebalance_schedule(index, frequency)` monthly rather than daily, which is what keeps costs survivable
- **Tests** The top ranked ticker gets a long position. Ranks are computed per date,
  never pooled across dates. Skipping a month shifts the signal by exactly 21 bars.
  Monthly rebalancing produces roughly a twelfth of the turnover of daily.
- **Why it matters** Momentum is the most documented anomaly in finance, which makes
  it the right sanity check: if my pipeline cannot reproduce a weak version of a
  known effect, the pipeline is broken, not the market. It also demonstrates the cost
  lesson from the other direction, because monthly rebalancing is cheap enough to
  survive while daily ML flipping is not. The skipped month matters because the most
  recent month tends to reverse, so including it weakens the signal.

### 6.7 Evaluation pipeline

- [ ] **File** `src/pipelines/evaluate.py`
- **What it does** Runs every strategy through CPCV and produces the Sharpe
  distributions.
- **Functions**
  - `run_evaluation(config) -> EvaluationResult`
  - `evaluate_strategy(strategy, splits, config)` one strategy across every path
  - `compare_strategies(results, benchmark)` the full comparison
  - `regime_breakdown(results, regimes)` performance per volatility regime
- **Tests** Runs end to end on a synthetic panel. Every strategy produces one result
  per CPCV path. Every run appends to the trial log.

### 6.8 Evaluate CLI

- [ ] **File** `entrypoint/evaluate.py`
- **What it does** Command line entry point for the full evaluation.
- **Arguments** `--config`, `--strategies`, `--n-paths`, `--monte-carlo-runs`,
  `--output`, `--log-level`
- **Tests** `--help` works. Writes results to the configured output path.

---

## Phase 7: Portfolio and Risk Analysis

The brief asks for Monte Carlo risk simulation and efficient frontier optimization.
Both go here, along with the portfolio construction that sits between a signal and a
position.

### 7.1 Monte Carlo resampling

- [ ] **File** `src/evaluation/monte_carlo.py`
- **What it does** Resamples returns to produce a distribution of outcomes instead
  of a single equity curve. This is the evaluation method the brief explicitly asks
  for.
- **Functions**
  - `iid_bootstrap(returns, n_runs, seed)` the naive version, kept for comparison
  - `block_bootstrap(returns, block_size, n_runs, seed)` 21 day blocks, the one to use
  - `stationary_bootstrap(returns, mean_block, n_runs, seed)` random block lengths
  - `simulate_paths(returns, method, config) -> np.ndarray` shape `(n_runs, n_periods)`
  - `path_statistics(paths)` the distribution of Sharpe, return, drawdown
  - `probability_of_beating(strategy_paths, benchmark)` the share of paths that win
  - `confidence_interval(statistic_samples, level)` percentile based
- **Tests** Block bootstrap with `block_size=1` matches IID bootstrap. A fixed seed
  is reproducible. Resampled mean return is close to the original mean. Volatility
  clustering, measured as the autocorrelation of squared returns, is preserved by the
  block bootstrap and destroyed by IID.
- **Why it matters** This is the most important test in the file, and notebook 4
  verified it numerically: squared return autocorrelation was 0.062 under block
  bootstrap and -0.0006 under IID. Volatility clusters in real markets, calm follows
  calm and panic follows panic. IID resampling shuffles that away and produces
  drawdowns that are far too small, which makes every strategy look safer than it is.
  21 days is about one trading month, long enough to hold a volatility regime
  together. The headline output is `probability_of_beating`: my long only ML strategy
  beat buy and hold on 22.1% of paths, which is a far more honest statement than
  "Sharpe 0.70 against 0.85".

### 7.2 Monte Carlo risk simulation

- [ ] **File** `src/evaluation/risk_simulation.py`
- **What it does** Forward looking risk: simulates future paths from the fitted
  return distribution and reports the tail.
- **Functions**
  - `simulate_forward(returns, horizon_days, n_runs, method)` normal, Student t, and bootstrap methods
  - `value_at_risk(paths, level)` the loss not exceeded at the given confidence
  - `conditional_var(paths, level)` the average loss given that VaR is breached
  - `drawdown_distribution(paths)` how deep a drawdown to expect, and how often
  - `time_to_recovery(paths)` how long underwater
  - `ruin_probability(paths, threshold)` the chance of falling below a floor
  - `stress_scenarios(returns, scenarios)` replay 2008 and 2020 volatility against the current portfolio
- **Tests** VaR at 95% on a normal distribution matches the analytic value. CVaR is
  always worse than VaR at the same level. A Student t simulation produces fatter
  tails than a normal one. A fixed seed is reproducible.
- **Why it matters** Backtest risk and forward risk are different questions. The
  backtest says what happened. This says what could happen next. Student t matters
  because daily equity returns have excess kurtosis, so a normal assumption
  understates the tail badly, which is the single most consequential modelling error
  in risk management. The stress scenarios are the honest version: instead of
  simulating a crash, replay a real one.

### 7.3 Portfolio construction

- [ ] **File** `src/portfolio/construction.py`
- **What it does** Turns per ticker signals into portfolio weights. Starts simple,
  as it should.
- **Functions**
  - `equal_weight(tickers)` the baseline, and a genuinely hard one to beat
  - `signal_weighted(signals, config)` weights proportional to signal strength
  - `inverse_volatility(returns, window)` risk parity's simple cousin
  - `risk_parity(returns, window)` equal risk contribution per asset
  - `apply_constraints(weights, config)` leverage cap, position cap, long only option
  - `rebalance(weights, schedule, current)` only trade on rebalance dates
- **Tests** Weights sum to 1 (or to the leverage limit). Equal weight gives `1/n`
  each. Inverse volatility gives the low volatility asset a larger weight.
  Constraints are never violated. Rebalancing on a monthly schedule produces no
  trades in between.
- **Why it matters** Equal weight beats most optimised portfolios out of sample, and
  that is not a joke, it is the headline finding of a large literature. The reason is
  that optimisation needs an estimate of expected returns, and those estimates are so
  noisy that the optimiser mostly fits noise. Starting from equal weight means I can
  measure whether optimisation actually helps rather than assuming it does.

### 7.4 Efficient frontier optimization

- [ ] **File** `src/portfolio/optimisation.py`
- **What it does** Mean variance optimisation, the efficient frontier, and the
  tangency portfolio.
- **Functions**
  - `expected_returns(returns, method)` historical mean, and a shrunk estimate
  - `covariance_matrix(returns, method)` sample, Ledoit Wolf shrinkage, and exponentially weighted
  - `minimum_variance_portfolio(cov, constraints)`
  - `maximum_sharpe_portfolio(mu, cov, risk_free, constraints)` the tangency portfolio
  - `efficient_frontier(mu, cov, n_points, constraints)` the frontier curve
  - `frontier_figure(frontier, assets, portfolios)` the plot, with individual assets marked
  - `out_of_sample_frontier(returns, config)` fit on the training window, evaluate on the test window
- **Tests** The minimum variance portfolio has lower variance than equal weight, in
  sample. Weights sum to 1. The frontier is monotonic: higher target return means
  higher variance. The maximum Sharpe portfolio sits on the frontier. Shrunk
  covariance is better conditioned than the sample estimate.
- **Why it matters** `out_of_sample_frontier` is the function that matters and most
  tutorials skip it. The in sample frontier always looks beautiful because it was
  fitted to that exact data. Refitting on the training window and plotting where
  those weights actually landed in the test window shows the gap, and that gap is the
  real lesson of Markowitz optimisation. Covariance shrinkage exists because the
  sample covariance of 11 assets from 5,000 observations is noisy enough to make the
  optimiser concentrate everything in two names. Ledoit Wolf pulls it toward a
  structured target and produces weights that survive.

### 7.5 Risk analysis

- [ ] **File** `src/portfolio/risk.py`
- **What it does** Decomposes portfolio risk so I can see where it is actually coming
  from.
- **Functions**
  - `portfolio_volatility(weights, cov)`
  - `risk_contribution(weights, cov)` each asset's share of total risk
  - `marginal_risk(weights, cov)` the effect of a small weight change
  - `diversification_ratio(weights, cov)` weighted average volatility over portfolio volatility
  - `rolling_correlation(returns, window)` correlation rising in a crisis is the thing to watch
  - `beta_to_benchmark(returns, benchmark)` and `tracking_error(returns, benchmark)`
- **Tests** Risk contributions sum to the portfolio volatility. A single asset
  portfolio has a diversification ratio of 1. An equal weight portfolio of
  uncorrelated assets has a ratio near `sqrt(n)`.
- **Why it matters** Weights are not risk. A 10% weight in a high volatility name can
  carry 30% of the portfolio's risk, and only a decomposition shows that. Rolling
  correlation is the crisis variable: correlations go to 1 exactly when
  diversification is needed most, which is why a correlation matrix computed over the
  full sample is misleading.

---

## Phase 8: Experiment Tracking and Reporting

This phase exists because of pitfall 10, backtest overfitting. Without a trial count
I cannot compute a deflated Sharpe, and without a deflated Sharpe every number in the
project is unfalsifiable.

### 8.1 Trial log

- [ ] **File** `src/evaluation/trial_log.py`
- **What it does** Records every configuration tried, with enough detail to count
  trials honestly.
- **Functions**
  - `TrialLog` appends to a CSV or parquet file, never overwrites
  - `log_trial(config, features, model_params, label, metrics)` one row per attempt
  - `trial_count(filters)` how many trials, used by the deflated Sharpe
  - `trial_table()` every trial, for the report and dashboard
  - `best_trial(metric)` and `sharpe_variance()` the inputs the DSR needs
- **Tests** Appending does not lose earlier rows. The trial count matches the row
  count. The same configuration logged twice is recorded twice, because it was tried
  twice. The file survives a process restart.
- **Why it matters** Build this before the first model run. Notebook 3 and 4 counted
  42 trials between them, and that number is the input that moved the luck threshold
  to around 0.46 and sank both ML strategies. Building the trial log on day three
  costs an hour. Building it on day thirty costs the credibility of every number in
  the project, because the count cannot be reconstructed after the fact.

### 8.2 Deflated Sharpe ratio

- [ ] **File** `src/evaluation/deflated_sharpe.py`
- **What it does** Corrects the Sharpe ratio for sample size, non normality, and the
  number of trials run.
- **Functions**
  - `probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, benchmark)` the PSR
  - `expected_max_sharpe(n_trials, sharpe_variance)` the luck threshold, roughly `sqrt(2·log(I)) · σ[SR]`
  - `deflated_sharpe_ratio(sharpe, trials, n, skew, kurtosis, variance)` the DSR
  - `minimum_track_record_length(sharpe, skew, kurtosis, target)` how much data is needed to be confident
  - `luck_threshold(n_trials, variance)` the number a strategy must clear
- **Tests** PSR rises with sample size for a fixed Sharpe. PSR falls with negative
  skew. The expected maximum Sharpe rises with the trial count. A strategy with
  Sharpe below the luck threshold gets a DSR near zero. The published worked example
  reproduces.
- **Why it matters** This is the honesty engine of the project. Try 42 strategies and
  the best one will look good even if every single one is worthless, because the
  maximum of 42 random draws is not near zero. The expected maximum Sharpe formula
  quantifies exactly how good the best of `I` trials looks by luck alone, and the
  strategy must clear that bar, not zero. Notebook 4 found a threshold near 0.46 at
  42 trials, and both ML strategies were below it. That is the finding, and it is
  worth more than a strategy that looks profitable because the trial count went
  unreported.

### 8.3 MLflow tracking

- [ ] **File** `src/tracking/mlflow_tracking.py`
- **What it does** Logs every run's parameters, metrics and artifacts to MLflow,
  degrading gracefully when no server is reachable.
- **Functions**
  - `MLflowTracker` with `is_available`, `start_run`, `log_params`, `log_metrics`, `log_artifact`
  - `get_tracker()` module level singleton
  - `log_backtest(result, config)` the whole backtest as one run
  - `log_figure(figure, name)` equity curves and frontier plots as artifacts
  - `recent_runs()` and `run_url(run_id)` for the dashboard
- **Tests** The tracker works with no server reachable and sets `is_available` to
  False rather than raising. Parameters and metrics round trip. A run with no metrics
  does not crash.
- **Why it matters** The freight project taught two lessons here. MLflow 3 removed
  the file backend, so `file:./mlruns` raises and the default must be
  `sqlite:///mlflow.db`. And a tracker that crashes the pipeline when the server is
  down is worse than no tracker, so every call is wrapped and `is_available` is
  checked rather than assumed.

### 8.4 Report generation

- [ ] **File** `src/reporting/report.py`
- **What it does** Writes the research findings to a file, reading every number from
  the actual run rather than from numbers I typed in.
- **Functions**
  - `build_report(research_result, evaluation_result, config, path)`
  - `summary_table(results)` the strategy comparison
  - `trial_summary(trial_log)` how many trials, and what the luck threshold was
  - `findings_section(results)` the honest conclusions
- **Tests** Runs end to end and writes a file. Every number comes from a result
  object, with no hardcoded figures anywhere.
- **Why it matters** The freight report quoted the training run rather than numbers
  copied by hand, and that is what made it defensible. Same rule here.

### 8.5 Results notebook

- [ ] **File** `notebooks/05_results.ipynb`
- **What it does** The one notebook that does import from `src/`, presenting the final
  results from the production pipeline.
- **Why it matters** Notebooks 1 to 4 were deliberately self contained with no
  imports, because they were about working things out by hand. This one is different:
  its job is to show that the production code reproduces what I found by hand. If
  notebook 5 disagrees with notebook 4, one of them is wrong and that is worth
  knowing.

### 8.6 Tests and tooling

- [ ] **File** `tests/` one test module per source module
- [ ] **File** `Makefile` targets: `install`, `fetch`, `research`, `evaluate`, `test`,
  `lint`, `format`, `dashboard`, `all`
- [ ] **File** `.github/workflows/test.yml` ruff check, ruff format check, pytest
- [ ] **File** `.gitignore` data, models, mlruns, caches, notebook checkpoints
- **Note on ruff** Configure the rule set explicitly in `pyproject.toml`. The freight
  project failed CI because there was no config at all, so ruff ran its full default
  rule set. Pin the rules, ignore `BLE001` with a written reason if blind exception
  catching is needed for graceful degradation, and run `ruff format` locally before
  pushing so `ruff format --check` cannot fail in CI.

---

## Phase 9: Dashboard, built last

Built only after every number it displays exists and has been verified. A dashboard
built early becomes a place to display numbers I have not checked.

### 9.1 Backend

- [ ] **File** `dashboard/data.py` loads results, trials and figures from disk with caching
- [ ] **File** `dashboard/theme.py` the colour system, layout helpers, card, metric, row, section
- [ ] **File** `dashboard/charts.py` every chart builder, shared across pages

### 9.2 Pages

- [ ] **File** `dashboard/pages/01_overview.py` the headline findings and the honest verdict
- [ ] **File** `dashboard/pages/02_market_data.py` prices, returns, volatility, correlation, ADF results
- [ ] **File** `dashboard/pages/03_feature_research.py` importance, IC over time, regime stability, redundancy
- [ ] **File** `dashboard/pages/04_model_performance.py` baselines against models, IC, hit rate, walk forward results
- [ ] **File** `dashboard/pages/05_backtesting.py` equity curves, gross against net, costs, turnover, drawdown
- [ ] **File** `dashboard/pages/06_portfolio_risk.py` weights, frontier, risk contribution, VaR and CVaR
- [ ] **File** `dashboard/pages/07_experiments.py` the trial log, the luck threshold, the deflated Sharpe
- [ ] **File** `dashboard/app.py` the shell, sidebar navigation, global controls

### 9.3 Rules for the dashboard

- **Never show a gross figure without the net figure beside it.** The gap is the
  lesson.
- **Label every prediction with its horizon and the period it came from.** A number
  with no horizon is meaningless.
- **Always show the trial count next to any Sharpe.** A Sharpe with no trial count is
  a claim with no evidence.
- **Never present a prediction as a certainty.** Show the distribution, not the point
  estimate.
- **Every number comes from a result file on disk.** No hardcoded performance claims
  anywhere in the dashboard code.
- **Keep the refresh pattern from the freight project.** Split into a fast figure only
  update and a slow structure rebuild, so the page does not blank while data loads.

---

## 13. Build order

The order matters more than usual here, because the wrong order lets me fool myself.

1. **`src/evaluation/trial_log.py` first.** Before any model. The trial count cannot
   be reconstructed later.
2. **`src/data/universe.py` and `src/data/cleaning.py`.** Trustworthy bars before
   anything reads them.
3. **`src/features/technical.py`, `fracdiff.py`, `build.py`.** With the leak
   assertion running inside `build`.
4. **`src/labelling/fixed_horizon.py`.** The simple label first. Triple barrier and
   sample weights after the pipeline works end to end.
5. **`src/models/baseline.py`.** Get the 53.4% random hit rate and the 55% always up
   rate on my own data, before seeing any model output.
6. **`src/validation/purged_cv.py` and `metrics.py`, with tests.** Then and only then
   the estimator.
7. **`src/models/estimator.py` and `persistence.py`.**
8. **`src/backtest/costs.py` before `engine.py`.** Deliberately. Costs cannot be
   added later if they were never left out.
9. **`src/backtest/strategy.py`, `engine.py`, `performance.py`.** With the constant
   long equals buy and hold test passing before anything else is trusted.
10. **`src/backtest/momentum.py` and `pairs.py`.** The two non ML strategies, for
    contrast.
11. **`src/validation/cpcv.py` and `src/evaluation/monte_carlo.py`.** The
    distribution, not the single curve.
12. **`src/evaluation/deflated_sharpe.py`.** Apply the luck threshold to everything
    above.
13. **`src/portfolio/`.** Construction, then optimisation, then risk.
14. **`src/pipelines/` and `entrypoint/`.** Wire it together.
15. **`src/tracking/` and `src/reporting/`.** Tracking and reports.
16. **`tests/`, `Makefile`, CI.** Running throughout, finished here.
17. **`dashboard/` last.** Only once every number exists and has been checked.

---

## 14. Complete folder structure

```
stock-return-predictor/
│
├── config/
│   └── config.yaml                  universe, dates, costs, CV settings, thresholds
│
├── data/
│   ├── 01-raw/                      cached price pulls (parquet)
│   ├── 02-bars/                     cleaned, adjusted OHLCV
│   └── 03-features/                 feature matrices
│
├── src/
│   ├── config.py                    typed, validated settings
│   ├── logger.py                    logging setup and helpers
│   │
│   ├── data/
│   │   ├── loaders.py               yfinance primary, Stooq fallback, cache once
│   │   ├── universe.py              which tickers, from when, survivorship note
│   │   └── cleaning.py              splits, gaps, zero-volume days, outliers
│   │
│   ├── features/
│   │   ├── technical.py             MA, RSI, rolling volatility, lagged returns
│   │   ├── fracdiff.py              fractional differentiation (Pitfall 4)
│   │   └── build.py                 assembles the matrix, no look-ahead
│   │
│   ├── labelling/
│   │   ├── fixed_horizon.py         next-day return, the simple baseline
│   │   ├── triple_barrier.py        profit / stop / time, vol-scaled (Pitfall 5)
│   │   └── sample_weights.py        uniqueness weighting (Pitfall 7)
│   │
│   ├── validation/
│   │   ├── purged_cv.py             purging + embargo (Pitfall 8)
│   │   ├── cpcv.py                  combinatorial paths (Pitfall 9)
│   │   └── metrics.py               IC, hit rate, R-squared, not just accuracy
│   │
│   ├── models/
│   │   ├── baseline.py              buy-and-hold, always-up, random forecaster
│   │   ├── estimator.py             RandomForest / LightGBM wrapper
│   │   ├── sequence.py              LSTM comparison, optional
│   │   └── persistence.py           save and load with full metadata
│   │
│   ├── backtest/
│   │   ├── strategy.py              signal to position sizing
│   │   ├── costs.py                 commission, slippage, borrow, from day one
│   │   ├── engine.py                vectorised backtest over a path
│   │   ├── performance.py           Sharpe, drawdown, turnover, hit rate
│   │   ├── momentum.py              cross-sectional momentum strategy
│   │   └── pairs.py                 cointegration and mean-reversion pairs
│   │
│   ├── portfolio/
│   │   ├── construction.py          equal weight, inverse vol, risk parity
│   │   ├── optimisation.py          efficient frontier, tangency portfolio
│   │   └── risk.py                  risk contribution, diversification, beta
│   │
│   ├── evaluation/
│   │   ├── trial_log.py             every experiment recorded (Pitfall 10)
│   │   ├── deflated_sharpe.py       PSR and DSR
│   │   ├── monte_carlo.py           block bootstrap, resampled distributions
│   │   └── risk_simulation.py       forward VaR, CVaR, ruin, stress scenarios
│   │
│   ├── tracking/
│   │   └── mlflow_tracking.py       runs, params, metrics, artifacts
│   │
│   ├── reporting/
│   │   ├── figures.py               every figure, regenerated from data
│   │   ├── stationarity.py          ADF and fracdiff order tables
│   │   └── report.py                the written findings, numbers from the run
│   │
│   └── pipelines/
│       ├── research.py              feature importance, not backtest loops
│       └── evaluate.py              full CPCV run to Sharpe distribution
│
├── entrypoint/
│   ├── fetch.py                     pull and cache the data, once
│   ├── research.py                  feature importance across regimes
│   └── evaluate.py                  CPCV + deflated Sharpe
│
├── dashboard/
│   ├── app.py                       the shell and navigation
│   ├── theme.py                     colours, layout helpers, cards
│   ├── charts.py                    shared chart builders
│   ├── data.py                      loads results from disk, cached
│   └── pages/
│       ├── 01_overview.py
│       ├── 02_market_data.py
│       ├── 03_feature_research.py
│       ├── 04_model_performance.py
│       ├── 05_backtesting.py
│       ├── 06_portfolio_risk.py
│       └── 07_experiments.py
│
├── notebooks/
│   ├── 01_data_quality.ipynb        what the raw data actually looks like
│   ├── 02_stationarity.ipynb        ADF across d, pick the fracdiff order
│   ├── 03_feature_research.ipynb    IC, leak checks, importance stability
│   ├── 04_backtest.ipynb            costs, position rules, bootstrap, PSR
│   └── 05_results.ipynb             final results from the production pipeline
│
├── tests/
│   ├── conftest.py                  shared fixtures, synthetic panels
│   ├── test_config.py
│   ├── test_loaders.py
│   ├── test_cleaning.py
│   ├── test_technical.py
│   ├── test_fracdiff.py
│   ├── test_build.py
│   ├── test_labelling.py
│   ├── test_purged_cv.py
│   ├── test_cpcv.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_costs.py
│   ├── test_engine.py
│   ├── test_performance.py
│   ├── test_momentum.py
│   ├── test_pairs.py
│   ├── test_portfolio.py
│   ├── test_monte_carlo.py
│   ├── test_risk_simulation.py
│   ├── test_deflated_sharpe.py
│   └── test_trial_log.py
│
├── reports/
│   ├── figures/                     every figure, regenerated not screenshotted
│   └── results/                     trial log, backtest results, metric tables
│
├── models/                          trained bundles and metadata
│
├── .github/
│   └── workflows/
│       └── test.yml                 ruff check, ruff format check, pytest
│
├── pyproject.toml                   deps, ruff config, pytest config (uv managed)
├── Makefile
├── .gitignore
└── README.md
```

---

## 15. Definition of done

A module is not done until all six of these are true.

1. **It has a test module** in `tests/`, and the tests assert behaviour rather than
   just that the code runs.
2. **The known cases pass.** Every file with maths in it has at least one case where
   I know the right answer by hand: `d=0` returns the input, a perfect prediction
   gives IC 1.0, constant long reproduces buy and hold.
3. **`ruff check` and `ruff format --check` both pass** with the project's own rule
   set, not the default one.
4. **It logs what it did.** Shape and date range of every frame crossing a module
   boundary.
5. **Its docstring says why, not just what.** Especially where I made a choice that
   could have gone the other way.
6. **Any experiment it ran is in the trial log.** No exceptions, including the ones
   that failed.

---

## 16. Stitch design prompt for the dashboard

Paste the block below into Google Stitch.

---

### Prompt

Design a multi page web dashboard for a **Quantitative Research and Machine
Learning Analytics Platform**. This is not a simple stock price prediction website.
It is a research tool that shows whether a predictive signal is statistically real
and whether a trading strategy built on it would have survived transaction costs.
Design it the way a quantitative research desk or a hedge fund research team would
use it, not the way a retail trading app looks.

**The four questions every screen serves**

The dashboard exists to answer four questions, in this order. The navigation and
the page hierarchy should make that order obvious.

1. **What is happening in the market?** Prices, returns, volatility, correlations,
   stationarity.
2. **Can we predict future returns?** Features, signals, model predictions,
   information coefficient.
3. **Are the predictions statistically useful?** Significance, sample sizes, trial
   counts, deflated Sharpe ratios, confidence intervals.
4. **Would a strategy have performed well historically?** Backtests, equity curves,
   costs, drawdowns, comparison against buy and hold.

**Visual style**

- **Dark navy background.** Deep, near black navy as the base, with slightly lighter
  navy panels and cards sitting on it. Calm and low contrast overall, so the data is
  the brightest thing on the screen.
- **Restrained green and red accents.** Green for positive returns and gains, red for
  negative returns and losses. Use them sparingly and only where they carry meaning.
  Neutral grey and muted blue for benchmarks, baselines and supporting data. Never
  rely on colour alone to communicate a value, always pair it with a number or a
  label.
- **Clean charts.** Thin lines, generous whitespace, minimal gridlines, no
  decoration, no 3D, no gradients behind data, no drop shadows on chart elements.
  Every chart has a clear title and axis labels with units.
- **Compact data tables.** Dense monospaced numbers, right aligned, tight row height,
  subtle row separators rather than heavy borders. Tables are for exact values and
  should look like they belong in a research terminal.
- **A clear navigation sidebar** on the left, fixed, listing the seven pages with the
  current page clearly marked. Global controls (asset selector, date range, model
  selector, strategy selector) sit in a compact strip under the top bar rather than
  scattered across the pages.

**What to avoid**

- **Do not fill every screen with charts.** Each page should have a small number of
  charts that matter, surrounded by whitespace. An empty region is better than a
  chart nobody needs. Use KPI cards for summary numbers, charts for trends, and
  tables for exact values, and do not use a chart where a single number would do.
- **Do not make predictions look like guaranteed outcomes.** No large confident
  arrows, no price targets presented as facts, no gauges that read like a
  recommendation. Predictions are estimates with uncertainty, so show distributions,
  confidence intervals and sample sizes next to every forecast. Every prediction
  should carry its horizon and the period it came from.
- No gamified elements, no celebration states, no retail trading app styling, no
  flashing live tickers, no large profit numbers in bright green.

**The seven pages**

1. **Overview.** The landing page. Selected ticker, date range and frequency at the
   top. Then a row of KPI cards: directional accuracy, information coefficient, net
   Sharpe ratio, maximum drawdown. Then actual against predicted returns, and strategy
   performance against the benchmark. Then a panel of research findings and data
   quality warnings written in plain sentences. The headline finding on this page is
   a negative one, so the layout must present a negative result with the same
   authority as a positive one.
2. **Market Data and EDA.** Candlestick or close price chart, returns at daily,
   weekly and monthly frequency, return distribution, rolling volatility,
   autocorrelation, a correlation heatmap across the universe, a table of ADF
   stationarity test results, and a data coverage panel showing missing values and
   the last update time.
3. **Feature and Signal Research.** The most research heavy page. Feature importance
   with stability across folds, momentum and lagged return and rolling volatility
   charts, a feature correlation and redundancy view, information coefficient over
   time and across assets, signal performance split by market regime, and a table of
   hypothesis tests with sample sizes and significance.
4. **Model Performance.** A comparison of baseline, linear, tree based and optional
   deep learning models. MAE, RMSE, directional accuracy and information coefficient
   per model. Predicted against actual returns. Performance across different test
   periods. The parameters and feature set used. Walk forward validation results. The
   user selects a model and inspects it, and training metrics are visually separated
   from unseen test period results so the two can never be confused.
5. **Backtesting.** One of the most important pages. Strategy selector for momentum,
   mean reversion and the machine learning signal. Date range, asset and position
   sizing controls. A cumulative equity curve. The strategy against buy and hold and
   against the benchmark. Annualised return, volatility, Sharpe ratio and maximum
   drawdown. An explicit panel stating the transaction cost and slippage assumptions
   in basis points. Trades, turnover, exposure and a drawdown chart. Gross and net
   figures always appear side by side, because the gap between them is the finding.
6. **Portfolio and Risk.** Asset allocation and portfolio weights. An efficient
   frontier chart with individual assets marked and the equal weight, minimum
   variance and maximum Sharpe portfolios plotted on it, showing both the in sample
   and the out of sample frontier. A correlation matrix. Portfolio volatility and
   drawdown. Exposure by asset. Risk contribution per holding, shown separately from
   weight per holding. A Monte Carlo risk panel with value at risk, conditional value
   at risk and the distribution of simulated drawdowns.
7. **Research Reports and Experiments.** Saved experiments and model versions. A
   comparison of research hypotheses and findings. The dataset and feature
   configuration used for each run. Experiment metrics and backtest assumptions. The
   trial count and the luck threshold it implies, displayed prominently. Exportable
   CSV results and PDF research reports.

**Global controls, consistent across pages**

Asset selector (single or multiple), date range, data frequency (daily, weekly,
monthly), model selector, strategy selector, a run analysis action, and an export
action. Show the last data update, the selected test period and the model version
wherever they are relevant.

**Interaction**

Hovering a chart reveals the date and the exact values. Clicking a model updates its
detail panel. Training, validation and out of sample test periods are visually
distinguished on every time series chart, with shaded bands and a legend.

---

## 17. What the README should contain

The README is the first and often the only thing a reviewer reads. It should be
possible to understand what the project does, what it found, and whether to trust it
without cloning anything. Structure it in this order.

### 1. Title and one line summary

The project name and one sentence saying what it does. Not a tagline. State the
subject and the method plainly, for example: predicting next day equity returns with
tree based models, and measuring honestly whether the result survives transaction
costs and the number of strategies tried.

### 2. The finding, stated up front

**This section goes before the methodology, not after it.** The most important thing
about this project is the result, and the result is negative. Lead with it.

State plainly that daily equity direction is close to unpredictable, that the small
edge found does not survive transaction costs, and that both machine learning
strategies failed the deflated Sharpe luck threshold. Then give the table:

| Strategy | Gross Sharpe | Net Sharpe | Max drawdown | Paths beating buy and hold |
|---|---|---|---|---|

Follow it with two or three sentences explaining why a negative result is the right
outcome of a rigorous process and why it is more credible than the alternative. A
reviewer who sees a portfolio project claiming a profitable daily equity strategy
assumes a methodology error. A reviewer who sees a well measured negative result
assumes competence.

### 3. Results

Every number the project produced, with the comparison that makes it meaningful.

- The model metrics: information coefficient, hit rate, edge over always predicting
  up, MAE and RMSE, each against the baselines.
- The strategy comparison table, gross and net side by side.
- The deflated Sharpe section: the trial count, the luck threshold it implies, and
  where each strategy landed relative to it.
- The Monte Carlo result stated as a probability, not a single number. "Beat buy and
  hold on 22.1% of 1,000 resampled paths" says more than any Sharpe ratio.
- The regime breakdown, including the one regime where the model did win, stated
  without overselling it.

### 4. Features

What the project actually implements, as a short list a reader can scan. Group it:
data pipeline, feature engineering including fractional differentiation, three
labelling methods, purged cross validation and CPCV, baseline and tree and sequence
models, four strategies, portfolio optimisation, Monte Carlo evaluation, deflated
Sharpe, experiment tracking, dashboard.

### 5. Tech stack

Python version, uv for environment management, and the libraries grouped by purpose:
data, modelling, statistics, backtesting, visualisation, tracking, testing and
linting. Say what each group is for in a few words. Name anything hand rolled rather
than imported, and say why, because the hand rolled ADF test and fractional
differentiation are deliberate choices to understand the mathematics rather than
call a function.

### 6. Architecture

A diagram showing the flow from raw price data through cleaning, feature building,
labelling, model fitting, validation, backtesting and evaluation, out to the report
and the dashboard. Mark clearly where the look ahead guards sit and where costs are
charged, because those two points are what make the architecture defensible.

Below the diagram, a short paragraph on the one design rule that shapes everything:
nothing computed at time `t` may use data from after `t`, and that rule is enforced
by assertions inside the pipeline rather than by careful coding.

### 7. Project structure

The folder tree with a one line comment on each directory saying what lives there.
Copy it from section 14 of this document.

### 8. Installation and setup

Python version. The uv commands to create the environment and install everything.
Note that `pyproject.toml` holds the dependencies and there is no
`requirements.txt`. Mention that `readme = "README.md"` in `pyproject.toml` means
the README must exist before `uv sync` will run, because hatchling validates it at
build time and the error message is not obvious.

### 9. Usage

The commands in the order someone should run them, with what each one produces.

```
make fetch        pull and cache the price data
make research     feature importance and information coefficient
make evaluate     CPCV, Monte Carlo, deflated Sharpe
make test         the full test suite
make dashboard    the research dashboard
make all          everything in order
```

Note that `make` is not available on Windows by default and give the plain Python
equivalents, as in the freight project.

### 10. Screenshots and demo

The dashboard overview, the backtesting page with the equity curve, the efficient
frontier, and the Monte Carlo distribution. Each with a one line caption saying what
it shows. If there is a demo video, link it here.

### 11. Engineering decisions

The section that shows judgement. For each decision: what the choice was, what the
alternative was, and why this one won. Decisions worth writing up:

- Why Random Forest rather than an LSTM, given 5,000 bars per ticker.
- Why fractional differentiation rather than first differences or raw prices.
- Why purging and embargo rather than a plain chronological split, and why the
  embargo does the real work when labels do not overlap but features do.
- Why CPCV rather than a single walk forward.
- Why a block bootstrap with 21 day blocks rather than IID resampling, with the
  numbers showing that volatility clustering was preserved (0.062) rather than
  destroyed (-0.0006).
- Why costs were built before the backtest engine rather than added afterwards.
- Why the trial log was built before the first model.
- Why monthly rebalancing for momentum and daily for the machine learning signal, and
  what that does to the cost drag.
- Why equal weight is the portfolio baseline rather than the optimised portfolio.

Where a decision was made by running the alternatives and comparing, say so and give
both numbers. A decision made by argument is weaker than one made by measurement.

### 12. Limitations

Honest and specific, with the size of each problem where it can be estimated.

- **Survivorship bias.** The universe is fixed as of 2005, so every company in it
  survived. Worth roughly 1% to 4% of annual return. Not removable without a point in
  time constituent list.
- **A single asset class and a single market.** US equities, 11 tickers, daily bars.
  No commodities, no currencies, no intraday data.
- **Daily frequency.** The lowest frequency at which the cost problem is severe.
  Weekly or monthly rebalancing would change the conclusion, and testing that is the
  obvious next step.
- **Transaction costs are an assumption, not a measurement.** 5 bps per side is a
  reasonable estimate for liquid US equities but it is not my fill data.
- **No market impact model.** Position size is assumed not to move the price, which
  holds at small size and fails at scale.
- **The trial count is a lower bound.** It counts the configurations the log
  recorded. Exploratory work in the notebooks that was not logged means the true
  number of trials is higher, so the luck threshold is if anything too generous.
- **Shorting is assumed available and is charged a flat borrow rate.** Real borrow
  availability and cost vary by name and by date.

### 13. Future improvements

Short and concrete. What would actually be done next, not a wish list.

- Test weekly and monthly rebalancing, since the cost arithmetic says that is where
  an edge would survive.
- Add meta labelling: let the primary model call direction and a secondary model
  decide whether to take the bet, which addresses the hit rate problem directly.
- Expand to a point in time universe to remove survivorship bias.
- Add intraday data, where the signal to noise ratio is different.
- Walk the deflated Sharpe forward, recomputing the threshold as trials accumulate.

### 14. References

The papers and resources the project is built on, with a line each on what was taken
from them. The López de Prado pitfalls paper, the fractional differentiation
material, the deflated Sharpe ratio paper, and the baseline hit rate reference.

### 15. Notes

Small practical things a reviewer will appreciate: that intermediate artifacts are
gitignored while results and figures are committed so they can be seen without
running anything, that every figure can be regenerated with one command, and that
model metadata records what was trained on which rows and how it scored, so any
number can be traced back to the run that produced it.
