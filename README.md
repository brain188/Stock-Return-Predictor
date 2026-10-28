# stock-return-predictor

Predicting next-day returns of 11 large-cap US equities with tree-based models,
and measuring honestly whether the result survives transaction costs and the
number of strategies tried.

## The finding

**No convincing evidence of an edge.** The predeclared primary strategy,
minimum-variance allocation, was judged once on a 2020–2024 holdout that no
model, parameter or choice had seen. The decision rule was fixed and approved
before the holdout was opened.

| Holdout, 2020-01-03 → 2024-12-30 | Value | Rule | Passed |
| --- | --- | --- | --- |
| Net Sharpe, minimum variance | 0.54 | — | — |
| Net Sharpe, equal-weight buy and hold | 0.80 | — | — |
| Difference, paired 95% interval | −0.26 [−0.74, +0.14] | entirely above 0 | no |
| Deflated Sharpe ratio (N = 98 trials) | 0.149 | above 0.95 | no |

In development (2005–2019) the model's forecasts carried almost no
information:

- pooled rank IC was +0.0059;
- the hit rate was 51.78%, against 51.75% for always predicting up.

The strategies built on those forecasts did not beat simply holding the
universe, net of costs:

| Development, CPCV (mean of 5 paths) | Gross Sharpe | Net Sharpe | Max drawdown |
| --- | --- | --- | --- |
| ML long/short | 0.82 | 0.76 | −34% |
| ML long only | 0.78 | 0.75 | −40% |
| ML thresholded | 0.72 | 0.70 | −39% |
| Momentum 12-1 | 0.10 | 0.05 | −35% |
| Pairs mean reversion | −0.10 | −0.17 | −30% |
| SPY buy and hold | 0.56 | 0.56 | −55% |
| Equal-weight buy and hold | 0.86 | 0.86 | −48% |

A negative result from a careful process is the credible outcome here. Daily
equity direction is close to unpredictable. The trial count, the luck
threshold it implies and the one-time holdout are what make the result
trustworthy, rather than a lucky backtest.

**Read more:**

- the development report: [`reports/research_report.md`](reports/research_report.md);
- the holdout report: [`reports/results/holdout/attempt-1/report.md`](reports/results/holdout/attempt-1/report.md);
- the holdout protocol: [`reports/methodology/holdout_protocol.md`](reports/methodology/holdout_protocol.md).

**Known limitations:**

- **Survivorship bias.** The 11-ticker universe was chosen with hindsight.
- **Earlier exploration saw the holdout years.** The notebooks that shaped the ML
  features explored data through 2024.

- **Price adjustments leak a little forward information.** Yahoo's back-adjusted
  prices carry later dividends and splits into earlier prices.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```text
uv sync --locked          # or: make install
```

## Run the research pipeline

The cached price data and every result are committed, so you don't need to
run anything to read the results or open the dashboard.

| What | make | Plain command (Windows has no make) |
| --- | --- | --- |
| Pull and cache prices | `make fetch` | `uv run python entrypoint/fetch.py` |
| CPCV research run | `make research` | `uv run python entrypoint/research.py` |
| CPCV strategy backtests | `make evaluate` | `uv run python entrypoint/evaluate.py` |
| Portfolio and risk | `make portfolio` | `uv run python -m src.pipelines.portfolio` |
| Research report | `make report` | `uv run python -m src.reporting.report` |
| Tests | `make test` | `uv run pytest` |
| Lint and format check | `make lint` | `uv run ruff check . && uv run ruff format --check .` |
| Dashboard | `make dashboard` | `uv run python -m dashboard.app` |

`research`, `evaluate` and `portfolio` fit models and append to the trial log.

**The holdout has already been run, once.** After it, any new development
trial would make its preflight fail: only holdout rows may follow the frozen
175-row trial-log prefix.

## Dashboard

```text
uv run python -m dashboard.app                 # http://127.0.0.1:8050
uv run python -m dashboard.app --port 8060     # another port
uv run python -m dashboard.app --debug         # Dash dev tools and hot reload
```

It is a [Dash](https://dash.plotly.com/) app implementing the seven-page Google
Stitch design. It is **read-only**:

- every number comes from a saved file in `reports/results/`, `data/` or the
  trial log;

- it never fits a model, reruns a backtest, logs a trial or opens the holdout;
- **Reload** re-reads the files, and **Export** downloads the main table of the
  current page as CSV.

| Page | Answers | Main sources |
| --- | --- | --- |
| Overview | The question, the headline numbers, the holdout verdict | `research/…/comparison.csv`, `holdout/attempt-1/`, `holdout_freeze.json` |
| Market Data & EDA | Prices, returns, volatility, autocorrelation, correlation, ADF | `data/02-bars/bars.parquet`, `stationarity.csv` |
| Feature & Signal | Importance and its fold stability, redundancy, IC over time and by regime, the tests run | `research/…/importance.csv`, `redundancy.csv`, `cs_ic_daily.csv` |
| Model Performance | The random forest against four baselines, out-of-fold, by path and year | `research/…/comparison.csv`, `paths.csv`, `predictions.parquet` |
| Backtesting | Gross beside net, costs in bps, growth, drawdown, years, paired intervals | `evaluation/path_returns.parquet`, `portfolio/allocation_returns.parquet`, holdout returns |
| Portfolio & Risk | Weight against risk share, frontier in- and out-of-sample, VaR/CVaR, stress | `portfolio/*.csv` |
| Experiments Registry | The trial log, the frozen N and luck threshold, files and MLflow runs | `trial_log.csv`, `holdout_freeze.json`, `holdout_declaration.json` |

**Controls.** The header holds the shared controls: asset, date window,
evaluation period, sampling, model, strategy and benchmark. Each page dims the
ones it doesn't use. Strategy, model and period tabs on a page drive the same
global values.

**Gaps it states rather than fills:**

- The holdout run recorded strategy returns, not model forecasts, so there is no
  holdout IC.

- Only one model was fitted; LightGBM and the LSTM are configured but never ran.
- Monte Carlo drawdowns were saved as percentiles, not paths.
- No PDF export exists; reports are Markdown.

**Configuration.** Paths and settings come from `config/config.yaml`. The
optional MLflow panel reads `mlflow.db` (read-only) if it is present. Fonts
(Inter, JetBrains Mono and a Material Symbols subset) are bundled in
`dashboard/assets/fonts/`, so the app needs no network.

## Layout

```text
src/            data, features, labelling, models, validation, backtest, portfolio, evaluation, pipelines
entrypoint/     command-line entry points (fetch, research, evaluate, holdout)
dashboard/      the Dash app: app.py, data.py, theme.py, charts.py, views.py, pages/
reports/        methodology, research report, trial log and every saved result
notebooks/      exploration (01–04) and the production results notebook (05)
tests/          the test suite, synthetic data unless a test says otherwise
```
