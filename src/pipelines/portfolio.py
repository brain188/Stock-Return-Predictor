"""Portfolio construction, optimisation and risk on the development period.

Runs end to end and writes its tables under reports/results/portfolio:

1. Allocation rules backtested with costs, monthly rebalanced, from a common
   start: equal weight (the benchmark), inverse volatility, risk parity,
   minimum variance and maximum Sharpe. Each rebalance uses only the trailing
   window up to that date, the trade happens a bar later, and every rule is
   compared with equal weight with a standard error, a paired bootstrap
   interval and the Monte Carlo probability of beating it.
2. The out-of-sample frontier: fitted on the first half of the development
   period, judged on the second.
3. Risk decomposition of each rule's latest weights.
4. Rolling correlation, by volatility regime.
5. Forward risk of each rule under normal, Student t and bootstrap models.
6. Stress scenarios: the pre-2020 crises replayed on the latest weights.
7. A resampling check: does the block bootstrap keep volatility clustering
   on real data, as notebook 4 found?

Nothing here reads a date on or after validation.test_start. Every allocation
rule is logged as a trial (equal weight as the benchmark).

    python -m src.pipelines.portfolio
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.momentum import rebalance_schedule
from src.backtest.performance import (
    exposure,
    performance_stats,
    sharpe_difference_interval,
    sharpe_standard_error,
)
from src.backtest.strategy import delay
from src.config import PROJECT_ROOT, Config, load_config
from src.data.cleaning import load_clean_panel
from src.evaluation.monte_carlo import (
    path_statistics,
    resample_indices,
    simulate_paths,
    squared_return_autocorrelation,
)
from src.evaluation.risk_simulation import (
    DEFAULT_SCENARIOS,
    FORWARD_METHODS,
    risk_summary,
    simulate_forward,
    stress_scenarios,
)
from src.evaluation.trial_log import TrialLog, trial_fingerprint
from src.logger import get_logger, log_frame, log_step, setup_logging
from src.portfolio.construction import (
    apply_constraints,
    inverse_volatility,
    rebalance,
    risk_parity,
)
from src.portfolio.optimisation import (
    Constraints,
    OptimisationError,
    covariance_matrix,
    expected_returns,
    frontier_figure,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    out_of_sample_frontier,
)
from src.portfolio.risk import (
    diversification_ratio,
    portfolio_volatility,
    risk_contribution,
    rolling_correlation,
)
from src.reporting.figures import FIGURES_DIR, _save, volatility_regimes

logger = get_logger(__name__)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "results" / "portfolio"
REFERENCE = "equal_weight"
ALLOCATIONS = (
    "equal_weight",
    "inverse_volatility",
    "risk_parity",
    "minimum_variance",
    "maximum_sharpe",
)

# Forward risk needs more paths than the backtest Monte Carlo: a 99% VaR from
# 1,000 paths rests on 10 of them.
FORWARD_RUNS = 10_000
CORRELATION_WINDOW = 63


@dataclass
class PortfolioResult:
    """Every Phase 7 table."""

    comparison: pd.DataFrame
    risk: pd.DataFrame
    risk_contributions: pd.DataFrame
    correlation_by_regime: pd.DataFrame
    forward_risk: pd.DataFrame
    stress: pd.DataFrame
    resampling_check: pd.DataFrame
    frontier: pd.DataFrame
    frontier_portfolios: pd.DataFrame
    frontier_assets: pd.DataFrame
    rolling_correlation: pd.Series
    period: tuple[str, str]
    figures: list[Path] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    def save(self, out_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
        """Write every table as CSV."""
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        tables = {
            "comparison.csv": self.comparison,
            "risk.csv": self.risk,
            "risk_contributions.csv": self.risk_contributions,
            "correlation_by_regime.csv": self.correlation_by_regime,
            "forward_risk.csv": self.forward_risk,
            "stress.csv": self.stress,
            "resampling_check.csv": self.resampling_check,
            "frontier.csv": self.frontier,
            "frontier_portfolios.csv": self.frontier_portfolios,
            "frontier_assets.csv": self.frontier_assets,
            "rolling_correlation.csv": self.rolling_correlation.to_frame(),
        }
        self.files = []
        for name, table in tables.items():
            table.to_csv(folder / name)
            self.files.append(folder / name)
        logger.info("Saved %s portfolio tables to %s", len(self.files), folder)

        return self.files


def development_returns(config: Config, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily asset returns before the holdout, one column per ticker."""
    panel = load_clean_panel(config) if panel is None else panel
    returns = panel["return"].unstack("ticker").sort_index()
    returns = returns[returns.index < pd.Timestamp(config.validation.test_start)]
    returns = returns.iloc[1:].fillna(0.0)

    if returns.index.max() >= pd.Timestamp(config.validation.test_start):
        raise AssertionError("development returns reach the holdout")
    log_frame(logger, returns, "Development returns")

    return returns


def first_rebalance(returns: pd.DataFrame, config: Config) -> pd.Timestamp:
    """The first scheduled date with a full covariance window behind it: every rule starts here."""
    schedule = rebalance_schedule(pd.DatetimeIndex(returns.index), config.portfolio.rebalance)
    ready = np.arange(len(returns)) + 1 >= config.portfolio.covariance_window

    return returns.index[(schedule.to_numpy() & ready)][0]


def allocation_weights(returns: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
    """Each rule's weights on its rebalance dates (NaN elsewhere), from the common start.

    Every estimate on a rebalance date t uses the trailing window ending at t.
    """
    settings = config.portfolio
    schedule = rebalance_schedule(pd.DatetimeIndex(returns.index), settings.rebalance)
    start = first_rebalance(returns, config)
    on = schedule & (returns.index >= start)
    dates = returns.index[on.to_numpy()]
    constraints = Constraints.from_config(config)
    blank = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    weights = {name: blank.copy() for name in ALLOCATIONS}
    weights["equal_weight"].loc[dates] = 1 / returns.shape[1]
    weights["inverse_volatility"].loc[dates] = inverse_volatility(
        returns, settings.volatility_window
    ).loc[dates]
    parity = risk_parity(returns, settings.covariance_window, on)
    weights["risk_parity"].loc[dates] = parity.loc[dates]

    for date in dates:
        end = returns.index.get_loc(date) + 1
        trailing = returns.iloc[end - settings.covariance_window : end]
        cov = covariance_matrix(trailing, settings.covariance_method)
        mu = expected_returns(trailing, settings.expected_return_method)
        try:
            weights["minimum_variance"].loc[date] = minimum_variance_portfolio(cov, constraints)
            weights["maximum_sharpe"].loc[date] = maximum_sharpe_portfolio(
                mu, cov, settings.risk_free_rate, constraints
            )
        except OptimisationError as exc:
            # Keep the previous weights rather than trading on a failed solve.
            logger.warning("Optimisation failed on %s (%s); weights held", date.date(), exc)

    for name, frame in weights.items():
        fitted = frame.loc[dates].dropna(how="all")
        weights[name].loc[fitted.index] = apply_constraints(fitted.fillna(0.0), config)

    return weights


def backtest_allocation(
    name: str,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: Config,
) -> BacktestResult:
    """Hold each rebalance's weights until the next, a bar later, with costs.

    The schedule forces a trade back to target on each rebalance even when
    the target did not change, so equal weight is rebalanced monthly rather
    than left to drift into buy and hold.
    """
    lag = config.backtest.execution_lag
    set_on = weights.notna().any(axis=1)
    start = set_on.idxmax()
    targets = rebalance(weights, set_on)
    positions = delay(targets, lag).loc[start:]
    trade_on = set_on.shift(lag, fill_value=False).loc[start:]

    return run_backtest(positions, returns.loc[start:], config, name=name, rebalance=trade_on)


def portfolio_settings(config: Config, name: str, period: tuple[str, str]) -> dict[str, Any]:
    """The settings that decide an allocation's result: its trial identity."""
    backtest = config.backtest
    universe = config.universe

    return {
        "pipeline": "portfolio",
        "allocation": name,
        "portfolio": dataclasses.asdict(config.portfolio),
        "costs": [backtest.commission_bps, backtest.slippage_bps, backtest.borrow_rate_annual],
        "limits": [backtest.max_position_size, backtest.max_gross_leverage, backtest.allow_short],
        "execution_lag": backtest.execution_lag,
        "universe": sorted(universe.tickers),
        "price_column": config.data.price_column,
        "cleaning": dataclasses.asdict(config.cleaning),
        "period": list(period),
    }


def compare_allocations(results: dict[str, BacktestResult], config: Config) -> pd.DataFrame:
    """Net performance of every rule against equal weight, with its uncertainty."""
    seed = config.project.random_seed
    base = results[REFERENCE].net
    rows = {}

    for name, result in results.items():
        stats = performance_stats(result.net)
        interval = sharpe_difference_interval(
            result.net, base, block=config.evaluation.bootstrap_block_days, seed=seed
        )
        pair = pd.concat([result.net, base], axis=1, join="inner")
        sims = simulate_paths(pair, "block", config)
        ours = path_statistics(sims[:, :, 0])["sharpe"].to_numpy()
        theirs = path_statistics(sims[:, :, 1])["sharpe"].to_numpy()
        held = exposure(result.weights)
        rows[name] = {
            "sharpe_gross": performance_stats(result.gross)["sharpe"],
            "sharpe_net": stats["sharpe"],
            "sharpe_se": sharpe_standard_error(result.net),
            "vs_equal_weight": interval["difference"],
            "vs_equal_weight_low": interval["low"],
            "vs_equal_weight_high": interval["high"],
            "p_beat_equal_weight": float(np.mean(ours > theirs)),
            "annual_return_net": stats["annual_return"],
            "annual_volatility_net": stats["annual_volatility"],
            "max_drawdown_net": stats["max_drawdown"],
            "turnover_annual": float(result.turnover.mean() * 252),
            "cost_annual": float(result.costs["total"].mean() * 252),
            "largest_weight_mean": float(result.weights.max(axis=1).mean()),
            "invested_mean": float(held["gross"].mean()),
        }

    table = pd.DataFrame(rows).T
    table.index.name = "allocation"

    return table


def risk_decomposition(
    weights: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where each rule's latest weights carry their risk, on the latest covariance."""
    cov = covariance_matrix(
        returns.iloc[-config.portfolio.covariance_window :], config.portfolio.covariance_method
    )
    summary, shares = {}, {}

    for name, frame in weights.items():
        latest = frame.dropna(how="all").iloc[-1].fillna(0.0)
        share = risk_contribution(latest, cov, as_share=True)
        shares[name] = share
        summary[name] = {
            "volatility": portfolio_volatility(latest, cov),
            "diversification_ratio": diversification_ratio(latest, cov),
            "largest_weight": float(latest.max()),
            "largest_weight_ticker": latest.idxmax(),
            "largest_risk_share": float(share.max()),
            "largest_risk_ticker": share.idxmax(),
        }

    return pd.DataFrame(summary).T, pd.DataFrame(shares).T


def correlation_regimes(returns: pd.DataFrame, config: Config) -> tuple[pd.Series, pd.DataFrame]:
    """Average pairwise correlation over time, and its level in each volatility regime.

    Both the 63-day correlation and the regime label use data up to each
    date, so this describes how co-movement and volatility coincide; it is
    not a trading signal.
    """
    average = rolling_correlation(returns, CORRELATION_WINDOW)
    regimes = volatility_regimes(returns[config.universe.benchmark]).reindex(average.index)
    table = average.groupby(regimes).agg(["mean", "median", "min", "max", "count"])
    table.loc["whole period"] = [
        average.mean(),
        average.median(),
        average.min(),
        average.max(),
        float(average.notna().sum()),
    ]
    table.loc["one matrix, whole period"] = [
        float(returns.corr().to_numpy()[np.triu_indices(returns.shape[1], 1)].mean()),
        *[np.nan] * 4,
    ]
    table.index.name = "regime"

    return average, table


def forward_risk(results: dict[str, BacktestResult], config: Config) -> pd.DataFrame:
    """Next-year risk of each rule, under each forward model."""
    settings = config.portfolio
    rows = {}

    for name, result in results.items():
        for method in FORWARD_METHODS:
            paths = simulate_forward(
                result.net,
                settings.horizon_days,
                FORWARD_RUNS,
                method,
                seed=config.project.random_seed,
                block_size=config.evaluation.bootstrap_block_days,
            )
            rows[(name, method)] = risk_summary(
                paths, tuple(settings.var_levels), settings.ruin_floor
            )

    table = pd.DataFrame(rows).T
    table.index.names = ["allocation", "method"]

    return table


def stress_tests(
    weights: dict[str, pd.DataFrame], returns: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """The pre-2020 crises replayed on each rule's latest weights, and on SPY alone."""
    latest = {name: frame.dropna(how="all").iloc[-1].fillna(0.0) for name, frame in weights.items()}
    latest["spy_only"] = pd.Series({config.universe.benchmark: 1.0})

    # Only the crises the data covers; a window outside it is reported, not invented.
    covered = {
        name: window
        for name, window in DEFAULT_SCENARIOS.items()
        if not returns.loc[window[0] : window[1]].empty
    }
    if skipped := sorted(set(DEFAULT_SCENARIOS) - set(covered)):
        logger.info("Stress scenarios outside the data, skipped: %s", ", ".join(skipped))

    tables = {
        name: stress_scenarios(returns, w, covered, holdout_start=config.validation.test_start)
        for name, w in latest.items()
    }

    return pd.concat(tables, names=["allocation", "scenario"])


def resampling_check(returns: pd.Series, config: Config, n_runs: int = 300) -> pd.DataFrame:
    """Volatility clustering in the real series and in each kind of resample."""
    values = returns.dropna().to_numpy()
    seed = config.project.random_seed
    block = config.evaluation.bootstrap_block_days
    rows = {"original": squared_return_autocorrelation(values)}

    for method in ("block", "stationary", "iid"):
        indices = resample_indices(len(values), method, n_runs, seed, block)
        rows[method] = squared_return_autocorrelation(values[indices])

    return pd.Series(rows, name="squared_return_autocorrelation").to_frame()


def run_portfolio_analysis(
    config: Config,
    panel: pd.DataFrame | None = None,
    *,
    trial_log: TrialLog | None = None,
    figures_dir: Path | None = None,
) -> PortfolioResult:
    """Run every Phase 7 analysis on the development period.

    Args:
        config: Loaded project configuration.
        panel: Cleaned panel. Loaded from disk when None.
        trial_log: Where the allocation trials go. Defaults to the config's.
        figures_dir: Where the frontier figure goes. Defaults to reports/figures.

    Returns:
        Every table, plus the figure paths.
    """
    trial_log = trial_log or TrialLog.from_config(config)
    returns = development_returns(config, panel)

    with log_step(logger, "Allocation weights"):
        weights = allocation_weights(returns, config)

    results = {
        name: backtest_allocation(name, frame, returns, config) for name, frame in weights.items()
    }
    period = (
        str(results[REFERENCE].net.index[0].date()),
        str(results[REFERENCE].net.index[-1].date()),
    )

    with log_step(logger, "Allocation comparison"):
        comparison = compare_allocations(results, config)

    for name in results:
        trial_log.log_trial(
            f"allocation_{name}",
            kind="benchmark" if name == REFERENCE else "backtest",
            fingerprint=trial_fingerprint(portfolio_settings(config, name, period)),
            status="ok",
            config=config,
            model_params={"allocation": name, **dataclasses.asdict(config.portfolio)},
            metrics={
                "sharpe": float(comparison.loc[name, "sharpe_net"]),
                **{k: float(v) for k, v in comparison.loc[name].items()},
            },
            notes=f"Monthly-rebalanced allocation, development period {period[0]} to {period[1]}",
        )

    with log_step(logger, "Out-of-sample frontier"):
        frontier = out_of_sample_frontier(returns, config)
        figure = frontier_figure(
            frontier.frontier,
            frontier.assets,
            frontier.portfolios,
            title=(
                f"Fitted {frontier.train_period[0][:4]}-{frontier.train_period[1][:4]}, "
                f"held {frontier.test_period[0][:4]}-{frontier.test_period[1][:4]}"
            ),
        )
        figures = [_save(figure, "portfolio_frontier_out_of_sample", figures_dir or FIGURES_DIR)]

    risk, contributions = risk_decomposition(weights, returns, config)
    average, by_regime = correlation_regimes(returns, config)

    with log_step(logger, "Forward risk"):
        forward = forward_risk(results, config)

    return PortfolioResult(
        comparison=comparison,
        risk=risk,
        risk_contributions=contributions,
        correlation_by_regime=by_regime,
        forward_risk=forward,
        stress=stress_tests(weights, returns, config),
        resampling_check=resampling_check(returns[config.universe.benchmark], config),
        frontier=frontier.frontier,
        frontier_portfolios=frontier.portfolios,
        frontier_assets=frontier.assets,
        rolling_correlation=average,
        period=period,
        figures=figures,
    )


def main() -> None:
    """Run the analysis with the project config and save the tables."""
    setup_logging()
    config = load_config()
    result = run_portfolio_analysis(config)
    result.save()

    pd.set_option("display.width", 200)
    print(f"\nAllocations, {result.period[0]} to {result.period[1]}")
    print(result.comparison.round(3).to_string())


if __name__ == "__main__":
    main()
