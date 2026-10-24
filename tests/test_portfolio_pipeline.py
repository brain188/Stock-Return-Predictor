"""Checks on the Phase 7 runner: development data only, no look-ahead, every rule logged."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.momentum import rebalance_schedule
from src.evaluation.trial_log import TrialLog
from src.pipelines import portfolio
from src.pipelines.portfolio import (
    ALLOCATIONS,
    allocation_weights,
    backtest_allocation,
    development_returns,
    run_portfolio_analysis,
)

TICKERS = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE"]
TEST_START = "2009-06-01"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Six correlated synthetic tickers from 2005, running past the holdout date."""
    rng = np.random.default_rng(0)
    n = 1400
    dates = pd.bdate_range("2005-01-03", periods=n)
    common = rng.normal(0, 0.008, n)
    frames = {}
    for i, ticker in enumerate(TICKERS):
        r = 0.0003 + common + rng.normal(0, 0.006 + 0.002 * i, n)
        price = 50 * np.cumprod(1 + r)
        frames[ticker] = pd.DataFrame(
            {
                "adj_close": price,
                "close": price,
                "return": pd.Series(price).pct_change().to_numpy(),
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def small_config(config, tmp_path):
    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY"),
        validation=dataclasses.replace(
            config.validation, test_start=pd.Timestamp(TEST_START).date()
        ),
        evaluation=dataclasses.replace(
            config.evaluation, trial_log=tmp_path / "trials.csv", monte_carlo_runs=200
        ),
    )


@pytest.fixture(scope="module")
def run(panel, config, tmp_path_factory):
    folder = tmp_path_factory.mktemp("portfolio")
    small = dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY"),
        validation=dataclasses.replace(
            config.validation, test_start=pd.Timestamp(TEST_START).date()
        ),
        evaluation=dataclasses.replace(
            config.evaluation, trial_log=folder / "trials.csv", monte_carlo_runs=200
        ),
    )
    original = portfolio.FORWARD_RUNS
    portfolio.FORWARD_RUNS = 400
    try:
        result = run_portfolio_analysis(small, panel, figures_dir=folder)
    finally:
        portfolio.FORWARD_RUNS = original

    return result, TrialLog.from_config(small), folder


def test_development_returns_stop_before_the_holdout(panel, small_config):
    returns = development_returns(small_config, panel)

    assert returns.index.max() < pd.Timestamp(TEST_START)
    assert panel.index.get_level_values("date").max() > pd.Timestamp(TEST_START)


def test_weights_use_only_past_returns(panel, small_config):
    returns = development_returns(small_config, panel)
    cut = returns.index[700]

    full = allocation_weights(returns, small_config)
    truncated = allocation_weights(returns.loc[:cut], small_config)

    for name in ALLOCATIONS:
        pd.testing.assert_frame_equal(full[name].loc[:cut], truncated[name], check_freq=False)


def test_equal_weight_trades_only_on_rebalance_days(panel, small_config):
    returns = development_returns(small_config, panel)
    weights = allocation_weights(returns, small_config)["equal_weight"]

    result = backtest_allocation("equal_weight", weights, returns, small_config)
    schedule = rebalance_schedule(returns.index, "monthly").shift(1, fill_value=False)
    traded = result.turnover > 0

    assert traded.sum() > 3  # rebalanced back to 1/n, not left to drift
    assert not (traded & ~schedule.reindex(traded.index)).any()


def test_every_rule_starts_on_the_same_day(run):
    result, _, _ = run

    assert set(result.comparison.index) == set(ALLOCATIONS)
    assert result.comparison.loc["equal_weight", "vs_equal_weight"] == 0


def test_every_rule_is_logged_with_equal_weight_as_the_benchmark(run):
    _, log, _ = run
    table = log.trial_table().set_index("name")

    assert log.trial_count() == len(ALLOCATIONS)
    assert table.loc["allocation_equal_weight", "kind"] == "benchmark"
    assert (table.drop("allocation_equal_weight")["kind"] == "backtest").all()


def test_risk_contributions_are_shares_of_risk(run):
    result, _, _ = run

    assert np.allclose(result.risk_contributions.sum(axis=1), 1.0)
    # Risk parity spreads risk most evenly.
    spread = result.risk_contributions.std(axis=1)
    assert spread["risk_parity"] <= spread.min() + 1e-6


def test_forward_risk_and_stress_tables(run):
    result, _, _ = run

    assert set(result.forward_risk.index.get_level_values("method")) == {
        "normal",
        "student_t",
        "bootstrap",
    }
    assert (result.forward_risk["cvar_95"] > result.forward_risk["var_95"]).all()
    # Only the 2008 crisis falls inside the synthetic data.
    assert set(result.stress.index.get_level_values("scenario")) == {"2008 financial crisis"}


def test_the_resampling_check_shows_clustering_survives_blocks_only(run):
    check = run[0].resampling_check["squared_return_autocorrelation"]

    assert abs(check["iid"]) < abs(check["block"]) + 1e-9


def test_save_writes_every_table_and_the_figure(run, tmp_path):
    result, _, folder = run
    files = result.save(tmp_path)

    assert len(files) == 11
    assert (folder / "portfolio_frontier_out_of_sample.png").is_file()
