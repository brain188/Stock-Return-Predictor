"""Checks on the performance statistics against hand-computed values."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import performance
from src.backtest.costs import TRADING_DAYS
from src.backtest.performance import (
    annualised_return,
    annualised_volatility,
    block_bootstrap_indices,
    calmar_ratio,
    drawdown_series,
    equity_from_returns,
    exposure,
    hit_rate_trades,
    max_drawdown,
    performance_table,
    sharpe_difference_interval,
    sharpe_ratio,
    sharpe_standard_error,
    sortino_ratio,
    trade_returns,
    turnover_annual,
)

DATES = pd.date_range("2020-01-01", periods=6, freq="B")


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(
        rng.normal(0.0005, 0.01, 2520), index=pd.bdate_range("2010-01-01", periods=2520)
    )


def test_annualisation_uses_252_days_everywhere(config):
    assert TRADING_DAYS == 252 == config.evaluation.trading_days_per_year
    # Every function defaults to the same constant.
    for function in (annualised_return, annualised_volatility, sharpe_ratio, sortino_ratio):
        assert function.__defaults__[-1] == 252


def test_sharpe_of_a_constant_series_is_guarded():
    assert math.isnan(sharpe_ratio(pd.Series(0.001, index=range(100))))
    assert math.isnan(sortino_ratio(pd.Series(0.001, index=range(100))))


def test_sharpe_matches_the_formula(returns):
    expected = returns.mean() / returns.std() * np.sqrt(252)

    assert sharpe_ratio(returns) == pytest.approx(expected)
    # A risk-free rate is subtracted per day.
    shifted = (returns.mean() - 0.02 / 252) / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns, risk_free=0.02) == pytest.approx(shifted)


def test_annual_return_compounds():
    # 1% a day for 252 days.
    daily = pd.Series(0.01, index=range(252))

    assert annualised_return(daily) == pytest.approx(1.01**252 - 1)
    # Half a year at the same pace annualises to the same figure.
    assert annualised_return(daily.iloc[:126]) == pytest.approx(1.01**252 - 1)


def test_volatility_scales_by_root_252(returns):
    assert annualised_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_sortino_ignores_upside():
    returns = pd.Series([0.02, -0.01, 0.03, -0.01])
    downside = np.sqrt((0.01**2 + 0.01**2) / 4)

    assert sortino_ratio(returns) == pytest.approx(returns.mean() / downside * np.sqrt(252))


def test_max_drawdown_of_a_rising_curve_is_zero():
    equity = pd.Series(np.linspace(1, 3, 100))

    assert max_drawdown(equity) == 0.0
    assert (drawdown_series(equity) == 0).all()
    assert math.isnan(calmar_ratio(pd.Series(0.001, index=range(100))))


def test_max_drawdown_matches_a_hand_value():
    # Peak 120, trough 90: -25%. A later peak of 130 and fall to 117 is only -10%.
    equity = pd.Series([100, 120, 90, 110, 130, 117], dtype=float)

    assert max_drawdown(equity) == pytest.approx(-0.25)
    assert drawdown_series(equity).tolist() == pytest.approx([0, 0, -0.25, -1 / 12, 0, -0.1])


def test_calmar_is_return_over_worst_drawdown():
    returns = pd.Series([0.1, -0.2, 0.05, 0.1])
    equity = equity_from_returns(returns)

    assert calmar_ratio(returns) == pytest.approx(
        annualised_return(returns) / abs(max_drawdown(equity))
    )


def test_equity_curve_from_known_returns():
    equity = equity_from_returns(pd.Series([0.1, -0.1, 0.05]), initial=100)

    assert equity.tolist() == pytest.approx([110, 99, 103.95])


def test_trades_are_runs_of_one_side_and_earn_the_next_days_returns():
    weights = pd.DataFrame({"A": [1, 1, 0, -1, -1, 0]}, index=DATES, dtype=float)
    returns = pd.DataFrame({"A": [0.0, 0.02, -0.01, 0.03, -0.02, -0.01]}, index=DATES)

    # Long held on days 0-1 earns the returns of days 1-2: +0.02 - 0.01.
    # Short held on days 3-4 earns minus the returns of days 4-5: +0.02 + 0.01.
    assert trade_returns(weights, returns).tolist() == pytest.approx([0.01, 0.03])
    assert hit_rate_trades(weights, returns) == 1.0


def test_a_flip_is_two_trades():
    weights = pd.DataFrame({"A": [1, -1, -1]}, index=DATES[:3], dtype=float)
    returns = pd.DataFrame({"A": [0.0, 0.01, 0.02]}, index=DATES[:3])

    assert trade_returns(weights, returns).tolist() == pytest.approx([0.01, -0.02])
    assert hit_rate_trades(weights, returns) == 0.5


def test_no_trades_gives_no_hit_rate():
    flat = pd.DataFrame({"A": 0.0}, index=DATES)

    assert math.isnan(hit_rate_trades(flat, flat))


def test_turnover_annual_and_exposure():
    positions = pd.DataFrame({"A": [0.5, 0.5, -0.5], "B": [-0.5, 0.0, 0.0]}, index=DATES[:3])

    # Turnover 1.0 + 0.5 + 1.0 over 3 days.
    assert turnover_annual(positions) == pytest.approx(2.5 / 3 * 252)
    assert turnover_annual(traded=pd.Series([0.1, 0.1])) == pytest.approx(25.2)
    table = exposure(positions)
    assert table["gross"].tolist() == [1.0, 0.5, 0.5]
    assert table["net"].tolist() == [0.0, 0.5, -0.5]


def test_performance_table_shows_gross_and_net_and_the_benchmark_gap(returns):
    def result(gross, cost):
        frame = gross.to_frame("A")
        return SimpleNamespace(
            gross=gross,
            net=gross - cost,
            turnover=pd.Series(cost / 0.0005, index=gross.index),
            costs=pd.DataFrame({"total": cost}, index=gross.index),
            weights=pd.DataFrame({"A": 1.0}, index=gross.index),
            returns=frame,
        )

    table = performance_table(
        {"hold": result(returns, 0.0), "trader": result(returns, 0.0002)}, benchmark="hold"
    )

    assert table.loc["hold", "sharpe_gross"] == table.loc["hold", "sharpe_net"]
    assert table.loc["trader", "sharpe_net"] < table.loc["trader", "sharpe_gross"]
    assert table.loc["trader", "sharpe_net_vs_benchmark"] < 0
    assert table.loc["hold", "sharpe_net_vs_benchmark"] == 0
    assert table.loc["trader", "cost_annual"] == pytest.approx(0.0002 * 252)
    with pytest.raises(KeyError):
        performance_table({"x": result(returns, 0.0)}, benchmark="missing")


def test_module_states_its_conventions():
    assert "252" in performance.__doc__


# --- Uncertainty ----------------------------------------------------------------


def test_sharpe_standard_error_for_normal_returns(returns):
    daily = returns.mean() / returns.std()
    expected = np.sqrt((1 + daily**2 / 2) / len(returns)) * np.sqrt(252)

    # Normal returns have no skew and no excess kurtosis, so Mertens reduces to Lo.
    assert sharpe_standard_error(returns) == pytest.approx(expected, rel=0.05)


def test_a_thousand_days_cannot_separate_close_sharpes():
    noise = pd.Series(np.random.default_rng(3).normal(0.0003, 0.01, 1000))

    assert 0.4 < sharpe_standard_error(noise) < 0.6


def test_bootstrap_blocks_are_consecutive_and_reproducible():
    first = block_bootstrap_indices(100, 5, 10, np.random.default_rng(0))
    second = block_bootstrap_indices(100, 5, 10, np.random.default_rng(0))

    assert first.shape == (5, 100)
    assert np.array_equal(first, second)
    # Within a block, each index follows the last, wrapping at the end.
    assert np.all((np.diff(first[:, :10], axis=1) % 100) == 1)


def test_identical_series_differ_by_exactly_nothing(returns):
    interval = sharpe_difference_interval(returns, returns, n_boot=200)

    assert interval["difference"] == 0
    assert interval["low"] == interval["high"] == 0


def test_a_clearly_better_series_has_an_interval_above_zero(returns):
    worse = returns - 0.0008  # 20% a year less, same risk

    interval = sharpe_difference_interval(returns, worse, n_boot=500, seed=1)

    assert interval["low"] > 0
    assert interval["share_positive"] == 1.0


def test_pairing_detects_a_small_but_consistent_gap(returns):
    # Same days, half a percent a year less: tiny, but true on every day.
    slightly_worse = returns - 0.00002

    assert sharpe_difference_interval(returns, slightly_worse, n_boot=500, seed=2)["low"] > 0


def test_equal_strategies_get_intervals_that_cover_zero():
    # Twenty pairs of independent series with the same true Sharpe: a 95%
    # interval should contain the true difference, zero, nearly every time.
    covered = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        a, b = (pd.Series(rng.normal(0.0004, 0.01, 1500)) for _ in range(2))
        interval = sharpe_difference_interval(a, b, n_boot=300, seed=seed)
        covered += interval["low"] <= 0 <= interval["high"]

    assert covered >= 17


def test_an_idle_strategy_has_no_interval(returns):
    idle = pd.Series(0.0, index=returns.index)

    interval = sharpe_difference_interval(idle, returns, n_boot=100)

    assert all(math.isnan(v) for v in interval.values())


def test_a_first_day_loss_counts_as_a_drawdown_from_the_starting_capital():
    returns = pd.Series([-0.1, 0.05])

    # Equity 0.9 then 0.945: from the start of 1.0 the worst fall is 10%.
    assert performance.performance_stats(returns)["max_drawdown"] == pytest.approx(-0.1)
    # Without a starting level the first day is its own peak.
    assert max_drawdown(equity_from_returns(returns)) == 0.0
    assert max_drawdown(equity_from_returns(returns), start=1.0) == pytest.approx(-0.1)


def test_drawdown_conventions_agree_with_the_monte_carlo():
    from src.evaluation.monte_carlo import max_drawdowns

    rng = np.random.default_rng(11)
    for _ in range(20):
        returns = pd.Series(rng.normal(0, 0.02, 60))
        assert performance.performance_stats(returns)["max_drawdown"] == pytest.approx(
            float(max_drawdowns(returns.to_numpy()[None, :])[0])
        )
