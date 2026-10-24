"""Checks on the backtest engine. The constant long test comes first, on purpose.

If an always-long position does not reproduce buy and hold exactly, gross,
the engine has an alignment bug and every number it ever produced is wrong.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestError,
    equity_curve,
    gross_returns,
    hold_weights,
    run_backtest,
)

N = 500


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=N)
    walk = np.exp(np.cumsum(rng.normal(0.0003, 0.012, (N, 3)), axis=0))

    return pd.DataFrame(100 * walk, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def returns(prices) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


# --- The test that matters most ---------------------------------------------


def test_constant_long_reproduces_buy_and_hold_exactly(prices, returns, config):
    positions = pd.DataFrame({"A": 1.0}, index=prices.index)

    result = run_backtest(positions, returns[["A"]], config)
    hold = prices["A"] / prices["A"].iloc[0]

    # Equity is the price relative, to floating point, every single day.
    assert np.allclose(result.equity_gross / config.backtest.initial_capital, hold, rtol=1e-12)
    # One entry, then never another trade.
    assert result.turnover.iloc[0] == 1.0
    assert (result.turnover.iloc[1:] == 0).all()


def test_equal_weight_buy_and_hold_drifts_like_a_real_holding(prices, returns, config):
    positions = pd.DataFrame(1 / 3, index=prices.index, columns=prices.columns)

    result = run_backtest(positions, returns, config)
    portfolio = (prices / prices.iloc[0]).mean(axis=1)

    assert np.allclose(result.equity_gross / config.backtest.initial_capital, portfolio)
    assert result.turnover.iloc[1:].sum() == 0


# --- Alignment ----------------------------------------------------------------


def test_position_at_t_earns_the_return_from_t_to_t_plus_1(returns, config):
    # Long only on date 10.
    positions = pd.DataFrame({"A": 0.0}, index=returns.index)
    positions.iloc[10] = 1.0

    gross = run_backtest(positions, returns[["A"]], config).gross

    assert gross.iloc[11] == pytest.approx(returns["A"].iloc[11])
    assert gross.drop(gross.index[11]).abs().max() == 0
    # Never the return that ended on the day the position was taken.
    assert gross.iloc[10] == 0


def test_flat_positions_give_zero_returns(returns, config):
    flat = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    result = run_backtest(flat, returns, config)

    assert (result.gross == 0).all()
    assert (result.net == 0).all()
    assert (result.turnover == 0).all()


def test_net_is_never_above_gross(returns, config):
    rng = np.random.default_rng(1)
    signs = rng.choice([-1.0, 0.0, 1.0], size=returns.shape)
    positions = pd.DataFrame(signs / 3, index=returns.index, columns=returns.columns)

    result = run_backtest(positions, returns, config)

    assert (result.net <= result.gross + 1e-15).all()
    assert (result.costs["total"] >= 0).all()


def test_equity_curve_matches_a_hand_value():
    returns = pd.Series([0.0, 0.1, -0.05, 0.02])

    assert equity_curve(returns, 1000).tolist() == pytest.approx([1000, 1100, 1045, 1065.9])


def test_gross_returns_use_yesterdays_weights():
    dates = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [1.0, -1.0, 0.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.5, 0.1, 0.2]}, index=dates)

    assert gross_returns(weights, returns).tolist() == pytest.approx([0.0, 0.1, -0.2])


# --- Trading and costs ----------------------------------------------------------


def test_turnover_is_measured_from_the_drifted_weights():
    dates = pd.bdate_range("2020-01-01", periods=2)
    targets = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, -0.10]}, index=dates)

    weights, traded = hold_weights(targets, returns)

    # After the move A is 0.55 / 1.0 = 0.55 and B is 0.45 of the book. Only
    # B's target changed, so only B trades: 0.45 to 0, a turnover of 0.45.
    assert weights.iloc[1].tolist() == pytest.approx([0.55, 0.0])
    assert traded.iloc[1] == pytest.approx(0.45)


def test_a_full_flip_costs_the_round_trip(returns, config):
    positions = pd.DataFrame({"A": 1.0}, index=returns.index[:3])
    positions.iloc[2] = -1.0

    result = run_backtest(positions, returns[["A"]], config)
    trading = result.costs["commission"] + result.costs["slippage"]

    assert result.turnover.iloc[2] == pytest.approx(2.0)
    assert trading.iloc[2] == pytest.approx(config.backtest.round_trip_cost)


def test_shorts_pay_borrow(returns, config):
    positions = pd.DataFrame({"A": -1.0}, index=returns.index[:5])

    borrow = run_backtest(positions, returns[["A"]], config).costs["borrow"]

    assert borrow.iloc[0] == 0
    assert (borrow.iloc[1:] > 0).all()


# --- Limits ---------------------------------------------------------------------


def test_leverage_beyond_the_limit_is_refused(returns, config):
    positions = pd.DataFrame(0.6, index=returns.index, columns=["A", "B"])

    with pytest.raises(BacktestError, match="gross exposure"):
        run_backtest(positions, returns, config)


def test_shorts_are_refused_when_not_allowed(returns, config):
    long_only = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, allow_short=False)
    )
    positions = pd.DataFrame({"A": -0.5}, index=returns.index)

    with pytest.raises(BacktestError, match="short"):
        run_backtest(positions, returns[["A"]], long_only)


def test_a_wiped_out_book_stops_the_run(config):
    dates = pd.bdate_range("2020-01-01", periods=3)
    positions = pd.DataFrame({"A": 1.0}, index=dates)
    returns = pd.DataFrame({"A": [0.0, -1.0, 0.1]}, index=dates)

    with pytest.raises(BacktestError, match="lost everything"):
        run_backtest(positions, returns, config)


def test_result_carries_capital_and_summary(returns, config):
    positions = pd.DataFrame({"A": 1.0}, index=returns.index)

    result = run_backtest(positions, returns[["A"]], config, name="hold", metadata={"k": 1})
    summary = result.summary()

    assert result.equity_net.iloc[0] < config.backtest.initial_capital
    assert summary["gross_sharpe"] > summary["net_sharpe"]
    assert result.metadata == {"k": 1}
