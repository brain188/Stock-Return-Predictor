"""Checks on trading costs, against the arithmetic from notebook 4."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import (
    borrow_cost,
    breakeven_return,
    commission,
    cost_breakdown,
    short_exposure,
    slippage,
    total_cost,
    turnover,
)

DATES = pd.date_range("2020-01-01", periods=6, freq="B")


def test_config_is_five_bps_a_side(config):
    # The rest of this file assumes the configured costs.
    assert config.backtest.cost_per_side == pytest.approx(0.0005)
    assert config.backtest.round_trip_cost == pytest.approx(0.0010)


def test_turnover_of_a_constant_position_is_zero_after_entry():
    positions = pd.Series(1.0, index=DATES)

    assert turnover(positions).tolist() == [1.0, 0, 0, 0, 0, 0]


def test_holding_a_position_charges_nothing(config):
    positions = pd.DataFrame({"A": 0.6, "B": 0.4}, index=DATES)

    costs = total_cost(positions, config)

    # Entry on day one, nothing after.
    assert costs.iloc[0] == pytest.approx(1.0 * 0.0005)
    assert (costs.iloc[1:] == 0).all()


def test_a_full_flip_charges_the_round_trip(config):
    positions = pd.Series([1.0, -1.0], index=DATES[:2])
    costs = cost_breakdown(positions, config)

    # Long to short is a turnover of 2: 2 x 5 bps = 10 bps.
    assert turnover(positions).iloc[1] == 2.0
    assert (costs["commission"] + costs["slippage"]).iloc[1] == pytest.approx(0.0010)


def test_commission_and_slippage_scale_with_turnover():
    assert commission(2.0, 2.0) == pytest.approx(0.0004)
    assert slippage(2.0, 3.0) == pytest.approx(0.0006)


def test_borrow_is_charged_for_the_period_the_short_was_held(config):
    positions = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [-0.5, -0.5, 0.0]}, index=DATES[:3])

    borrow = cost_breakdown(positions, config)["borrow"]
    daily = 0.5 * config.backtest.borrow_rate_annual / 252

    # Short from day 0's close: charged on days 1 and 2, nothing on day 0.
    assert borrow.tolist() == pytest.approx([0.0, daily, daily])


def test_longs_pay_no_borrow(config):
    positions = pd.Series(1.0, index=DATES)

    assert (cost_breakdown(positions, config)["borrow"] == 0).all()


def test_short_exposure_counts_only_shorts():
    positions = pd.DataFrame({"A": [0.5, -0.2], "B": [-0.3, -0.1]}, index=DATES[:2])

    assert short_exposure(positions).tolist() == pytest.approx([0.3, 0.3])
    assert borrow_cost(0.3, 0.005, 252) == pytest.approx(0.3 * 0.005 / 252)


def test_daily_flipping_needs_about_25_percent_a_year(config):
    # Notebook 4: flipping long to short every day at 5 bps a side.
    flips = pd.Series(np.where(np.arange(253) % 2 == 0, 1.0, -1.0))

    assert breakeven_return(turnover(flips).iloc[1:], config) == pytest.approx(0.252)
    assert breakeven_return(2.0, config) == pytest.approx(0.252)


def test_buy_and_hold_breaks_even_almost_for_free(config):
    hold = pd.Series(1.0, index=pd.date_range("2005-01-03", periods=5032, freq="B"))

    assert breakeven_return(turnover(hold), config) < 0.0001


def test_daily_flipping_costs_thousands_of_times_buy_and_hold(config):
    # Notebook 4's comparison: a daily coin flip against holding.
    days = pd.date_range("2005-01-03", periods=5032, freq="B")
    flip = pd.Series(np.sign(np.random.default_rng(0).normal(size=len(days))), index=days)
    hold = pd.Series(1.0, index=days)

    ratio = total_cost(flip, config).sum() / total_cost(hold, config).sum()

    # About n_days: half the days flip at a turnover of 2, against a single entry.
    assert 4000 < ratio < 6000


def test_engine_turnover_can_override_the_estimate(config):
    positions = pd.Series(1.0, index=DATES)
    traded = pd.Series(0.1, index=DATES)

    costs = cost_breakdown(positions, config, traded)

    assert costs["commission"].tolist() == pytest.approx([0.1 * 0.0002] * len(DATES))


def test_costs_scale_with_the_configured_bps(config):
    expensive = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, slippage_bps=8.0)
    )
    positions = pd.Series([1.0, -1.0], index=DATES[:2])

    assert total_cost(positions, expensive).iloc[1] == pytest.approx(2 * 0.0010)
