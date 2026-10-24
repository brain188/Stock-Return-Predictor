"""Checks on cross-sectional momentum."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import turnover
from src.backtest.momentum import (
    DAYS_PER_MONTH,
    momentum_positions,
    momentum_signal,
    momentum_targets,
    rank_cross_section,
    rebalance_schedule,
    skip_month,
    trailing_return,
)

TICKERS = list("ABCDEFGH")


@pytest.fixture
def prices() -> pd.DataFrame:
    """Eight random walks with different drifts, over four years."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-01", periods=1008)
    drift = np.linspace(-0.0005, 0.0010, len(TICKERS))
    walk = np.cumsum(rng.normal(drift, 0.015, (len(dates), len(TICKERS))), axis=0)

    return pd.DataFrame(100 * np.exp(walk), index=dates, columns=TICKERS)


def test_trailing_return_uses_only_past_prices(prices):
    signal = trailing_return(prices, 12)

    assert signal.iloc[251].isna().all()
    assert signal.iloc[252]["A"] == pytest.approx(prices["A"].iloc[252] / prices["A"].iloc[0] - 1)


def test_skipping_a_month_shifts_by_exactly_21_bars(prices):
    signal = trailing_return(prices, 3)
    skipped = skip_month(signal)

    assert DAYS_PER_MONTH == 21
    pd.testing.assert_frame_equal(skipped.iloc[21:], signal.iloc[:-21].set_axis(skipped.index[21:]))
    assert skipped.iloc[:21].isna().all().all()


def test_12_1_momentum_spans_252_to_21_days_ago(prices):
    signal = momentum_signal(prices, lookback_months=12, skip_months=1)
    t = 600

    expected = prices.iloc[t - 21] / prices.iloc[t - 252] - 1

    pd.testing.assert_series_equal(signal.iloc[t], expected, check_names=False)


def test_ranks_are_taken_within_each_date():
    dates = pd.bdate_range("2020-01-01", periods=2)
    # Day 2's values are all far larger than day 1's; pooling would rank them all above.
    signal = pd.DataFrame({"A": [0.30, 10.0], "B": [0.10, 30.0], "C": [0.20, 20.0]}, index=dates)

    ranks = rank_cross_section(signal)

    assert ranks.iloc[0].tolist() == [1, 3, 2]
    assert ranks.iloc[1].tolist() == [3, 1, 2]
    assert rank_cross_section(signal, dates[1]).tolist() == [3, 1, 2]


def test_the_top_ranked_ticker_goes_long_and_the_bottom_short(config):
    dates = pd.bdate_range("2020-01-01", periods=1)
    ranks = pd.DataFrame([[1, 2, 3, 4, 5, 6]], index=dates, columns=list("ABCDEF"), dtype=float)

    weights = momentum_positions(ranks, top_k=2, bottom_k=2, config=config).iloc[0]

    assert weights.tolist() == pytest.approx([0.25, 0.25, 0, 0, -0.25, -0.25])
    assert weights.abs().sum() == pytest.approx(config.backtest.max_gross_leverage)


def test_without_shorts_the_long_side_takes_the_budget(config):
    long_only = dataclasses.replace(
        config, backtest=dataclasses.replace(config.backtest, allow_short=False)
    )
    dates = pd.bdate_range("2020-01-01", periods=1)
    ranks = pd.DataFrame([[1, 2, 3, 4]], index=dates, columns=list("ABCD"), dtype=float)

    weights = momentum_positions(ranks, 2, 2, long_only).iloc[0]

    assert weights.tolist() == pytest.approx([0.5, 0.5, 0, 0])


def test_too_few_ranked_names_means_flat(config):
    dates = pd.bdate_range("2020-01-01", periods=1)
    ranks = pd.DataFrame([[1, 2, np.nan, np.nan]], index=dates, columns=list("ABCD"))

    assert (momentum_positions(ranks, 2, 2, config) == 0).all().all()


def test_rebalance_on_the_last_trading_day_of_each_month():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")

    monthly = rebalance_schedule(dates, "monthly")

    assert monthly.sum() == 12
    assert monthly[pd.Timestamp("2020-01-31")]
    assert monthly[pd.Timestamp("2020-02-28")]
    assert rebalance_schedule(dates, "daily").all()
    assert rebalance_schedule(dates, "weekly").sum() == 53


def test_targets_change_only_on_rebalance_dates(prices, config):
    targets = momentum_targets(prices, config)
    moves = targets.diff().abs().sum(axis=1) > 0
    schedule = rebalance_schedule(prices.index, "monthly")

    assert not (moves & ~schedule).any()


def test_monthly_rebalancing_trades_far_less_than_daily(prices, config):
    daily_config = dataclasses.replace(
        config,
        strategies=dataclasses.replace(
            config.strategies,
            momentum=dataclasses.replace(config.strategies.momentum, rebalance="daily"),
        ),
    )
    monthly = turnover(momentum_targets(prices, config)).iloc[300:].sum()
    daily = turnover(momentum_targets(prices, daily_config)).iloc[300:].sum()

    # Twelve chances to trade a year instead of 252. The exact ratio depends
    # on how often the ranking changes, so only the order of magnitude is fixed.
    assert monthly < daily / 3


def test_the_best_drift_is_held_long(config):
    # Drifts far enough apart that 11 months of trend outweighs 11 months of noise.
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-01", periods=1008)
    drift = np.linspace(-0.002, 0.002, len(TICKERS))
    walk = np.cumsum(rng.normal(drift, 0.01, (len(dates), len(TICKERS))), axis=0)
    prices = pd.DataFrame(100 * np.exp(walk), index=dates, columns=TICKERS)

    targets = momentum_targets(prices, config).iloc[400:]

    # H has the strongest drift and A the weakest.
    assert (targets["H"] > 0).mean() > 0.5
    assert (targets["A"] < 0).mean() > 0.5


def test_a_book_larger_than_the_universe_is_refused(config):
    prices = pd.DataFrame(
        100.0, index=pd.bdate_range("2020-01-01", periods=10), columns=["A", "B", "C"]
    )

    with pytest.raises(ValueError, match="top_k"):
        momentum_targets(prices, config)
