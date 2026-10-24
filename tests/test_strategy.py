"""Checks on the strategies: the delay, the limits, and what each one holds."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.strategy import (
    STRATEGIES,
    BuyAndHold,
    EqualWeightHold,
    MeanReversionPairs,
    MLLongOnly,
    MLLongShort,
    MLThresholded,
    Momentum,
    delay,
    enforce_limits,
)

TICKERS = ["SPY", "A", "B", "C", "D", "E", "F", "G"]
N = 700


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2012-01-02", periods=N)
    walk = np.cumsum(rng.normal(0.0003, 0.012, (N, len(TICKERS))), axis=0)

    return pd.DataFrame(100 * np.exp(walk), index=dates, columns=TICKERS)


@pytest.fixture
def signal(prices) -> pd.DataFrame:
    rng = np.random.default_rng(1)

    return pd.DataFrame(rng.normal(0, 0.002, prices.shape), index=prices.index, columns=TICKERS)


@pytest.fixture
def small_config(config):
    return dataclasses.replace(
        config, universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY")
    )


def every_strategy(prices, config):
    """One fitted instance of every strategy."""
    out = []
    for cls in STRATEGIES.values():
        strategy = cls()
        out.append(strategy.fit(prices, config))
    return out


def test_every_strategy_respects_the_limits(prices, signal, small_config):
    limits = small_config.backtest

    for strategy in every_strategy(prices, small_config):
        weights = strategy.positions(signal, prices, small_config)

        assert weights.abs().max().max() <= limits.max_position_size + 1e-12, strategy.name
        assert (weights.abs().sum(axis=1) <= limits.max_gross_leverage + 1e-12).all(), strategy.name
        # And the engine agrees.
        run_backtest(weights, prices.pct_change(), small_config, name=strategy.name)


def test_positions_change_on_the_bar_after_the_signal(prices, small_config):
    flat = pd.DataFrame(0.0, index=prices.index, columns=TICKERS)
    signal = flat.copy()
    signal.iloc[100:, 1] = 0.01  # A turns positive at the close of day 100

    for cls in (MLLongShort, MLLongOnly):
        weights = cls().positions(signal, prices, small_config)["A"]

        assert weights.iloc[100] == 0, cls.name
        assert weights.iloc[101] > 0, cls.name


def test_the_lag_follows_the_config(prices, small_config):
    slower = dataclasses.replace(
        small_config, backtest=dataclasses.replace(small_config.backtest, execution_lag=3)
    )
    signal = pd.DataFrame(0.0, index=prices.index, columns=TICKERS)
    signal.iloc[100:, 1] = 0.01

    weights = MLLongShort().positions(signal, prices, slower)["A"]

    assert weights.iloc[102] == 0
    assert weights.iloc[103] > 0


def test_a_flat_signal_gives_flat_positions(prices, small_config):
    flat = pd.DataFrame(0.0, index=prices.index, columns=TICKERS)

    for cls in (MLLongShort, MLLongOnly, MLThresholded):
        assert (cls().positions(flat, prices, small_config) == 0).all().all(), cls.name


def test_missing_forecasts_mean_no_position(prices, small_config):
    missing = pd.DataFrame(np.nan, index=prices.index, columns=TICKERS)

    for cls in (MLLongShort, MLLongOnly, MLThresholded):
        assert (cls().positions(missing, prices, small_config) == 0).all().all(), cls.name


def test_strategies_are_reproducible(prices, signal, small_config):
    for first, second in zip(
        every_strategy(prices, small_config), every_strategy(prices, small_config), strict=True
    ):
        pd.testing.assert_frame_equal(
            first.positions(signal, prices, small_config),
            second.positions(signal, prices, small_config),
        )


def test_long_short_uses_the_whole_budget_equally(prices, signal, small_config):
    weights = MLLongShort().targets(signal, prices, small_config)

    assert np.allclose(weights.abs(), 1 / len(TICKERS))
    assert np.allclose(weights.abs().sum(axis=1), 1.0)


def test_long_only_never_shorts(prices, signal, small_config):
    weights = MLLongOnly().positions(signal, prices, small_config)

    assert (weights >= 0).all().all()
    assert set(np.unique(weights)) <= {0.0, 1 / len(TICKERS)}


def test_thresholded_holds_until_the_forecast_pays_for_the_trade(prices, small_config):
    dates = prices.index[:6]
    signal = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    # Round trip is 10 bps: 0.0015 clears it, 0.0005 and -0.0005 do not.
    signal["A"] = [0.0005, 0.0015, 0.0005, -0.0005, -0.0020, 0.0]

    strategy = MLThresholded()
    targets = strategy.targets(signal, prices.loc[dates], small_config)["A"]
    size = 1 / len(TICKERS)

    assert strategy.resolved_threshold(small_config) == pytest.approx(0.0010)
    assert targets.tolist() == pytest.approx([0, size, size, size, -size, -size])


def test_thresholded_trades_less_than_long_short(prices, signal, small_config):
    shifted = signal + 0.0008  # some forecasts clear the bar, most do not

    def total_turnover(strategy):
        weights = strategy.positions(shifted, prices, small_config)
        return run_backtest(weights, prices.pct_change(), small_config).turnover.sum()

    assert total_turnover(MLThresholded()) < total_turnover(MLLongShort())


def test_buy_and_hold_holds_the_benchmark_only(prices, small_config):
    weights = BuyAndHold().positions(None, prices, small_config)

    assert (weights["SPY"].iloc[1:] == 1.0).all()
    assert (weights.drop(columns="SPY") == 0).all().all()
    assert weights["SPY"].iloc[0] == 0  # the delay applies to everyone


def test_equal_weight_hold_never_trades_after_entry(prices, small_config):
    weights = EqualWeightHold().positions(None, prices, small_config)
    result = run_backtest(weights, prices.pct_change(), small_config)

    assert (result.turnover.iloc[2:] == 0).all()


def test_momentum_and_pairs_go_through_the_same_delay(prices, small_config):
    momentum = Momentum()

    pd.testing.assert_frame_equal(
        momentum.positions(None, prices, small_config),
        enforce_limits(delay(momentum.targets(None, prices, small_config), 1), small_config),
    )
    with pytest.raises(RuntimeError, match="fitted"):
        MeanReversionPairs().targets(None, prices, small_config)


def test_pairs_select_on_training_dates_only(prices, small_config):
    early = MeanReversionPairs().fit(prices, small_config, prices.index[:400])
    cut = MeanReversionPairs().fit(prices.iloc[:400], small_config)

    pd.testing.assert_frame_equal(early.selection_, cut.selection_)


def test_shorts_are_removed_when_banned(prices, signal, small_config):
    long_only = dataclasses.replace(
        small_config, backtest=dataclasses.replace(small_config.backtest, allow_short=False)
    )

    assert (MLLongShort().positions(signal, prices, long_only) >= 0).all().all()


def test_over_levered_dates_are_scaled_down(small_config):
    dates = pd.bdate_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"A": [0.8, 0.3], "B": [-0.8, 0.3]}, index=dates)

    capped = enforce_limits(weights, small_config)

    assert capped.abs().sum(axis=1).tolist() == pytest.approx([1.0, 0.6])
    assert capped.iloc[0].tolist() == pytest.approx([0.5, -0.5])


def test_same_close_execution_is_possible_only_as_a_diagnostic(prices, small_config):
    signal = pd.DataFrame(0.0, index=prices.index, columns=TICKERS)
    signal.iloc[100:, 1] = 0.01

    weights = MLLongShort(execution_lag=0).positions(signal, prices, small_config)["A"]

    assert weights.iloc[100] > 0
