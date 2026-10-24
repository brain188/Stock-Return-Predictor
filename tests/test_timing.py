"""Execution timing, end to end: signal, target, delay, position, return.

The chain being tested:

    close t:      the forecast for t is known (features use data up to t;
                  the forecast is about the return from t to t + 1)
    close t:      targets() turns it into a target weight
    close t + 1:  positions() applies the one-bar delay, so the target is
                  first held after the close of t + 1
    close t + 2:  the engine credits that holding with the return dated
                  t + 2, the move from close t + 1 to close t + 2

So a forecast made at the close of t can never earn the return it forecast
(dated t + 1). The canaries below prove it: a signal that knows tomorrow's
return exactly is worthless after the delay, and only becomes a money machine
if the delay is removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.performance import sharpe_ratio
from src.backtest.strategy import MLLongShort

DATES = pd.bdate_range("2021-01-04", periods=6)


@pytest.fixture
def two_assets():
    """Returns and signals for the hand calculation."""
    returns = pd.DataFrame(
        {
            "A": [0.00, 0.10, -0.05, 0.02, 0.04, -0.01],
            "B": [0.00, -0.02, 0.03, 0.01, -0.06, 0.05],
        },
        index=DATES,
    )
    signal = pd.DataFrame(
        {
            "A": [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            "B": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        },
        index=DATES,
    )

    return returns, signal


def test_positions_are_the_targets_one_bar_later(two_assets, config):
    returns, signal = two_assets
    positions = MLLongShort().positions(signal, returns, config)

    # Targets are sign / 2; the position on day t is the target of day t - 1.
    assert positions["A"].tolist() == [0.0, 0.5, -0.5, -0.5, 0.5, 0.5]
    assert positions["B"].tolist() == [0.0, 0.5, 0.5, -0.5, -0.5, 0.5]


def test_gross_returns_match_a_hand_calculation(two_assets, config):
    returns, signal = two_assets
    result = run_backtest(MLLongShort().positions(signal, returns, config), returns, config)

    # Day 0: nothing held.                                   gross = 0
    # Day 1: weights set at close 1 are [0.5, 0.5]; nothing was held over
    #        day 1 itself.                                   gross = 0
    # Day 2: held [0.5, 0.5] over day 2: 0.5(-0.05) + 0.5(0.03) = -0.01
    #        Then A's target flips to -0.5; B's is unchanged, so B drifts:
    #        0.5 x 1.03 / 0.99 = 0.520202.
    # Day 3: held [-0.5, 0.520202]: -0.5(0.02) + 0.520202(0.01) = -0.004798
    #        A unchanged drifts: -0.5 x 1.02 / 0.995202 = -0.512459; B -> -0.5.
    # Day 4: held [-0.512459, -0.5]: -0.512459(0.04) - 0.5(-0.06) = 0.009502
    #        A -> 0.5; B unchanged drifts: -0.5 x 0.94 / 1.009502 = -0.465576.
    # Day 5: held [0.5, -0.465576]: 0.5(-0.01) - 0.465576(0.05) = -0.028279
    expected = [0.0, 0.0, -0.01, -0.004798, 0.009502, -0.028279]

    assert result.gross.tolist() == pytest.approx(expected, abs=1e-6)
    # The signal at the close of day 0 earns its first return on day 2,
    # never on day 1, the return it was forecasting.
    assert result.weights.iloc[1].tolist() == [0.5, 0.5]


def test_costs_are_charged_on_the_day_of_the_trade(two_assets, config):
    returns, signal = two_assets
    result = run_backtest(MLLongShort().positions(signal, returns, config), returns, config)

    # Day 1: entry [0, 0] -> [0.5, 0.5], turnover 1.0.
    # Day 2: A from its drifted 0.479798 to -0.5, turnover 0.979798.
    assert result.turnover.iloc[0] == 0
    assert result.turnover.iloc[1] == pytest.approx(1.0)
    assert result.turnover.iloc[2] == pytest.approx(0.5 * 0.95 / 0.99 + 0.5)
    trading = result.costs["commission"] + result.costs["slippage"]
    assert trading.iloc[1] == pytest.approx(1.0 * config.backtest.cost_per_side)


def test_perfect_foresight_is_worthless_after_the_delay(config):
    # Independent returns, and a signal that knows tomorrow's return exactly.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2000-01-03", periods=5000)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, len(dates))}, index=dates)
    knows_tomorrow = returns.shift(-1)

    delayed = run_backtest(
        MLLongShort().positions(knows_tomorrow, returns, config), returns, config
    )
    same_close = run_backtest(
        MLLongShort(execution_lag=0).positions(knows_tomorrow, returns, config), returns, config
    )

    # With the delay the foresight lands one day late, on an independent
    # return: no edge. Without it, the backtest prints an absurd Sharpe.
    assert abs(sharpe_ratio(delayed.gross)) < 0.5
    assert sharpe_ratio(same_close.gross) > 10


def test_a_signal_that_only_knows_today_is_also_worthless(config):
    # The same check one step further back: the delay leaves no route for a
    # same-day return to be earned.
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2000-01-03", periods=5000)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, len(dates))}, index=dates)

    knows_today = run_backtest(
        MLLongShort(execution_lag=0).positions(returns, returns, config), returns, config
    )

    assert abs(sharpe_ratio(knows_today.gross)) < 0.5
