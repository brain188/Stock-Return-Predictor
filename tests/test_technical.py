"""Checks on the technical indicators.

Each indicator has a case whose answer is known by hand, keeps its input's
index, and is NaN, never 0, during its warm-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    distance_from_high,
    lagged_returns,
    ma_ratio,
    momentum,
    moving_average,
    rolling_volatility,
    rsi,
    volatility_ratio,
    volume_ratio,
)
from tests.conftest import make_bars

N = 300


@pytest.fixture
def close() -> pd.Series:
    """A synthetic adjusted close."""
    return make_bars(n=N)["adj_close"]


@pytest.fixture
def rising() -> pd.Series:
    """A price that goes up every day."""
    return pd.Series(np.linspace(100, 200, N), index=pd.date_range("2010", periods=N, freq="B"))


# Each indicator, called on a price-like series, with the number of leading
# NaN values its docstring promises.
CASES = [
    ("momentum_5", lambda c: momentum(c, 5), 5),
    ("ma_10", lambda c: moving_average(c, 10), 9),
    ("ma_ratio_10", lambda c: ma_ratio(c, 10), 9),
    ("rsi_14", lambda c: rsi(c, 14), 14),
    ("vol_21", lambda c: rolling_volatility(c.pct_change(), 21), 21),
    ("vol_ratio_10_63", lambda c: volatility_ratio(c.pct_change(), 10, 63), 63),
    ("high_63", lambda c: distance_from_high(c, 63), 62),
    ("volume_21", lambda c: volume_ratio(c, 21), 20),
]


@pytest.mark.parametrize(("name", "function", "warmup"), CASES)
def test_same_index_as_input(name, function, warmup, close):
    """Output index equals input index: no rows dropped, added or reordered."""
    assert function(close).index.equals(close.index)


@pytest.mark.parametrize(("name", "function", "warmup"), CASES)
def test_warmup_is_nan_never_zero(name, function, warmup, close):
    """The warm-up is NaN, and the first value after it exists."""
    out = function(close)

    assert out.iloc[:warmup].isna().all()
    assert not (out.iloc[:warmup] == 0).any()
    assert out.iloc[warmup:].notna().all()


@pytest.mark.parametrize(("name", "function", "warmup"), CASES)
def test_no_future_data(name, function, warmup, close):
    """Changing a later price never changes an earlier value."""
    changed = close.copy()
    changed.iloc[200:] *= 1.5

    pd.testing.assert_series_equal(function(close).iloc[:200], function(changed).iloc[:200])


def test_rsi_is_100_when_only_rising(rising):
    """Only gains, no losses: RSI is exactly 100."""
    assert (rsi(rising).dropna() == 100).all()


def test_rsi_is_0_when_only_falling(rising):
    """Only losses, no gains: RSI is exactly 0."""
    assert (rsi(rising[::-1].set_axis(rising.index)).dropna() == 0).all()


def test_rsi_is_50_when_flat():
    """No movement at all is neither strength nor weakness."""
    flat = pd.Series(100.0, index=range(50))

    assert (rsi(flat).dropna() == 50).all()


def test_rsi_uses_wilder_smoothing(close):
    """Matches a hand-written Wilder recursion, not a simple rolling mean."""
    window = 14
    delta = close.diff().to_numpy()
    gains, losses = np.clip(delta, 0, None), np.clip(-delta, 0, None)

    avg_gain, avg_loss = gains[1], losses[1]
    expected = [np.nan] * len(close)
    for t in range(2, len(close)):
        avg_gain = avg_gain + (gains[t] - avg_gain) / window
        avg_loss = avg_loss + (losses[t] - avg_loss) / window
        if t >= window:
            expected[t] = 100 - 100 / (1 + avg_gain / avg_loss)

    np.testing.assert_allclose(rsi(close, window).to_numpy(), expected, rtol=1e-10)


def test_rsi_is_bounded(close):
    """RSI never leaves [0, 100]."""
    values = rsi(close).dropna()

    assert values.between(0, 100).all()


def test_constant_series_has_zero_volatility():
    """No movement, no volatility."""
    returns = pd.Series(0.0, index=range(60))

    assert (rolling_volatility(returns, 21).dropna() == 0).all()


def test_volatility_is_annualised():
    """Alternating +/-1% has daily std 1%, so about 15.9% a year."""
    returns = pd.Series([0.01, -0.01] * 50)
    daily = returns.iloc[:21].std()

    assert rolling_volatility(returns, 21).iloc[-1] == pytest.approx(daily * np.sqrt(252))


def test_momentum_matches_hand_calculation(close):
    """The 5 day return is the price ratio over five bars."""
    assert momentum(close, 5).iloc[10] == pytest.approx(close.iloc[10] / close.iloc[5] - 1)


def test_lagged_returns_columns(close):
    """One named column per horizon, each equal to momentum at that horizon."""
    frame = lagged_returns(close, [1, 5, 21])

    assert list(frame.columns) == ["ret_1d", "ret_5d", "ret_21d"]
    pd.testing.assert_series_equal(frame["ret_5d"], momentum(close, 5), check_names=False)


def test_ma_ratio_is_zero_on_a_constant_price():
    """A price equal to its own average sits 0% above it."""
    flat = pd.Series(50.0, index=range(30))

    assert (ma_ratio(flat, 10).dropna() == 0).all()


def test_distance_from_high_is_zero_on_new_highs(rising):
    """A price making new highs every day is never below its high."""
    assert (distance_from_high(rising, 20).dropna() == 0).all()


def test_distance_from_high_after_a_drop():
    """A 10% fall from the peak reads -10%."""
    prices = pd.Series([100.0, 110.0, 99.0])

    assert distance_from_high(prices, 3).iloc[-1] == pytest.approx(-0.1)


def test_volume_ratio_spike():
    """Double the usual volume reads close to 2."""
    volume = pd.Series([1_000.0] * 20 + [2_000.0])

    assert volume_ratio(volume, 20).iloc[-1] == pytest.approx(2_000 / 1_050)


def test_volatility_ratio_above_one_when_heating_up():
    """Calm then rough: short volatility exceeds long."""
    rng = np.random.default_rng(0)
    returns = pd.Series(np.r_[rng.normal(0, 0.005, 100), rng.normal(0, 0.03, 10)])

    assert volatility_ratio(returns, 10, 63).iloc[-1] > 1
