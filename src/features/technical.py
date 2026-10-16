"""Technical indicators, each a standalone function from series to series.

Every function takes one ticker's series and returns a series (or frame) on
exactly the same index, so each one is testable on its own and none of them
can quietly drop or reorder rows.

Two rules hold throughout. First, the value at time t uses only data up to and
including t: rolling and exponential windows look backwards, and nothing is
centred or shifted forward. Second, the warm-up is NaN, never 0. A zero is a
real value that a model will happily learn from; NaN is honest about there
being no value yet.

Levels are never used directly. A price of 400 in 2024 and 40 in 2005 mean
nothing next to each other, and a tree will split on "price above 400" and call
it a signal. Every feature here is a ratio or a return, so it is comparable
across time and across tickers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Cumulative return over the last `window` bars.

    Args:
        close: Adjusted close prices.
        window: Bars to look back.

    Returns:
        close[t] / close[t - window] - 1. The first `window` values are NaN.
    """
    return close.pct_change(window, fill_method=None)


def lagged_returns(close: pd.Series, lags: Iterable[int] = (1, 2, 3, 5, 10, 21)) -> pd.DataFrame:
    """Returns over several horizons, all ending today.

    ret_1d is the latest daily return, ret_5d the last week, ret_21d the last
    month. Each is momentum() at a short horizon.

    Args:
        close: Adjusted close prices.
        lags: Horizons in bars.

    Returns:
        One column per horizon, named ret_{lag}d.
    """
    return pd.DataFrame({f"ret_{lag}d": momentum(close, lag) for lag in lags}, index=close.index)


def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average of the last `window` bars, today included.

    Returns:
        The mean. The first `window - 1` values are NaN.
    """
    return close.rolling(window, min_periods=window).mean()


def ma_ratio(close: pd.Series, window: int) -> pd.Series:
    """How far price sits above its moving average, as a fraction.

    The raw average is a price level and not stationary; the ratio is.

    Returns:
        close / moving_average - 1. The first `window - 1` values are NaN.
    """
    return close / moving_average(close, window) - 1


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing.

    Wilder's smoothing is an exponential average with alpha = 1 / window, not a
    simple mean, so a single large move fades out rather than dropping off a
    cliff `window` days later.

    Edge cases are set explicitly rather than left to division: only gains
    gives 100, only losses gives 0, and no movement at all gives 50.

    Args:
        close: Adjusted close prices.
        window: Smoothing period.

    Returns:
        RSI between 0 and 100. The first `window` values are NaN, because the
        first price change only exists from the second bar.
    """
    delta = close.diff()
    smooth = {"alpha": 1 / window, "adjust": False, "min_periods": window}
    gain = delta.clip(lower=0).ewm(**smooth).mean()
    loss = (-delta.clip(upper=0)).ewm(**smooth).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        value = 100 - 100 / (1 + gain / loss)

    value = value.mask((loss == 0) & (gain > 0), 100.0)
    value = value.mask((loss == 0) & (gain == 0), 50.0)

    return value.where(gain.notna() & loss.notna())


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Realised volatility over the last `window` returns, annualised.

    Args:
        returns: Daily simple returns.
        window: Returns in each estimate.

    Returns:
        Standard deviation times sqrt(252). NaN until `window` returns exist.
    """
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def volatility_ratio(returns: pd.Series, short: int, long: int) -> pd.Series:
    """Short-window volatility over long-window volatility, a regime proxy.

    Above 1, recent conditions are rougher than the longer baseline.

    Returns:
        The ratio. NaN until the long window is full, and where the long
        volatility is zero.
    """
    long_vol = rolling_volatility(returns, long)

    return rolling_volatility(returns, short) / long_vol.where(long_vol > 0)


def distance_from_high(close: pd.Series, window: int) -> pd.Series:
    """Drawdown from the highest close of the last `window` bars.

    Returns:
        close / rolling max - 1, so 0 at a new high and negative below it.
        The first `window - 1` values are NaN.
    """
    return close / close.rolling(window, min_periods=window).max() - 1


def volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """Today's volume against its average over the last `window` bars.

    Returns:
        volume / rolling mean. NaN during the warm-up and where volume is
        missing or the average is zero.
    """
    average = volume.rolling(window, min_periods=window).mean()

    return volume / average.where(average > 0)
