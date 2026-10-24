"""Cross-sectional momentum: long the recent winners, short the recent losers.

The most documented anomaly in finance, which makes it the right sanity
check: if this pipeline cannot reproduce a weak version of a known effect,
the pipeline is broken, not the market. It also shows the cost lesson from
the other side, because rebalancing monthly is cheap enough to survive where
daily ML flipping is not.

The signal is the classic 12-1: the return over the past 12 months excluding
the most recent one, i.e. from 252 trading days ago to 21 days ago. The most
recent month is skipped because it tends to reverse (short-term reversal),
which would weaken the signal. A month is 21 trading days throughout.

With 11 tickers this is a thin cross-section, far from the hundreds of stocks
in the published studies, so expect a noisy version of the effect at best.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

DAYS_PER_MONTH = 21


def trailing_return(prices: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    """Return over the last lookback_months, per ticker, using only past prices."""
    bars = lookback_months * DAYS_PER_MONTH

    return prices / prices.shift(bars) - 1


def skip_month(signal: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """The signal as it stood `months` ago: shifted by exactly 21 bars per month."""
    return signal.shift(months * DAYS_PER_MONTH)


def momentum_signal(prices: pd.DataFrame, lookback_months: int, skip_months: int) -> pd.DataFrame:
    """The ranking signal: the return from lookback_months ago to skip_months ago."""
    return skip_month(trailing_return(prices, lookback_months - skip_months), skip_months)


def rank_cross_section(
    signal: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> pd.DataFrame | pd.Series:
    """Rank tickers against each other on each date; 1 is the strongest.

    Ranks are taken within a date, never pooled across dates: a 30% return
    in 2009 and a 30% return in 2017 are not comparable, but the best name
    on each day is. Missing signals are left unranked.

    Args:
        signal: One row per date, one column per ticker.
        date: Rank this date only.

    Returns:
        Ranks per date, or for the one date asked for.
    """
    ranks = signal.rank(axis=1, ascending=False, method="first")

    return ranks.loc[date] if date is not None else ranks


def momentum_positions(
    ranks: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    config: Config,
) -> pd.DataFrame:
    """Long the top_k ranks and short the bottom_k, equal weight, per date.

    Each name gets max_gross_leverage / (top_k + bottom_k), so the book is
    fully used and, with equal counts, dollar neutral. With shorts banned
    the long side takes the whole budget. A date with too few ranked names
    to fill both sides is flat.

    Returns:
        Target weights, one column per ticker.
    """
    limits = config.backtest
    bottom_k = bottom_k if limits.allow_short else 0
    size = limits.max_gross_leverage / (top_k + bottom_k)
    size = min(size, limits.max_position_size)

    ranked = ranks.notna().sum(axis=1)
    worst = ranked.to_numpy()[:, None] - bottom_k

    weights = np.where(ranks.to_numpy() <= top_k, size, 0.0)
    weights = np.where(ranks.to_numpy() > worst, -size, weights)
    weights[(ranked < top_k + bottom_k).to_numpy()] = 0.0
    weights[np.isnan(ranks.to_numpy())] = 0.0

    return pd.DataFrame(weights, index=ranks.index, columns=ranks.columns)


def rebalance_schedule(index: pd.DatetimeIndex, frequency: str = "monthly") -> pd.Series:
    """True on the last trading date of each period, where the book is reset.

    Args:
        index: Trading dates.
        frequency: "daily", "weekly" or "monthly".

    Returns:
        A boolean per date.
    """
    if frequency == "daily":
        return pd.Series(True, index=index)

    alias = {"weekly": "W", "monthly": "M"}[frequency]
    period = pd.Series(index.to_period(alias), index=index)

    return period != period.shift(-1)


def momentum_targets(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Target weights set on each rebalance date and held until the next.

    The delay between the rebalance date's close and the trade is applied
    by the strategy, not here.
    """
    settings = config.strategies.momentum
    if settings.top_k + settings.bottom_k > prices.shape[1]:
        raise ValueError(
            f"momentum needs top_k + bottom_k = {settings.top_k + settings.bottom_k} tickers, "
            f"the universe has {prices.shape[1]}"
        )

    signal = momentum_signal(prices, settings.lookback_months, settings.skip_months)
    ranks = rank_cross_section(signal)
    weights = momentum_positions(ranks, settings.top_k, settings.bottom_k, config)

    on_rebalance = rebalance_schedule(prices.index, settings.rebalance)

    return weights.where(on_rebalance, np.nan).ffill().fillna(0.0)
