"""What trading costs: commission, slippage and the fee for borrowing shorts.

Built before the engine on purpose, so no backtest in the project can exist
without it. This is the module that sank every ML strategy in notebook 4:
flipping a position every day costs thousands of times what buying and
holding costs, and at 5 bps a side a strategy that flips daily needs about
25% a year of gross return just to stand still. breakeven_return() puts that
number next to every strategy instead of leaving it to be discovered at the
end.

Positions are portfolio weights, one column per ticker. Turnover is the sum
of absolute weight changes, so going from fully long to fully short is a
turnover of 2 and costs the round trip: 2 x (2 + 3) bps = 10 bps. Commission
and slippage scale with turnover; borrow scales with how much is held short
and for how long.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

TRADING_DAYS = 252


def turnover(positions: pd.DataFrame | pd.Series) -> pd.Series:
    """Sum of absolute weight changes per period; the first row is the entry.

    This is the turnover of trading to the target weights every period. The
    engine refines it by measuring changes against the weights after the
    market has moved them (drift), which is what a trade actually costs.

    Args:
        positions: Weights, one column per ticker, or a single series.

    Returns:
        Turnover per period, as a fraction of capital.
    """
    frame = positions.to_frame() if isinstance(positions, pd.Series) else positions
    frame = frame.fillna(0.0)
    change = frame.diff()
    change.iloc[0] = frame.iloc[0]

    return change.abs().sum(axis=1).rename("turnover")


def commission(turnover: pd.Series | float, bps: float) -> pd.Series | float:
    """Broker commission: bps per unit of turnover, per side."""
    return turnover * bps / 10_000


def slippage(turnover: pd.Series | float, bps: float) -> pd.Series | float:
    """Price impact and spread: bps per unit of turnover, per side."""
    return turnover * bps / 10_000


def short_exposure(positions: pd.DataFrame | pd.Series) -> pd.Series:
    """Total weight held short per period, as a positive number."""
    frame = positions.to_frame() if isinstance(positions, pd.Series) else positions

    return (-frame.fillna(0.0)).clip(lower=0).sum(axis=1).rename("short_exposure")


def borrow_cost(
    short_exposure: pd.Series | float,
    annual_rate: float,
    days: int = TRADING_DAYS,
) -> pd.Series | float:
    """Fee for holding shorts for one period: exposure x annual rate / days."""
    return short_exposure * annual_rate / days


def cost_breakdown(
    positions: pd.DataFrame | pd.Series,
    config: Config,
    traded: pd.Series | None = None,
) -> pd.DataFrame:
    """Commission, slippage and borrow per period, and their total.

    A trade at the close of t is charged at t. A short held from t to t + 1
    is charged its borrow fee at t + 1, the period it was held for.

    Args:
        positions: Weights held after trading at each close.
        config: Supplies the bps and the borrow rate.
        traded: Turnover per period, when the caller measured it more
            precisely (the engine passes drift-adjusted turnover). Defaults to
            turnover(positions).

    Returns:
        One row per period with commission, slippage, borrow and total, all
        as fractions of capital.
    """
    backtest = config.backtest
    traded = turnover(positions) if traded is None else traded
    held_short = short_exposure(positions).shift(1).fillna(0.0)

    costs = pd.DataFrame(
        {
            "commission": commission(traded, backtest.commission_bps),
            "slippage": slippage(traded, backtest.slippage_bps),
            "borrow": borrow_cost(held_short, backtest.borrow_rate_annual),
        }
    )
    costs["total"] = costs.sum(axis=1)

    return costs


def total_cost(
    positions: pd.DataFrame | pd.Series,
    config: Config,
    traded: pd.Series | None = None,
) -> pd.Series:
    """The full per-period cost drag, as a fraction of capital."""
    return cost_breakdown(positions, config, traded)["total"].rename("cost")


def breakeven_return(
    turnover: pd.Series | float,
    config: Config,
    days: int = TRADING_DAYS,
) -> float:
    """Annual gross return a strategy needs just to pay its trading costs.

    Mean daily turnover x cost per side x trading days. A strategy that flips
    from fully long to fully short every day has a turnover of 2 a day, so at
    5 bps a side it needs 2 x 0.0005 x 252 = 25.2% a year before it earns
    anything. Borrow is left out: it depends on how long shorts are held,
    not on how often the book trades.

    Args:
        turnover: Turnover per period, or its mean.
        config: Supplies the cost per side.
        days: Periods per year.

    Returns:
        The annual return that only breaks even.
    """
    mean = float(np.nanmean(turnover)) if isinstance(turnover, pd.Series) else float(turnover)

    return mean * config.backtest.cost_per_side * days
