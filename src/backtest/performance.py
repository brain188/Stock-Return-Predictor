"""Every statistic needed to judge a strategy, gross and net side by side.

The gap between gross and net is the most informative number in the project:
it is the part of what the model found that the market charges for. So
performance_table() always shows both.

Conventions, used everywhere:

- Returns are simple daily returns, one per trading day.
- Annualisation uses 252 trading days, imported from costs.TRADING_DAYS so
  there is exactly one copy of the number.
- Sharpe and Sortino are annualised by sqrt(252) and use ddof=1. When the
  return series has no variation they are undefined and returned as NaN,
  rather than an infinity that would poison every average it entered.
- Drawdowns are negative fractions: -0.25 is a 25% fall from the peak.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import TRADING_DAYS, turnover

# Standard deviations below this count as zero.
ZERO_VOLATILITY = 1e-15


def _clean(returns: pd.Series) -> pd.Series:
    """The return series without missing values."""
    return pd.Series(returns, dtype=float).dropna()


def annualised_return(returns: pd.Series, days: int = TRADING_DAYS) -> float:
    """Compound annual growth rate: the constant yearly return with the same end value."""
    series = _clean(returns)

    if series.empty:
        return float("nan")

    growth = float((1 + series).prod())
    if growth <= 0:
        return -1.0

    return growth ** (days / len(series)) - 1


def annualised_volatility(returns: pd.Series, days: int = TRADING_DAYS) -> float:
    """Standard deviation of daily returns, scaled by sqrt(days)."""
    series = _clean(returns)

    return float(series.std() * np.sqrt(days)) if len(series) > 1 else float("nan")


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    days: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio: mean excess return over its volatility.

    Args:
        returns: Daily returns.
        risk_free: Annual risk-free rate, converted to daily by / days.
        days: Periods per year.

    Returns:
        The ratio, or NaN when the returns do not vary (a constant series has
        no risk to divide by; the ratio is guarded rather than infinite).
    """
    excess = _clean(returns) - risk_free / days

    if len(excess) < 2 or excess.std() < ZERO_VOLATILITY:
        return float("nan")

    return float(excess.mean() / excess.std() * np.sqrt(days))


def sortino_ratio(returns: pd.Series, target: float = 0.0, days: int = TRADING_DAYS) -> float:
    """Annualised Sortino ratio: mean excess return over downside deviation.

    Downside deviation is the root mean square of the shortfalls below the
    target, averaged over every day (days above target count as zero), as
    in Sortino and Price (1994). Only losses are treated as risk.

    Returns:
        The ratio, or NaN when there is no downside at all.
    """
    excess = _clean(returns) - target / days

    if excess.empty:
        return float("nan")

    downside = np.sqrt((excess.clip(upper=0) ** 2).mean())

    if downside < ZERO_VOLATILITY:
        return float("nan")

    return float(excess.mean() / downside * np.sqrt(days))


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Growth of `initial` under the returns, compounded."""
    return initial * (1 + _clean(returns)).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fall from the running peak at every date, as a negative fraction."""
    return (equity / equity.cummax() - 1).rename("drawdown")


def max_drawdown(equity: pd.Series) -> float:
    """The worst fall from a peak, as a negative fraction; 0 if it never fell."""
    if equity.empty:
        return float("nan")

    return float(min(drawdown_series(equity).min(), 0.0))


def calmar_ratio(
    returns: pd.Series,
    equity: pd.Series | None = None,
    days: int = TRADING_DAYS,
) -> float:
    """Annual return over the size of the worst drawdown; NaN with no drawdown."""
    equity = equity_from_returns(returns) if equity is None else equity
    worst = max_drawdown(equity)

    if not worst:
        return float("nan")

    return annualised_return(returns, days) / abs(worst)


def trade_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Profit of every trade: each run of same-sign holding in one ticker.

    A weight held after the close of t earns the return of t + 1, so a trade
    held on dates t0..t1 earns the returns of t0 + 1..t1 + 1.

    Args:
        weights: Weights held after each close, one column per ticker.
        returns: Daily returns on the same grid.

    Returns:
        One contribution to portfolio return per trade.
    """
    weights = weights.fillna(0.0)
    earned = weights.shift(1).fillna(0.0) * returns.reindex_like(weights).fillna(0.0)
    profits = []

    for ticker in weights.columns:
        side = np.sign(weights[ticker].to_numpy())
        # A new trade starts whenever the side changes; flat periods are not trades.
        trade_id = np.cumsum(np.r_[True, side[1:] != side[:-1]])
        # The profit of a holding lands on the next date, so shift the ids with it.
        held_id = pd.Series(trade_id, index=weights.index).shift(1)
        held_side = pd.Series(side, index=weights.index).shift(1)
        mask = held_side.fillna(0).to_numpy() != 0
        grouped = earned[ticker][mask].groupby(held_id[mask].to_numpy()).sum()
        profits.extend(grouped.tolist())

    return pd.Series(profits, dtype=float, name="trade_return")


def hit_rate_trades(weights: pd.DataFrame, returns: pd.DataFrame) -> float:
    """Share of trades that made money, NaN when there were none."""
    profits = trade_returns(weights, returns)

    return float((profits > 0).mean()) if len(profits) else float("nan")


def turnover_annual(
    positions: pd.DataFrame | None = None,
    *,
    traded: pd.Series | None = None,
    days: int = TRADING_DAYS,
) -> float:
    """Average yearly turnover: 1.0 means the whole book traded once a year.

    Args:
        positions: Weights; turnover is measured as their absolute changes.
        traded: Turnover per period, when already measured (the engine's
            drift-adjusted figure). Takes precedence over positions.
        days: Periods per year.
    """
    if traded is None:
        if positions is None:
            raise ValueError("give positions or traded turnover")
        traded = turnover(positions)

    return float(traded.mean() * days)


def exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """Long, short, net and gross exposure per date, as fractions of capital."""
    frame = positions.fillna(0.0)
    long = frame.clip(lower=0).sum(axis=1)
    short = (-frame).clip(lower=0).sum(axis=1)

    return pd.DataFrame({"long": long, "short": short, "net": long - short, "gross": long + short})


def performance_stats(returns: pd.Series, days: int = TRADING_DAYS) -> dict[str, float]:
    """The headline statistics of one daily return series."""
    equity = equity_from_returns(returns)

    return {
        "annual_return": annualised_return(returns, days),
        "annual_volatility": annualised_volatility(returns, days),
        "sharpe": sharpe_ratio(returns, days=days),
        "sortino": sortino_ratio(returns, days=days),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(returns, equity, days),
    }


def performance_table(
    results: Mapping[str, Any],
    benchmark: str | None = None,
    days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """One row per strategy, gross and net side by side.

    Args:
        results: Name to BacktestResult (anything with gross, net, turnover,
            costs, weights and returns attributes).
        benchmark: Name of the result every other is compared with.
        days: Periods per year.

    Returns:
        Columns: Sharpe gross and net and the Sharpe lost to costs; annual
        return gross and net; net volatility, Sortino, max drawdown and
        Calmar; annual turnover and annual cost; trade hit rate; and, with a
        benchmark, the net Sharpe and annual return relative to it.
    """
    rows = {}

    for name, result in results.items():
        gross = performance_stats(result.gross, days)
        net = performance_stats(result.net, days)
        rows[name] = {
            "sharpe_gross": gross["sharpe"],
            "sharpe_net": net["sharpe"],
            "sharpe_lost_to_costs": gross["sharpe"] - net["sharpe"],
            "annual_return_gross": gross["annual_return"],
            "annual_return_net": net["annual_return"],
            "annual_volatility_net": net["annual_volatility"],
            "sortino_net": net["sortino"],
            "max_drawdown_net": net["max_drawdown"],
            "calmar_net": net["calmar"],
            "turnover_annual": turnover_annual(traded=result.turnover, days=days),
            "cost_annual": float(result.costs["total"].mean() * days),
            "trade_hit_rate": hit_rate_trades(result.weights, result.returns),
        }

    table = pd.DataFrame(rows).T
    table.index.name = "strategy"

    if benchmark is not None:
        if benchmark not in table.index:
            raise KeyError(f"benchmark '{benchmark}' is not among the results")
        base = table.loc[benchmark]
        table["sharpe_net_vs_benchmark"] = table["sharpe_net"] - base["sharpe_net"]
        table["annual_return_net_vs_benchmark"] = (
            table["annual_return_net"] - base["annual_return_net"]
        )

    return table
