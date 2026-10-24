"""Runs positions against returns: equity curves gross and net of costs.

Timing, the source of most backtest bugs, is fixed in one place:

- returns.loc[t] is the return from the close of t - 1 to the close of t.
- positions.loc[t] is the target weight chosen at the close of t, after any
  execution delay the strategy applied. It is held from close t to close
  t + 1, so it earns returns.loc[t + 1] and never returns.loc[t].
- A trade made at the close of t is charged at t.

The engine trades a ticker only when its target changes. Between changes the
weight is left to drift with the market, as a real holding would. That one
rule makes an always-long book exactly buy and hold, lets a monthly strategy
hold still between rebalances, and makes turnover what a trade really costs:
the distance from the drifted weight to the new target, not from yesterday's
target. If the constant long test in tests/test_engine.py ever fails, every
number this engine has produced is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import cost_breakdown
from src.backtest.performance import performance_stats
from src.config import Config
from src.logger import get_logger, log_frame

logger = get_logger(__name__)

# Weight changes smaller than this are not trades.
TARGET_TOLERANCE = 1e-12

# Slack on the leverage limits, for floating point.
LIMIT_TOLERANCE = 1e-9


class BacktestError(Exception):
    """Raised when positions break a limit or the book is wiped out."""


@dataclass
class BacktestResult:
    """Everything one backtest produced, on one date index."""

    name: str
    gross: pd.Series
    net: pd.Series
    weights: pd.DataFrame
    targets: pd.DataFrame
    turnover: pd.Series
    costs: pd.DataFrame
    returns: pd.DataFrame
    initial_capital: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def equity_gross(self) -> pd.Series:
        """Capital over time before costs."""
        return equity_curve(self.gross, self.initial_capital)

    @property
    def equity_net(self) -> pd.Series:
        """Capital over time after costs."""
        return equity_curve(self.net, self.initial_capital)

    def summary(self) -> dict[str, float]:
        """Headline statistics, gross_ and net_ prefixed."""
        gross = performance_stats(self.gross)
        net = performance_stats(self.net)

        return {
            **{f"gross_{k}": v for k, v in gross.items()},
            **{f"net_{k}": v for k, v in net.items()},
            "turnover_annual": float(self.turnover.mean() * 252),
            "cost_annual": float(self.costs["total"].mean() * 252),
        }


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Capital over time: initial compounded by the returns."""
    return (initial * (1 + returns.fillna(0.0)).cumprod()).rename("equity")


def gross_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Portfolio return before costs: weights held after t - 1 times returns of t.

    Args:
        weights: Weights actually held after each close.
        returns: Asset returns on the same grid.

    Returns:
        One return per date; the first is 0, since nothing was held before it.
    """
    held = weights.shift(1).fillna(0.0)

    return (held * returns.reindex_like(weights).fillna(0.0)).sum(axis=1).rename("gross")


def net_returns(
    gross: pd.Series,
    weights: pd.DataFrame,
    config: Config,
    traded: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Gross minus commission, slippage and borrow.

    Returns:
        The net return series and the cost breakdown it subtracted.
    """
    costs = cost_breakdown(weights, config, traded)

    return (gross - costs["total"]).rename("net"), costs


def check_limits(positions: pd.DataFrame, config: Config) -> None:
    """Refuse targets above the per-ticker or gross leverage limits, or shorts if banned.

    Raises:
        BacktestError: Naming the first date that breaks a limit.
    """
    limits = config.backtest
    frame = positions.fillna(0.0)
    largest = frame.abs().max(axis=1)
    gross = frame.abs().sum(axis=1)

    if (bad := largest > limits.max_position_size + LIMIT_TOLERANCE).any():
        raise BacktestError(
            f"weight above max_position_size on {bad.idxmax().date()}: {largest[bad].iloc[0]:.4f}"
        )
    if (bad := gross > limits.max_gross_leverage + LIMIT_TOLERANCE).any():
        raise BacktestError(
            f"gross exposure above max_gross_leverage on {bad.idxmax().date()}: "
            f"{gross[bad].iloc[0]:.4f}"
        )
    if not limits.allow_short and (frame < -LIMIT_TOLERANCE).to_numpy().any():
        raise BacktestError("short positions while backtest.allow_short is false")


def hold_weights(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Weights actually held, trading a ticker only when its target changes.

    Args:
        targets: Target weights per date, NaN treated as 0.
        returns: Asset returns on the same grid.

    Returns:
        The weights held after each close, and the turnover traded to reach
        them, measured from the drifted weights.

    Raises:
        BacktestError: If the book loses everything, after which weights are
            undefined.
    """
    target = targets.fillna(0.0).to_numpy(dtype=float)
    r = returns.reindex_like(targets).fillna(0.0).to_numpy(dtype=float)
    held = np.zeros_like(target)
    traded = np.zeros(len(target))
    previous_weight = np.zeros(target.shape[1])
    previous_target = np.zeros(target.shape[1])

    for t in range(len(target)):
        growth = 1.0 + previous_weight @ r[t]
        if growth <= 0:
            raise BacktestError(f"the book lost everything on {targets.index[t].date()}")

        drifted = previous_weight * (1.0 + r[t]) / growth
        changed = np.abs(target[t] - previous_target) > TARGET_TOLERANCE
        held[t] = np.where(changed, target[t], drifted)
        traded[t] = np.abs(held[t] - drifted).sum()
        previous_weight, previous_target = held[t], target[t]

    weights = pd.DataFrame(held, index=targets.index, columns=targets.columns)

    return weights, pd.Series(traded, index=targets.index, name="turnover")


def run_backtest(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    config: Config,
    *,
    name: str = "strategy",
    metadata: dict[str, Any] | None = None,
) -> BacktestResult:
    """Backtest target weights against asset returns, gross and net.

    Args:
        positions: Target weights from a strategy, delay already applied,
            one column per ticker.
        returns: Daily asset returns; columns not in positions are ignored.
        config: Supplies the costs, the limits and the initial capital.
        name: Label for the result.
        metadata: Anything to keep with the result, for example parameters.

    Returns:
        The backtest, on the positions' date index.

    Raises:
        BacktestError: For a broken limit or a wiped-out book.
    """
    positions = positions.sort_index()
    check_limits(positions, config)
    asset_returns = returns.reindex(index=positions.index, columns=positions.columns).fillna(0.0)

    weights, traded = hold_weights(positions, asset_returns)
    gross = gross_returns(weights, asset_returns)
    net, costs = net_returns(gross, weights, config, traded)

    log_frame(logger, weights, f"Backtest weights ({name})")

    return BacktestResult(
        name=name,
        gross=gross,
        net=net,
        weights=weights,
        targets=positions.fillna(0.0),
        turnover=traded,
        costs=costs,
        returns=asset_returns,
        initial_capital=config.backtest.initial_capital,
        metadata=dict(metadata or {}),
    )
