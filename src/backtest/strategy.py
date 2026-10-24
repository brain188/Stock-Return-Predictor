"""Signal to position: every strategy the project tests, behind one interface.

Three traps this module exists to avoid:

1. The one bar delay. A signal computed from today's close cannot be traded
   at today's close. targets() says what the signal at the close of t wants;
   positions() shifts it by config.backtest.execution_lag bars, so the
   weight held from close t to close t + 1 only uses signals up to t - 1.
   Skipping this is the single most common backtest error, and notebook 4
   did skip it: its ML positions earned the very return they forecast.
2. Sizing is where the cost battle is won, so the three ML variants share
   one forecast and differ only in how they turn it into weights.
3. Side and size are kept apart (pitfall 6). The model calls direction; the
   strategy decides how much. That separation is what makes meta labelling
   possible later.

Positions are portfolio weights, one column per ticker, within the limits in
config.backtest. The ML strategies give each ticker an equal slice,
max_gross_leverage / number of tickers, so a full long/short book uses the
whole budget and no single forecast dominates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.momentum import momentum_targets
from src.backtest.pairs import find_pairs, pairs_targets
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def delay(targets: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Targets moved `lag` bars later, flat before the first."""
    return targets.shift(lag).fillna(0.0)


class Strategy:
    """Shared interface for every strategy.

    Subclasses implement targets(). fit() is for strategies that choose
    something from training data (the pairs to trade); the rest learn
    nothing and return self.

    Args:
        execution_lag: Bars between signal and trade. None uses
            config.backtest.execution_lag. 0 is allowed only for the same-close
            diagnostic, which deliberately reproduces notebook 4's timing.
    """

    name = "strategy"
    uses_model = False

    def __init__(self, execution_lag: int | None = None) -> None:
        self.execution_lag = execution_lag

    @property
    def parameters(self) -> dict[str, Any]:
        """Settings recorded in the trial log."""
        return {"execution_lag": self.execution_lag}

    def fit(
        self,
        prices: pd.DataFrame,
        config: Config,
        dates: pd.Index | None = None,
    ) -> Strategy:
        """Learn from the training dates. Most strategies learn nothing."""
        return self

    def targets(
        self,
        signal: pd.DataFrame | None,
        prices: pd.DataFrame,
        config: Config,
    ) -> pd.DataFrame:
        """Weights the signal at each close asks for, before any delay."""
        raise NotImplementedError

    def lag(self, config: Config) -> int:
        """The execution delay in bars."""
        return config.backtest.execution_lag if self.execution_lag is None else self.execution_lag

    def positions(
        self,
        signal: pd.DataFrame | None,
        prices: pd.DataFrame,
        config: Config,
    ) -> pd.DataFrame:
        """Weights to hold from each close, delay applied and limits enforced.

        Args:
            signal: Forecasts, one column per ticker (None for rule-based
                strategies).
            prices: Prices, one column per ticker.
            config: Supplies the delay and the limits.

        Returns:
            Weights in [-max_position_size, max_position_size], gross at most
            max_gross_leverage, and no shorts when allow_short is false.
        """
        lag = self.lag(config)
        if lag == 0:
            logger.warning(
                "%s trades at the signal's own close: look-ahead, diagnostic only", self.name
            )

        return enforce_limits(delay(self.targets(signal, prices, config), lag), config)


def enforce_limits(weights: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Clip shorts if banned, cap each weight, and scale down any over-levered date."""
    limits = config.backtest
    frame = weights.fillna(0.0)

    if not limits.allow_short:
        frame = frame.clip(lower=0.0)

    frame = frame.clip(-limits.max_position_size, limits.max_position_size)
    gross = frame.abs().sum(axis=1)
    scale = (limits.max_gross_leverage / gross).where(gross > limits.max_gross_leverage, 1.0)

    return frame.mul(scale, axis=0)


def _slice(signal: pd.DataFrame, config: Config) -> float:
    """Equal weight per ticker: the gross budget shared across the columns."""
    return config.backtest.max_gross_leverage / signal.shape[1]


class MLLongShort(Strategy):
    """Long when the forecast is positive, short when negative: the brief's main strategy."""

    name = "ml_long_short"
    uses_model = True

    def targets(self, signal, prices, config):
        return np.sign(signal).fillna(0.0) * _slice(signal, config)


class MLLongOnly(Strategy):
    """Long when the forecast is positive, flat otherwise.

    Never shorting halves the trading and removes the borrow fee, which is
    why it lost the least to costs in notebook 4.
    """

    name = "ml_long_only"
    uses_model = True

    def targets(self, signal, prices, config):
        return (signal > 0).astype(float).where(signal.notna(), 0.0) * _slice(signal, config)


class MLThresholded(Strategy):
    """Trade only when the forecast move pays for itself; otherwise keep the position.

    The threshold defaults to cost_multiple x the round-trip cost: with the
    configured 1.0 the forecast return must at least cover getting in and
    out. Below it the current position is held rather than closed, since
    closing is itself a trade. This is the cheapest cost control there is.

    The threshold is in return units, so it suits a regression forecast; a
    classification forecast (probability minus 0.5) needs its own scale.

    Args:
        threshold: Smallest absolute forecast that triggers a trade. None
            derives it from config.
        execution_lag: As in Strategy.
    """

    name = "ml_thresholded"
    uses_model = True

    def __init__(self, threshold: float | None = None, execution_lag: int | None = None) -> None:
        super().__init__(execution_lag)
        self.threshold = threshold

    @property
    def parameters(self):
        return {**super().parameters, "threshold": self.threshold}

    def resolved_threshold(self, config: Config) -> float:
        """The threshold in force: the one given, or cost_multiple x round trip."""
        if self.threshold is not None:
            return self.threshold

        return config.strategies.ml_thresholded["cost_multiple"] * config.backtest.round_trip_cost

    def targets(self, signal, prices, config):
        threshold = self.resolved_threshold(config)
        decided = np.sign(signal).where(signal.abs() > threshold)

        return decided.ffill().fillna(0.0) * _slice(signal, config)


class BuyAndHold(Strategy):
    """The benchmark: fully long one ticker from the first day, never trading again.

    Args:
        ticker: What to hold. None uses config.universe.benchmark.
    """

    name = "buy_and_hold"

    def __init__(self, ticker: str | None = None, execution_lag: int | None = None) -> None:
        super().__init__(execution_lag)
        self.ticker = ticker

    @property
    def parameters(self):
        return {**super().parameters, "ticker": self.ticker}

    def targets(self, signal, prices, config):
        ticker = self.ticker or config.universe.benchmark
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[ticker] = min(1.0, config.backtest.max_gross_leverage)

        return weights


class EqualWeightHold(Strategy):
    """Equal weight in every ticker on day one, then left to drift.

    The fair passive comparison for strategies that trade the whole
    universe: same names, no forecasting, one entry.
    """

    name = "equal_weight_hold"

    def targets(self, signal, prices, config):
        size = config.backtest.max_gross_leverage / prices.shape[1]

        return pd.DataFrame(size, index=prices.index, columns=prices.columns)


class Momentum(Strategy):
    """Cross-sectional 12-1 momentum, rebalanced monthly; settings in config."""

    name = "momentum"

    def targets(self, signal, prices, config):
        return momentum_targets(prices, config)


class MeanReversionPairs(Strategy):
    """Trade the spread of every pair that passes selection on the training dates."""

    name = "pairs"

    def __init__(self, execution_lag: int | None = None) -> None:
        super().__init__(execution_lag)
        self.selection_: pd.DataFrame | None = None
        self.pairs_: list[tuple[str, str]] = []
        # Every split's selection table, filled in by the evaluation pipeline.
        self.selections_: pd.DataFrame | None = None

    def fit(
        self,
        prices: pd.DataFrame,
        config: Config,
        dates: pd.Index | None = None,
    ) -> MeanReversionPairs:
        """Select pairs on the given dates only; the config supplies the screen."""
        self.selection_ = find_pairs(prices, config, dates)
        chosen = self.selection_.loc[self.selection_["selected"].astype(bool)]
        self.pairs_ = list(zip(chosen["a"], chosen["b"], strict=True))

        return self

    def targets(self, signal, prices, config):
        if self.selection_ is None:
            raise RuntimeError("MeanReversionPairs must be fitted before use")

        return pairs_targets(prices, self.pairs_, config)[0]

    def leg_targets(self, prices: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
        """Each selected pair's own weights, before any delay, keyed "A_B"."""
        return pairs_targets(prices, self.pairs_, config)[1]


ML_STRATEGIES = {
    "ml_long_short": MLLongShort,
    "ml_long_only": MLLongOnly,
    "ml_thresholded": MLThresholded,
}

RULE_STRATEGIES = {
    "momentum": Momentum,
    "pairs": MeanReversionPairs,
}

BENCHMARKS = {
    "buy_and_hold": BuyAndHold,
    "equal_weight_hold": EqualWeightHold,
}

STRATEGIES: dict[str, type[Strategy]] = {**ML_STRATEGIES, **RULE_STRATEGIES, **BENCHMARKS}
