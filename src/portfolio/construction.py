"""From per-ticker signals or returns to portfolio weights, starting simple.

Equal weight beats most optimised portfolios out of sample. That is not a
joke but the headline of a large literature (DeMiguel, Garlappi and Uppal,
2009): optimisation needs expected returns, those estimates are so noisy
that the optimiser mostly fits noise, and 1/n has no estimation error at
all. So every allocation here is measured against equal weight rather than
assumed to improve on it.

Every rule that estimates something (volatility, covariance) uses only the
trailing window up to the date the weights are set, and weights change only
on rebalance dates. Between them a holding drifts with the market, which the
backtest engine models when it sees an unchanged target.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.backtest.momentum import rebalance_schedule
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

RISK_PARITY_TOLERANCE = 1e-10
RISK_PARITY_ITERATIONS = 500


def equal_weight(tickers: Iterable[str], budget: float = 1.0) -> pd.Series:
    """1/n in every ticker, scaled to the budget: the baseline that is hard to beat."""
    tickers = list(tickers)

    if not tickers:
        raise ValueError("equal_weight needs at least one ticker")

    return pd.Series(budget / len(tickers), index=tickers, name="weight")


def signal_weighted(signals: pd.DataFrame | pd.Series, config: Config) -> pd.DataFrame | pd.Series:
    """Weights proportional to signal strength, scaled to the gross leverage limit.

    Positive signals go long and negative short; with long_only (or shorts
    banned) negatives are dropped. A date with no signal at all is flat.
    """
    long_only = config.portfolio.long_only or not config.backtest.allow_short
    frame = signals.to_frame().T if isinstance(signals, pd.Series) else signals
    frame = frame.astype(float).fillna(0.0)

    if long_only:
        frame = frame.clip(lower=0.0)

    gross = frame.abs().sum(axis=1)
    weights = (
        frame.div(gross.where(gross > 0), axis=0).fillna(0.0) * config.backtest.max_gross_leverage
    )

    return weights.iloc[0].rename("weight") if isinstance(signals, pd.Series) else weights


def inverse_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Weights proportional to 1 / trailing volatility, summing to 1, per date.

    Risk parity's simple cousin: it equalises each asset's stand-alone risk
    but ignores correlation. The weight at t uses returns up to t only.
    """
    volatility = returns.rolling(window).std()
    inverse = 1 / volatility.where(volatility > 0)

    return inverse.div(inverse.sum(axis=1), axis=0)


def risk_parity_weights(cov: pd.DataFrame | np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Long-only weights where every asset contributes the same share of risk.

    Solves w_i (Cov w)_i = sigma_p^2 / n for all i by the cyclical coordinate
    method of Griveau-Billion, Richard and Roncalli (2013), which converges
    for any positive definite covariance.
    """
    sigma = np.asarray(cov, dtype=float)
    n = len(sigma)
    weights = np.full(n, 1.0 / n)

    for _ in range(RISK_PARITY_ITERATIONS):
        previous = weights.copy()
        for i in range(n):
            # Positive root of sigma_ii w_i^2 + (sum_j!=i sigma_ij w_j) w_i - 1/n = 0.
            b = sigma[i] @ weights - sigma[i, i] * weights[i]
            weights[i] = (-b + np.sqrt(b * b + 4 * sigma[i, i] / n)) / (2 * sigma[i, i])
        if np.abs(weights - previous).max() < RISK_PARITY_TOLERANCE:
            break

    return budget * weights / weights.sum()


def risk_parity(
    returns: pd.DataFrame,
    window: int,
    schedule: pd.Series | None = None,
) -> pd.DataFrame:
    """Equal-risk-contribution weights, refitted on each rebalance date.

    Args:
        returns: Daily returns, one column per ticker.
        window: Trailing days for the covariance.
        schedule: Dates to refit on (True); every date when None.

    Returns:
        Weights per date, NaN before the first full window and on dates
        not in the schedule (to be held by rebalance()).
    """
    schedule = pd.Series(True, index=returns.index) if schedule is None else schedule
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for position in np.flatnonzero(schedule.reindex(returns.index, fill_value=False).to_numpy()):
        if position + 1 < window:
            continue
        trailing = returns.iloc[position + 1 - window : position + 1].dropna(axis=1, how="any")
        if trailing.shape[1] == 0:
            continue
        solved = risk_parity_weights(trailing.cov().to_numpy())
        weights.loc[returns.index[position], trailing.columns] = solved

    return weights


def apply_constraints(
    weights: pd.DataFrame | pd.Series,
    config: Config,
    *,
    long_only: bool | None = None,
) -> pd.DataFrame | pd.Series:
    """Enforce long-only, the per-position cap and the gross leverage cap.

    For a long-only book the capped weight is redistributed across the other
    names in proportion to their weights (repeatedly, until nothing exceeds
    the cap), so a fully invested book stays fully invested. A long/short book
    is scaled down when its gross exposure is over the limit.

    Args:
        weights: Weights per ticker, for one date (Series) or many (DataFrame).
        config: Supplies the caps.
        long_only: Overrides config.portfolio.long_only.
    """
    limits = config.backtest
    long_only = config.portfolio.long_only if long_only is None else long_only
    long_only = long_only or not limits.allow_short
    frame = weights.to_frame().T if isinstance(weights, pd.Series) else weights.copy()

    out = []
    for _, row in frame.iterrows():
        w = row.fillna(0.0).to_numpy(dtype=float)
        if long_only:
            w = np.clip(w, 0.0, None)
            total = min(w.sum(), limits.max_gross_leverage)
            w = _cap_and_redistribute(w, limits.max_position_size, total)
        else:
            w = np.clip(w, -limits.max_position_size, limits.max_position_size)
            gross = np.abs(w).sum()
            if gross > limits.max_gross_leverage:
                w = w * limits.max_gross_leverage / gross
        out.append(w)

    result = pd.DataFrame(out, index=frame.index, columns=frame.columns)

    return result.iloc[0].rename("weight") if isinstance(weights, pd.Series) else result


def _cap_and_redistribute(w: np.ndarray, cap: float, total: float) -> np.ndarray:
    """Long-only weights summing to `total` with none above `cap`.

    When too few names hold weight for the cap to allow full investment
    (two names, a 30% cap, a 100% budget), each is capped and the rest is
    held as cash rather than breaking the cap.
    """
    if w.sum() <= 0:
        return w

    w = w / w.sum() * total
    for _ in range(len(w)):
        over = w > cap + 1e-15
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = (w > 0) & (w < cap - 1e-15)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    return np.minimum(w, cap)


def rebalance(
    weights: pd.DataFrame,
    schedule: pd.Series | str,
    current: pd.Series | None = None,
) -> pd.DataFrame:
    """Target weights that change only on rebalance dates.

    Args:
        weights: Desired weights per date (NaN where not computed).
        schedule: A boolean Series of rebalance dates, or "daily",
            "weekly" or "monthly".
        current: Weights held before the first rebalance; flat when None.

    Returns:
        Each rebalance date's weights, held until the next. The engine then
        lets the holding drift in between, so no trades happen off schedule.
    """
    if isinstance(schedule, str):
        schedule = rebalance_schedule(pd.DatetimeIndex(weights.index), schedule)

    on = schedule.reindex(weights.index, fill_value=False)
    held = weights.where(on, np.nan).ffill()
    start = pd.Series(0.0, index=weights.columns) if current is None else current

    return held.fillna(start.reindex(weights.columns).fillna(0.0))
