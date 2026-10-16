"""The simple label: the return over the next `horizon` days, and its sign.

This is the label the brief asks for and the baseline every other label is
compared against. It is also pitfall 5, fixed time horizon labelling: a 1%
move in a calm market and a 1% move in a panic get the same label, although
they are not the same event. Triple barrier labelling scales by volatility to
fix that, which is why both exist.

The negative shift in forward_return is the one place in the pipeline where
future data is used on purpose, because the label is by definition the future.

Both labellers return the same columns, label, touch_time and return, so
sample weights and purged cross-validation treat them identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger, log_frame

logger = get_logger(__name__)

LABEL_COLUMNS = ["label", "touch_time", "return"]


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """The return from close at t to close at t + horizon.

    Args:
        close: Adjusted close prices for one ticker.
        horizon: Bars ahead.

    Returns:
        close[t + horizon] / close[t] - 1. The last `horizon` values are NaN,
        because their future does not exist yet.
    """
    return close.shift(-horizon) / close - 1


def label_sign(forward: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Up or not up.

    The rule, stated: +1 when the forward return is strictly above the
    threshold, -1 otherwise, including a return of exactly zero. A flat day
    earns nothing for a long position, so it counts as "not up", which is
    also what the AlwaysUp baseline assumes. NaN stays NaN.

    Args:
        forward: Forward returns.
        threshold: Smallest return that counts as up.

    Returns:
        +1.0, -1.0 or NaN.
    """
    labels = np.where(forward > threshold, 1.0, -1.0)

    return pd.Series(labels, index=forward.index).where(forward.notna())


def label_fixed_horizon(panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Fixed horizon labels for every ticker in the cleaned panel.

    Args:
        panel: Cleaned panel indexed by (date, ticker).
        config: Supplies the horizon, threshold and price column.

    Returns:
        Indexed like the panel, with label (+1 / -1), touch_time (the date
        the label's return ends) and return (the forward return).
    """
    settings = config.labelling.fixed_horizon
    horizon = settings.horizon_days
    prices = panel[config.data.price_column].unstack("ticker")

    forward = forward_return(prices, horizon).stack(future_stack=True).reindex(panel.index)

    # The date `horizon` trading days ahead, from the shared calendar.
    dates = pd.Series(prices.index, index=prices.index)
    touch = dates.shift(-horizon).reindex(panel.index, level="date")

    labels = pd.DataFrame(
        {
            "label": label_sign(forward, settings.threshold),
            "touch_time": touch.where(forward.notna()),
            "return": forward,
        }
    )

    up = (labels["label"] == 1).sum() / labels["label"].notna().sum()
    logger.info("Fixed horizon labels: %s day horizon, %.1f%% up", horizon, up * 100)
    log_frame(logger, labels, "Labels")

    return labels[LABEL_COLUMNS]
