"""Triple barrier labels: profit target, stop loss, or time limit, whichever comes first.

A fixed percentage target is a different bet in every regime: 2% is a quiet
month in 2008 and a large move in 2017, so a model trained on fixed targets
quietly learns the volatility regime instead of the signal. Here both
horizontal barriers are a multiple of the volatility on the day the trade
starts, so a label means the same thing whatever the regime.

Everything is measured in log returns. Volatility, barrier widths and the
path are then on one scale, and multiplying a series' log returns by k
multiplies its barriers by k too, leaving every label unchanged. The return
column reports the simple return at the touch, which is what a position
would have earned.

The width on day t uses volatility up to t only. The path after t is the
label, and is the only future data this module reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import Config
from src.labelling.fixed_horizon import LABEL_COLUMNS
from src.logger import get_logger, log_frame

logger = get_logger(__name__)

UPPER, LOWER, VERTICAL = "upper", "lower", "vertical"


def daily_volatility(close: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted volatility of daily log returns, the barrier scale.

    Args:
        close: Adjusted close prices for one ticker.
        span: EWM span in bars; also the warm-up length.

    Returns:
        Daily (not annualised) volatility. NaN until `span` returns exist.
    """
    log_returns = np.log(close).diff()

    return log_returns.ewm(span=span, min_periods=span).std()


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    barriers: pd.DataFrame,
    vertical: int,
) -> pd.DataFrame:
    """Find which barrier each event touches first, and when.

    Args:
        close: Adjusted close prices for one ticker.
        events: Dates on which a position is opened.
        barriers: Indexed by event, with upper and lower columns holding the
            distance to each barrier as a positive log return. A NaN width
            skips the event; a zero width switches that barrier off.
        vertical: Bars until the time limit.

    Returns:
        Indexed by event: touch_time, barrier (upper, lower or vertical),
        return (simple, at the touch) and label. Upper is +1, lower is -1, and
        a vertical touch takes the sign of its return, with a zero return
        counted as -1, the same rule as fixed horizon labels. Events without
        `vertical` bars ahead of them, or without a width, are NaN.
    """
    log_price = np.log(close.to_numpy(dtype=float))
    out = pd.DataFrame(
        {
            "touch_time": pd.Series(pd.NaT, index=events, dtype="datetime64[ns]"),
            "barrier": pd.Series(None, index=events, dtype=object),
            "return": np.nan,
            "label": np.nan,
        },
        index=events,
    )

    position = close.index.get_indexer(events)
    upper = barriers["upper"].reindex(events).to_numpy(dtype=float)
    lower = barriers["lower"].reindex(events).to_numpy(dtype=float)

    # A NaN width is unknown (volatility warm-up), so the event is skipped. A
    # zero width is known: no volatility, so only the time limit applies.
    usable = (position >= 0) & (position + vertical < len(close))
    usable &= ~np.isnan(upper) & ~np.isnan(lower)
    upper[~(upper > 0)] = np.inf
    lower[~(lower > 0)] = np.inf

    if vertical < 1 or not usable.any():
        return out

    # paths[i, j] is the log return from event i to j + 1 bars later.
    windows = sliding_window_view(log_price, vertical + 1)
    rows = position[usable]
    paths = windows[rows, 1:] - windows[rows, :1]

    hit_upper = paths >= upper[usable, None]
    hit_lower = paths <= -lower[usable, None]
    first_upper = np.where(hit_upper.any(axis=1), hit_upper.argmax(axis=1), vertical)
    first_lower = np.where(hit_lower.any(axis=1), hit_lower.argmax(axis=1), vertical)
    step = np.minimum(np.minimum(first_upper, first_lower), vertical - 1)

    kind = np.where(
        first_upper == step, UPPER, np.where(first_lower == step, LOWER, VERTICAL)
    ).astype(object)
    log_return = paths[np.arange(len(rows)), step]

    label = np.where(kind == UPPER, 1.0, np.where(kind == LOWER, -1.0, np.nan))
    timed_out = kind == VERTICAL
    label[timed_out] = np.where(log_return[timed_out] > 0, 1.0, -1.0)
    label[np.isnan(log_return)] = np.nan

    index = events[usable]
    out.loc[index, "touch_time"] = close.index[rows + step + 1]
    out.loc[index, "barrier"] = kind
    out.loc[index, "return"] = np.expm1(log_return)
    out.loc[index, "label"] = label

    return out


def label_triple_barrier(panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Triple barrier labels for every ticker in the cleaned panel.

    Args:
        panel: Cleaned panel indexed by (date, ticker).
        config: Supplies the barrier multiples, holding period, volatility
            window and price column.

    Returns:
        Indexed like the panel, with label, touch_time, return and barrier.
    """
    settings = config.labelling.triple_barrier
    frames = {}

    for ticker in panel.index.unique("ticker"):
        close = panel.xs(ticker, level="ticker")[config.data.price_column]
        vol = daily_volatility(close, settings.volatility_window)
        barriers = pd.DataFrame(
            {"upper": vol * settings.profit_multiple, "lower": vol * settings.stop_multiple}
        )
        frames[ticker] = apply_triple_barrier(
            close, close.index, barriers, settings.max_holding_days
        )

    labels = pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()
    labels = labels.reindex(panel.index)

    shares = labels["barrier"].value_counts(normalize=True)
    logger.info(
        "Triple barrier labels: %s, %.1f%% up",
        ", ".join(f"{name} {share:.1%}" for name, share in shares.items()),
        labels["label"].eq(1).sum() / labels["label"].notna().sum() * 100,
    )
    log_frame(logger, labels, "Labels")

    return labels[[*LABEL_COLUMNS, "barrier"]]
