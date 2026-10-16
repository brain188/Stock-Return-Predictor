"""Sample weights that stop overlapping labels counting as independent evidence.

Every machine learning library assumes its samples are independent. Labels
whose returns overlap in time are not: two 5 day labels starting one day apart
share four days of return, so a model fitted to both sees most of that
information twice and grows overconfident. This is pitfall 7, non-IID sample
weighting.

Each label is weighted by its average uniqueness: over the bars its return
spans, the average of 1 / (number of labels live on that bar). A label alone
on every bar gets 1; one of n labels spanning exactly the same bars gets 1/n.
The weights therefore sum to the effective number of independent labels.

For the 1 day fixed horizon label no two returns overlap, so every weight is
1 and this module changes nothing. The features still overlap through their
rolling windows, which is why the embargo in cross-validation is needed even
when weighting is not. Overlap is computed per ticker: labels on different
tickers are different bets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


def _span_positions(
    close_index: pd.Index, touch_times: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bar positions of each label's start and end, and which labels are usable."""
    starts = close_index.get_indexer(touch_times.index)
    ends = close_index.get_indexer(pd.DatetimeIndex(touch_times))
    valid = (starts >= 0) & (ends > starts)

    return starts, ends, valid


def concurrency(close_index: pd.Index, touch_times: pd.Series) -> pd.Series:
    """How many labels are live on each bar.

    A label starting at t and touching at t1 is live on bars t+1 to t1: those
    are the bars whose returns make up its outcome.

    Args:
        close_index: Every bar of one ticker.
        touch_times: End date of each label, indexed by its start date. NaT
            means no label.

    Returns:
        Number of live labels per bar.
    """
    starts, ends, valid = _span_positions(close_index, touch_times)
    change = np.zeros(len(close_index) + 1)
    np.add.at(change, starts[valid] + 1, 1)
    np.add.at(change, ends[valid] + 1, -1)

    return pd.Series(np.cumsum(change[:-1]), index=close_index, name="concurrency")


def average_uniqueness(touch_times: pd.Series, concurrency: pd.Series) -> pd.Series:
    """Mean of 1 / concurrency over the bars each label spans.

    Args:
        touch_times: End date of each label, indexed by its start date.
        concurrency: Output of concurrency() on the same bars.

    Returns:
        Uniqueness per label, between 0 and 1. NaN where there is no label.
    """
    starts, ends, valid = _span_positions(concurrency.index, touch_times)
    inverse = np.where(concurrency > 0, 1 / concurrency.where(concurrency > 0), 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])

    uniqueness = np.full(len(touch_times), np.nan)
    s, e = starts[valid], ends[valid]
    uniqueness[valid] = (cumulative[e + 1] - cumulative[s + 1]) / (e - s)

    return pd.Series(uniqueness, index=touch_times.index, name="uniqueness")


def time_decay_weights(uniqueness: pd.Series, decay: float = 1.0) -> pd.Series:
    """Linear decay so older labels count less.

    The decay runs along cumulative uniqueness rather than calendar time, so a
    stretch of heavily overlapping labels does not age the sample faster. The
    newest label gets 1 and the oldest gets `decay`.

    Args:
        uniqueness: Per-label uniqueness, in time order.
        decay: Weight of the oldest label, in (0, 1]. 1 means no decay.

    Returns:
        A multiplier per label, strictly positive. NaN where uniqueness is NaN.

    Raises:
        ValueError: If decay is outside (0, 1], which would give zero or
            negative weights.
    """
    if not 0 < decay <= 1:
        raise ValueError(f"decay must be in (0, 1], got {decay}")

    clock = uniqueness.fillna(0).cumsum()
    total = clock.iloc[-1] if len(clock) else 0.0

    if total == 0 or decay == 1:
        return pd.Series(1.0, index=uniqueness.index).where(uniqueness.notna())

    factor = decay + (1 - decay) * clock / total

    return factor.where(uniqueness.notna())


def sample_weights(labels: pd.DataFrame, config: Config, decay: float = 1.0) -> pd.Series:
    """Weight for every label: uniqueness times time decay, per ticker.

    With decay = 1 the weights sum to the effective number of independent
    labels; non-overlapping labels each get 1 and n identical spans each get
    1/n. Decay below 1 lowers the older weights and so the total.

    Args:
        labels: Output of a labeller, indexed by (date, ticker), with touch_time.
        config: Loaded project configuration. Unused today; kept so every
            labelling step has the same signature for the pipeline.
        decay: Weight of the oldest label, in (0, 1].

    Returns:
        A weight per label, NaN where there is no label.
    """
    weights = []

    for ticker in labels.index.unique("ticker"):
        touch = labels.xs(ticker, level="ticker")["touch_time"]
        live = concurrency(touch.index, touch)
        uniqueness = average_uniqueness(touch, live)
        weight = uniqueness * time_decay_weights(uniqueness, decay)
        weights.append(pd.concat({ticker: weight}, names=["ticker", "date"]))

    out = pd.concat(weights).swaplevel().sort_index().reindex(labels.index)
    out.name = "weight"

    # The labels say which method made them; the config may name another.
    method = "triple_barrier" if "barrier" in labels.columns else "fixed_horizon"
    logger.info(
        "Sample weights (%s, decay %.2f): mean %.3f, %s effective of %s labels",
        method,
        decay,
        out.mean(),
        f"{out.sum():,.0f}",
        f"{out.notna().sum():,}",
    )

    return out
