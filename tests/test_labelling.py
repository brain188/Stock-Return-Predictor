"""Checks on labelling and sample weights.

The label is the one place future data is used on purpose, so its alignment is
tested exactly: the label at t is the return from t to t + horizon, and
nothing else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.labelling.fixed_horizon import forward_return, label_fixed_horizon, label_sign
from src.labelling.sample_weights import (
    average_uniqueness,
    concurrency,
    sample_weights,
    time_decay_weights,
)
from src.labelling.triple_barrier import (
    apply_triple_barrier,
    daily_volatility,
    label_triple_barrier,
)
from tests.conftest import make_bars

TICKERS = ["SPY", "AAA"]


def make_panel(n: int = 400, drift: float | None = None) -> pd.DataFrame:
    """A cleaned-panel-shaped frame for two synthetic tickers.

    Args:
        n: Bars per ticker.
        drift: Mean daily log return to impose on adj_close, if not the default.
    """
    frames = {}
    for seed, ticker in enumerate(TICKERS):
        bars = make_bars(n=n, seed=seed)
        if drift is not None:
            rng = np.random.default_rng(seed)
            bars["adj_close"] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
        frames[ticker] = bars

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two synthetic tickers on one calendar."""
    return make_panel()


@pytest.fixture
def small_config(config):
    """Config for the synthetic panel, reading adjusted prices."""
    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY"),
        data=dataclasses.replace(config.data, price_column="adj_close"),
    )


# --- Fixed horizon ----------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 5])
def test_forward_return_is_t_to_t_plus_horizon(horizon, bars):
    """The value at t is close[t + h] / close[t] - 1, exactly."""
    close = bars["adj_close"]
    forward = forward_return(close, horizon)

    assert forward.iloc[10] == pytest.approx(close.iloc[10 + horizon] / close.iloc[10] - 1)


@pytest.mark.parametrize("horizon", [1, 5])
def test_last_horizon_rows_are_nan(horizon, bars):
    """The final h rows have no future, so no label."""
    forward = forward_return(bars["adj_close"], horizon)

    assert forward.iloc[-horizon:].isna().all()
    assert forward.iloc[:-horizon].notna().all()


def test_label_sign_rule():
    """Strictly positive is up; zero and negative are not up; NaN stays NaN."""
    forward = pd.Series([0.01, 0.0, -0.02, np.nan])

    labels = label_sign(forward)

    assert labels.iloc[:3].tolist() == [1.0, -1.0, -1.0]
    assert np.isnan(labels.iloc[3])


def test_label_sign_threshold():
    """With a threshold, small gains are not counted as up."""
    labels = label_sign(pd.Series([0.001, 0.003]), threshold=0.002)

    assert labels.tolist() == [-1.0, 1.0]


def test_panel_labels_align_per_ticker(panel, small_config):
    """Each ticker's label uses its own next close, and touch_time is the next date."""
    labels = label_fixed_horizon(panel, small_config)
    aaa = panel.xs("AAA", level="ticker")["adj_close"]
    dates = aaa.index

    row = labels.loc[(dates[10], "AAA")]

    assert row["return"] == pytest.approx(aaa.iloc[11] / aaa.iloc[10] - 1)
    assert row["touch_time"] == dates[11]
    assert labels.index.equals(panel.index)


def test_panel_last_row_per_ticker_is_nan(panel, small_config):
    """Every ticker loses exactly its last row, no more."""
    labels = label_fixed_horizon(panel, small_config)

    assert labels["label"].isna().sum() == len(TICKERS)
    assert labels.loc[panel.index.get_level_values("date").max(), "label"].isna().all()


def test_class_balance_tracks_the_drift(small_config):
    """A strongly drifting series is labelled up on most days."""
    labels = label_fixed_horizon(make_panel(n=3_000, drift=0.003), small_config)

    assert labels["label"].eq(1).mean() > 0.55


# --- Triple barrier ---------------------------------------------------------


def series(values) -> pd.Series:
    """A price series on business days."""
    return pd.Series(values, index=pd.date_range("2010-01-04", periods=len(values), freq="B"))


def run(close: pd.Series, width: float, vertical: int = 5) -> pd.DataFrame:
    """Triple barrier on every bar with a constant barrier width."""
    barriers = pd.DataFrame({"upper": width, "lower": width}, index=close.index)

    return apply_triple_barrier(close, close.index, barriers, vertical)


def test_rising_series_hits_profit_barrier():
    """Straight up: the upper barrier, label +1, touched on the first bar past it."""
    close = series(100 * np.exp(0.01 * np.arange(30)))

    out = run(close, width=0.025).dropna()

    assert (out["barrier"] == "upper").all()
    assert (out["label"] == 1).all()
    assert out["touch_time"].iloc[0] == close.index[3]


def test_falling_series_hits_stop():
    """Straight down: the lower barrier, label -1."""
    close = series(100 * np.exp(-0.01 * np.arange(30)))

    out = run(close, width=0.025).dropna()

    assert (out["barrier"] == "lower").all()
    assert (out["label"] == -1).all()


def test_flat_series_hits_time_barrier():
    """No movement: zero volatility, so only the time limit can end the trade."""
    close = series(np.full(40, 100.0))
    vol = daily_volatility(close, 10)

    out = apply_triple_barrier(
        close, close.index, pd.DataFrame({"upper": vol, "lower": vol}), vertical=5
    ).dropna(subset=["label"])

    assert len(out)
    assert (out["barrier"] == "vertical").all()
    assert (out["return"] == 0).all()
    assert (out["label"] == -1).all()


def test_vertical_touch_is_holding_period_later():
    """A timed-out trade ends exactly `vertical` bars after it starts."""
    close = series(np.full(20, 100.0))

    out = run(close, width=0.0, vertical=5)

    assert out["touch_time"].iloc[0] == close.index[5]


def test_events_without_room_are_nan():
    """The last `vertical` events have no full holding period, so no label."""
    close = series(100 * np.exp(0.01 * np.arange(30)))

    out = run(close, width=0.5, vertical=5)

    assert out["label"].iloc[-5:].isna().all()
    assert out["label"].iloc[:-5].notna().all()


def test_barriers_scale_with_volatility(small_config):
    """Triple the volatility of a path and its labels do not change."""
    rng = np.random.default_rng(11)
    shocks = rng.normal(0, 0.01, 600)
    dates = pd.date_range("2005-01-03", periods=600, freq="B", name="date")

    def panel_from(scale: float) -> pd.DataFrame:
        bars = pd.DataFrame({"adj_close": 100 * np.exp(np.cumsum(scale * shocks))}, index=dates)
        return pd.concat({"SPY": bars}, names=["ticker", "date"]).swaplevel()

    calm = label_triple_barrier(panel_from(1.0), small_config)
    wild = label_triple_barrier(panel_from(3.0), small_config)

    pd.testing.assert_series_equal(calm["label"], wild["label"])
    pd.testing.assert_series_equal(calm["barrier"], wild["barrier"])
    assert wild["return"].abs().mean() > 2 * calm["return"].abs().mean()


def test_volatility_warmup_gives_no_label(panel, small_config):
    """Before the volatility estimate exists there is no barrier and no label."""
    labels = label_triple_barrier(panel, small_config)
    window = small_config.labelling.triple_barrier.volatility_window

    first = labels.xs("SPY", level="ticker")["label"]

    assert first.iloc[:window].isna().all()
    assert first.iloc[window:-10].notna().all()


def test_triple_barrier_shares_label_columns(panel, small_config):
    """Same columns and index as the fixed horizon labeller, plus barrier."""
    fixed = label_fixed_horizon(panel, small_config)
    barrier = label_triple_barrier(panel, small_config)

    assert list(barrier.columns[:3]) == list(fixed.columns)
    assert barrier.index.equals(fixed.index)


# --- Sample weights ---------------------------------------------------------


def spans(starts: list[int], ends: list[int], n: int = 20) -> tuple[pd.Index, pd.Series]:
    """A bar index and touch times for labels from bar s to bar e."""
    index = pd.date_range("2010-01-04", periods=n, freq="B")
    touch = pd.Series(index[ends], index=index[starts])

    return index, touch


def test_concurrency_counts_live_labels():
    """Two labels covering bars 1-3 and 2-4 overlap on bars 2 and 3."""
    index, touch = spans([0, 1], [3, 4], n=6)

    assert concurrency(index, touch).tolist() == [0, 1, 2, 2, 1, 0]


def test_non_overlapping_labels_get_weight_one():
    """Back-to-back one day labels never share a bar."""
    index, touch = spans(list(range(10)), list(range(1, 11)), n=11)

    uniqueness = average_uniqueness(touch, concurrency(index, touch))

    assert (uniqueness == 1).all()


@pytest.mark.parametrize("n", [2, 3, 5])
def test_fully_overlapping_labels_get_one_over_n(n):
    """n labels over exactly the same bars each get 1/n."""
    index = pd.date_range("2010-01-04", periods=10, freq="B")
    touch = pd.Series([index[6]] * n, index=pd.DatetimeIndex([index[1]] * n))

    uniqueness = average_uniqueness(touch, concurrency(index, touch))

    np.testing.assert_allclose(uniqueness, 1 / n)


def test_partial_overlap_hand_calculation():
    """Bars 1-3 and 2-4: each label has one bar alone and two shared, so 2/3."""
    index, touch = spans([0, 1], [3, 4], n=6)

    uniqueness = average_uniqueness(touch, concurrency(index, touch))

    np.testing.assert_allclose(uniqueness, [(1 + 0.5 + 0.5) / 3] * 2)


def test_time_decay():
    """No decay leaves 1s; decay gives the oldest `decay` and the newest 1."""
    uniqueness = pd.Series([1.0, 1.0, 1.0, 1.0])

    assert (time_decay_weights(uniqueness, 1.0) == 1).all()

    decayed = time_decay_weights(uniqueness, 0.5)

    assert decayed.iloc[-1] == pytest.approx(1.0)
    assert decayed.is_monotonic_increasing
    assert (decayed > 0).all()


def test_decay_outside_range_raises():
    """A decay of 0 or below would zero out old samples."""
    with pytest.raises(ValueError):
        time_decay_weights(pd.Series([1.0]), 0.0)


def test_fixed_horizon_weights_are_all_one(panel, small_config):
    """One day labels never overlap, so every weight is 1 and they sum to the count."""
    labels = label_fixed_horizon(panel, small_config)

    weights = sample_weights(labels, small_config)

    assert (weights.dropna() == 1).all()
    assert weights.sum() == labels["label"].notna().sum()
    assert weights.isna().equals(labels["label"].isna())


def test_triple_barrier_weights_positive_and_sum_to_effective_count(panel, small_config):
    """Overlapping labels: weights in (0, 1], summing to fewer than the label count."""
    labels = label_triple_barrier(panel, small_config)

    weights = sample_weights(labels, small_config).dropna()

    assert (weights > 0).all()
    assert (weights <= 1).all()
    assert weights.sum() < len(weights)
