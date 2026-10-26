"""The development period ends before the holdout, for features and labels alike.

The strongest check is invariance: change every price on or after the
holdout start and the development data must not change at all. If any
feature or label read a holdout price, it would.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.pipelines.evaluate import build_context
from src.pipelines.research import (
    BoundaryError,
    check_development_boundary,
    development_panel,
    prepare_research_data,
    with_overrides,
)
from tests.conftest import make_bars

TICKERS = ["SPY", "AAA", "BBB"]
HOLDOUT = "2008-06-02"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Three synthetic tickers running well past the holdout start."""
    frames = {}
    for seed, ticker in enumerate(TICKERS):
        bars = make_bars(n=1000, seed=seed)
        bars["return"] = bars["adj_close"].pct_change()
        frames[ticker] = bars

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def small_config(config):
    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS),
        validation=dataclasses.replace(config.validation, test_start=pd.Timestamp(HOLDOUT).date()),
    )


def shocked(panel: pd.DataFrame) -> pd.DataFrame:
    """The panel with every price and return on or after the holdout start changed."""
    out = panel.copy()
    later = out.index.get_level_values("date") >= pd.Timestamp(HOLDOUT)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        out.loc[later, column] = out.loc[later, column] * 3
    out.loc[later, "return"] = 0.5

    return out


def test_the_panel_is_cut_before_anything_is_built(panel, small_config):
    cut = development_panel(panel, small_config)

    assert cut.index.get_level_values("date").max() < pd.Timestamp(HOLDOUT)
    with pytest.raises(BoundaryError):
        development_panel(
            panel[panel.index.get_level_values("date") >= pd.Timestamp(HOLDOUT)], small_config
        )


@pytest.mark.parametrize("method", ["fixed_horizon", "triple_barrier"])
def test_no_label_is_dated_or_resolved_in_the_holdout(panel, small_config, method):
    data = prepare_research_data(panel, with_overrides(small_config, label_method=method))

    assert data.X.index.get_level_values("date").max() < pd.Timestamp(HOLDOUT)
    assert (data.labels["touch_time"] < pd.Timestamp(HOLDOUT)).all()


@pytest.mark.parametrize("method", ["fixed_horizon", "triple_barrier"])
def test_holdout_prices_cannot_change_development_data(panel, small_config, method):
    config = with_overrides(small_config, label_method=method)

    base = prepare_research_data(panel, config)
    moved = prepare_research_data(shocked(panel), config)

    pd.testing.assert_frame_equal(base.X, moved.X)
    pd.testing.assert_frame_equal(base.labels, moved.labels)
    pd.testing.assert_series_equal(base.weights, moved.weights)


def test_the_last_development_dates_lose_their_labels_rather_than_borrow_2020(panel, small_config):
    dates = panel.index.get_level_values("date").unique()
    last_dev = dates[dates < pd.Timestamp(HOLDOUT)][-1]

    fixed = prepare_research_data(panel, small_config)
    barrier = prepare_research_data(
        panel, with_overrides(small_config, label_method="triple_barrier")
    )
    holding = small_config.labelling.triple_barrier.max_holding_days

    # Fixed horizon: the last date has no next close inside the period.
    assert fixed.X.index.get_level_values("date").max() < last_dev
    # Triple barrier: the last max_holding_days dates have no full window.
    dev_dates = dates[dates < pd.Timestamp(HOLDOUT)]
    assert barrier.X.index.get_level_values("date").max() <= dev_dates[-holding - 1]


def test_a_label_reaching_the_holdout_is_refused(small_config):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2008-05-30"), "AAA")], names=["date", "ticker"]
    )
    labels = pd.DataFrame({"touch_time": [pd.Timestamp(HOLDOUT)]}, index=index)

    with pytest.raises(BoundaryError, match="resolved"):
        check_development_boundary(labels, small_config)


def test_the_evaluation_never_sees_a_holdout_price(panel, small_config):
    context = build_context(small_config, panel, fit_models=False)

    assert context.prices.index.max() < pd.Timestamp(HOLDOUT)
    assert context.returns.index.max() < pd.Timestamp(HOLDOUT)
    assert np.isfinite(context.prices.to_numpy()).all()
