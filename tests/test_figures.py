"""Checks on figure generation.

Every figure must come back as an object and land on disk, and a second run
must overwrite rather than fail, so one command always regenerates them all.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.reporting.figures import (
    build_all_figures,
    correlation_heatmap,
    price_and_cumulative_return,
    return_autocorrelation,
    return_distribution,
    rolling_volatility,
    volatility_regime_chart,
    volatility_regimes,
)
from tests.conftest import make_bars

TICKERS = ["SPY", "AAA", "BBB"]


@pytest.fixture
def cleaned() -> pd.DataFrame:
    """A cleaned-panel-shaped frame for three synthetic tickers."""
    frames = {}
    for seed, ticker in enumerate(TICKERS):
        bars = make_bars(n=800, seed=seed)
        bars["return"] = bars["adj_close"].pct_change()
        frames[ticker] = bars

    return pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()


@pytest.fixture
def small_config(config):
    """Config with a three-ticker universe and adjusted prices."""
    return dataclasses.replace(
        config,
        universe=dataclasses.replace(config.universe, tickers=TICKERS, benchmark="SPY"),
        data=dataclasses.replace(config.data, price_column="adj_close"),
    )


FIGURE_FUNCTIONS = [
    (price_and_cumulative_return, "eda_cumulative_returns.png"),
    (return_distribution, "eda_return_distribution_SPY.png"),
    (rolling_volatility, "eda_rolling_volatility_SPY.png"),
    (return_autocorrelation, "eda_autocorrelation_SPY.png"),
    (correlation_heatmap, "eda_correlation.png"),
    (volatility_regime_chart, "eda_volatility_regimes.png"),
]


@pytest.mark.parametrize(("function", "filename"), FIGURE_FUNCTIONS)
def test_returns_figure_and_writes_file(function, filename, cleaned, small_config, tmp_path):
    """Each function returns a Figure and writes its PNG."""
    figure = function(cleaned, small_config, out_dir=tmp_path)

    assert isinstance(figure, Figure)
    assert (tmp_path / filename).stat().st_size > 0


@pytest.mark.parametrize(("function", "filename"), FIGURE_FUNCTIONS)
def test_second_run_overwrites(function, filename, cleaned, small_config, tmp_path):
    """Running twice replaces the file instead of raising."""
    function(cleaned, small_config, out_dir=tmp_path)
    function(cleaned, small_config, out_dir=tmp_path)

    assert len(list(tmp_path.glob(filename))) == 1


def test_build_all_writes_every_figure(cleaned, small_config, tmp_path):
    """The one command produces all six figures."""
    figures = build_all_figures(cleaned, small_config, tmp_path)

    assert len(figures) == 6
    assert len(list(tmp_path.glob("*.png"))) == 6


def test_volatility_regimes_use_only_the_past():
    """Changing late returns leaves every earlier regime label unchanged."""
    returns = make_bars(n=1_000, seed=5)["adj_close"].pct_change()
    changed = returns.copy()
    changed.iloc[800:] *= 5

    before = volatility_regimes(returns)
    after = volatility_regimes(changed)

    pd.testing.assert_series_equal(before.iloc[:800], after.iloc[:800])


def test_volatility_regimes_labels_and_warmup():
    """Only the three labels appear, and the warm-up is unlabelled rather than guessed."""
    returns = pd.Series(np.random.default_rng(6).normal(0, 0.01, 1_000))

    regimes = volatility_regimes(returns, window=21, min_history=252)

    assert set(regimes.dropna()) <= {"low", "normal", "high"}
    assert regimes.iloc[: 21 + 252 - 1].isna().all()
